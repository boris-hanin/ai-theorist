#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Optional

from ai_theorist.autoscaler.forecast_campaigns import compile_real_text_scaling_plan
from ai_theorist.autoscaler.forecast_fleet import (
    ForecastFleetTask,
    build_forecast_fleet_tasks,
)
from ai_theorist.autoscaler.study import atomic_write_json


@dataclass(frozen=True)
class Campaign:
    name: str
    config_path: Path
    root: Path
    selected_learning_rate: Optional[float]
    selected_weight_decay_tau_ema: Optional[float]


@dataclass(frozen=True)
class WorkItem:
    campaign: Campaign
    task: ForecastFleetTask
    output: Path


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value)


def _load_campaign(name: str, config_path: Path, root: Path, phase: str) -> Campaign:
    selected_learning_rate = None
    selected_weight_decay_tau_ema = None
    if phase == "ladder":
        selection = json.loads((root / "reference-selection.json").read_text())
        selected_learning_rate = float(selection["selected_learning_rate"])
        raw_tau = selection.get("selected_weight_decay_tau_ema")
        selected_weight_decay_tau_ema = (
            None if raw_tau is None else float(raw_tau)
        )
    return Campaign(
        name=name,
        config_path=config_path,
        root=root,
        selected_learning_rate=selected_learning_rate,
        selected_weight_decay_tau_ema=selected_weight_decay_tau_ema,
    )


def _tasks(campaign: Campaign, phase: str) -> list[WorkItem]:
    config = json.loads(campaign.config_path.read_text())
    plan = compile_real_text_scaling_plan(config)
    tasks = build_forecast_fleet_tasks(
        plan,
        phase=phase,
        selected_learning_rate=campaign.selected_learning_rate,
        selected_weight_decay_tau_ema=campaign.selected_weight_decay_tau_ema,
        run_negative_control=bool(config.get("run_negative_control", True)),
    )
    return [
        WorkItem(
            campaign=campaign,
            task=task,
            output=(
                campaign.root
                / phase
                / "tasks"
                / f"{task.ordinal:04d}-{_safe_name(task.task_id)}"
            ),
        )
        for task in tasks
    ]


def _completed(item: WorkItem, phase: str) -> bool:
    manifest = item.output / f"{phase}-shard-000.json"
    if not manifest.is_file():
        return False
    payload = json.loads(manifest.read_text())
    assigned = payload.get("assigned_tasks", ())
    return (
        payload.get("status") == "completed"
        and len(assigned) == 1
        and assigned[0].get("task_id") == item.task.task_id
    )


def _round_robin(groups: list[list[WorkItem]]) -> list[WorkItem]:
    result: list[WorkItem] = []
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if index < len(group):
                result.append(group[index])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple forecast campaign tasks through one shared GPU pool."
    )
    parser.add_argument("--phase", choices=("tune", "ladder"), required=True)
    parser.add_argument(
        "--campaign",
        nargs=3,
        action="append",
        metavar=("NAME", "CONFIG", "ROOT"),
        required=True,
    )
    parser.add_argument("--cli", default=".venv-forecast/bin/ai-theorist-autoscale")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        metavar="CAMPAIGN=TASK_ID",
        help=(
            "Restrict one named campaign to explicit tasks; campaigns without "
            "filters still run their full phase. May be repeated."
        ),
    )
    args = parser.parse_args()

    gpus = [int(value) for value in args.gpus.split(",")]
    if len(gpus) != len(set(gpus)) or not gpus:
        raise ValueError("--gpus must contain unique GPU indices")
    campaigns = [
        _load_campaign(name, Path(config), Path(root), args.phase)
        for name, config, root in args.campaign
    ]
    known_campaigns = {campaign.name for campaign in campaigns}
    task_filters: dict[str, set[str]] = {}
    for value in args.task_id:
        campaign_name, separator, task_id = value.partition("=")
        if not separator or not campaign_name or not task_id:
            raise ValueError("--task-id must have the form CAMPAIGN=TASK_ID")
        if campaign_name not in known_campaigns:
            raise ValueError(f"--task-id names unknown campaign: {campaign_name}")
        task_filters.setdefault(campaign_name, set()).add(task_id)
    groups = []
    for campaign in campaigns:
        group = _tasks(campaign, args.phase)
        requested = task_filters.get(campaign.name)
        if requested is not None:
            known_tasks = {item.task.task_id for item in group}
            unknown = requested - known_tasks
            if unknown:
                raise ValueError(
                    f"unknown task IDs for {campaign.name}: "
                    + ", ".join(sorted(unknown))
                )
            group = [item for item in group if item.task.task_id in requested]
        groups.append(group)
    if args.phase == "tune":
        queue = _round_robin(groups)
    else:
        queue = sorted(
            [item for group in groups for item in group],
            key=lambda item: (-item.task.estimated_flops, item.task.ordinal),
        )

    args.status.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.status.with_suffix(args.status.suffix + ".lock")
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another task pool holds {lock_path}") from error

    total = len(queue)
    already_complete = [item for item in queue if _completed(item, args.phase)]
    pending = [item for item in queue if item not in already_complete]
    completed = len(already_complete)
    running: dict[int, tuple[int, WorkItem, subprocess.Popen[Any], Any]] = {}
    free_gpus = list(gpus)

    def write_status(status: str) -> None:
        atomic_write_json(
            args.status,
            {
                "schema_version": 1,
                "status": status,
                "phase": args.phase,
                "total_tasks": total,
                "completed_tasks": completed,
                "running_tasks": [
                    {
                        "pid": pid,
                        "gpu": gpu,
                        "campaign": item.campaign.name,
                        "task": item.task.to_dict(),
                        "output": str(item.output),
                    }
                    for pid, (gpu, item, _process, _log) in sorted(running.items())
                ],
                "pending_tasks": len(pending),
                "gpu_pool": gpus,
            },
        )

    try:
        while pending or running:
            while pending and free_gpus:
                gpu = free_gpus.pop(0)
                item = pending.pop(0)
                item.output.mkdir(parents=True, exist_ok=True)
                command = [
                    args.cli,
                    "forecast-shard",
                    str(item.campaign.config_path),
                    "--phase",
                    args.phase,
                    "--shard-index",
                    "0",
                    "--shard-count",
                    "1",
                    "--task-id",
                    item.task.task_id,
                    "--device",
                    "cuda",
                    "--output",
                    str(item.output),
                    "--progress-jsonl",
                ]
                if item.campaign.selected_learning_rate is not None:
                    command.extend(
                        [
                            "--selected-learning-rate",
                            str(item.campaign.selected_learning_rate),
                        ]
                    )
                if item.campaign.selected_weight_decay_tau_ema is not None:
                    command.extend(
                        [
                            "--selected-weight-decay-tau-ema",
                            str(item.campaign.selected_weight_decay_tau_ema),
                        ]
                    )
                log = (item.output / "worker.log").open("a", encoding="utf-8")
                environment = dict(os.environ)
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                running[process.pid] = (gpu, item, process, log)
            write_status("running")
            if not running:
                break
            time.sleep(1.0)
            for pid, (gpu, item, process, log) in list(running.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                log.close()
                del running[pid]
                free_gpus.append(gpu)
                free_gpus.sort()
                if return_code != 0 or not _completed(item, args.phase):
                    raise RuntimeError(
                        f"{item.campaign.name} task {item.task.task_id} failed on "
                        f"GPU {gpu}; see {item.output / 'worker.log'}"
                    )
                completed += 1
        write_status("completed")
    except BaseException:
        for _pid, (_gpu, _item, process, log) in running.items():
            process.terminate()
            log.close()
        write_status("failed")
        raise
    finally:
        lock_handle.close()


if __name__ == "__main__":
    main()
