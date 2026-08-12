#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from ai_theorist.autoscaler.forecast_critical_batch import (
    ForecastCriticalBatchTask,
    build_forecast_critical_batch_tasks,
    compile_forecast_critical_batch_plan,
)
from ai_theorist.autoscaler.study import atomic_write_json


@dataclass(frozen=True)
class Campaign:
    name: str
    config_path: Path
    root: Path
    plan_fingerprint: str
    selected_eta_multiplier: float | None


@dataclass(frozen=True)
class WorkItem:
    campaign: Campaign
    task: ForecastCriticalBatchTask


def _campaign(name: str, config_path: Path, root: Path, phase: str) -> Campaign:
    config = json.loads(config_path.read_text())
    plan = compile_forecast_critical_batch_plan(config)
    selected = None
    if phase in {"baseline", "branch"}:
        selection_path = root / "pilot-selection.json"
        if not selection_path.is_file():
            raise ValueError(f"missing pilot selection for {name}: {selection_path}")
        selection = json.loads(selection_path.read_text())
        if (
            selection.get("plan_fingerprint") != plan["fingerprint"]
            or not selection.get("optimum_is_interior")
        ):
            raise ValueError(f"{name} pilot selection is not an interior match")
        selected = float(selection["selected_eta_multiplier"])
    return Campaign(name, config_path, root, plan["fingerprint"], selected)


def _complete(item: WorkItem) -> bool:
    path = item.campaign.root / item.task.phase / item.task.task_id / "result.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text())
    return (
        payload.get("plan_fingerprint") == item.campaign.plan_fingerprint
        and payload.get("task", {}).get("task_id") == item.task.task_id
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
        description="Run faithful critical-batch tasks through one shared GPU pool."
    )
    parser.add_argument(
        "--phase", choices=("pilot", "baseline", "branch"), required=True
    )
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
    args = parser.parse_args()

    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must contain unique GPU indices")
    campaigns = [
        _campaign(name, Path(config), Path(root), args.phase)
        for name, config, root in args.campaign
    ]
    groups = [
        [
            WorkItem(campaign, task)
            for task in build_forecast_critical_batch_tasks(
                compile_forecast_critical_batch_plan(
                    json.loads(campaign.config_path.read_text())
                ),
                phase=args.phase,
            )
        ]
        for campaign in campaigns
    ]
    queue = _round_robin(groups)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.status.with_suffix(args.status.suffix + ".lock")
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another pool holds {lock_path}") from error

    completed = sum(_complete(item) for item in queue)
    pending = [item for item in queue if not _complete(item)]
    total = len(queue)
    free_gpus = list(gpus)
    running: dict[int, tuple[int, WorkItem, subprocess.Popen[Any], Any]] = {}

    def write_status(status: str) -> None:
        atomic_write_json(
            args.status,
            {
                "schema_version": 1,
                "status": status,
                "phase": args.phase,
                "total_tasks": total,
                "completed_tasks": completed,
                "pending_tasks": len(pending),
                "running_tasks": [
                    {
                        "pid": pid,
                        "gpu": gpu,
                        "campaign": item.campaign.name,
                        "task": item.task.to_dict(),
                    }
                    for pid, (gpu, item, _process, _log) in sorted(running.items())
                ],
                "gpu_pool": gpus,
            },
        )

    try:
        while pending or running:
            while pending and free_gpus:
                gpu = free_gpus.pop(0)
                item = pending.pop(0)
                log_path = (
                    item.campaign.root
                    / item.task.phase
                    / item.task.task_id
                    / "worker.log"
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    args.cli,
                    "forecast-cbs-task",
                    str(item.campaign.config_path),
                    "--phase",
                    item.task.phase,
                    "--seed",
                    str(item.task.seed),
                    "--device",
                    "cuda",
                    "--root",
                    str(item.campaign.root),
                ]
                if item.task.eta_multiplier is not None:
                    command.extend(["--eta-multiplier", str(item.task.eta_multiplier)])
                if item.task.checkpoint_tokens is not None:
                    command.extend(
                        ["--checkpoint-tokens", str(item.task.checkpoint_tokens)]
                    )
                if item.task.batch_examples is not None:
                    command.extend(["--batch-examples", str(item.task.batch_examples)])
                if item.campaign.selected_eta_multiplier is not None:
                    command.extend(
                        [
                            "--selected-eta-multiplier",
                            str(item.campaign.selected_eta_multiplier),
                        ]
                    )
                log = log_path.open("a", encoding="utf-8")
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
                if return_code or not _complete(item):
                    raise RuntimeError(
                        f"{item.campaign.name} task {item.task.task_id} failed on "
                        f"GPU {gpu}; see "
                        f"{item.campaign.root / item.task.phase / item.task.task_id / 'worker.log'}"
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
