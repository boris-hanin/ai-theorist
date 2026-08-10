from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Dict, Mapping, Optional

from .batch_campaigns import (
    run_constant_tpp_campaign,
    run_transformer_batch_census,
)
from .pretraining import (
    PretrainingRuntimeSpec,
    compile_standard_pretraining_plan,
    run_standard_pretraining_batch_census,
)
from .pretraining_worker import PROGRESS_PREFIX
from .study import atomic_write_json


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]
CAMPAIGNS = {
    "transformer_census",
    "constant_tpp",
    "standard_pretraining_census",
}


def compile_campaign_plan(campaign: str, config: Mapping[str, Any]) -> Dict[str, Any]:
    if campaign not in CAMPAIGNS:
        raise ValueError(f"unknown campaign: {campaign}")
    if campaign == "standard_pretraining_census":
        return compile_standard_pretraining_plan(config)
    required = (
        ("architecture", "dataset", "scales", "batch_examples", "total_tokens", "optimizers")
        if campaign == "transformer_census"
        else ("architecture", "dataset", "scales", "optimizer", "tokens_per_parameter")
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"missing campaign field(s): {', '.join(missing)}")
    if campaign == "transformer_census":
        planned_grid_trials = sum(
            len(config["scales"])
            * len(config["batch_examples"])
            * len(optimizer["learning_rates"])
            * len(config.get("seeds", [11, 29]))
            for optimizer in config["optimizers"]
        )
    else:
        planned_grid_trials = len(config["scales"]) + 2
    return {
        "schema_version": 1,
        "campaign": campaign,
        "scale_count": len(config["scales"]),
        "planned_grid_trials": planned_grid_trials,
        "resumable": True,
    }


def compile_fsdp_launch(
    config_path: Path,
    output_path: Path,
    num_processes: int,
) -> list:
    if num_processes < 2:
        raise ValueError("FSDP launch requires at least two processes")
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={num_processes}",
        "-m",
        "ai_theorist.autoscaler.pretraining_worker",
        "--config",
        str(config_path),
        "--output",
        str(output_path),
    ]


def _run_fsdp(
    config: Mapping[str, Any],
    output_dir: Path,
    progress: ProgressCallback,
) -> Dict[str, Any]:
    runtime = PretrainingRuntimeSpec.from_dict(config.get("runtime", {}))
    config_path = output_dir / "worker-config.json"
    result_path = output_dir / "result.json"
    atomic_write_json(config_path, dict(config))
    command = compile_fsdp_launch(config_path, result_path, runtime.num_processes)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    tail = deque(maxlen=80)
    assert process.stdout is not None
    for line in process.stdout:
        stripped = line.rstrip()
        if stripped.startswith(PROGRESS_PREFIX):
            event = json.loads(stripped[len(PROGRESS_PREFIX) :])
            if progress is not None:
                progress(event)
        elif stripped:
            tail.append(stripped)
    return_code = process.wait()
    if return_code:
        raise RuntimeError(
            "FSDP worker failed with exit code "
            f"{return_code}: " + "\n".join(tail)
        )
    if not result_path.is_file():
        raise RuntimeError("FSDP worker completed without a result")
    with result_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_campaign_job(
    campaign: str,
    config: Mapping[str, Any],
    *,
    device: str,
    output_dir: Path,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    compile_campaign_plan(campaign, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    configured = dict(config)
    configured.setdefault("cache_directory", str(output_dir / "trials"))
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "campaign": campaign,
            "device": device,
            "config": configured,
        },
    )
    if campaign == "transformer_census":
        result = run_transformer_batch_census(
            configured, device=device, progress=progress
        )
    elif campaign == "constant_tpp":
        result = run_constant_tpp_campaign(
            configured, device=device, progress=progress
        )
    else:
        runtime = PretrainingRuntimeSpec.from_dict(configured.get("runtime", {}))
        if runtime.distributed == "fsdp":
            result = _run_fsdp(configured, output_dir, progress)
        else:
            result = run_standard_pretraining_batch_census(
                configured, device=device, progress=progress
            )
    atomic_write_json(output_dir / "result.json", result)
    return result
