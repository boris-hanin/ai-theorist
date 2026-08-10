#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

from ai_theorist.autoscaler.study import atomic_write_json
from ai_theorist.autoscaler.transfer_campaign import (
    analyze_followup_trials,
    analyze_lr_trials,
    campaign_fingerprint,
    compile_campaign_plan,
    load_trial_rows,
    run_campaign_phase,
)


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _progress(event: Dict[str, Any]) -> None:
    print(json.dumps(event, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the gated hard-task MLP+Adam transfer campaign"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("config", type=Path)
    compile_parser.add_argument(
        "--phase",
        choices=("lr", "lr-extension", "batch", "batch-extension", "horizon"),
        default="lr",
    )
    compile_parser.add_argument("--analysis", type=Path)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument(
        "--phase",
        choices=("lr", "lr-extension", "batch", "batch-extension", "horizon"),
        required=True,
    )
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--shard-count", type=int, default=1)
    run_parser.add_argument("--analysis", type=Path)
    run_parser.add_argument("--only-scale", action="append")
    run_parser.add_argument("--only-seed", type=int, action="append")

    analyze_lr_parser = subparsers.add_parser("analyze-lr")
    analyze_lr_parser.add_argument("config", type=Path)
    analyze_lr_parser.add_argument("--trials", type=Path, action="append", required=True)
    analyze_lr_parser.add_argument("--output", type=Path, required=True)

    analyze_followup_parser = subparsers.add_parser("analyze-followup")
    analyze_followup_parser.add_argument("config", type=Path)
    analyze_followup_parser.add_argument("--phase", choices=("batch", "horizon"), required=True)
    analyze_followup_parser.add_argument("--trials", type=Path, action="append", required=True)
    analyze_followup_parser.add_argument("--output", type=Path, required=True)

    arguments = parser.parse_args()
    config = _load(arguments.config)
    if arguments.command == "compile":
        analysis = _load(arguments.analysis) if arguments.analysis else None
        print(json.dumps(compile_campaign_plan(config, arguments.phase, analysis=analysis), indent=2))
        return 0
    if arguments.command == "run":
        analysis = _load(arguments.analysis) if arguments.analysis else None
        result = run_campaign_phase(
            config,
            arguments.phase,
            arguments.output,
            device=arguments.device,
            shard_index=arguments.shard_index,
            shard_count=arguments.shard_count,
            analysis=analysis,
            progress=_progress,
            only_scales=arguments.only_scale,
            only_seeds=arguments.only_seed,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    fingerprint = campaign_fingerprint(config)
    rows = load_trial_rows(arguments.trials, fingerprint)
    if arguments.command == "analyze-lr":
        result = analyze_lr_trials(config, rows)
    else:
        result = analyze_followup_trials(config, rows, arguments.phase)
    atomic_write_json(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
