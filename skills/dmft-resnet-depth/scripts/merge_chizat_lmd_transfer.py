#!/usr/bin/env python3
"""Merge Chizat L/M/D shards and verify duplicated common-seed trials."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, Tuple

from chizat_lmd_transfer import (
    Shape,
    Trial,
    build_report,
    json_safe,
    validate_shapes,
)


def _trial(payload) -> Trial:
    return Trial(
        label=str(payload["label"]),
        L=int(payload["L"]),
        M=int(payload["M"]),
        D=int(payload["D"]),
        dial=float(payload["dial"]),
        seed=int(payload["seed"]),
        normalized_eta=float(payload["normalized_eta"]),
        rule=str(payload["rule"]),
        train_groups=str(payload["train_groups"]),
        raw_learning_rates={
            str(key): float(value) for key, value in payload["raw_learning_rates"].items()
        },
        checkpoints={
            int(step): float("inf") if value is None else float(value)
            for step, value in payload["checkpoints"].items()
        },
        final_loss=(
            float("inf") if payload["final_loss"] is None else float(payload["final_loss"])
        ),
        diverged=bool(payload["diverged"]),
    )


def _key(trial: Trial) -> Tuple[str, str, int]:
    return trial.rule, trial.label, trial.seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.inputs) < 2:
        raise ValueError("at least two shards are required")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    invariant_fields = (
        "shapes",
        "d0",
        "P",
        "steps",
        "normalized_eta",
        "reference_D_for_single_rate_control",
        "train_groups",
        "parameterization",
    )
    for field in invariant_fields:
        values = [shard[field] for shard in shards]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"shard mismatch for {field}")

    by_key: Dict[Tuple[str, str, int], Trial] = {}
    duplicate_count = 0
    for shard in shards:
        for payload in shard["trials"]:
            trial = _trial(payload)
            key = _key(trial)
            if key in by_key:
                if asdict(by_key[key]) != asdict(trial):
                    raise ValueError(f"cross-worker duplicate mismatch: {key}")
                duplicate_count += 1
            else:
                by_key[key] = trial

    trials = list(by_key.values())
    shapes = validate_shapes(
        [
            Shape(
                str(row["label"]),
                int(row["L"]),
                int(row["M"]),
                int(row["D"]),
                float(row["dial"]),
            )
            for row in shards[0]["shapes"]
        ]
    )
    seeds = sorted({trial.seed for trial in trials})
    rules = sorted({trial.rule for trial in trials})
    primary_rule = str(shards[0]["parameterization"]["primary_rule"])
    result = build_report(
        trials,
        shapes,
        seeds,
        eta=float(shards[0]["normalized_eta"]),
        steps=int(shards[0]["steps"]),
        d0=int(shards[0]["d0"]),
        P=int(shards[0]["P"]),
        reference_D=int(shards[0]["reference_D_for_single_rate_control"]),
        finite_size_exponent=float(shards[0]["fixed_eta_trajectory"]["finite_size_exponent"]),
        rules=rules,
        train_groups=str(shards[0]["train_groups"]),
        primary_rule=primary_rule,
        minimum_progress=float(
            shards[0]["learning_progress"].get("minimum_progress", 1e-3)
        ),
    )
    result.update(
        {
            "experiment": "chizat_joint_L_M_D_fixed_eta_transfer_merged",
            "source_files": [str(path) for path in args.inputs],
            "source_hosts": [shard["host"] for shard in shards],
            "cross_worker_duplicate_trials_verified": duplicate_count,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "transfer_accepted": result["transfer_verdict"]["accepted"],
                "duplicate_trials_verified": duplicate_count,
                "controls": {
                    key: value["rejected"]
                    for key, value in result["negative_controls"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
