#!/usr/bin/env python3
"""Merge common-design Chizat transfer shards and verify duplicate trials."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, Tuple

from chizat_lr_transfer import (
    Trial,
    edge_report,
    fixed_eta_report,
    json_safe,
    learning_progress_report,
)


INVARIANTS = ("axis", "dials", "D", "P", "alpha", "steps", "raw_lr_rule")


def _trial(payload: Dict[str, object]) -> Trial:
    checkpoints = {
        int(step): float("inf") if value is None else float(value)
        for step, value in payload["checkpoints"].items()
    }
    return Trial(
        dial=int(payload["dial"]),
        depth=int(payload["depth"]),
        width=int(payload["width"]),
        seed=int(payload["seed"]),
        normalized_eta=float(payload["normalized_eta"]),
        raw_learning_rate=float(payload["raw_learning_rate"]),
        rule=str(payload["rule"]),
        checkpoints=checkpoints,
        final_loss=(
            float("inf") if payload["final_loss"] is None else float(payload["final_loss"])
        ),
        diverged=bool(payload["diverged"]),
    )


def _key(trial: Trial) -> Tuple[str, int, float, int]:
    return trial.rule, trial.dial, trial.normalized_eta, trial.seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.inputs) < 2:
        raise ValueError("at least two shards are required")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    for field in INVARIANTS:
        values = [shard[field] for shard in shards]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"shard mismatch for {field}: {values}")

    by_key: Dict[Tuple[str, int, float, int], Trial] = {}
    duplicate_count = 0
    for shard in shards:
        rows = list(shard["trials"])
        if shard.get("negative_control"):
            rows.extend(shard["negative_control"]["trials"])
        for payload in rows:
            trial = _trial(payload)
            key = _key(trial)
            if key in by_key:
                if asdict(by_key[key]) != asdict(trial):
                    raise ValueError(f"cross-worker duplicate mismatch: {key}")
                duplicate_count += 1
            else:
                by_key[key] = trial

    trials = list(by_key.values())
    correct = [trial for trial in trials if trial.rule == "correct_LM"]
    controls = [trial for trial in trials if trial.rule != "correct_LM"]
    seeds = sorted({trial.seed for trial in correct})
    etas = sorted({trial.normalized_eta for trial in correct})
    transfer_eta = float(shards[0]["fixed_eta_transfer"]["normalized_eta"])
    exponent = float(shards[0]["fixed_eta_transfer"]["finite_size_exponent"])
    fixed = fixed_eta_report(
        correct,
        shards[0]["dials"],
        seeds,
        transfer_eta,
        finite_size_exponent=exponent,
    )
    progress = learning_progress_report(
        correct, shards[0]["dials"], seeds, transfer_eta
    )
    control = None
    if controls:
        rule = controls[0].rule
        control_report = fixed_eta_report(
            controls,
            shards[0]["dials"],
            seeds,
            transfer_eta,
            rule=rule,
            finite_size_exponent=exponent,
        )
        control_progress = learning_progress_report(
            controls,
            shards[0]["dials"],
            seeds,
            transfer_eta,
            rule=rule,
        )
        control = {
            "rule": rule,
            "rejected": not bool(control_progress["accepted"]),
            "fixed_eta_diagnostics": control_report,
            "learning_progress": control_progress,
            "trials": [asdict(trial) for trial in controls],
        }
    result = {
        **{field: shards[0][field] for field in INVARIANTS},
        "schema_version": 1,
        "experiment": "chizat_fixed_eta_transfer_and_stability_edge_merged",
        "source_files": [str(path) for path in args.inputs],
        "source_hosts": [shard["host"] for shard in shards],
        "cross_worker_duplicate_trials_verified": duplicate_count,
        "seeds": seeds,
        "transfer_verdict": {
            "accepted": bool(fixed["accepted"]) and bool(progress["accepted"]),
            "requires": ["fixed_eta_trajectory_settling", "nonzero_M0_learning_progress"],
        },
        "fixed_eta_transfer": fixed,
        "learning_progress": progress,
        "edge_of_stability": edge_report(correct, shards[0]["dials"], etas),
        "negative_control": control,
        "trials": [asdict(trial) for trial in correct],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "transfer_accepted": result["transfer_verdict"]["accepted"],
        "negative_control_rejected": None if control is None else control["rejected"],
        "duplicate_trials_verified": duplicate_count,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
