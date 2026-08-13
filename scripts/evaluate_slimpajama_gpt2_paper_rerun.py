#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping

from ai_theorist.autoscaler.study import atomic_write_json


def _load(path: Path, label: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the SlimPajama/GPT-2 paper-coordinate rerun."
    )
    parser.add_argument("preregistration", type=Path)
    parser.add_argument("jiang_result", type=Path)
    parser.add_argument("anchor_shards", nargs=1, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prereg = _load(args.preregistration, "preregistration")
    jiang = _load(args.jiang_result, "Jiang aggregate")
    anchor_rows = []
    anchor_hashes = []
    for path in args.anchor_shards:
        shard = _load(path, f"CompleteP anchor shard {path}")
        records = shard.get("records")
        if not isinstance(records, list) or len(records) != 1:
            raise ValueError(f"anchor shard must contain exactly one record: {path}")
        record = records[0]
        task = shard.get("assigned_tasks", [None])[0]
        if not isinstance(record, Mapping) or not isinstance(task, Mapping):
            raise ValueError("anchor shard record/task is malformed")
        anchor_rows.append(
            {
                "seed": int(record["seed"]),
                "validation_loss": float(record["final_validation_loss"]),
                "run_id": str(record["run_id"]),
                "task_id": str(task["task_id"]),
                "wall_time_seconds": float(record["wall_time_seconds"]),
            }
        )
        anchor_hashes.append({"path": str(path), "sha256": _sha256(path)})
    anchor_rows.sort(key=lambda row: row["seed"])
    anchor_losses = [row["validation_loss"] for row in anchor_rows]
    expected_anchor_tasks = {
        "tune-S1-theory-eta0.00390625-seed11",
    }
    gates = {
        "preregistration_passed": prereg.get("status") == "preregistered"
        and all(prereg.get("gates", {}).values()),
        "jiang_plan_matches_preregistration": jiang.get("plan_fingerprint")
        == prereg["jiang"]["plan_fingerprint"],
        "jiang_aggregate_completed": jiang.get("status") == "completed",
        "jiang_has_one_hidden_backtest": isinstance(
            jiang.get("hidden_scale_backtests"), list
        )
        and len(jiang["hidden_scale_backtests"]) == 1,
        "anchor_plan_matches_preregistration": all(
            _load(path, "anchor shard").get("plan_fingerprint")
            == prereg["completep_anchor"]["plan_fingerprint"]
            for path in args.anchor_shards
        ),
        "anchor_exact_published_eta_tasks": {
            row["task_id"] for row in anchor_rows
        }
        == expected_anchor_tasks,
        "anchor_exact_fixed_seed": [row["seed"] for row in anchor_rows] == [11],
        "all_losses_finite": all(math.isfinite(value) for value in anchor_losses),
    }
    payload = {
        "schema_version": 1,
        "status": "passed" if all(gates.values()) else "failed",
        "claim_scope": prereg["claim_scope"],
        "jiang": {
            "result": str(args.jiang_result),
            "result_sha256": _sha256(args.jiang_result),
            "selected_learning_rate": jiang.get("reference_tuning", {}).get(
                "selected_learning_rate"
            ),
            "scales": jiang.get("scales"),
            "hidden_scale_backtests": jiang.get("hidden_scale_backtests"),
            "forecastable": jiang.get("forecastable"),
            "refusal_reasons": jiang.get("refusal_reasons"),
        },
        "completep_paper_anchor": {
            "geometry": {"width": 256, "depth": 2},
            "learning_rate": 0.00390625,
            "seed_results": anchor_rows,
            "mean_validation_loss": mean(anchor_losses),
            "sample_standard_deviation": (
                stdev(anchor_losses) if len(anchor_losses) > 1 else None
            ),
            "mean_perplexity": math.exp(mean(anchor_losses)),
            "source_files": anchor_hashes,
        },
        "interpretation": (
            "The CompleteP anchor calibrates the runtime against the paper's literal "
            "architecture and training coordinates using the preserved 6B sample of "
            "SlimPajama. Jiang raw loss is reported on the identical token stream "
            "and optimization budget, but is not treated as a numerical reproduction "
            "of CompleteP because its architecture is deliberately different and the "
            "paper's exact example order was not published."
        ),
        "gates": gates,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise SystemExit("paper rerun evaluation failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
