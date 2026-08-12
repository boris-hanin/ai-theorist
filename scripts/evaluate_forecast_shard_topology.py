#!/usr/bin/env python3
"""Compare one-GPU and multi-GPU versions of one explicit forecast task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ai_theorist.autoscaler.forecast_qualification import (
    compare_forecast_topologies,
)
from ai_theorist.autoscaler.study import atomic_write_json


def _read(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("single_shard", type=Path)
    parser.add_argument("ddp_shard", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--maximum-loss-delta", type=float, default=0.001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    single = _read(args.single_shard)
    ddp = _read(args.ddp_shard)
    plan = _read(args.plan)

    def wrapper(shard: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": shard.get("status"),
            "campaign": "real_text_scaling_ladder",
            "dataset": plan["dataset_identity"],
            "architecture_contract": plan["architecture_contract"],
            "reference_tuning": {
                "selected_learning_rate": shard.get("selected_learning_rate")
            },
            "records": shard.get("records", []),
        }

    result = compare_forecast_topologies(
        wrapper(single),
        wrapper(ddp),
        maximum_absolute_loss_delta=args.maximum_loss_delta,
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
