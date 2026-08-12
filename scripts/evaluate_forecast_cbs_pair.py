#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_theorist.autoscaler.study import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the paired faithful critical-batch censuses."
    )
    parser.add_argument("preregistration", type=Path)
    parser.add_argument("jiang_result", type=Path)
    parser.add_argument("completep_result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preregistration = json.loads(args.preregistration.read_text())
    jiang = json.loads(args.jiang_result.read_text())
    completep = json.loads(args.completep_result.read_text())
    gates = {
        "preregistration_passed": preregistration.get("status") == "passed",
        "jiang_census_passed": jiang.get("status") == "completed"
        and all(jiang.get("gates", {}).values()),
        "completep_census_passed": completep.get("status") == "completed"
        and all(completep.get("gates", {}).values()),
        "jiang_plan_matches": (
            jiang.get("plan", {}).get("fingerprint")
            == preregistration.get("jiang_plan_fingerprint")
        ),
        "completep_plan_matches": (
            completep.get("plan", {}).get("fingerprint")
            == preregistration.get("completep_plan_fingerprint")
        ),
        "neither_schedule_uses_extrapolated_batch": (
            jiang.get("batch_warmup", {}).get("uses_extrapolated_batch") is False
            and completep.get("batch_warmup", {}).get("uses_extrapolated_batch")
            is False
        ),
    }
    passed = all(gates.values())
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "gates": gates,
        "jiang": {
            "selected_eta_reference": jiang.get("pilot_selection", {}).get(
                "selected_eta_reference"
            ),
            "growth_fit": jiang.get("growth_fit"),
            "batch_warmup": jiang.get("batch_warmup"),
        },
        "completep": {
            "selected_eta_reference": completep.get("pilot_selection", {}).get(
                "selected_eta_reference"
            ),
            "growth_fit": completep.get("growth_fit"),
            "batch_warmup": completep.get("batch_warmup"),
        },
        "next_action": (
            "Freeze each architecture's batch-warmup and per-group Adam sqrt LR "
            "schedule into a fresh 100M ladder."
            if passed
            else "Do not launch a new ladder; expand the failed CBS bracket or LR grid."
        ),
        "certified_forecast": False,
        "interpretation": "This qualifies optimizer geometry; it is not a loss forecast.",
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
