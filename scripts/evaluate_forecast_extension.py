#!/usr/bin/env python3
"""Evaluate a completed one-seed extension against its frozen preregistration."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

from ai_theorist.autoscaler.study import atomic_write_json


def _read_object(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _require(label: str, condition: bool) -> None:
    if not condition:
        raise ValueError(f"extension result gate failed: {label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    preregistration = _read_object(args.run_root / "preregistration.json")
    plan = _read_object(args.run_root / "plan.json")
    shard = _read_object(args.run_root / "trial" / "ladder-shard-000.json")
    _require("preregistration status", preregistration.get("status") == "preregistered")
    unsigned_preregistration = dict(preregistration)
    preregistration_fingerprint = unsigned_preregistration.pop("fingerprint", None)
    _require(
        "preregistration fingerprint",
        preregistration_fingerprint == _fingerprint(unsigned_preregistration),
    )
    _require("shard completion", shard.get("status") == "completed")
    _require("ladder phase", shard.get("phase") == "ladder")
    _require(
        "plan fingerprint",
        shard.get("plan_fingerprint") == plan.get("fingerprint")
        == preregistration.get("extension_plan_fingerprint"),
    )
    _require(
        "selected learning rate",
        float(shard.get("selected_learning_rate"))
        == float(preregistration["selected_learning_rate"]),
    )
    records = shard.get("records")
    _require("exactly one revealed record", isinstance(records, list) and len(records) == 1)
    record = records[0]
    target = preregistration["target"]
    _require("target scale", record["metadata"]["scale"]["name"] == target["name"])
    _require("target parameters", int(record["parameter_count"]) == int(target["parameters"]))
    _require("target seed", int(record["seed"]) == int(preregistration["seed"]))
    _require("theory optimizer mode", record["metadata"]["optimizer_mode"] == "theory")
    _require(
        "dataset fingerprint",
        record["metadata"]["dataset_fingerprint"]
        == preregistration["extension_dataset_fingerprint"],
    )
    audit = record["metadata"]["optimizer_group_audit"]
    _require("complete optimizer group audit", bool(audit.get("complete")))
    _require("disjoint optimizer groups", bool(audit.get("disjoint")))
    _require("seven CompleteP groups", len(audit.get("groups", [])) == 7)
    _require(
        "full token horizon",
        int(record["nonpadding_tokens_seen"]) == int(target["presented_tokens"]),
    )
    observed_loss = float(record["final_validation_loss"])
    _require("finite validation loss", math.isfinite(observed_loss) and observed_loss > 0.0)

    frozen = preregistration["frozen_prediction"]
    exploratory = float(frozen["exploratory_prediction"])
    interval = [float(value) for value in frozen["prediction_interval_95"]]
    candidate_errors = [
        {
            "kind": row["kind"],
            "prediction": float(row["target_prediction"]),
            "relative_error": abs(float(row["target_prediction"]) / observed_loss - 1.0),
        }
        for row in frozen["candidate_fits"]
        if row.get("qualified")
    ]
    result: Dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "scientific_status": "one_seed_exploratory_prediction_test",
        "certified_forecast": False,
        "preregistration_fingerprint": preregistration_fingerprint,
        "plan_fingerprint": plan["fingerprint"],
        "dataset_fingerprint": preregistration["extension_dataset_fingerprint"],
        "target_parameters": int(target["parameters"]),
        "seed": int(record["seed"]),
        "observed_validation_loss": observed_loss,
        "frozen_exploratory_prediction": exploratory,
        "relative_prediction_error": abs(exploratory / observed_loss - 1.0),
        "prediction_interval_95": interval,
        "prediction_interval_contains_observation": interval[0]
        <= observed_loss
        <= interval[1],
        "candidate_family_errors": candidate_errors,
        "optimizer_group_audit": audit,
        "record": record,
    }
    result["fingerprint"] = _fingerprint(result)
    atomic_write_json(args.run_root / "result.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
