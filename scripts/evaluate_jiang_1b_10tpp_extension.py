#!/usr/bin/env python3
"""Reveal and score the preregistered 1B/10-TPP Jiang endpoint."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ai_theorist.autoscaler.study import atomic_write_json


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preregistration", type=Path)
    parser.add_argument("shard", type=Path)
    parser.add_argument("topology_comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    preregistration = _load(args.preregistration)
    shard = _load(args.shard)
    topology = _load(args.topology_comparison)
    if preregistration.get("status") != "preregistered":
        raise ValueError("1B preregistration is invalid")
    records = list(shard.get("records", ()))
    if shard.get("status") != "completed" or len(records) != 1:
        raise ValueError("1B shard is incomplete or ambiguous")
    record = records[0]
    target = preregistration["target"]
    frozen = preregistration["frozen_prediction"]
    observed = float(record["final_validation_loss"])
    predicted = float(frozen["predicted_validation_loss"])
    interval = [float(value) for value in frozen["calibrated_prediction_interval_95"]]
    relative_error = abs(predicted / observed - 1.0)

    gates = {
        "shard_complete": shard["status"] == "completed",
        "task_matches_preregistration": (
            len(shard.get("assigned_tasks", ())) == 1
            and shard["assigned_tasks"][0]["task_id"]
            == preregistration["task_id"]
        ),
        "geometry_matches_preregistration": (
            int(record["parameter_count"]) == int(target["parameters"])
            and int(record["metadata"]["scale"]["non_embedding_parameters"])
            == int(target["non_embedding_parameters"])
            and int(record["depth"]) == int(target["depth"])
            and int(record["width"]) == int(target["width"])
        ),
        "tokens_match_preregistration": int(record["total_tokens"])
        == int(target["presented_tokens"]),
        "frozen_eta_applied": float(record["optimizer"]["learning_rate"])
        == float(preregistration["selected_learning_rate"]),
        "zero_decay_applied": float(record["optimizer"]["weight_decay"]) == 0.0,
        "optimizer_groups_complete_and_disjoint": (
            record["metadata"]["optimizer_group_audit"]["complete"] is True
            and record["metadata"]["optimizer_group_audit"]["disjoint"] is True
        ),
        "finite_loss": math.isfinite(observed) and observed > 0.0,
        "topology_diagnostic_passed": topology.get("status") == "passed",
    }
    payload = {
        "schema_version": 1,
        "status": "completed" if all(gates.values()) else "failed",
        "scientific_status": "prospective_single_seed_1b_10tpp_prediction_test",
        "preregistration_sha256": _sha256(args.preregistration),
        "shard_sha256": _sha256(args.shard),
        "predicted_validation_loss": predicted,
        "observed_validation_loss": observed,
        "relative_prediction_error": relative_error,
        "prediction_interval_95": interval,
        "prediction_interval_contains_observation": interval[0]
        <= observed
        <= interval[1],
        "record": record,
        "topology_comparison": topology,
        "gates": gates,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
