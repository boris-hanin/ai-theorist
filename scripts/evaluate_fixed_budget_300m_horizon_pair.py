#!/usr/bin/env python3
"""Evaluate the preregistered 300M Jiang 1x/10x token-horizon pair."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

from ai_theorist.autoscaler.study import atomic_write_json


def _read(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    prereg = _read(args.root / "preregistration.json")
    topology_path = args.root / "topology" / "comparison.json"
    topology = _read(topology_path)
    one_plan = _read(args.root / "one-x" / "plan.json")
    ten_plan = _read(args.root / "ten-x" / "plan.json")
    one_shard = _read(args.root / "one-x" / "trial" / "ladder-shard-000.json")
    ten_shard = _read(args.root / "ten-x" / "trial" / "ladder-shard-000.json")

    unsigned = dict(prereg)
    prereg_fingerprint = unsigned.pop("fingerprint", None)
    one_records = list(one_shard.get("records", ()))
    ten_records = list(ten_shard.get("records", ()))
    one_record = one_records[0] if len(one_records) == 1 else {}
    ten_record = ten_records[0] if len(ten_records) == 1 else {}
    one_target = prereg["one_x"]["target"]
    ten_target = prereg["ten_x"]["target"]

    def audit_passes(record: Mapping[str, Any]) -> bool:
        audit = record.get("metadata", {}).get("optimizer_group_audit", {})
        return bool(audit.get("complete")) and bool(audit.get("disjoint"))

    gates = {
        "preregistration_integrity": prereg_fingerprint == _fingerprint(unsigned),
        "topology_qualification_passed": topology.get("status") == "passed"
        and int(topology.get("ddp_replicas", 0)) == 8,
        "plans_match_preregistration": one_plan.get("fingerprint")
        == prereg["one_x"]["plan_fingerprint"]
        and ten_plan.get("fingerprint") == prereg["ten_x"]["plan_fingerprint"],
        "both_shards_complete": one_shard.get("status") == "completed"
        and ten_shard.get("status") == "completed",
        "exactly_one_record_per_horizon": len(one_records) == len(ten_records) == 1,
        "same_geometry": one_record.get("parameter_count")
        == ten_record.get("parameter_count")
        == one_target["parameters"]
        == ten_target["parameters"],
        "same_seed": one_record.get("seed")
        == ten_record.get("seed")
        == prereg["seed"],
        "same_learning_rate": float(
            one_record.get("optimizer", {}).get("learning_rate", float("nan"))
        )
        == float(ten_record.get("optimizer", {}).get("learning_rate", float("nan")))
        == float(prereg["selected_learning_rate"]),
        "eight_data_parallel_replicas": one_record.get("data_parallel_replicas")
        == ten_record.get("data_parallel_replicas")
        == 8,
        "exact_token_horizons": one_record.get("nonpadding_tokens_seen")
        == one_target["presented_tokens"]
        and ten_record.get("nonpadding_tokens_seen")
        == ten_target["presented_tokens"]
        == 10 * one_target["presented_tokens"],
        "optimizer_groups_complete_and_disjoint": audit_passes(one_record)
        and audit_passes(ten_record),
        "finite_losses": all(
            math.isfinite(float(record.get("final_validation_loss", float("nan"))))
            for record in (one_record, ten_record)
        ),
    }
    passed = all(gates.values())
    one_loss = float(one_record.get("final_validation_loss", float("nan")))
    ten_loss = float(ten_record.get("final_validation_loss", float("nan")))
    frozen = prereg["frozen_one_x_prediction"]
    prediction = float(frozen["exploratory_prediction"])
    interval = [float(value) for value in frozen["prediction_interval_95"]]
    result: Dict[str, Any] = {
        "schema_version": 1,
        "status": "completed" if passed else "failed",
        "scientific_status": "one_seed_exploratory_300m_horizon_pair",
        "certified_forecast": False,
        "preregistration_fingerprint": prereg_fingerprint,
        "topology_qualification_sha256": _sha256(topology_path),
        "gates": gates,
        "one_x": {
            "presented_tokens": int(one_target["presented_tokens"]),
            "tokens_per_parameter": float(one_target["tokens_per_parameter"]),
            "observed_validation_loss": one_loss,
            "frozen_prediction": prediction,
            "relative_prediction_error": abs(prediction / one_loss - 1.0),
            "prediction_interval_95": interval,
            "prediction_interval_contains_observation": interval[0]
            <= one_loss
            <= interval[1],
            "record": one_record,
        },
        "ten_x": {
            "presented_tokens": int(ten_target["presented_tokens"]),
            "tokens_per_parameter": float(ten_target["tokens_per_parameter"]),
            "observed_validation_loss": ten_loss,
            "record": ten_record,
        },
        "horizon_comparison": {
            "validation_loss_delta_ten_x_minus_one_x": ten_loss - one_loss,
            "relative_loss_change": ten_loss / one_loss - 1.0,
            "perplexity_ratio_ten_x_over_one_x": math.exp(ten_loss - one_loss),
            "token_ratio": 10.0,
            "eta_retuned_at_ten_x": False,
        },
    }
    result["fingerprint"] = _fingerprint(result)
    atomic_write_json(args.root / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    if not passed:
        failed = [name for name, value in gates.items() if not value]
        raise ValueError("300M horizon pair failed gates: " + ", ".join(failed))


if __name__ == "__main__":
    main()
