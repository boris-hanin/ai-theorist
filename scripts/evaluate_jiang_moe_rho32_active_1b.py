#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ai_theorist.autoscaler.scaling import fit_scaling_ensemble
from ai_theorist.autoscaler.study import atomic_write_json


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _fingerprint(value: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _selected_reference_record(
    tune_root: Path, selected_eta: float
) -> Mapping[str, Any]:
    matches: list[Mapping[str, Any]] = []
    for path in tune_root.glob("shard-*/trials/*.json"):
        row = _load(path)
        if (
            row.get("metadata", {}).get("scale", {}).get("name") == "S1"
            and row.get("metadata", {}).get("optimizer_mode") == "theory"
            and float(row.get("optimizer", {}).get("learning_rate", math.nan))
            == selected_eta
        ):
            matches.append(row)
    if len(matches) != 1:
        raise ValueError("selected S1 tuning record is missing or ambiguous")
    return matches[0]


def _ddp_record(root: Path, name: str) -> Mapping[str, Any]:
    shard = _load(root / "ladder" / name / "ladder-shard-000.json")
    records = list(shard.get("records", ()))
    if shard.get("status") != "completed" or len(records) != 1:
        raise ValueError(f"{name} shard is incomplete or ambiguous")
    return records[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "result.json"

    prereg_path = args.root / "preregistration.json"
    frozen_path = args.root / "frozen-prediction.json"
    plan_path = args.root / "plan-single.json"
    config_path = args.root / "config-single.json"
    selection_path = args.root / "reference-selection.json"
    topology_path = args.root / "topology" / "comparison.json"
    prereg = _load(prereg_path)
    frozen = _load(frozen_path)
    plan = _load(plan_path)
    config = _load(config_path)
    selection = _load(selection_path)
    topology = _load(topology_path)

    prereg_unsigned = dict(prereg)
    prereg_fingerprint = prereg_unsigned.pop("fingerprint", None)
    frozen_unsigned = dict(frozen)
    frozen_fingerprint = frozen_unsigned.pop("fingerprint", None)
    selected_eta = float(selection["selected_learning_rate"])
    records: dict[str, Mapping[str, Any]] = {
        "S1": _selected_reference_record(args.root / "tune", selected_eta)
    }
    scales = [dict(row) for row in plan["scales"]]
    for scale in scales[1:]:
        name = str(scale["name"])
        records[name] = _ddp_record(args.root, name)

    rows: list[dict[str, Any]] = []
    integrity: dict[str, bool] = {}
    for scale in scales:
        name = str(scale["name"])
        record = records[name]
        audit = record.get("metadata", {}).get("optimizer_group_audit", {})
        diagnostics = record.get("metadata", {}).get("diagnostics", {})
        row_ok = (
            record.get("metadata", {}).get("scale", {}).get("name") == name
            and int(record["parameter_count"]) == int(scale["parameters"])
            and int(record["nonpadding_tokens_seen"]) == int(scale["presented_tokens"])
            and float(record["optimizer"]["learning_rate"]) == selected_eta
            and float(record["optimizer"]["weight_decay"]) == 0.0
            and audit.get("complete") is True
            and audit.get("disjoint") is True
            and len(audit.get("groups", ())) == 8
            and math.isfinite(float(record["final_validation_loss"]))
            and float(record["final_validation_loss"]) > 0.0
            and float(diagnostics.get("maximum_absolute_expert_bias", 0.0)) > 0.0
        )
        integrity[name] = row_ok
        rows.append(
            {
                "scale": name,
                "L": int(scale["depth"]),
                "D": int(scale["width"]),
                "M": int(scale["hidden_width"]),
                "rho": float(scale["rho_lm_over_d"]),
                "active_non_embedding_parameters": int(
                    scale["active_non_embedding_parameters"]
                ),
                "active_parameters": int(scale["active_parameters"]),
                "total_parameters": int(scale["parameters"]),
                "presented_tokens": int(scale["presented_tokens"]),
                "tokens_per_active_parameter": float(
                    scale["tokens_per_active_parameter"]
                ),
                "validation_loss": float(record["final_validation_loss"]),
                "perplexity": math.exp(float(record["final_validation_loss"])),
                "data_parallel_replicas": int(record["data_parallel_replicas"]),
            }
        )

    endpoint = rows[-1]
    endpoint_name = str(scales[-1]["name"])
    frozen_fit = frozen["prediction"]
    predicted = float(frozen_fit["exploratory_prediction"])
    observed = float(endpoint["validation_loss"])
    holdout_relative_error = abs(predicted / observed - 1.0)
    interval = frozen_fit.get("prediction_interval_95")
    interval_contains = bool(
        interval is not None and float(interval[0]) <= observed <= float(interval[1])
    )

    target = float(plan["target_forecasts"][0])
    ladder = config["ladder"]
    final_fit = fit_scaling_ensemble(
        [float(row["active_non_embedding_parameters"]) for row in rows],
        [float(row["validation_loss"]) for row in rows],
        [0.0] * len(rows),
        target_size=target,
        maximum_extrapolation_factor=float(ladder["maximum_extrapolation_factor"]),
        maximum_family_spread=float(ladder["maximum_family_spread"]),
        maximum_backtest_relative_error=float(
            ladder["maximum_backtest_relative_error"]
        ),
        bootstrap_samples=400,
    )
    execution_gates = {
        "preregistration_integrity": prereg_fingerprint
        == _fingerprint(prereg_unsigned),
        "frozen_prediction_integrity": frozen_fingerprint
        == _fingerprint(frozen_unsigned),
        "prediction_was_frozen_before_reveal": frozen.get("status")
        == f"frozen_before_{endpoint_name}_reveal",
        "reference_eta_is_interior": selection.get(
            "learning_rate_optimum_is_interior"
        )
        is True,
        "single_vs_eight_gpu_topology_passed": topology.get("status") == "passed"
        and int(topology.get("ddp_replicas", 0)) == 8,
        "all_nine_rungs_are_complete_and_faithful": all(integrity.values()),
        "S1_is_single_gpu_tuning_evidence": rows[0]["data_parallel_replicas"] == 1,
        "nonreference_rungs_are_eight_gpu_ddp": all(
            row["data_parallel_replicas"] == 8 for row in rows[1:]
        ),
        "constant_rho32_L16_alpha2": all(
            row["L"] == 16
            and row["M"] == 2 * row["D"]
            and row["rho"] == 32.0
            for row in rows
        ),
        "endpoint_is_one_billion_active": (
            int(endpoint["active_parameters"])
            == int(prereg["endpoint"]["active_parameters"])
            and 1_000_000_000
            <= int(endpoint["active_parameters"])
            <= 1_030_000_000
        ),
        "no_corpus_repetition": all(
            row["presented_tokens"]
            <= int(plan["dataset_identity"]["training_tokens"])
            for row in rows
        ),
    }
    scientific_gates = {
        "endpoint_holdout_relative_error_within_15_percent": holdout_relative_error
        <= 0.15,
        "endpoint_holdout_interval_contains_observation": interval_contains,
        "prefix_scaling_ensemble_certified": frozen_fit.get("certified") is True,
        "full_scaling_ensemble_certified": final_fit.get("certified") is True,
    }
    execution_passed = all(execution_gates.values())
    payload = {
        "schema_version": 1,
        "status": "completed" if execution_passed else "failed",
        "scientific_status": (
            "single_seed_exploratory_scaling_law_with_preregistered_1b_active_holdout"
        ),
        "certified_forecast": False,
        "claim_restriction": prereg["claim_restriction"],
        "selected_learning_rate": selected_eta,
        "rows": rows,
        "holdout": {
            "scale": endpoint_name,
            "fit_parameter_axis": "active_non_embedding_parameters",
            "predicted_validation_loss": predicted,
            "observed_validation_loss": observed,
            "relative_prediction_error": holdout_relative_error,
            "prediction_interval_95": interval,
            "prediction_interval_contains_observation": interval_contains,
            "prefix_fit": frozen_fit,
        },
        "full_ladder_forecast": final_fit,
        "execution_gates": execution_gates,
        "scientific_gates": scientific_gates,
        "integrity_by_scale": integrity,
        "evidence_sha256": {
            "preregistration": _sha(prereg_path),
            "frozen_prediction": _sha(frozen_path),
            "reference_selection": _sha(selection_path),
            "topology_comparison": _sha(topology_path),
        },
    }
    payload["fingerprint"] = _fingerprint(payload)
    atomic_write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not execution_passed:
        failed = [name for name, passed in execution_gates.items() if not passed]
        raise SystemExit("1B-active MoE execution failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
