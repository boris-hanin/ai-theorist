#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ai_theorist.autoscaler.study import atomic_write_json


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _load(path: Path, name: str) -> Mapping[str, Any]:
    return _object(json.loads(path.read_text()), name)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _selection(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return _object(result.get("reference_tuning"), "reference tuning")


def _heldout(result: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = result.get("hidden_scale_backtests")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("result must have exactly one hidden-scale backtest")
    return _object(rows[0], "hidden-scale backtest")


def _audit_gate(result: Mapping[str, Any], expected_groups: int) -> bool:
    records = result.get("records")
    if not isinstance(records, list) or not records:
        return False
    selected = _selection(result)
    selected_eta = float(selected["selected_learning_rate"])
    selected_tau = selected.get("selected_weight_decay_tau_ema")
    scales = result.get("scales")
    if not isinstance(scales, list):
        return False
    matching = [
        row
        for row in records
        if float(row["optimizer"]["learning_rate"]) == selected_eta
        and row["metadata"].get("weight_decay_tau_ema") == selected_tau
        and row["metadata"].get("optimizer_mode") == "theory"
    ]
    seen_scales = {row["metadata"]["scale"]["name"] for row in matching}
    seeds = {int(row["seed"]) for row in matching}
    complete_grid = {
        (str(scale["name"]), seed) for scale in scales for seed in seeds
    }
    observed_grid = {
        (str(row["metadata"]["scale"]["name"]), int(row["seed"]))
        for row in matching
    }
    return (
        seen_scales == {str(scale["name"]) for scale in scales}
        and len(seeds) >= 3
        and observed_grid == complete_grid
        and all(
        row["metadata"]["optimizer_group_audit"].get("complete") is True
        and row["metadata"]["optimizer_group_audit"].get("disjoint") is True
        and len(row["metadata"]["optimizer_group_audit"].get("groups", ()))
        == expected_groups
        and row["metadata"].get("software_contract", {}).get(
            "optimizer_backend"
        )
        == "fused_adamw"
        for row in matching
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the paired fixed-budget 100M scaling scans."
    )
    parser.add_argument("preregistration", type=Path)
    parser.add_argument("jiang_result", type=Path)
    parser.add_argument("completep_result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prereg = _load(args.preregistration, "preregistration")
    jiang = _load(args.jiang_result, "Jiang result")
    completep = _load(args.completep_result, "CompleteP result")
    jiang_selection = _selection(jiang)
    completep_selection = _selection(completep)
    jiang_holdout = _heldout(jiang)
    completep_holdout = _heldout(completep)
    jiang_last = _object(jiang["scales"][-1], "Jiang largest scale")
    completep_last = _object(completep["scales"][-1], "CompleteP largest scale")

    gates = {
        "preregistration_passed": prereg.get("status") == "preregistered"
        and all(_object(prereg.get("gates"), "preregistration gates").values()),
        "both_results_completed": jiang.get("status") == completep.get("status")
        == "completed",
        "plans_match_preregistration": jiang.get("plan_fingerprint")
        == prereg["jiang"]["plan_fingerprint"]
        and completep.get("plan_fingerprint")
        == prereg["completep"]["plan_fingerprint"],
        "dataset_and_tokenizer_match": jiang["dataset"]["fingerprint"]
        == completep["dataset"]["fingerprint"]
        == prereg["dataset_fingerprint"]
        and jiang["dataset"]["tokenizer_fingerprint"]
        == completep["dataset"]["tokenizer_fingerprint"]
        == prereg["tokenizer_fingerprint"],
        "both_reference_optima_valid": jiang_selection.get(
            "optimum_is_interior"
        )
        is True
        and completep_selection.get("optimum_is_interior") is True,
        "all_selected_group_contracts_complete": _audit_gate(jiang, 8)
        and _audit_gate(completep, 6),
        "both_hidden_predictions_pass": jiang_holdout.get("passed") is True
        and completep_holdout.get("passed") is True,
        "both_ladders_monotone_within_two_sem": all(
            row.get("accepted") is True for row in jiang["monotonicity_checks"]
        )
        and all(
            row.get("accepted") is True
            for row in completep["monotonicity_checks"]
        ),
        "both_primary_scans_forecastable": jiang.get("forecastable") is True
        and completep.get("forecastable") is True,
        "finite_largest_scale_losses": math.isfinite(
            float(jiang_last["mean_validation_loss"])
        )
        and math.isfinite(float(completep_last["mean_validation_loss"])),
    }
    passed = all(gates.values())
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "claim_scope": prereg["claim_scope"],
        "preregistration_sha256": _sha256(args.preregistration),
        "jiang": {
            "plan_fingerprint": jiang["plan_fingerprint"],
            "selected_learning_rate": jiang_selection[
                "selected_learning_rate"
            ],
            "selected_weight_decay_tau_ema": jiang_selection[
                "selected_weight_decay_tau_ema"
            ],
            "largest_scale": jiang_last,
            "hidden_backtest": jiang_holdout,
            "parameter_axis_backtests": jiang["parameter_axis_backtests"],
            "refusal_reasons": jiang["refusal_reasons"],
        },
        "completep": {
            "plan_fingerprint": completep["plan_fingerprint"],
            "selected_learning_rate": completep_selection[
                "selected_learning_rate"
            ],
            "selected_weight_decay_tau_ema": completep_selection[
                "selected_weight_decay_tau_ema"
            ],
            "selected_zero_weight_decay_endpoint": completep_selection.get(
                "selected_weight_decay_tau_ema"
            )
            is None,
            "largest_scale": completep_last,
            "hidden_backtest": completep_holdout,
            "parameter_axis_backtests": completep[
                "parameter_axis_backtests"
            ],
            "refusal_reasons": completep["refusal_reasons"],
        },
        "descriptive_architecture_comparison": {
            "validation_loss_completep_minus_jiang_at_largest_rungs": float(
                completep_last["mean_validation_loss"]
            )
            - float(jiang_last["mean_validation_loss"]),
            "parameter_ratio_completep_over_jiang": int(
                completep_last["parameters"]
            )
            / int(jiang_last["parameters"]),
            "interpretation": (
                "descriptive only: the ladders use different architectures and "
                "nearby, not identical, largest parameter counts"
            ),
        },
        "source_results": {
            "jiang": {
                "path": str(args.jiang_result),
                "sha256": _sha256(args.jiang_result),
            },
            "completep": {
                "path": str(args.completep_result),
                "sha256": _sha256(args.completep_result),
            },
        },
        "gates": gates,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit("fixed-budget pair failed gates: " + ", ".join(failed))


if __name__ == "__main__":
    main()
