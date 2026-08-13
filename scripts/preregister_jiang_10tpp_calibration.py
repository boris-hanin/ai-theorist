#!/usr/bin/env python3
"""Preregister the constant-10-TPP Jiang calibration before any trials run."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from ai_theorist.autoscaler.forecast_campaigns import compile_real_text_scaling_plan
from ai_theorist.autoscaler.jiang_chizat import JIANG_DENSE_REPORTED_LR_MULTIPLIERS
from ai_theorist.autoscaler.study import atomic_write_json


EXPECTED_PARAMETERS = (51_080_832, 106_984_192, 200_020_480)
EXPECTED_FORECAST_COORDINATES = (245_929_600, 906_295_296)


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("runtime_qualification", type=Path)
    parser.add_argument("known_300m_result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = _load(args.config)
    qualification = _load(args.runtime_qualification)
    # Bind the already-revealed result by digest without using its loss to
    # choose this campaign's model sizes, fit family, or acceptance gates.
    _load(args.known_300m_result)
    plan = compile_real_text_scaling_plan(config)
    scales = list(plan["scales"])
    reference = scales[int(plan["architecture_contract"]["reference_scale_index"])]
    batch_tokens = int(config["batch_examples"]) * int(
        config["architecture"]["context_length"]
    )
    qualified_fingerprints = {
        str(row["plan_fingerprint"])
        for row in qualification.get("campaigns", ())
    }
    forecast_geometries = list(config.get("forecast_target_geometries", ()))

    gates = {
        "runtime_qualification_passed": qualification.get("status") == "passed",
        "qualified_exact_plan": qualified_fingerprints == {plan["fingerprint"]},
        "forecast_profile": plan["run_profile"] == "forecast",
        "constant_10_tpp_with_only_batch_rounding": all(
            abs(int(row["presented_tokens"]) - 10 * int(row["parameters"]))
            <= batch_tokens
            for row in scales
        ),
        "fixed_global_batch_512": int(config["batch_examples"]) == 512,
        "requested_calibration_sizes_present": all(
            any(int(row["parameters"]) == target for row in scales)
            for target in EXPECTED_PARAMETERS
        ),
        "exact_rho32_everywhere": all(
            float(row["rho_lm_over_d"]) == 32.0 for row in scales
        ),
        "reference_geometry_exact": (
            int(reference["depth"]),
            int(reference["hidden_width"]),
            int(reference["width"]),
        )
        == (4, 2560, 320),
        "largest_200m_rung_is_internal_holdout": (
            int(scales[-1]["parameters"]) == 200_020_480
            and bool(scales[-1]["heldout"])
            and sum(bool(row["heldout"]) for row in scales) == 1
        ),
        "primary_axis_is_nonembedding_parameters": plan["fit_parameter_axis"]
        == "non_embedding_parameters",
        "forecast_coordinates_match_declared_geometries": (
            tuple(int(value) for value in plan["target_forecasts"])
            == EXPECTED_FORECAST_COORDINATES
            and tuple(
                int(row["non_embedding_parameters"])
                for row in forecast_geometries
            )
            == EXPECTED_FORECAST_COORDINATES
        ),
        "jiang_parameterization_complete": (
            plan["architecture_contract"]["tied_embeddings"] is True
            and plan["architecture_contract"]["residual_branch_scale"] == "1/L"
            and plan["architecture_contract"]["unembedding_forward_scale"]
            == "(D/D0)^(-1)"
            and plan["optimizer_contract"]["learning_rate_multipliers"]
            == JIANG_DENSE_REPORTED_LR_MULTIPLIERS
        ),
        "adamw_epsilon_and_zero_decay_explicit": (
            plan["optimizer_contract"]["name"] == "adamw"
            and float(plan["optimizer_contract"]["epsilon"]) == 1e-12
            and float(plan["optimizer_contract"]["weight_decay"]) == 0.0
        ),
        "single_frozen_seed": (
            list(plan["seeds"]) == [11]
            and plan.get("exploratory_single_seed") is True
        ),
        "bf16_flash_fused_runtime": (
            plan["runtime"]["precision"] == "bf16"
            and plan["runtime"]["attention_backend"] == "flash"
            and plan["optimizer_contract"]["fused"] is True
        ),
        "no_wrong_lr_control": int(plan["negative_control_trials"]) == 0,
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError("10-TPP preregistration failed: " + ", ".join(failed))

    payload = {
        "schema_version": 1,
        "status": "preregistered",
        "claim_scope": (
            "single-seed constant-10-TPP Jiang-Chizat calibration through 200M; "
            "the 200M rung is a hidden internal test, the already observed 300M "
            "result is retrospective only, and the 1B prediction is prospective"
        ),
        "adaptation_disclosure": {
            "known_300m_outcome_existed_before_preregistration": True,
            "known_300m_check_is_not_blind": True,
            "known_300m_result_sha256": _sha256(args.known_300m_result),
            "prospective_1b_outcome_unseen": True,
        },
        "selection_rule": (
            "minimum single-seed reference loss over the frozen dense eta grid; "
            "eta must be interior; exact zero AdamW decay remains fixed"
        ),
        "fit_rule": {
            "axis": "non_embedding_parameters",
            "families": ["pure_power_law", "floor_power_law", "broken_power_law"],
            "maximum_backtest_relative_error": 0.1,
            "maximum_family_spread": 0.08,
            "maximum_extrapolation_factor": 6.0,
            "internal_hidden_rung_parameters": 200_020_480,
        },
        "execution": {
            "scheduler": "dynamic_eight_gpu_single_process_task_pool",
            "gpu_count": 8,
            "seeds": [11],
        },
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "plan_fingerprint": plan["fingerprint"],
        "dataset_fingerprint": plan["dataset_identity"]["fingerprint"],
        "tokenizer_fingerprint": plan["dataset_identity"]["tokenizer_fingerprint"],
        "reference": reference,
        "scales": scales,
        "forecast_target_geometries": forecast_geometries,
        "learning_rates": plan["learning_rates"],
        "runtime_qualification_sha256": _sha256(args.runtime_qualification),
        "gates": gates,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
