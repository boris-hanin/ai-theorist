#!/usr/bin/env python3
"""Compile and preregister the prospective 1B/10-TPP Jiang endpoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from ai_theorist.autoscaler.forecast_campaigns import compile_real_text_scaling_plan
from ai_theorist.autoscaler.study import atomic_write_json


TARGET_PARAMETERS = 1_008_531_456
TARGET_NONEMBEDDING_PARAMETERS = 906_295_296
TARGET_DEPTH = 8
TARGET_WIDTH = 3072
TARGET_HIDDEN_WIDTH = 12_288
TARGET_SEED = 11


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_config", type=Path)
    parser.add_argument("calibration_aggregate", type=Path)
    parser.add_argument("calibration_result", type=Path)
    parser.add_argument("continuation_manifest", type=Path)
    parser.add_argument("continuation_verification", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    calibration_config = _load(args.calibration_config)
    calibration_aggregate = _load(args.calibration_aggregate)
    calibration_result = _load(args.calibration_result)
    continuation_verification = _load(args.continuation_verification)
    if calibration_result.get("status") != "passed":
        raise ValueError("the 10-TPP calibration did not pass")
    if _sha256(args.calibration_aggregate) != calibration_result.get(
        "aggregate_sha256"
    ):
        raise ValueError("the calibration aggregate changed after evaluation")

    args.output_root.mkdir(parents=True, exist_ok=True)
    revealed = list(args.output_root.glob("**/ladder-shard-*.json"))
    if revealed:
        raise ValueError("a 1B outcome already exists; refusing to rewrite preregistration")

    parent_plan = compile_real_text_scaling_plan(calibration_config)
    selected_eta = float(calibration_result["selected_learning_rate"])
    config = deepcopy(dict(calibration_config))
    config["run_profile"] = "extension"
    config["dataset"]["token_stream_manifest_path"] = str(
        args.continuation_manifest.resolve()
    )
    config["ladder"]["target_parameters"] = [
        *config["ladder"]["target_parameters"],
        TARGET_PARAMETERS,
    ]
    config["ladder"]["depths"] = [*config["ladder"]["depths"], TARGET_DEPTH]
    config["ladder"]["target_forecasts"] = []
    config["ladder"]["heldout_scale_count"] = 1
    config["validation_interval_steps"] = 4809
    config["runtime"].update(
        {
            "distributed": "ddp",
            "num_processes": 8,
            "gradient_accumulation_steps": 32,
            "activation_checkpointing": False,
            "checkpoint_interval_steps": 0,
            "checkpoint_interval_seconds": 900,
            "resume": True,
        }
    )
    config["extension_contract"] = {
        "parent_plan_fingerprint": parent_plan["fingerprint"],
        "parent_dataset_fingerprint": parent_plan["dataset_identity"]["fingerprint"],
        "parent_aggregate_sha256": _sha256(args.calibration_aggregate),
        "selected_learning_rate": selected_eta,
        "target_scale": "S10",
        "target_seed": TARGET_SEED,
        "expected_target_parameters": TARGET_PARAMETERS,
    }
    config_path = args.output_root / "config.json"
    atomic_write_json(config_path, config)
    plan = compile_real_text_scaling_plan(config)
    target = plan["scales"][-1]
    expected_geometry = (
        TARGET_PARAMETERS,
        TARGET_NONEMBEDDING_PARAMETERS,
        TARGET_DEPTH,
        TARGET_WIDTH,
        TARGET_HIDDEN_WIDTH,
    )
    observed_geometry = (
        int(target["parameters"]),
        int(target["non_embedding_parameters"]),
        int(target["depth"]),
        int(target["width"]),
        int(target["hidden_width"]),
    )
    if observed_geometry != expected_geometry:
        raise ValueError(
            f"compiled 1B geometry {observed_geometry} != {expected_geometry}"
        )
    if float(target["rho_lm_over_d"]) != 32.0:
        raise ValueError("compiled 1B endpoint does not have exact rho=32")
    if abs(float(target["tokens_per_parameter"]) - 10.0) > 0.001:
        raise ValueError("compiled 1B endpoint is not at 10 TPP")
    plan_path = args.output_root / "plan.json"
    atomic_write_json(plan_path, plan)

    prospective = calibration_result["prospective_1b"]
    if (
        int(prospective["parameters"]) != TARGET_PARAMETERS
        or int(prospective["non_embedding_parameters"])
        != TARGET_NONEMBEDDING_PARAMETERS
        or prospective.get("outcome_seen") is not False
    ):
        raise ValueError("calibration result does not contain the sealed 1B forecast")

    gates = {
        "calibration_passed": calibration_result["status"] == "passed",
        "parent_plan_matches_calibration_aggregate": calibration_aggregate.get(
            "plan_fingerprint"
        )
        == parent_plan["fingerprint"],
        "verified_same_distribution_continuation": (
            continuation_verification.get("status") == "passed"
            and continuation_verification.get("base_fingerprint")
            == parent_plan["dataset_identity"]["fingerprint"]
            and continuation_verification.get("continuation_fingerprint")
            == plan["dataset_identity"]["fingerprint"]
        ),
        "same_pinned_tokenizer": plan["dataset_identity"]["tokenizer_fingerprint"]
        == parent_plan["dataset_identity"]["tokenizer_fingerprint"],
        "exact_1b_geometry": observed_geometry == expected_geometry,
        "exact_rho32": float(target["rho_lm_over_d"]) == 32.0,
        "constant_10_tpp": abs(float(target["tokens_per_parameter"]) - 10.0)
        <= 0.001,
        "frozen_eta_in_original_grid": selected_eta
        in {float(value) for value in plan["learning_rates"]},
        "zero_weight_decay": float(plan["optimizer_contract"]["weight_decay"])
        == 0.0,
        "eight_gpu_ddp": (
            plan["runtime"]["distributed"] == "ddp"
            and int(plan["runtime"]["num_processes"]) == 8
        ),
        "single_frozen_seed": plan["seeds"] == [TARGET_SEED],
        "outcome_unseen": not revealed,
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError("1B preregistration failed: " + ", ".join(failed))

    preregistration = {
        "schema_version": 1,
        "status": "preregistered",
        "scientific_status": (
            "prospective_single_seed_1b_10tpp_prediction_test; topology remains "
            "an engineering diagnostic rather than certification"
        ),
        "config_sha256": _sha256(config_path),
        "plan_fingerprint": plan["fingerprint"],
        "calibration_aggregate_sha256": _sha256(args.calibration_aggregate),
        "calibration_result_sha256": _sha256(args.calibration_result),
        "continuation_verification_sha256": _sha256(
            args.continuation_verification
        ),
        "selected_learning_rate": selected_eta,
        "selected_weight_decay_tau_ema": None,
        "task_id": f"ladder-S10-theory-eta{selected_eta:g}-seed{TARGET_SEED}",
        "target": target,
        "frozen_prediction": prospective,
        "topology_diagnostic": {
            "single_vs_eight_gpu_canary_steps": 3,
            "maximum_absolute_loss_delta": 0.005,
            "tolerance_predeclared_from_prior_300m_delta": True,
        },
        "gates": gates,
    }
    atomic_write_json(args.output_root / "preregistration.json", preregistration)
    print(json.dumps(preregistration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
