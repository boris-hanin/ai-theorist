#!/usr/bin/env python3
"""Preregister exact rho=32 300M runs at 1x and 10x token horizons."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Dict, Mapping

from ai_theorist.autoscaler.forecast_campaigns import (
    bind_real_text_scaling_config,
    compile_real_text_scaling_plan,
)
from ai_theorist.autoscaler.scaling import fit_scaling_ensemble
from ai_theorist.autoscaler.study import atomic_write_json
from ai_theorist.autoscaler.tokenization import load_token_stream_manifest


TARGET_PARAMETERS = 299_177_600
TARGET_NON_EMBEDDING_PARAMETERS = 245_929_600
TARGET_DEPTH = 8
TARGET_WIDTH = 1_600
TARGET_HIDDEN_WIDTH = 6_400
TARGET_HEADS = 25
TARGET_RHO = 32.0
TARGET_SEED = 11
SELECTED_LEARNING_RATE = 0.03
ONE_X_STEPS = 1_144
TEN_X_STEPS = 11_440
BATCH_TOKENS = 512 * 512
ONE_X_TOKENS = ONE_X_STEPS * BATCH_TOKENS
TEN_X_TOKENS = TEN_X_STEPS * BATCH_TOKENS


def _read(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require(label: str, condition: bool) -> None:
    if not condition:
        raise ValueError(f"fixed-budget 300M horizon gate failed: {label}")


def _validation_contract(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "packing": manifest["packing"],
        "split": manifest["splits"]["validation"],
        "tokenizer_fingerprint": manifest["tokenizer_fingerprint"],
    }


def _extension_config(
    parent_config: Mapping[str, Any],
    parent_plan: Mapping[str, Any],
    parent_result_path: Path,
    *,
    optimizer_steps: int,
    validation_interval_steps: int,
) -> Dict[str, Any]:
    result = deepcopy(dict(parent_config))
    result.pop("cache_directory", None)
    result["run_profile"] = "extension"
    result["seeds"] = [TARGET_SEED]
    refinement = result["optimizer"].pop("learning_rate_refinement", None)
    if refinement is not None:
        result["optimizer"]["learning_rates"] = sorted(
            {
                *(
                    float(value)
                    for value in result["optimizer"]["learning_rates"]
                ),
                *(
                    float(value)
                    for value in refinement["learning_rates"]
                ),
            }
        )
    result["runtime"].update(
        distributed="ddp",
        num_processes=8,
        gradient_accumulation_steps=32,
    )
    result["validation_interval_steps"] = validation_interval_steps
    ladder = result["ladder"]
    ladder["target_parameters"] = [
        *ladder["target_parameters"],
        TARGET_PARAMETERS,
    ]
    ladder["depths"] = [*ladder["depths"], TARGET_DEPTH]
    ladder["optimizer_steps"] = optimizer_steps
    ladder["target_forecasts"] = [500_000_000]
    ladder["maximum_extrapolation_factor"] = 4.0
    result["extension_contract"] = {
        "parent_plan_fingerprint": parent_plan["fingerprint"],
        "parent_dataset_fingerprint": parent_plan["dataset_identity"]["fingerprint"],
        "parent_aggregate_sha256": _sha256(parent_result_path),
        "selected_learning_rate": SELECTED_LEARNING_RATE,
        "target_scale": f"S{len(parent_plan['scales']) + 1}",
        "target_seed": TARGET_SEED,
        "expected_target_parameters": TARGET_PARAMETERS,
    }
    return result


def _target_gate(plan: Mapping[str, Any], steps: int, tokens: int) -> Dict[str, Any]:
    target = dict(plan["scales"][-1])
    _require("target parameters", int(target["parameters"]) == TARGET_PARAMETERS)
    _require(
        "target non-embedding parameters",
        int(target["non_embedding_parameters"])
        == TARGET_NON_EMBEDDING_PARAMETERS,
    )
    _require("target depth", int(target["depth"]) == TARGET_DEPTH)
    _require("target residual width", int(target["width"]) == TARGET_WIDTH)
    _require(
        "target hidden width",
        int(target["hidden_width"]) == TARGET_HIDDEN_WIDTH,
    )
    _require("target heads", int(target["num_heads"]) == TARGET_HEADS)
    _require("target rho", float(target["rho_lm_over_d"]) == TARGET_RHO)
    _require("target rho exact", float(target["rho_relative_error"]) == 0.0)
    _require("target steps", int(target["optimizer_steps"]) == steps)
    _require("target tokens", int(target["presented_tokens"]) == tokens)
    _require("target held out", target["heldout"] is True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_run", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    parent_config = _read(args.parent_run / "bound-config.json")
    parent_plan = _read(args.parent_run / "plan.json")
    selection = _read(args.parent_run / "reference-selection.json")
    parent_result_path = args.parent_run / "aggregate" / "result.json"
    parent_result = _read(parent_result_path)
    manifest = load_token_stream_manifest(args.manifest, verify_files=True)

    scales = list(parent_plan["scales"])
    hidden = list(parent_result.get("hidden_scale_backtests", ()))
    _require("parent profile", parent_config.get("run_profile") == "fixed_budget_scan")
    _require(
        "Jiang architecture",
        parent_config["architecture"]["block_type"]
        == "jiang_chizat_transformer",
    )
    _require(
        "exact rho=32 parent ladder",
        bool(scales)
        and all(float(row["rho_lm_over_d"]) == TARGET_RHO for row in scales),
    )
    _require("completed parent", parent_result.get("status") == "completed")
    _require("forecastable parent", parent_result.get("forecastable") is True)
    _require(
        "passed parent hidden rung",
        bool(hidden) and all(bool(row.get("passed")) for row in hidden),
    )
    _require(
        "matched single-seed selection",
        selection.get("selection_mode")
        == "matched_single_seed_across_all_learning_rates",
    )
    _require(
        "eta=0.03",
        math.isclose(
            float(selection["selected_learning_rate"]),
            SELECTED_LEARNING_RATE,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
    )
    _require("interior LR", selection.get("optimum_is_interior") is True)
    _require(
        "zero weight decay",
        float(parent_config["optimizer"].get("weight_decay", 0.0)) == 0.0,
    )
    _require(
        "parent fixed budget",
        int(parent_plan["fixed_budget_contract"]["presented_tokens"])
        == ONE_X_TOKENS,
    )
    _require(
        "same dataset",
        manifest["fingerprint"] == parent_plan["dataset_identity"]["fingerprint"],
    )
    _require(
        "enough unique training tokens",
        int(parent_plan["dataset_identity"]["training_tokens"]) >= TEN_X_TOKENS,
    )

    one_template = _extension_config(
        parent_config,
        parent_plan,
        parent_result_path,
        optimizer_steps=ONE_X_STEPS,
        validation_interval_steps=143,
    )
    ten_template = _extension_config(
        parent_config,
        parent_plan,
        parent_result_path,
        optimizer_steps=TEN_X_STEPS,
        validation_interval_steps=1_430,
    )
    one_config, one_binding = bind_real_text_scaling_config(
        one_template, args.manifest
    )
    ten_config, ten_binding = bind_real_text_scaling_config(
        ten_template, args.manifest
    )
    one_plan = compile_real_text_scaling_plan(one_config)
    ten_plan = compile_real_text_scaling_plan(ten_config)
    one_target = _target_gate(one_plan, ONE_X_STEPS, ONE_X_TOKENS)
    ten_target = _target_gate(ten_plan, TEN_X_STEPS, TEN_X_TOKENS)
    geometry_fields = {
        "parameters",
        "non_embedding_parameters",
        "depth",
        "width",
        "hidden_width",
        "num_heads",
        "rho_lm_over_d",
        "rho_relative_error",
    }
    _require(
        "identical 1x/10x geometry",
        {key: one_target[key] for key in geometry_fields}
        == {key: ten_target[key] for key in geometry_fields},
    )
    _require(
        "exact tenfold tokens",
        int(ten_target["presented_tokens"])
        == 10 * int(one_target["presented_tokens"]),
    )

    axis = str(parent_result.get("fit_parameter_axis"))
    _require("non-embedding fit axis", axis == "non_embedding_parameters")
    fit = fit_scaling_ensemble(
        [row[axis] for row in parent_result["scales"]],
        [row["mean_validation_loss"] for row in parent_result["scales"]],
        [row["sem_validation_loss"] for row in parent_result["scales"]],
        target_size=float(one_target[axis]),
        maximum_extrapolation_factor=4.0,
        maximum_family_spread=float(
            parent_config["ladder"]["maximum_family_spread"]
        ),
        maximum_backtest_relative_error=float(
            parent_config["ladder"]["maximum_backtest_relative_error"]
        ),
        bootstrap_samples=int(parent_config.get("bootstrap_samples", 400)),
    )
    _require("finite frozen 1x prediction", fit.get("certified") is True)

    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    preregistration_core: Dict[str, Any] = {
        "schema_version": 1,
        "status": "preregistered",
        "scientific_status": "one_seed_exploratory_300m_horizon_pair",
        "code_commit": code_commit,
        "parent_plan_fingerprint": parent_plan["fingerprint"],
        "parent_aggregate_sha256": _sha256(parent_result_path),
        "dataset_fingerprint": parent_plan["dataset_identity"]["fingerprint"],
        "validation_contract_sha256": _fingerprint(
            _validation_contract(manifest)
        ),
        "selected_learning_rate": SELECTED_LEARNING_RATE,
        "seed": TARGET_SEED,
        "fit_parameter_axis": axis,
        "frozen_one_x_prediction": fit,
        "one_x": {
            "plan_fingerprint": one_plan["fingerprint"],
            "target": one_target,
            "schedule_horizon_steps": ONE_X_STEPS,
            "validation_interval_steps": 143,
        },
        "ten_x": {
            "plan_fingerprint": ten_plan["fingerprint"],
            "target": ten_target,
            "schedule_horizon_steps": TEN_X_STEPS,
            "validation_interval_steps": 1_430,
            "scaling_law_prediction": None,
            "interpretation": "learning-rate and schedule-horizon transfer measurement",
        },
        "pair_contract": {
            "same_model_geometry": True,
            "same_seed": True,
            "same_eta_and_parameter_group_rules": True,
            "same_global_batch": True,
            "same_validation_windows": True,
            "independent_schedules_with_matched_relative_warmup": True,
            "token_horizon_ratio": 10.0,
            "topology": "8_gpu_ddp_pending_qualification",
            "wrong_lr_control": False,
        },
        "claim_limitations": {
            "not_a_published_benchmark_dataset": True,
            "one_x_is_about_one_token_per_parameter": True,
            "ten_x_is_about_ten_tokens_per_parameter": True,
            "no_lr_retuning_at_ten_x": True,
            "not_certified_scaling_law_evidence_at_ten_x": True,
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    preregistration_path = args.output / "preregistration.json"
    if preregistration_path.is_file():
        preregistration = _read(preregistration_path)
        unsigned = dict(preregistration)
        existing_fingerprint = unsigned.pop("fingerprint", None)
        existing_core = dict(unsigned)
        existing_core.pop("created_at", None)
        _require(
            "immutable preregistration fingerprint",
            existing_fingerprint == _fingerprint(unsigned),
        )
        _require(
            "immutable preregistration contract",
            existing_core == preregistration_core,
        )
    else:
        preregistration = {
            **preregistration_core,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        preregistration["fingerprint"] = _fingerprint(preregistration)
        atomic_write_json(preregistration_path, preregistration)

    for label, config, plan, binding in (
        ("one-x", one_config, one_plan, one_binding),
        ("ten-x", ten_config, ten_plan, ten_binding),
    ):
        root = args.output / label
        root.mkdir(parents=True, exist_ok=True)
        for path, payload in (
            (root / "config.json", config),
            (root / "plan.json", plan),
            (root / "binding.json", binding),
        ):
            if path.is_file():
                _require(f"immutable {path}", _read(path) == payload)
            else:
                atomic_write_json(path, payload)
    print(json.dumps(preregistration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
