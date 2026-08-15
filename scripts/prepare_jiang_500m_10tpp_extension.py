#!/usr/bin/env python3
"""Compile and preregister an adaptive 500M/10-TPP Jiang endpoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ai_theorist.autoscaler.forecast_campaigns import compile_real_text_scaling_plan
from ai_theorist.autoscaler.scaling import fit_scaling_ensemble
from ai_theorist.autoscaler.study import atomic_write_json


TARGET_PARAMETERS = 498_723_456
TARGET_NONEMBEDDING_PARAMETERS = 428_436_096
TARGET_DEPTH = 8
TARGET_WIDTH = 2_112
TARGET_HIDDEN_WIDTH = 8_448
TARGET_SEED = 11
SOURCE_1B_PARAMETERS = 1_008_531_456
KNOWN_300M_PARAMETERS = 299_177_600
KNOWN_300M_NONEMBEDDING_PARAMETERS = 245_929_600


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _freeze_prediction(
    aggregate: Mapping[str, Any], known_300m: Mapping[str, Any]
) -> dict[str, Any]:
    if aggregate.get("status") != "completed":
        raise ValueError("the 10-TPP calibration aggregate is incomplete")
    if aggregate.get("fit_parameter_axis") != "non_embedding_parameters":
        raise ValueError("the calibration must fit non-embedding parameters")
    rows = list(aggregate.get("scales", ()))
    if len(rows) < 5:
        raise ValueError("the calibration ladder is too short")
    sizes = [float(row["non_embedding_parameters"]) for row in rows]
    losses = [float(row["mean_validation_loss"]) for row in rows]
    sems = [float(row["sem_validation_loss"]) for row in rows]

    ten_x = known_300m.get("ten_x")
    if not isinstance(ten_x, Mapping):
        raise ValueError("the bound 300M result has no 10-TPP outcome")
    record = ten_x.get("record")
    if not isinstance(record, Mapping):
        raise ValueError("the bound 300M 10-TPP record is missing")
    scale = record.get("metadata", {}).get("scale", {})
    if (
        int(record.get("parameter_count", -1)) != KNOWN_300M_PARAMETERS
        or int(scale.get("non_embedding_parameters", -1))
        != KNOWN_300M_NONEMBEDDING_PARAMETERS
        or abs(float(scale.get("tokens_per_parameter", 0.0)) - 10.0) > 0.03
    ):
        raise ValueError("the bound 300M result is not the expected 10-TPP rung")
    sizes.append(float(KNOWN_300M_NONEMBEDDING_PARAMETERS))
    losses.append(float(ten_x["observed_validation_loss"]))
    sems.append(0.0)

    fit = fit_scaling_ensemble(
        sizes,
        losses,
        sems,
        target_size=float(TARGET_NONEMBEDDING_PARAMETERS),
        maximum_extrapolation_factor=6.0,
        maximum_family_spread=0.08,
        maximum_backtest_relative_error=0.10,
        bootstrap_samples=400,
    )
    prediction = float(fit["exploratory_prediction"])
    if not math.isfinite(prediction) or prediction <= 0.0:
        raise ValueError("the frozen 500M prediction is not finite")
    historical_errors = [
        float(row["relative_error"])
        for row in fit.get("rolling_backtests", ())
        if math.isfinite(float(row["relative_error"]))
    ]
    relative_half_width = max(0.05, *historical_errors)
    interval = [
        prediction * (1.0 - relative_half_width),
        prediction * (1.0 + relative_half_width),
    ]
    raw_interval = fit.get("prediction_interval_95")
    if raw_interval is not None:
        interval[0] = min(interval[0], float(raw_interval[0]))
        interval[1] = max(interval[1], float(raw_interval[1]))
    return {
        "outcome_seen": False,
        "parameters": TARGET_PARAMETERS,
        "non_embedding_parameters": TARGET_NONEMBEDDING_PARAMETERS,
        "predicted_validation_loss": prediction,
        "raw_prediction_interval_95": raw_interval,
        "calibrated_prediction_interval_95": interval,
        "relative_interval_half_width": relative_half_width,
        "raw_fit": fit,
        "fit_includes_known_300m_10tpp_rung": True,
        "fit_source_sizes": [int(value) for value in sizes],
        "fit_source_losses": losses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_1b_config", type=Path)
    parser.add_argument("source_1b_plan", type=Path)
    parser.add_argument("source_1b_preregistration", type=Path)
    parser.add_argument("calibration_aggregate", type=Path)
    parser.add_argument("known_300m_result", type=Path)
    parser.add_argument("continuation_manifest", type=Path)
    parser.add_argument("--partial-1b-step", type=int, required=True)
    parser.add_argument("--partial-1b-tokens", type=int, required=True)
    parser.add_argument("--partial-1b-validation-loss", type=float, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source_config = _load(args.source_1b_config)
    source_plan = _load(args.source_1b_plan)
    source_preregistration = _load(args.source_1b_preregistration)
    aggregate = _load(args.calibration_aggregate)
    known_300m = _load(args.known_300m_result)
    if source_preregistration.get("status") != "preregistered":
        raise ValueError("the source 1B preregistration is invalid")
    if _sha256(args.source_1b_config) != source_preregistration.get("config_sha256"):
        raise ValueError("the source 1B config changed after preregistration")
    if source_plan.get("fingerprint") != source_preregistration.get(
        "plan_fingerprint"
    ):
        raise ValueError("the source 1B plan changed after preregistration")
    if _sha256(args.calibration_aggregate) != source_preregistration.get(
        "calibration_aggregate_sha256"
    ):
        raise ValueError("the calibration aggregate differs from the 1B binding")
    source_target = source_plan.get("scales", ())[-1]
    if int(source_target.get("parameters", -1)) != SOURCE_1B_PARAMETERS:
        raise ValueError("the source plan is not the pinned 1B campaign")
    if float(source_target.get("rho_lm_over_d", 0.0)) != 32.0:
        raise ValueError("the source 1B campaign is not exact rho=32")
    if not (
        args.partial_1b_step > 0
        and args.partial_1b_tokens > 0
        and math.isfinite(args.partial_1b_validation_loss)
        and args.partial_1b_validation_loss > 0.0
    ):
        raise ValueError("partial 1B adaptation evidence must be finite and positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    if list(args.output_root.glob("**/ladder-shard-*.json")):
        raise ValueError("a 500M outcome already exists; refusing to rewrite the freeze")

    selected_eta = float(source_preregistration["selected_learning_rate"])
    config = deepcopy(dict(source_config))
    targets = list(config["ladder"]["target_parameters"])
    depths = list(config["ladder"]["depths"])
    if int(targets[-1]) != SOURCE_1B_PARAMETERS or int(depths[-1]) != TARGET_DEPTH:
        raise ValueError("the source config does not end at the expected 1B geometry")
    targets[-1] = TARGET_PARAMETERS
    config["ladder"]["target_parameters"] = targets
    config["ladder"]["depths"] = depths
    config["ladder"]["target_forecasts"] = []
    config["ladder"]["heldout_scale_count"] = 1
    config["dataset"]["token_stream_manifest_path"] = str(
        args.continuation_manifest.resolve()
    )
    config["validation_interval_steps"] = 2_378
    config["runtime"].update(
        {
            "distributed": "ddp",
            "num_processes": 8,
            "gradient_accumulation_steps": 1,
            "activation_checkpointing": False,
            "checkpoint_interval_steps": 0,
            "checkpoint_interval_seconds": 600,
            "resume": True,
        }
    )
    config["extension_contract"] = {
        "parent_plan_fingerprint": source_plan["fingerprint"],
        "parent_dataset_fingerprint": source_plan["dataset_identity"]["fingerprint"],
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
            f"compiled 500M geometry {observed_geometry} != {expected_geometry}"
        )
    plan_path = args.output_root / "plan.json"
    atomic_write_json(plan_path, plan)
    frozen_prediction = _freeze_prediction(aggregate, known_300m)

    revealed = list(args.output_root.glob("**/ladder-shard-*.json"))
    gates = {
        "source_1b_preregistration_bound": source_plan["fingerprint"]
        == source_preregistration["plan_fingerprint"],
        "same_dataset_fingerprint": plan["dataset_identity"]["fingerprint"]
        == source_plan["dataset_identity"]["fingerprint"],
        "same_tokenizer_fingerprint": plan["dataset_identity"][
            "tokenizer_fingerprint"
        ]
        == source_plan["dataset_identity"]["tokenizer_fingerprint"],
        "exact_500m_geometry": observed_geometry == expected_geometry,
        "exact_rho32": float(target["rho_lm_over_d"]) == 32.0,
        "constant_10_tpp": abs(float(target["tokens_per_parameter"]) - 10.0)
        <= 0.001,
        "frozen_source_eta": selected_eta
        == float(source_preregistration["selected_learning_rate"]),
        "zero_weight_decay": float(plan["optimizer_contract"]["weight_decay"])
        == 0.0,
        "eight_gpu_ddp": plan["runtime"]["distributed"] == "ddp"
        and int(plan["runtime"]["num_processes"]) == 8,
        "full_local_batch_per_gpu": int(
            config["batch_examples"]
            / (
                plan["runtime"]["num_processes"]
                * plan["runtime"]["gradient_accumulation_steps"]
            )
        )
        == 64,
        "single_frozen_seed": plan["seeds"] == [TARGET_SEED],
        "partial_1b_adaptation_disclosed": True,
        "target_outcome_unseen": not revealed,
        "finite_frozen_prediction": math.isfinite(
            float(frozen_prediction["predicted_validation_loss"])
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError("500M preregistration failed: " + ", ".join(failed))

    preregistration = {
        "schema_version": 1,
        "status": "preregistered",
        "scientific_status": (
            "adaptive_exploratory_single_seed_500m_10tpp_intermediate_rung; "
            "chosen after observing the partial 1B learning curve"
        ),
        "source_training_commit": "f969cafb3738351ca93fa8d28f6e65abd74a83c5",
        "source_1b_config_sha256": _sha256(args.source_1b_config),
        "source_1b_plan_fingerprint": source_plan["fingerprint"],
        "source_1b_preregistration_sha256": _sha256(
            args.source_1b_preregistration
        ),
        "calibration_aggregate_sha256": _sha256(args.calibration_aggregate),
        "known_300m_result_sha256": _sha256(args.known_300m_result),
        "config_sha256": _sha256(config_path),
        "plan_fingerprint": plan["fingerprint"],
        "dataset_fingerprint": plan["dataset_identity"]["fingerprint"],
        "tokenizer_fingerprint": plan["dataset_identity"]["tokenizer_fingerprint"],
        "selected_learning_rate": selected_eta,
        "selected_weight_decay_tau_ema": None,
        "task_id": f"ladder-S10-theory-eta{selected_eta:g}-seed{TARGET_SEED}",
        "target": target,
        "frozen_prediction": frozen_prediction,
        "adaptation_disclosure": {
            "partial_1b_outcome_seen": True,
            "latest_seen_validation_step": args.partial_1b_step,
            "latest_seen_validation_tokens": args.partial_1b_tokens,
            "latest_seen_validation_loss": args.partial_1b_validation_loss,
            "500m_outcome_seen": False,
        },
        "topology_diagnostic": {
            "single_vs_eight_gpu_canary_steps": 3,
            "maximum_absolute_loss_delta": 0.005,
        },
        "gates": gates,
    }
    atomic_write_json(args.output_root / "preregistration.json", preregistration)
    print(json.dumps(preregistration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
