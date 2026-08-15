#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from ai_theorist.autoscaler.forecast_campaigns import compile_real_text_scaling_plan
from ai_theorist.autoscaler.forecast_fleet import build_forecast_fleet_tasks
from ai_theorist.autoscaler.study import atomic_write_json


TARGET_DEPTH = 16
TARGET_WIDTH = 768
TARGET_HIDDEN_WIDTH = 1_536
TARGET_HEADS = 12
TARGET_PARAMETERS = 115_804_416
TARGET_NON_EMBEDDING_PARAMETERS = 75_634_176
TARGET_RHO = 32.0
FROZEN_TRANSFER_ETA = 0.055
DIAGNOSTIC_ETA_GRID = [0.035, 0.045, 0.055, 0.07, 0.09]
EXPECTED_SEEDS = [11, 29, 47]


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_config", type=Path)
    parser.add_argument("parent_aggregate", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    parent_config = _load(args.parent_config)
    parent_aggregate = _load(args.parent_aggregate)
    if parent_aggregate.get("status") != "completed":
        raise ValueError("parent aggregate is not complete")
    if args.output_root.joinpath("tune").exists():
        raise ValueError("refusing to preregister after ablation outcomes exist")

    config = json.loads(json.dumps(parent_config))
    ladder = config["ladder"]
    # The matched-budget target is slightly smaller in total parameters than
    # the existing 8-layer endpoint because narrowing also reduces its tied
    # embedding table. Keep the ladder strictly increasing by inserting the
    # ablation immediately before that endpoint.
    insertion_index = len(ladder["depths"]) - 1
    ladder["depths"] = [
        *ladder["depths"][:insertion_index],
        TARGET_DEPTH,
        *ladder["depths"][insertion_index:],
    ]
    ladder["target_parameters"] = [
        *ladder["target_parameters"][:insertion_index],
        TARGET_PARAMETERS,
        *ladder["target_parameters"][insertion_index:],
    ]
    ladder["reference_scale_index"] = insertion_index
    # Preserve the original endpoint as the single held-out scale. The new
    # ablation is the final non-held-out/reference scale immediately below it.
    ladder["heldout_scale_count"] = 1
    ladder["target_forecasts"] = []
    config["optimizer"]["learning_rates"] = DIAGNOSTIC_ETA_GRID
    config["seeds"] = EXPECTED_SEEDS
    config["run_negative_control"] = False

    plan = compile_real_text_scaling_plan(config)
    target = plan["scales"][insertion_index]
    tune_tasks = build_forecast_fleet_tasks(
        plan,
        phase="tune",
        run_negative_control=False,
    )
    parent_scales = parent_aggregate["scales"]
    baseline = parent_scales[-1]
    gates = {
        "same_immutable_dataset": (
            plan["dataset_identity"]["fingerprint"]
            == parent_aggregate["dataset"]["fingerprint"]
        ),
        "same_pinned_tokenizer": (
            plan["dataset_identity"]["tokenizer_fingerprint"]
            == parent_aggregate["dataset"]["tokenizer_fingerprint"]
        ),
        "same_training_horizon": (
            target["optimizer_steps"] == baseline["optimizer_steps"] == 1_144
            and target["presented_tokens"]
            == baseline["presented_tokens"]
            == 299_892_736
            and config["batch_examples"] == 128
        ),
        "matched_non_embedding_budget": (
            target["non_embedding_parameters"]
            == TARGET_NON_EMBEDDING_PARAMETERS
            and baseline["non_embedding_parameters"] == 77_165_312
            and abs(
                target["non_embedding_parameters"]
                / baseline["non_embedding_parameters"]
                - 1.0
            )
            < 0.02
        ),
        "exact_deep_narrow_geometry": (
            target["depth"] == TARGET_DEPTH
            and target["width"] == TARGET_WIDTH
            and target["hidden_width"] == TARGET_HIDDEN_WIDTH
            and target["num_heads"] == TARGET_HEADS
            and target["parameters"] == TARGET_PARAMETERS
            and target["rho_lm_over_d"] == TARGET_RHO
            and target["rho_relative_error"] == 0.0
        ),
        "frozen_primary_transfer_eta_in_diagnostic_grid": (
            FROZEN_TRANSFER_ETA in plan["learning_rates"]
            and plan["learning_rates"] == DIAGNOSTIC_ETA_GRID
        ),
        "matched_three_seeds": plan["seeds"] == EXPECTED_SEEDS,
        "complete_optimizer_contract": (
            plan["optimizer_contract"]["name"] == "adamw"
            and plan["optimizer_contract"]["beta1"] == 0.9
            and plan["optimizer_contract"]["beta2"] == 0.95
            and plan["optimizer_contract"]["epsilon"] == 1e-16
            and plan["optimizer_contract"]["weight_decay"] == 0.0
        ),
        "fifteen_preregistered_diagnostic_trials": (
            len(tune_tasks) == len(DIAGNOSTIC_ETA_GRID) * len(EXPECTED_SEEDS)
            and {task.scale_name for task in tune_tasks} == {target["name"]}
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError("shape-ablation preregistration failed: " + ", ".join(failed))

    args.output_root.mkdir(parents=True, exist_ok=True)
    config_path = args.output_root / "config.json"
    atomic_write_json(config_path, config)
    # Recompile the exact bytes written to disk before freezing its fingerprint.
    frozen_plan = compile_real_text_scaling_plan(_load(config_path))
    atomic_write_json(args.output_root / "plan.json", frozen_plan)
    payload = {
        "schema_version": 1,
        "status": "preregistered",
        "scientific_status": "matched_budget_jiang_depth_shape_ablation",
        "hypothesis": (
            "At matched non-embedding parameters and fixed rho=32, doubling depth "
            "from 8 to 16 while narrowing the residual stream improves validation loss."
        ),
        "primary_estimand": {
            "metric": "three-seed mean final SlimPajama validation loss",
            "frozen_learning_rate": FROZEN_TRANSFER_ETA,
            "baseline_mean_validation_loss": baseline["mean_validation_loss"],
            "baseline_seed_losses": baseline["seed_losses"],
            "comparison": "deep_narrow_minus_existing_8_layer_rung",
        },
        "diagnostic_eta_grid": DIAGNOSTIC_ETA_GRID,
        "diagnostic_rule": (
            "The eta bracket diagnoses transfer detuning only; the primary result "
            "remains the preregistered eta=0.055 cell regardless of bracket outcome."
        ),
        "target_geometry": {
            "depth": TARGET_DEPTH,
            "width": TARGET_WIDTH,
            "hidden_width": TARGET_HIDDEN_WIDTH,
            "num_heads": TARGET_HEADS,
            "parameters": TARGET_PARAMETERS,
            "non_embedding_parameters": TARGET_NON_EMBEDDING_PARAMETERS,
            "rho_lm_over_d": TARGET_RHO,
        },
        "plan_fingerprint": frozen_plan["fingerprint"],
        "dataset_fingerprint": frozen_plan["dataset_identity"]["fingerprint"],
        "tokenizer_fingerprint": frozen_plan["dataset_identity"][
            "tokenizer_fingerprint"
        ],
        "parent_config_sha256": _digest(args.parent_config),
        "parent_aggregate_sha256": _digest(args.parent_aggregate),
        "gates": gates,
        "execution_order": [
            "wait_for_the_active_1b_job_and_all_eight_gpus_to_be_idle",
            "qualify_the_exact_deep_narrow_runtime_without_reading_loss_outcomes",
            "run_the_fifteen_preregistered_eta_by_seed_trials",
            "report_the_frozen_eta_cell_as_the_primary_transfer_test",
            "report_the_local_eta_optimum_only_as_a_detuning_diagnostic",
        ],
    }
    atomic_write_json(args.output_root / "preregistration.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
