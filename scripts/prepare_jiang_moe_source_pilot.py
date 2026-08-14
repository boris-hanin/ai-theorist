#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from ai_theorist.autoscaler.forecast_campaigns import (
    compile_real_text_scaling_plan,
)
from ai_theorist.autoscaler.forecast_fleet import build_forecast_fleet_tasks
from ai_theorist.autoscaler.study import atomic_write_json


SOURCE_ETA_GRID = [
    2.0**-7,
    2.0**-6,
    2.0**-5,
    2.0**-4,
    2.0**-3.5,
    2.0**-3,
    2.0**-2.5,
    2.0**-2,
]
REFERENCE_INITIALIZATION_STD = 2.0**-6
DECLARED_EXPERT_BIAS_LEARNING_RATE = 0.01


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(
            "configs/autoscaler/jiang_moe_slimpajama_source_pilot.json"
        ),
    )
    args = parser.parse_args()

    if args.output_root.joinpath("tune").exists() or args.output_root.joinpath(
        "ladder"
    ).exists():
        raise ValueError("refusing to preregister after MoE outcomes exist")
    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise ValueError(f"token-stream manifest does not exist: {manifest}")
    config = json.loads(json.dumps(_load(args.template)))
    config["dataset"]["token_stream_manifest_path"] = str(manifest)
    plan = compile_real_text_scaling_plan(config)
    reference_index = plan["architecture_contract"]["reference_scale_index"]
    reference = plan["scales"][reference_index]
    tuning_tasks = build_forecast_fleet_tasks(
        plan, phase="tune", run_negative_control=False
    )
    ladder_tasks = build_forecast_fleet_tasks(
        plan,
        phase="ladder",
        selected_learning_rate=SOURCE_ETA_GRID[4],
        run_negative_control=False,
    )
    gates = {
        "paper_base_geometry": (
            reference["depth"] == 8
            and reference["width"] == 512
            and reference["hidden_width"] == 512
            and reference["num_experts"] == 4
            and reference["active_experts"] == 1
            and reference["parameters"] == 51_504_672
            and reference["active_parameters"] == 38_897_184
        ),
        "fixed_kappa_one_quarter": all(
            row["active_experts"] / row["num_experts"] == 0.25
            for row in plan["scales"]
        ),
        "source_dimensions_independently_scaled": (
            plan["architecture_contract"][
                "rho_lm_over_d_is_not_a_source_transfer_invariant"
            ]
            is True
            and plan["architecture_contract"][
                "optional_declared_rho_lm_over_d"
            ]
            is None
        ),
        "source_initialization_coordinate": (
            config["architecture"]["initialization_std"]
            == REFERENCE_INITIALIZATION_STD
            and config["architecture"]["router_gamma"] == 1.0
        ),
        "source_eta_grid_with_interpolating_bracket_points": (
            plan["learning_rates"] == SOURCE_ETA_GRID
            and len(tuning_tasks) == 8
        ),
        "expert_bias_coordinate_explicit_not_inferred": (
            config["optimizer"]["expert_bias_learning_rate"]
            == DECLARED_EXPERT_BIAS_LEARNING_RATE
        ),
        "eight_nonreference_ladder_tasks": (
            len(ladder_tasks) == 8
            and all(task.optimizer_mode == "theory" for task in ladder_tasks)
        ),
        "fixed_approximately_one_billion_token_protocol": all(
            row["optimizer_steps"] == 2_000
            and row["presented_tokens"] == 999_424_000
            for row in plan["scales"]
        ),
        "paper_optimizer_and_schedule": (
            plan["optimizer_contract"]["name"] == "adam"
            and plan["optimizer_contract"]["beta1"] == 0.9
            and plan["optimizer_contract"]["beta2"] == 0.95
            and plan["optimizer_contract"]["epsilon"] == 1e-12
            and plan["optimizer_contract"]["weight_decay"] == 0.0
            and plan["schedule"]["family"] == "linear_warmup_constant"
            and plan["schedule"]["warmup_fraction"] == 0.5
            and plan["schedule"]["last_multiplier"] == 1.0
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError("MoE preregistration failed: " + ", ".join(failed))

    args.output_root.mkdir(parents=True, exist_ok=True)
    config_path = args.output_root / "config.json"
    atomic_write_json(config_path, config)
    frozen_plan = compile_real_text_scaling_plan(_load(config_path))
    atomic_write_json(args.output_root / "plan.json", frozen_plan)
    preregistration = {
        "schema_version": 1,
        "status": "preregistered",
        "scientific_status": "single_seed_source_faithful_moe_pilot",
        "source": {
            "title": "Hyperparameter Transfer with Mixture-of-Experts Layers",
            "url": "https://arxiv.org/abs/2601.20205",
            "version": "arXiv:2601.20205v3",
        },
        "claim_restriction": (
            "Exploratory one-seed reproduction on SlimPajama rather than the "
            "paper's FineWeb corpus; not certified evidence."
        ),
        "expert_bias_disclosure": (
            "eta_bias=0.01 is an explicit pilot reference coordinate. The "
            "paper proves eta_bias is Theta(1) across expert count at fixed "
            "kappa but does not publish a universal numeric value."
        ),
        "plan_fingerprint": frozen_plan["fingerprint"],
        "dataset_fingerprint": frozen_plan["dataset_identity"]["fingerprint"],
        "tokenizer_fingerprint": frozen_plan["dataset_identity"][
            "tokenizer_fingerprint"
        ],
        "template_sha256": _digest(args.template),
        "manifest_sha256": _digest(manifest),
        "gates": gates,
        "execution_order": [
            "verify_source_parameterization_and_immutable_dataset",
            "qualify_sparse_dispatch_memory_and_all_optimizer_groups",
            "run_eight_reference_learning_rate_trials_in_parallel",
            "require_an_interior_reference_eta_optimum",
            "run_eight_independent_dimension_scaling_tasks_in_parallel",
            "aggregate_active_and_total_parameter scaling evidence",
        ],
    }
    atomic_write_json(args.output_root / "preregistration.json", preregistration)
    print(json.dumps(preregistration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
