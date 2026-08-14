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


EXPECTED_SHAPES = (
    (2, 128, 2048, 16.0),
    (4, 256, 2048, 8.0),
    (6, 384, 2048, 2048 / 384),
    (8, 512, 2048, 4.0),
    (12, 768, 2048, 2048 / 768),
    (16, 1024, 2048, 2.0),
)
EXPECTED_ETA_GRID = (
    2.0**-7,
    2.0**-6,
    2.0**-5,
    2.0**-4,
    2.0**-3.5,
    2.0**-3,
    2.0**-2.5,
    2.0**-2,
)


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(
            "configs/autoscaler/jiang_moe_slimpajama_rho32_transfer_pilot.json"
        ),
    )
    args = parser.parse_args()

    if args.output_root.joinpath("fleet", "tune").exists() or args.output_root.joinpath(
        "fleet", "ladder"
    ).exists():
        raise ValueError("refusing to preregister after transfer outcomes exist")
    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise ValueError(f"token-stream manifest does not exist: {manifest}")
    config = json.loads(json.dumps(_load(args.template)))
    config["dataset"]["token_stream_manifest_path"] = str(manifest)
    plan = compile_real_text_scaling_plan(config)
    reference_index = int(plan["architecture_contract"]["reference_scale_index"])
    scales = plan["scales"]
    observed_shapes = tuple(
        (
            int(row["depth"]),
            int(row["width"]),
            int(row["hidden_width"]),
            float(row["hidden_width"]) / float(row["width"]),
        )
        for row in scales
    )
    tune_tasks = build_forecast_fleet_tasks(plan, phase="tune")
    ladder_tasks = build_forecast_fleet_tasks(
        plan,
        phase="ladder",
        selected_learning_rate=EXPECTED_ETA_GRID[4],
        run_negative_control=True,
    )
    gates = {
        "exact_constant_rho32_geometry": (
            observed_shapes == EXPECTED_SHAPES
            and all(float(row["rho_lm_over_d"]) == 32.0 for row in scales)
            and plan["architecture_contract"]["optional_declared_rho_lm_over_d"]
            == 32.0
        ),
        "finite_depth_stops_above_measured_crossover": (
            observed_shapes[-1][3] == 2.0
            and observed_shapes[-1][3] > 1.8
        ),
        "fixed_kappa_one_quarter_and_fixed_E": all(
            int(row["num_experts"]) == 4
            and int(row["active_experts"]) == 1
            for row in scales
        ),
        "fixed_alpha_star_one_over_128": all(
            float(row["width"])
            / (
                float(row["hidden_width"])
                * float(row["num_experts"])
                * float(row["depth"])
            )
            == 1.0 / 128.0
            for row in scales
        ),
        "reference_is_l4_d256_m2048": (
            reference_index == 1
            and observed_shapes[reference_index] == EXPECTED_SHAPES[reference_index]
        ),
        "three_paired_seeds": plan["seeds"] == [11, 29, 47],
        "eight_eta_grid_and_interior_required_downstream": (
            tuple(plan["learning_rates"]) == EXPECTED_ETA_GRID
            and len(tune_tasks) == 24
        ),
        "fifteen_theory_ladder_plus_three_wrong_global_controls": (
            len(ladder_tasks) == 18
            and sum(task.optimizer_mode == "theory" for task in ladder_tasks) == 15
            and sum(task.optimizer_mode == "wrong_global" for task in ladder_tasks)
            == 3
        ),
        "matched_short_token_horizon": all(
            int(row["optimizer_steps"]) == 200
            and int(row["presented_tokens"]) == 6_553_600
            for row in scales
        ),
        "exact_source_optimizer_and_schedule": (
            plan["optimizer_contract"]["name"] == "adam"
            and plan["optimizer_contract"]["beta1"] == 0.9
            and plan["optimizer_contract"]["beta2"] == 0.95
            and plan["optimizer_contract"]["epsilon"] == 1e-12
            and plan["optimizer_contract"]["weight_decay"] == 0.0
            and plan["schedule"]["family"] == "linear_warmup_constant"
            and plan["schedule"]["warmup_fraction"] == 0.5
        ),
        "source_main_text_routing_initialization": (
            plan["architecture_contract"]["router_gamma"] == 1.0
            and plan["architecture_contract"]["reference_initialization_std"]
            == 2.0**-6
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError("rho32 MoE transfer preregistration failed: " + ", ".join(failed))

    args.output_root.mkdir(parents=True, exist_ok=True)
    config_path = args.output_root / "config.json"
    atomic_write_json(config_path, config)
    frozen_plan = compile_real_text_scaling_plan(_load(config_path))
    atomic_write_json(args.output_root / "plan.json", frozen_plan)
    preregistration = {
        "schema_version": 1,
        "status": "preregistered",
        "scientific_status": "three_seed_short_horizon_constant_rho_transfer_pilot",
        "claim_restriction": (
            "Tests fixed-normalized-eta transfer over a short 6.55M-token horizon. "
            "It does not certify a scaling law, token-horizon transfer, or the full MoE DMFT."
        ),
        "source": {
            "title": "Hyperparameter Transfer with Mixture-of-Experts Layers",
            "url": "https://arxiv.org/abs/2601.20205",
            "version": "arXiv:2601.20205v3",
        },
        "derived_contract": {
            "rho": 32.0,
            "alpha_star": 1.0 / 128.0,
            "sqrtD_over_LM_interpretation": (
                "effective residualized down scale; raw down matrix remains sqrt(D)/M"
            ),
            "primary_transfer_metric": (
                "fixed selected reference eta: finite progress on every shape and "
                "absolute log-progress/log-active-nonembedding-parameter slope <= 0.30"
            ),
            "stability_edge_metric": (
                "reference eta grid is selected separately and must be interior"
            ),
        },
        "plan_fingerprint": frozen_plan["fingerprint"],
        "dataset_fingerprint": frozen_plan["dataset_identity"]["fingerprint"],
        "tokenizer_fingerprint": frozen_plan["dataset_identity"][
            "tokenizer_fingerprint"
        ],
        "template_sha256": _sha(args.template),
        "manifest_sha256": _sha(manifest),
        "gates": gates,
        "execution_order": [
            "verify_immutable_slimpajama_gpt2_stream",
            "qualify_l16_d1024_m2048_sparse_runtime_and_all_eight_optimizer_groups",
            "run_24_reference_eta_seed_trials_in_an_eight_gpu_pool",
            "require_interior_reference_eta",
            "freeze_eta_and_apply_every_table2_group_rule",
            "run_fifteen_nonreference_theory_trials_and_three_wrong_global_controls",
            "evaluate_fixed_eta_progress_routing_and_control_without_local_retuning",
        ],
    }
    atomic_write_json(args.output_root / "preregistration.json", preregistration)
    print(json.dumps(preregistration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
