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


EXPECTED_WIDTHS = (512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2624)
EXPECTED_ACTIVE_NONEMBEDDING = (
    33_678_400,
    75_683_392,
    134_465_600,
    210_025_024,
    302_361_664,
    411_475_520,
    537_366_592,
    680_034_880,
    881_963_200,
)
EXPECTED_ENDPOINT = {
    "parameters": 2_336_879_360,
    "active_parameters": 1_014_509_312,
    "non_embedding_parameters": 2_204_333_248,
    "active_non_embedding_parameters": 881_963_200,
}
EXPECTED_TRANSFER_SUMMARY_SHA256 = (
    "3131a43349882cba609ae2f7041c49948f5c3b91bcc794630ed18716c1c70eca"
)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(
            "configs/autoscaler/jiang_moe_slimpajama_rho32_active_1b.json"
        ),
    )
    parser.add_argument(
        "--transfer-summary",
        type=Path,
        default=Path(
            "rounds/018-jiang-moe-constant-rho/transfer-result-summary.json"
        ),
    )
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise ValueError(f"token-stream manifest does not exist: {manifest}")
    if not args.template.is_file() or not args.transfer_summary.is_file():
        raise ValueError("campaign template or transfer summary is missing")
    if _sha(args.transfer_summary) != EXPECTED_TRANSFER_SUMMARY_SHA256:
        raise ValueError("the frozen rho=32 transfer summary has changed")
    transfer = _load(args.transfer_summary)
    if transfer.get("status") != "passed" or not transfer.get(
        "all_preregistered_gates_passed"
    ):
        raise ValueError("the source rho=32 transfer pilot did not pass")

    config = json.loads(json.dumps(_load(args.template)))
    config["dataset"]["token_stream_manifest_path"] = str(manifest)
    single_plan = compile_real_text_scaling_plan(config)

    ddp_config = json.loads(json.dumps(config))
    ddp_config["runtime"].update(
        {
            "distributed": "ddp",
            "num_processes": 8,
            "gradient_accumulation_steps": 2,
        }
    )
    ddp_plan = compile_real_text_scaling_plan(ddp_config)

    scales = [dict(row) for row in single_plan["scales"]]
    ddp_scales = [dict(row) for row in ddp_plan["scales"]]
    tune_tasks = build_forecast_fleet_tasks(single_plan, phase="tune")
    probe_eta = float(single_plan["learning_rates"][len(single_plan["learning_rates"]) // 2])
    ladder_tasks = build_forecast_fleet_tasks(
        ddp_plan,
        phase="ladder",
        selected_learning_rate=probe_eta,
        run_negative_control=False,
    )
    endpoint = scales[-1]
    geometry_keys = (
        "depth",
        "width",
        "hidden_width",
        "parameters",
        "active_parameters",
        "non_embedding_parameters",
        "active_non_embedding_parameters",
        "presented_tokens",
        "optimizer_steps",
    )
    gates = {
        "source_transfer_passed_and_is_immutable": (
            transfer["contract"]["rho"] == 32.0
            and transfer["contract"]["alpha_star"] == 1.0 / 128.0
            and transfer["contract"]["selected_reference_eta"] == 0.00390625
        ),
        "single_and_ddp_geometry_are_identical": all(
            all(left[key] == right[key] for key in geometry_keys)
            for left, right in zip(scales, ddp_scales)
        ),
        "nine_exact_l16_alpha2_rungs": (
            len(scales) == 9
            and tuple(int(row["width"]) for row in scales) == EXPECTED_WIDTHS
            and tuple(int(row["active_non_embedding_parameters"]) for row in scales)
            == EXPECTED_ACTIVE_NONEMBEDDING
            and all(
                int(row["depth"]) == 16
                and int(row["hidden_width"]) == 2 * int(row["width"])
                and float(row["rho_lm_over_d"]) == 32.0
                for row in scales
            )
        ),
        "fixed_E4_A1_and_alpha_star": all(
            int(row["num_experts"]) == 4
            and int(row["active_experts"]) == 1
            and float(row["width"])
            / (
                float(row["hidden_width"])
                * float(row["num_experts"])
                * float(row["depth"])
            )
            == 1.0 / 128.0
            for row in scales
        ),
        "endpoint_is_one_billion_active": all(
            int(endpoint[key]) == expected
            for key, expected in EXPECTED_ENDPOINT.items()
        ),
        "endpoint_is_hidden_from_primary_fit": endpoint["heldout"] is True
        and all(row["heldout"] is False for row in scales[:-1]),
        "constant_0375_tpp_on_active_total_axis": (
            single_plan["scales"][0]["token_budget_parameter_axis"]
            == "active_parameters"
            and all(
                abs(
                    float(row["presented_tokens"])
                    / float(row["active_parameters"])
                    - 0.375
                )
                <= 0.0003
                for row in scales
            )
        ),
        "no_training_data_repetition": max(
            float(row["repetition_ratio"]) for row in scales
        )
        <= 1.0,
        "one_seed_eight_eta_reference_screen": len(tune_tasks) == 8
        and len({task.seed for task in tune_tasks}) == 1,
        "eight_nonreference_ddp_rungs": len(ladder_tasks) == 8
        and {task.scale_name for task in ladder_tasks}
        == {f"S{index}" for index in range(2, 10)},
        "exact_main_paper_optimizer": (
            single_plan["optimizer_contract"]["name"] == "adam"
            and single_plan["optimizer_contract"]["beta1"] == 0.9
            and single_plan["optimizer_contract"]["beta2"] == 0.95
            and single_plan["optimizer_contract"]["epsilon"] == 1e-12
            and single_plan["optimizer_contract"]["weight_decay"] == 0.0
            and single_plan["schedule"]["family"] == "linear_warmup_constant"
            and single_plan["schedule"]["warmup_fraction"] == 0.5
        ),
        "global_batch_is_topology_invariant": (
            config["batch_examples"] == ddp_config["batch_examples"] == 128
            and config["runtime"]["gradient_accumulation_steps"] == 16
            and ddp_config["runtime"]["num_processes"] == 8
            and ddp_config["runtime"]["gradient_accumulation_steps"] == 2
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError("1B-active MoE preregistration failed: " + ", ".join(failed))

    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "config-single.json", config)
    atomic_write_json(args.output_root / "config-ddp.json", ddp_config)
    atomic_write_json(args.output_root / "plan-single.json", single_plan)
    atomic_write_json(args.output_root / "plan-ddp.json", ddp_plan)
    preregistration = {
        "schema_version": 1,
        "status": "preregistered",
        "scientific_status": "single_seed_constant_active_tpp_rho32_moe_scaling_law",
        "certified_forecast": False,
        "claim_restriction": (
            "Exploratory one-seed scaling law at 0.375 tokens per active total "
            "parameter. The 1B-active endpoint is a preregistered holdout; this "
            "does not establish token-horizon transfer or an asymptotic law."
        ),
        "decision": {
            "primary_shape": "fixed L=16, fixed M/D=2, width-scaled",
            "rho_lm_over_d": 32.0,
            "reason": (
                "L=16 keeps alpha_ffn=2 above the measured crossover while "
                "preserving rho=32; deeper/narrower is reserved for a matched ablation."
            ),
            "headline_parameter_axis": "active_parameters",
            "fit_parameter_axis": "active_non_embedding_parameters",
        },
        "source_transfer_summary_sha256": _sha(args.transfer_summary),
        "dataset_fingerprint": single_plan["dataset_identity"]["fingerprint"],
        "tokenizer_fingerprint": single_plan["dataset_identity"][
            "tokenizer_fingerprint"
        ],
        "single_plan_fingerprint": single_plan["fingerprint"],
        "ddp_plan_fingerprint": ddp_plan["fingerprint"],
        "endpoint": endpoint,
        "tune_task_ids": [task.task_id for task in tune_tasks],
        "ddp_ladder_task_templates": [task.to_dict() for task in ladder_tasks],
        "gates": gates,
        "execution_order": [
            "verify_immutable_slimpajama_stream_and_no_repetition",
            "qualify_endpoint_memory_and_all_eight_optimizer_groups",
            "run_eight_single_seed_reference_eta_trials_across_eight_h100s",
            "require_an_interior_eta_and_freeze_it",
            "qualify_single_vs_eight_gpu_endpoint_topology",
            "run_complete_S2_through_S8_ddp_rungs_in_order",
            "freeze_the_S9_prediction_before_reveal",
            "run_the_1B_active_S9_holdout_on_eight_h100s",
            "score_holdout_error_and_final_scaling_families",
        ],
    }
    preregistration["fingerprint"] = _fingerprint(preregistration)
    atomic_write_json(args.output_root / "preregistration.json", preregistration)
    print(json.dumps(preregistration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
