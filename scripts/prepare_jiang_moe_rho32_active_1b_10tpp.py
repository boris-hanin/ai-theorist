#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from ai_theorist.autoscaler.forecast_campaigns import compile_real_text_scaling_plan
from ai_theorist.autoscaler.forecast_fleet import build_forecast_fleet_tasks
from ai_theorist.autoscaler.study import atomic_write_json


EXPECTED_DATASET_FINGERPRINT = (
    "25a2fcdd8d274875f31df97b6801e78a7a836d8e01be2c55f4875f6b7f46c409"
)
EXPECTED_TOKENIZER_FINGERPRINT = (
    "d52f662783555cbf11f6a0cd8af35016652cda033389db471813c7d30f6958c5"
)
EXPECTED_MANIFEST_SHA256 = (
    "8946edd2348167e4c69277ccb72bcc8e8f5aa678a6d6c896bb922ec04414e415"
)
EXPECTED_TRAINING_TOKENS = 11_290_549_008
EXPECTED_WIDTHS = (512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2688)
EXPECTED_ACTIVE_NONEMBEDDING = (
    33_678_400,
    75_683_392,
    134_465_600,
    210_025_024,
    302_361_664,
    411_475_520,
    537_366_592,
    680_034_880,
    925_494_592,
)
EXPECTED_ENDPOINT = {
    "parameters": 2_401_916_224,
    "active_parameters": 1_014_263_104,
    "non_embedding_parameters": 2_313_147_712,
    "active_non_embedding_parameters": 925_494_592,
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
    parser.add_argument("verification_receipt", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(
            "configs/autoscaler/"
            "jiang_moe_fineweb_mistral_rho32_active_1b_10tpp.json"
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
    receipt = args.verification_receipt.expanduser().resolve()
    if not manifest.is_file() or not receipt.is_file():
        raise ValueError("manifest and full verification receipt are required")
    repo_root = Path(__file__).resolve().parents[1]
    repo_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    if len(repo_commit) != 40:
        raise ValueError("could not bind the preregistration to a repo commit")
    if _sha(manifest) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("FineWeb-Mistral token stream manifest changed")
    if _sha(args.transfer_summary) != EXPECTED_TRANSFER_SUMMARY_SHA256:
        raise ValueError("the source rho=32 transfer summary changed")
    transfer = _load(args.transfer_summary)
    if transfer.get("status") != "passed":
        raise ValueError("source rho=32 transfer did not pass")

    config = json.loads(json.dumps(_load(args.template)))
    config["dataset"]["token_stream_manifest_path"] = str(manifest)
    config["dataset"]["token_stream_verification_receipt_path"] = str(receipt)
    single_plan = compile_real_text_scaling_plan(config)
    ddp_config = json.loads(json.dumps(config))
    ddp_config["runtime"].update(
        {
            "distributed": "ddp",
            "num_processes": 8,
            "gradient_accumulation_steps": 1,
        }
    )
    ddp_plan = compile_real_text_scaling_plan(ddp_config)
    scales = [dict(row) for row in single_plan["scales"]]
    ddp_scales = [dict(row) for row in ddp_plan["scales"]]
    endpoint = scales[-1]
    tune_tasks = build_forecast_fleet_tasks(single_plan, phase="tune")
    probe_eta = float(single_plan["learning_rates"][4])
    ladder_tasks = build_forecast_fleet_tasks(
        ddp_plan,
        phase="ladder",
        selected_learning_rate=probe_eta,
        run_negative_control=False,
    )
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
            and transfer["all_preregistered_gates_passed"] is True
        ),
        "exact_pinned_fineweb_mistral_stream": (
            single_plan["dataset_identity"]["fingerprint"]
            == EXPECTED_DATASET_FINGERPRINT
            and single_plan["dataset_identity"]["tokenizer_fingerprint"]
            == EXPECTED_TOKENIZER_FINGERPRINT
            and int(single_plan["dataset_identity"]["training_tokens"])
            == EXPECTED_TRAINING_TOKENS
        ),
        "single_and_ddp_geometry_are_identical": all(
            all(left[key] == right[key] for key in geometry_keys)
            for left, right in zip(scales, ddp_scales)
        ),
        "nine_exact_l16_alpha2_rungs": (
            tuple(int(row["width"]) for row in scales) == EXPECTED_WIDTHS
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
            int(endpoint[key]) == value for key, value in EXPECTED_ENDPOINT.items()
        ),
        "endpoint_is_hidden_from_primary_fit": endpoint["heldout"] is True
        and all(row["heldout"] is False for row in scales[:-1]),
        "constant_10_tpp_on_active_total_axis": (
            all(
                abs(
                    float(row["presented_tokens"])
                    / float(row["active_parameters"])
                    - 10.0
                )
                <= 0.002
                for row in scales
            )
        ),
        "no_training_data_repetition": max(
            float(row["repetition_ratio"]) for row in scales
        )
        <= 1.0,
        "endpoint_has_token_margin": int(endpoint["presented_tokens"])
        <= EXPECTED_TRAINING_TOKENS,
        "one_seed_eight_eta_reference_screen": len(tune_tasks) == 8
        and {task.seed for task in tune_tasks} == {11},
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
        "batch512_has_identical_microbatch64_across_topologies": (
            int(config["batch_examples"]) == 512
            and int(config["runtime"]["gradient_accumulation_steps"]) == 8
            and int(ddp_config["runtime"]["num_processes"]) == 8
            and int(ddp_config["runtime"]["gradient_accumulation_steps"]) == 1
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError("10-TPP MoE preregistration failed: " + ", ".join(failed))

    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "config-single.json", config)
    atomic_write_json(args.output_root / "config-ddp.json", ddp_config)
    atomic_write_json(args.output_root / "plan-single.json", single_plan)
    atomic_write_json(args.output_root / "plan-ddp.json", ddp_plan)
    preregistration = {
        "schema_version": 1,
        "status": "preregistered",
        "scientific_status": "single_seed_constant_10_active_tpp_rho32_moe_scaling_law",
        "certified_forecast": False,
        "claim_restriction": (
            "Exploratory one-seed 10-TPP scaling law on a pinned FineWeb-Edu "
            "Mistral-token stream. The 1B-active endpoint is a preregistered "
            "holdout; the dataset differs from the short SlimPajama transfer pilot."
        ),
        "repo_commit": repo_commit,
        "decision": {
            "primary_shape": "fixed L=16, fixed M/D=2, width-scaled",
            "rho_lm_over_d": 32.0,
            "headline_parameter_axis": "active_parameters",
            "fit_parameter_axis": "active_non_embedding_parameters",
            "tokens_per_active_parameter": 10.0,
            "global_batch_examples": 512,
            "context_length": 256,
        },
        "source_transfer_summary_sha256": _sha(args.transfer_summary),
        "manifest_sha256": _sha(manifest),
        "verification_receipt_sha256": _sha(receipt),
        "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        "tokenizer_fingerprint": EXPECTED_TOKENIZER_FINGERPRINT,
        "single_plan_fingerprint": single_plan["fingerprint"],
        "ddp_plan_fingerprint": ddp_plan["fingerprint"],
        "endpoint": endpoint,
        "tune_task_ids": [task.task_id for task in tune_tasks],
        "ddp_ladder_task_templates": [task.to_dict() for task in ladder_tasks],
        "gates": gates,
        "execution_order": [
            "fully_hash_corpus_once_and_bind_local_verification_receipt",
            "qualify_1B_active_batch512_memory_and_all_optimizer_groups",
            "benchmark_1B_active_eight_gpu_throughput",
            "run_eight_single_seed_10TPP_reference_eta_trials_across_eight_h100s",
            "require_an_interior_eta_and_freeze_it",
            "qualify_single_vs_eight_gpu_topology",
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
