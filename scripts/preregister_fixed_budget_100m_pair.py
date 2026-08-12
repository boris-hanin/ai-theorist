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
from ai_theorist.autoscaler.jiang_chizat import (
    JIANG_DENSE_REPORTED_LR_MULTIPLIERS,
)
from ai_theorist.autoscaler.study import atomic_write_json


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _load(path: Path, name: str) -> Mapping[str, Any]:
    return _object(json.loads(path.read_text()), name)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preregister the paired fixed-batch, fixed-step 100M scans."
    )
    parser.add_argument("jiang_config", type=Path)
    parser.add_argument("completep_config", type=Path)
    parser.add_argument("runtime_qualification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    jiang_config = _load(args.jiang_config, "Jiang config")
    completep_config = _load(args.completep_config, "CompleteP config")
    qualification = _load(args.runtime_qualification, "runtime qualification")
    jiang = compile_real_text_scaling_plan(jiang_config)
    completep = compile_real_text_scaling_plan(completep_config)
    jiang_scales = list(jiang["scales"])
    completep_scales = list(completep["scales"])
    jiang_reference = jiang_scales[
        int(jiang["architecture_contract"]["reference_scale_index"])
    ]
    completep_reference = completep_scales[
        int(completep["architecture_contract"]["reference_scale_index"])
    ]
    qualification_fingerprints = {
        row["plan_fingerprint"] for row in qualification.get("campaigns", ())
    }
    fixed_jiang = _object(jiang["fixed_budget_contract"], "Jiang fixed budget")
    fixed_completep = _object(
        completep["fixed_budget_contract"], "CompleteP fixed budget"
    )

    gates = {
        "runtime_qualification_passed": qualification.get("status") == "passed",
        "qualified_exact_plan_fingerprints": qualification_fingerprints
        == {jiang["fingerprint"], completep["fingerprint"]},
        "same_immutable_dataset": jiang["dataset_identity"]["fingerprint"]
        == completep["dataset_identity"]["fingerprint"],
        "same_pinned_tokenizer": jiang["dataset_identity"]["tokenizer_fingerprint"]
        == completep["dataset_identity"]["tokenizer_fingerprint"],
        "both_fixed_budget_profiles": jiang["run_profile"]
        == completep["run_profile"]
        == "fixed_budget_scan",
        "same_global_batch": fixed_jiang["batch_examples"]
        == fixed_completep["batch_examples"]
        == 512,
        "same_optimizer_steps": fixed_jiang["optimizer_steps"]
        == fixed_completep["optimizer_steps"],
        "same_presented_tokens": fixed_jiang["presented_tokens"]
        == fixed_completep["presented_tokens"],
        "budget_identical_at_every_scale": fixed_jiang[
            "identical_at_every_scale"
        ]
        is True
        and fixed_completep["identical_at_every_scale"] is True,
        "same_training_seeds": jiang["seeds"] == completep["seeds"],
        "same_fixed_validation_windows": jiang["measurement_contract"]
        == completep["measurement_contract"]
        and jiang["measurement_contract"][
            "validation_windows_are_identical_across_trials"
        ]
        is True,
        "eight_rungs_each_with_hidden_top_rung": len(jiang_scales)
        == len(completep_scales)
        == 8
        and sum(bool(row["heldout"]) for row in jiang_scales) == 1
        and sum(bool(row["heldout"]) for row in completep_scales) == 1
        and bool(jiang_scales[-1]["heldout"])
        and bool(completep_scales[-1]["heldout"]),
        "nonembedding_primary_fit_axis": jiang["fit_parameter_axis"]
        == completep["fit_parameter_axis"]
        == "non_embedding_parameters",
        "jiang_exact_rho32_everywhere": all(
            float(row["rho_lm_over_d"]) == 32.0 for row in jiang_scales
        ),
        "jiang_reference_geometry_exact": (
            int(jiang_reference["depth"]),
            int(jiang_reference["hidden_width"]),
            int(jiang_reference["width"]),
        )
        == (4, 2560, 320),
        "jiang_tied_embedding_and_scaled_unembedding": jiang[
            "architecture_contract"
        ]["tied_embeddings"]
        is True
        and jiang["architecture_contract"]["unembedding_forward_scale"]
        == "(D/D0)^(-1)",
        "jiang_all_reported_group_multipliers": jiang["optimizer_contract"].get(
            "learning_rate_multipliers"
        )
        == JIANG_DENSE_REPORTED_LR_MULTIPLIERS,
        "completep_reference_geometry_exact": (
            int(completep_reference["depth"]),
            int(completep_reference["width"]),
        )
        == (2, 256),
        "completep_untied_embedding_and_scaled_unembedding": completep[
            "architecture_contract"
        ]["tied_embeddings"]
        is False
        and completep["architecture_contract"]["unembedding_forward_scale"]
        == "(N/N0)^(-1)",
        "adamw_and_explicit_epsilon": jiang["optimizer_contract"]["name"]
        == completep["optimizer_contract"]["name"]
        == "adamw"
        and float(jiang["optimizer_contract"]["epsilon"]) == 1e-12
        and float(completep["optimizer_contract"]["epsilon"]) == 1e-16,
        "fused_bf16_flash_runtime": jiang["runtime"]["precision"]
        == completep["runtime"]["precision"]
        == "bf16"
        and jiang["runtime"]["attention_backend"]
        == completep["runtime"]["attention_backend"]
        == "flash"
        and jiang["optimizer_contract"]["fused"] is True
        and completep["optimizer_contract"]["fused"] is True,
        "no_wrong_lr_controls": jiang["negative_control_trials"]
        == completep["negative_control_trials"]
        == 0,
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError("fixed-budget preregistration failed: " + ", ".join(failed))

    payload = {
        "schema_version": 1,
        "status": "preregistered",
        "claim_scope": (
            "preliminary matched fixed-batch, fixed-step real-text scaling scans; "
            "not a constant-TPP or certified extrapolative scaling law"
        ),
        "primary_question": (
            "how well does validation loss scale with non-embedding parameters "
            "when global batch, optimizer steps, presented tokens, validation "
            "windows, and training seeds are fixed?"
        ),
        "selection_rule": (
            "minimum three-seed reference loss; eta must be interior; a finite "
            "tau_EMA must be interior, while exact zero decay is a valid endpoint"
        ),
        "hidden_test": "largest rung is withheld from every scaling-law fit",
        "execution": {
            "scheduler": "shared_dynamic_eight_gpu_single_process_task_pool",
            "gpu_count": 8,
            "tail_policy": "refill immediately until fewer than eight tasks remain",
        },
        "jiang": {
            "config": str(args.jiang_config),
            "config_sha256": _sha256(args.jiang_config),
            "plan_fingerprint": jiang["fingerprint"],
            "reference": jiang_reference,
            "scales": jiang_scales,
            "learning_rates": jiang["learning_rates"],
            "weight_decay_tau_ema_grid": jiang.get(
                "weight_decay_tau_ema_grid", []
            ),
        },
        "completep": {
            "config": str(args.completep_config),
            "config_sha256": _sha256(args.completep_config),
            "plan_fingerprint": completep["fingerprint"],
            "reference": completep_reference,
            "scales": completep_scales,
            "learning_rates": completep["learning_rates"],
            "weight_decay_tau_ema_grid": completep[
                "weight_decay_tau_ema_grid"
            ],
            "includes_exact_zero_decay_endpoint": completep[
                "optimizer_contract"
            ]["include_zero_weight_decay_control"],
        },
        "fixed_budget": fixed_jiang,
        "dataset_fingerprint": jiang["dataset_identity"]["fingerprint"],
        "tokenizer_fingerprint": jiang["dataset_identity"][
            "tokenizer_fingerprint"
        ],
        "seeds": jiang["seeds"],
        "measurement_contract": jiang["measurement_contract"],
        "runtime_qualification": {
            "path": str(args.runtime_qualification),
            "sha256": _sha256(args.runtime_qualification),
        },
        "gates": gates,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
