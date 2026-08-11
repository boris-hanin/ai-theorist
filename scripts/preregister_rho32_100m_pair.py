#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from ai_theorist.autoscaler.forecast_campaigns import compile_real_text_scaling_plan
from ai_theorist.autoscaler.study import atomic_write_json


def _object(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _read(path: Path, name: str) -> Dict[str, Any]:
    return _object(json.loads(path.read_text()), name)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the corrected rho=32 Jiang search and CompleteP ladder reuse."
    )
    parser.add_argument("base_jiang_config", type=Path)
    parser.add_argument("expanded_jiang_config", type=Path)
    parser.add_argument("zero_jiang_config", type=Path)
    parser.add_argument("completep_config", type=Path)
    parser.add_argument("completep_selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    configs = {
        "jiang_base_finite_tau": args.base_jiang_config,
        "jiang_expanded_finite_tau": args.expanded_jiang_config,
        "jiang_zero": args.zero_jiang_config,
        "completep": args.completep_config,
    }
    plans = {
        name: compile_real_text_scaling_plan(_read(path, f"{name} config"))
        for name, path in configs.items()
    }
    base = plans["jiang_base_finite_tau"]
    expanded = plans["jiang_expanded_finite_tau"]
    zero = plans["jiang_zero"]
    completep = plans["completep"]
    completep_selection = _read(args.completep_selection, "CompleteP selection")
    finite_scales = base["scales"]
    expanded_scales = expanded["scales"]
    zero_scales = zero["scales"]
    reference = finite_scales[0]
    endpoint = finite_scales[-1]
    architecture = _object(
        _read(args.base_jiang_config, "base Jiang config").get("architecture"),
        "Jiang architecture",
    )
    gates = {
        "jiang_parameterization_is_rho32": (
            base["architecture_contract"]["parameterization"]
            == "jiang_completep_adamw"
            and float(base["architecture_contract"]["rho_lm_over_d"]) == 32.0
        ),
        "reference_is_exact_L2_M1024_D64": (
            int(reference["depth"]),
            int(reference["hidden_width"]),
            int(reference["width"]),
        ) == (2, 1024, 64)
        and (
            int(architecture["reference_depth"]),
            int(architecture["reference_hidden_width"]),
            int(architecture["reference_residual_width"]),
        ) == (2, 1024, 64),
        "every_jiang_scale_has_exact_rho32": all(
            float(row["rho_lm_over_d"]) == 32.0
            and float(row["rho_relative_error"]) == 0.0
            for row in finite_scales
        ),
        "endpoint_is_L8_M3584_D896": (
            int(endpoint["depth"]),
            int(endpoint["hidden_width"]),
            int(endpoint["width"]),
        ) == (8, 3584, 896),
        "base_expanded_and_zero_geometry_match": (
            finite_scales == expanded_scales == zero_scales
        ),
        "base_finite_grid_matches_live_preregistration": base[
            "weight_decay_tau_ema_grid"
        ] == [0.035175, 0.07035, 0.1407, 0.2814, 0.5628],
        "expanded_finite_grid_is_nonduplicative": expanded[
            "weight_decay_tau_ema_grid"
        ] == [1.1256, 2.2512, 4.5024, 9.0048],
        "zero_arm_is_exact_adamw_zero_decay": (
            zero["optimizer_contract"]["name"] == "adamw"
            and not zero.get("weight_decay_tau_ema_grid")
            and float(zero["optimizer_contract"]["weight_decay"]) == 0.0
        ),
        "all_dataset_fingerprints_match": len(
            {
                plan["dataset_identity"]["fingerprint"]
                for plan in plans.values()
            }
        ) == 1,
        "all_tokenizer_fingerprints_match": len(
            {
                plan["dataset_identity"]["tokenizer_fingerprint"]
                for plan in plans.values()
            }
        ) == 1,
        "same_seeds_and_tpp": (
            base["seeds"]
            == expanded["seeds"]
            == zero["seeds"]
            == completep["seeds"]
            and abs(
                float(endpoint["tokens_per_parameter"])
                - float(completep["scales"][-1]["tokens_per_parameter"])
            ) <= 0.001
        ),
        "completep_selection_matches_frozen_plan": (
            completep_selection.get("plan_fingerprint")
            == completep["fingerprint"]
            and completep_selection.get("optimum_is_interior") is True
        ),
        "eight_gpu_dynamic_pool_declared": True,
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError("rho=32 preregistration failed gates: " + ", ".join(failed))

    payload = {
        "schema_version": 1,
        "status": "preregistered_rho32_correction",
        "claim_scope": (
            "post-correction exploratory Jiang-Chizat constant-rho=32 scaling ladder "
            "with a fresh AdamW eta/tau_EMA-or-zero reference search; the unaffected "
            "CompleteP reference selection is reused and its ladder is rerun"
        ),
        "correction": {
            "obsolete_rho": 4.0,
            "required_rho": 32.0,
            "reuse_of_obsolete_jiang_tuning": "forbidden",
        },
        "selection_rule": (
            "minimum three-seed mean reference validation loss across the broad finite "
            "tau_EMA grid and exact zero decay; eta must be interior and a selected "
            "finite tau_EMA must be interior (zero is an explicit valid endpoint)"
        ),
        "execution": {
            "scheduler": "shared_dynamic_eight_gpu_single_process_task_pool",
            "gpu_count": 8,
            "refill_policy": "maintain eight workers while at least eight tasks remain",
        },
        "plans": {
            name: {
                "config": str(configs[name]),
                "config_sha256": _sha256(configs[name]),
                "plan_fingerprint": plan["fingerprint"],
                "tuning_trials": plan["tuning_trials"],
                "scale_trials": plan["scale_trials"],
            }
            for name, plan in plans.items()
        },
        "completep_selection": {
            "path": str(args.completep_selection),
            "sha256": _sha256(args.completep_selection),
            "selected_learning_rate": completep_selection[
                "selected_learning_rate"
            ],
            "selected_weight_decay_tau_ema": completep_selection.get(
                "selected_weight_decay_tau_ema"
            ),
        },
        "dataset_fingerprint": base["dataset_identity"]["fingerprint"],
        "tokenizer_fingerprint": base["dataset_identity"][
            "tokenizer_fingerprint"
        ],
        "seeds": base["seeds"],
        "gates": gates,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
