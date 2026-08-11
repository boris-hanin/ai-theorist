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


def _same_except_optimizer(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = (
        "dataset_identity",
        "architecture_contract",
        "runtime",
        "schedule",
        "batch_examples",
        "seeds",
        "measurement_contract",
        "scales",
    )
    return all(left[key] == right[key] for key in keys)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preregister the adaptive weight-decay boundary extension."
    )
    parser.add_argument("original_pair_preregistration", type=Path)
    parser.add_argument("original_jiang_config", type=Path)
    parser.add_argument("original_completep_config", type=Path)
    parser.add_argument("expanded_jiang_config", type=Path)
    parser.add_argument("zero_jiang_config", type=Path)
    parser.add_argument("zero_completep_config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original_prereg = _read(
        args.original_pair_preregistration, "original pair preregistration"
    )
    configs = {
        "original_jiang": args.original_jiang_config,
        "original_completep": args.original_completep_config,
        "expanded_jiang": args.expanded_jiang_config,
        "zero_jiang": args.zero_jiang_config,
        "zero_completep": args.zero_completep_config,
    }
    plans = {
        name: compile_real_text_scaling_plan(_read(path, f"{name} config"))
        for name, path in configs.items()
    }
    original_jiang = plans["original_jiang"]
    original_completep = plans["original_completep"]
    expanded_jiang = plans["expanded_jiang"]
    zero_jiang = plans["zero_jiang"]
    zero_completep = plans["zero_completep"]
    expanded_grid = expanded_jiang["weight_decay_tau_ema_grid"]

    gates = {
        "original_preregistration_passed": (
            original_prereg.get("status") == "preregistered"
            and all(_object(original_prereg.get("gates"), "original gates").values())
        ),
        "original_plan_fingerprints_match": (
            original_prereg["jiang"]["plan_fingerprint"]
            == original_jiang["fingerprint"]
            and original_prereg["completep"]["plan_fingerprint"]
            == original_completep["fingerprint"]
        ),
        "jiang_expansion_changes_only_optimizer_contract": _same_except_optimizer(
            original_jiang, expanded_jiang
        ),
        "jiang_zero_changes_only_optimizer_contract": _same_except_optimizer(
            original_jiang, zero_jiang
        ),
        "completep_zero_changes_only_optimizer_contract": _same_except_optimizer(
            original_completep, zero_completep
        ),
        "expanded_grid_starts_at_old_boundary": (
            expanded_grid[0]
            == original_jiang["weight_decay_tau_ema_grid"][-1]
        ),
        "expanded_grid_has_four_new_log_spaced_points": (
            expanded_grid == [0.5628, 1.1256, 2.2512, 4.5024, 9.0048]
        ),
        "zero_controls_are_exact_adamw_zero_decay": all(
            plan["optimizer_contract"].get("name") == "adamw"
            and float(plan["optimizer_contract"].get("weight_decay", -1.0)) == 0.0
            and "weight_decay_tau_ema_grid" not in plan
            for plan in (zero_jiang, zero_completep)
        ),
        "all_dataset_fingerprints_match": len(
            {plan["dataset_identity"]["fingerprint"] for plan in plans.values()}
        )
        == 1,
        "all_use_same_seed_set": len(
            {tuple(plan["seeds"]) for plan in plans.values()}
        )
        == 1,
    }
    if not all(gates.values()):
        failed = [name for name, value in gates.items() if not value]
        raise ValueError("adaptive extension preregistration failed: " + ", ".join(failed))

    payload = {
        "schema_version": 1,
        "status": "preregistered_adaptive_extension",
        "claim_scope": (
            "adaptive follow-up triggered by the original Jiang finite-tau boundary; "
            "not a fresh confirmatory preregistration and not a certified forecast"
        ),
        "original_pair_preregistration": {
            "path": str(args.original_pair_preregistration),
            "sha256": _sha256(args.original_pair_preregistration),
        },
        "decision_rule": {
            "metric": "three-seed mean reference validation loss",
            "jiang_candidates": (
                "union of original finite tau grid, expanded finite tau grid, and "
                "exact AdamW zero decay (tau_EMA=infinity)"
            ),
            "completep_candidates": (
                "original finite tau grid and exact AdamW zero decay "
                "(tau_EMA=infinity)"
            ),
            "finite_tau_gate": (
                "selected eta and finite tau must be interior in the combined grid"
            ),
            "zero_decay_rule": (
                "zero decay may be selected exactly when it has the lowest mean; "
                "it is reported as an endpoint outcome, not an interior tau optimum"
            ),
            "overlap_gate": (
                "the duplicated Jiang tau=0.5628 cells must reproduce within 0.005 loss"
            ),
        },
        "execution": {
            "supplemental_tuning_trials": sum(
                plans[name]["tuning_trials"]
                for name in ("expanded_jiang", "zero_jiang", "zero_completep")
            ),
            "shared_gpu_pool": 8,
            "launch_ladders_only_after_adaptive_decision_passes": True,
        },
        "plans": {
            name: {
                "config_path": str(configs[name]),
                "config_sha256": _sha256(configs[name]),
                "plan_fingerprint": plan["fingerprint"],
                "tuning_trials": plan["tuning_trials"],
                "tau_grid": plan.get("weight_decay_tau_ema_grid"),
                "weight_decay": plan["optimizer_contract"].get("weight_decay"),
            }
            for name, plan in plans.items()
        },
        "dataset_fingerprint": original_jiang["dataset_identity"]["fingerprint"],
        "tokenizer_fingerprint": original_jiang["dataset_identity"][
            "tokenizer_fingerprint"
        ],
        "seeds": original_jiang["seeds"],
        "gates": gates,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
