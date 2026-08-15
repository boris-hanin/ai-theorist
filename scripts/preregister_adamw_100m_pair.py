#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from ai_theorist.autoscaler.forecast_campaigns import compile_real_text_scaling_plan
from ai_theorist.autoscaler.study import atomic_write_json


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _target(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    scales = plan.get("scales")
    if not isinstance(scales, list) or not scales:
        raise ValueError("compiled plan has no scales")
    return _object(scales[-1], "target scale")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the matched Jiang/CompleteP AdamW comparison before training."
    )
    parser.add_argument("jiang_config", type=Path)
    parser.add_argument("completep_config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    jiang_config = _object(json.loads(args.jiang_config.read_text()), "Jiang config")
    completep_config = _object(
        json.loads(args.completep_config.read_text()), "CompleteP config"
    )
    jiang = compile_real_text_scaling_plan(jiang_config)
    completep = compile_real_text_scaling_plan(completep_config)
    jiang_target = _target(jiang)
    completep_target = _target(completep)

    gates = {
        "same_dataset_fingerprint": (
            jiang["dataset_identity"]["fingerprint"]
            == completep["dataset_identity"]["fingerprint"]
        ),
        "same_tokenizer_fingerprint": (
            jiang["dataset_identity"]["tokenizer_fingerprint"]
            == completep["dataset_identity"]["tokenizer_fingerprint"]
        ),
        "same_seeds": jiang["seeds"] == completep["seeds"],
        "same_tokens_per_parameter": abs(
            float(jiang_target["tokens_per_parameter"])
            - float(completep_target["tokens_per_parameter"])
        )
        <= 0.001,
        "matched_parameter_count_within_one_percent": abs(
            int(completep_target["parameters"]) / int(jiang_target["parameters"])
            - 1.0
        )
        <= 0.01,
        "both_optimizers_are_adamw": (
            jiang["optimizer_contract"]["name"] == "adamw"
            and completep["optimizer_contract"]["name"] == "adamw"
        ),
        "both_tune_eta_and_tau_ema": (
            len(jiang["learning_rates"]) >= 3
            and len(completep["learning_rates"]) >= 3
            and len(jiang["weight_decay_tau_ema_grid"]) >= 3
            and len(completep["weight_decay_tau_ema_grid"]) >= 3
        ),
        "both_use_all_eight_independent_gpu_workers": True,
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError("pair preregistration gates failed: " + ", ".join(failed))

    payload = {
        "schema_version": 1,
        "status": "preregistered",
        "claim_scope": (
            "matched one-node comparison of separately eta/tau_EMA-tuned AdamW "
            "Jiang-Chizat and CompleteP runs; not a certified scaling-law forecast"
        ),
        "comparison_metric": "validation_loss_completep_minus_jiang_at_about_100m",
        "selection_rule": (
            "minimum mean reference validation loss over the preregistered eta x "
            "tau_EMA grid, requiring both coordinates to be interior"
        ),
        "execution": {
            "scheduler": "shared_dynamic_eight_gpu_single_process_task_pool",
            "gpu_count": 8,
            "tail_policy": "refill immediately until fewer than eight tasks remain",
        },
        "jiang": {
            "config": str(args.jiang_config),
            "config_sha256": _sha256(args.jiang_config),
            "plan_fingerprint": jiang["fingerprint"],
            "target": dict(jiang_target),
            "learning_rates": jiang["learning_rates"],
            "weight_decay_tau_ema_grid": jiang["weight_decay_tau_ema_grid"],
            "schedule": jiang["schedule"],
        },
        "completep": {
            "config": str(args.completep_config),
            "config_sha256": _sha256(args.completep_config),
            "plan_fingerprint": completep["fingerprint"],
            "target": dict(completep_target),
            "learning_rates": completep["learning_rates"],
            "weight_decay_tau_ema_grid": completep[
                "weight_decay_tau_ema_grid"
            ],
            "schedule": completep["schedule"],
        },
        "dataset_fingerprint": jiang["dataset_identity"]["fingerprint"],
        "tokenizer_fingerprint": jiang["dataset_identity"][
            "tokenizer_fingerprint"
        ],
        "seeds": jiang["seeds"],
        "gates": gates,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
