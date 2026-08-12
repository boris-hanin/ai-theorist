#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from ai_theorist.autoscaler.forecast_critical_batch import (
    compile_forecast_critical_batch_plan,
)
from ai_theorist.autoscaler.study import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preregister the paired Jiang/CompleteP critical-batch census."
    )
    parser.add_argument("jiang_config", type=Path)
    parser.add_argument("completep_config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    jiang_config = json.loads(args.jiang_config.read_text())
    completep_config = json.loads(args.completep_config.read_text())
    jiang = compile_forecast_critical_batch_plan(jiang_config)
    completep = compile_forecast_critical_batch_plan(completep_config)
    gates = {
        "same_dataset_fingerprint": (
            jiang["dataset_identity"]["fingerprint"]
            == completep["dataset_identity"]["fingerprint"]
        ),
        "same_tokenizer_fingerprint": (
            jiang["dataset_identity"]["tokenizer_fingerprint"]
            == completep["dataset_identity"]["tokenizer_fingerprint"]
        ),
        "same_batch_grid": jiang["batch_examples"] == completep["batch_examples"],
        "same_checkpoint_grid": (
            jiang["checkpoint_tokens"] == completep["checkpoint_tokens"]
        ),
        "same_continuation_window": (
            jiang["continuation_tokens"] == completep["continuation_tokens"]
        ),
        "same_fresh_seeds": jiang["seeds"] == completep["seeds"] == [101, 103, 107],
        "same_loss_tolerance": jiang["loss_tolerance"] == completep["loss_tolerance"],
        "same_safety_fraction": jiang["safety_fraction"] == completep["safety_fraction"],
        "jiang_is_rho32": (
            float(jiang["architecture_contract"]["rho_lm_over_d"]) == 32.0
        ),
        "completep_is_completep": (
            completep["architecture_contract"]["parameterization"]
            == "completep_alpha_1_adamw"
        ),
        "no_extrapolated_batch_is_preregistered": (
            jiang_config["critical_batch"]["method"]["extrapolated_batches_allowed"]
            is False
            and completep_config["critical_batch"]["method"][
                "extrapolated_batches_allowed"
            ]
            is False
        ),
        "gradient_noise_is_not_a_gate": (
            jiang_config["critical_batch"]["method"]["gradient_noise_is_gating"]
            is False
            and completep_config["critical_batch"]["method"][
                "gradient_noise_is_gating"
            ]
            is False
        ),
    }
    passed = all(gates.values())
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "adaptive_followup": True,
        "prior_failed_large_scale_losses_are_excluded_from_tuning": True,
        "gates": gates,
        "jiang_plan_fingerprint": jiang["fingerprint"],
        "completep_plan_fingerprint": completep["fingerprint"],
        "jiang_config_sha256": sha256(args.jiang_config.read_bytes()).hexdigest(),
        "completep_config_sha256": sha256(args.completep_config.read_bytes()).hexdigest(),
        "execution_order": jiang["execution_order"],
        "interpretation": (
            "Each architecture receives its own measured Bcrit(T) and horizon-safe "
            "reference LR. The production schedule uses only measured CBS lower "
            "bounds; cross-architecture agreement is evidence, not an assumption."
        ),
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
