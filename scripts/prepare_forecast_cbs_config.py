#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from ai_theorist.autoscaler.forecast_campaigns import compile_real_text_scaling_plan
from ai_theorist.autoscaler.forecast_critical_batch import (
    compile_forecast_critical_batch_plan,
)
from ai_theorist.autoscaler.study import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bind a completed forecast selection to a fresh CBS census."
    )
    parser.add_argument("source_config", type=Path)
    parser.add_argument("source_selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.source_config.read_text())
    selection = json.loads(args.source_selection.read_text())
    source_plan = compile_real_text_scaling_plan(config)
    selected_eta = float(selection["selected_learning_rate"])
    if selected_eta not in {float(value) for value in source_plan["learning_rates"]}:
        raise ValueError("selection LR is not in the source plan")
    selected_tau = selection.get("selected_weight_decay_tau_ema")
    if selection.get("plan_fingerprint") not in {None, source_plan["fingerprint"]}:
        # Adaptive combined selections deliberately bind several source grids;
        # their selected source is checked separately by the pair controller.
        if not selection.get("selected_source"):
            raise ValueError("selection plan fingerprint does not match source plan")

    configured = deepcopy(config)
    configured["critical_batch"] = {
        "source_plan_fingerprint": source_plan["fingerprint"],
        "source_selection_sha256": sha256(args.source_selection.read_bytes()).hexdigest(),
        "selected_learning_rate": selected_eta,
        "selected_weight_decay_tau_ema": selected_tau,
        "anchor_scale_index": 0,
        "reference_batch_examples": 16,
        "initial_batch_examples": 4,
        "microbatch_examples": 4,
        "pilot_batch_examples": 16,
        "batch_examples": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
        "checkpoint_tokens": [
            16_777_216,
            67_108_864,
            268_435_456,
            805_306_368,
        ],
        "continuation_tokens": 67_108_864,
        "pilot_tokens": 134_217_728,
        "eta_multipliers": [
            0.03125,
            0.0625,
            0.125,
            0.25,
            0.5,
            1.0,
            2.0,
        ],
        "seeds": [101, 103, 107],
        "pilot_seed_count": 2,
        "loss_tolerance": 0.01,
        "safety_fraction": 0.8,
        "schedule": {
            "family": "warmup_stable_decay",
            "warmup_fraction": 0.02,
            "stable_fraction": 0.78,
            "terminal_fraction": 0.0,
        },
        "method": {
            "primary": "Merrill_et_al_local_branched_training",
            "gradient_noise_is_gating": False,
            "batch_coordinate": "non_padding_tokens_per_optimizer_update",
            "adam_lr_rule": "eta(B)=eta(B_ref)*sqrt(B/B_ref)",
            "parameter_group_rule": "multiply_every_theory_group_lr_by_the_same_batch_factor",
            "production_batch_rule": "power_of_two_below_0.8_times_measured_lower_CBS_bound",
            "extrapolated_batches_allowed": False,
        },
    }
    plan = compile_forecast_critical_batch_plan(configured)
    atomic_write_json(args.output, configured)
    print(
        json.dumps(
            {
                "config": str(args.output.resolve()),
                "source_plan_fingerprint": source_plan["fingerprint"],
                "source_selection_sha256": configured["critical_batch"][
                    "source_selection_sha256"
                ],
                "critical_batch_plan_fingerprint": plan["fingerprint"],
                "planned_grid_trials": plan["planned_grid_trials"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
