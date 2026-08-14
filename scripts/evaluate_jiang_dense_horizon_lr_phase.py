#!/usr/bin/env python3
"""Evaluate one preregistered T^-1/3 dense horizon phase."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ai_theorist.autoscaler.study import atomic_write_json

from evaluate_jiang_dense_horizon_scaling import (
    _evaluate_record,
    _one_record,
    _sha256,
    _load,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preregistration", type=Path)
    parser.add_argument("campaign_key")
    parser.add_argument("shard", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    preregistration = _load(args.preregistration)
    if (
        preregistration.get("status") != "preregistered"
        or not all(preregistration["gates"].values())
    ):
        raise ValueError("target-horizon preregistration is invalid")
    campaign = preregistration["campaigns"][args.campaign_key]
    raw_record = _one_record(args.shard)
    result = _evaluate_record(raw_record, campaign)
    losses = {
        float(row["requested_tokens_per_parameter"]): float(
            row["validation_loss"]
        )
        for row in result["horizons"]
    }
    target_tpp = float(campaign["target_tokens_per_parameter"])
    post_10 = [loss for tpp, loss in losses.items() if tpp > 10.0]
    expected_eta = float(campaign["selected_learning_rate"])
    final_improvement = losses[target_tpp] < losses[10.0]
    if target_tpp == 40.0:
        final_improvement = final_improvement and losses[40.0] < losses[20.0]
    gates = {
        "fresh_from_initialization": int(
            raw_record["metadata"].get("resumed_from_step", -1)
        )
        == 0,
        "selected_learning_rate_exact": math.isclose(
            float(result["selected_learning_rate"]),
            expected_eta,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "all_retained_losses_finite": all(
            math.isfinite(loss) for loss in losses.values()
        ),
        "no_post_10tpp_instability": all(
            loss <= 1.10 * losses[10.0] for loss in post_10
        ),
        "target_horizon_improves_past_10tpp": final_improvement,
        "full_state_at_every_preregistered_horizon": all(
            row["full_model_optimizer_generator_state_verified"]
            for row in result["horizons"]
        ),
        "zero_corpus_repetition": result["total_tokens"]
        <= int(preregistration["dataset_identity"]["training_tokens"]),
    }
    payload = {
        "schema_version": 1,
        "status": "passed" if all(gates.values()) else "failed",
        "scientific_status": (
            "preregistered exploratory single-seed target-horizon phase"
        ),
        "campaign_key": args.campaign_key,
        "preregistration_sha256": _sha256(args.preregistration),
        "shard_sha256": _sha256(args.shard),
        "result": result,
        "losses_by_tokens_per_parameter": {
            f"{tpp:g}": loss for tpp, loss in sorted(losses.items())
        },
        "gates": gates,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
