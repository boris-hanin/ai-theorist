#!/usr/bin/env python3
"""Evaluate the 10-TPP ladder and freeze its prospective 1B forecast."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ai_theorist.autoscaler.study import atomic_write_json


RETROSPECTIVE_300M_AXIS = 245_929_600
PROSPECTIVE_1B_AXIS = 906_295_296


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _forecast(forecasts: list[Mapping[str, Any]], target: int) -> Mapping[str, Any]:
    matches = [row for row in forecasts if int(row["target_size"]) == target]
    if len(matches) != 1:
        raise ValueError(f"expected one forecast at primary-axis coordinate {target}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preregistration", type=Path)
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("known_300m_result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    preregistration = _load(args.preregistration)
    aggregate = _load(args.aggregate)
    known = _load(args.known_300m_result)
    if preregistration.get("status") != "preregistered":
        raise ValueError("calibration preregistration is not valid")
    if aggregate.get("plan_fingerprint") != preregistration.get("plan_fingerprint"):
        raise ValueError("aggregate does not match the preregistered plan")
    if _sha256(args.known_300m_result) != preregistration[
        "adaptation_disclosure"
    ]["known_300m_result_sha256"]:
        raise ValueError("the bound 300M result changed after preregistration")

    forecasts = list(aggregate.get("forecasts", ()))
    forecast_300m = _forecast(forecasts, RETROSPECTIVE_300M_AXIS)
    forecast_1b = _forecast(forecasts, PROSPECTIVE_1B_AXIS)
    observed_300m = float(known["ten_x"]["observed_validation_loss"])
    predicted_300m = float(forecast_300m["exploratory_prediction"])
    retrospective_error = abs(predicted_300m / observed_300m - 1.0)

    error_rows = [retrospective_error]
    error_rows.extend(
        float(row["relative_error"])
        for row in aggregate.get("hidden_scale_backtests", ())
    )
    error_rows.extend(
        float(row["relative_error"])
        for row in forecast_1b.get("rolling_backtests", ())
    )
    # The old bootstrap interval was visibly under-dispersed. Freeze a
    # prospective interval no narrower than +/-5%, enlarged by every available
    # upper-rung error. The point prediction itself remains untouched.
    relative_half_width = max(0.05, *error_rows)
    prediction_1b = float(forecast_1b["exploratory_prediction"])
    raw_interval = forecast_1b.get("prediction_interval_95")
    calibrated_interval = [
        prediction_1b * (1.0 - relative_half_width),
        prediction_1b * (1.0 + relative_half_width),
    ]
    if raw_interval is not None:
        calibrated_interval[0] = min(calibrated_interval[0], float(raw_interval[0]))
        calibrated_interval[1] = max(calibrated_interval[1], float(raw_interval[1]))

    gates = {
        "aggregate_completed": aggregate.get("status") == "completed",
        "reference_eta_is_interior": aggregate.get("reference_tuning", {}).get(
            "learning_rate_optimum_is_interior"
        )
        is True,
        "zero_decay_remained_selected": aggregate.get("reference_tuning", {}).get(
            "selected_weight_decay_tau_ema"
        )
        is None,
        "internal_200m_holdout_passed": (
            len(aggregate.get("hidden_scale_backtests", ())) == 1
            and aggregate["hidden_scale_backtests"][0].get("passed") is True
            and int(aggregate["hidden_scale_backtests"][0]["parameters"])
            == 200_020_480
        ),
        "all_loss_transitions_monotone": all(
            bool(row["accepted"])
            for row in aggregate.get("monotonicity_checks", ())
        ),
        "aggregate_forecastable": aggregate.get("forecastable") is True,
        "retrospective_300m_error_within_10_percent": retrospective_error <= 0.10,
        "prospective_1b_ensemble_qualified": forecast_1b.get("certified") is True,
        "finite_prospective_prediction_and_interval": (
            math.isfinite(prediction_1b)
            and prediction_1b > 0.0
            and all(math.isfinite(value) and value > 0.0 for value in calibrated_interval)
        ),
    }
    payload = {
        "schema_version": 1,
        "status": "passed" if all(gates.values()) else "failed",
        "scientific_status": (
            "single_seed_10tpp_calibration_with_retrospective_300m_check_and_"
            "prospective_1b_forecast"
        ),
        "aggregate_sha256": _sha256(args.aggregate),
        "preregistration_sha256": _sha256(args.preregistration),
        "selected_learning_rate": aggregate.get("reference_tuning", {}).get(
            "selected_learning_rate"
        ),
        "retrospective_300m": {
            "is_blind": False,
            "parameters": 299_177_600,
            "non_embedding_parameters": RETROSPECTIVE_300M_AXIS,
            "predicted_validation_loss": predicted_300m,
            "observed_validation_loss": observed_300m,
            "relative_error": retrospective_error,
            "raw_fit": forecast_300m,
        },
        "prospective_1b": {
            "outcome_seen": False,
            "parameters": 1_008_531_456,
            "non_embedding_parameters": PROSPECTIVE_1B_AXIS,
            "presented_tokens": 10_085_203_968,
            "predicted_validation_loss": prediction_1b,
            "raw_prediction_interval_95": raw_interval,
            "calibrated_prediction_interval_95": calibrated_interval,
            "relative_interval_half_width": relative_half_width,
            "raw_fit": forecast_1b,
        },
        "gates": gates,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
