from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr


@dataclass(frozen=True)
class ScalingLawFit:
    loss_floor: float
    amplitude: float
    exponent: float
    r_squared: float
    condition_number: float
    forecastable: bool
    short_range_forecastable: bool
    asymptotic_floor_identifiable: bool
    model_kind: str
    refusal_reasons: Tuple[str, ...]
    parameter_samples: Tuple[Tuple[float, float, float], ...] = ()

    def predict(self, compute: float) -> float:
        return float(self.loss_floor + self.amplitude * compute ** (-self.exponent))

    def prediction_interval(self, compute: float) -> Tuple[float, float]:
        if not self.parameter_samples:
            prediction = self.predict(compute)
            return prediction, prediction
        values = [floor + amplitude * compute ** (-exponent) for floor, amplitude, exponent in self.parameter_samples]
        return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.pop("parameter_samples", None)
        return payload


def _fit_once(compute: np.ndarray, losses: np.ndarray, sems: np.ndarray) -> Tuple[np.ndarray, float, float]:
    min_loss = float(np.min(losses))
    floor_upper = max(min_loss * (1.0 - 1e-7), 1e-12)
    initial_floor = max(0.0, min_loss - 0.25 * (float(np.max(losses)) - min_loss))
    initial_alpha = 0.25
    initial_amplitude = max(
        1e-12,
        (float(losses[0]) - initial_floor) * float(compute[0]) ** initial_alpha,
    )
    safe_sems = np.maximum(sems, max(1e-6, float(np.median(losses)) * 1e-4))

    def residuals(params: np.ndarray) -> np.ndarray:
        floor, log_amplitude, log_alpha = params
        amplitude = np.exp(log_amplitude)
        alpha = np.exp(log_alpha)
        predicted = floor + amplitude * compute ** (-alpha)
        return (predicted - losses) / safe_sems

    result = least_squares(
        residuals,
        x0=np.array([initial_floor, math.log(initial_amplitude), math.log(initial_alpha)]),
        bounds=(
            np.array([0.0, -40.0, math.log(1e-3)]),
            np.array([floor_upper, 60.0, math.log(3.0)]),
        ),
        max_nfev=20_000,
    )
    if not result.success:
        raise RuntimeError(f"Scaling-law optimization failed: {result.message}")
    floor, log_amplitude, log_alpha = result.x
    params = np.array([floor, math.exp(log_amplitude), math.exp(log_alpha)])
    jacobian_condition = float(np.linalg.cond(result.jac.T @ result.jac))
    residual_sum = float(np.sum((losses - (params[0] + params[1] * compute ** (-params[2]))) ** 2))
    total_sum = float(np.sum((losses - np.mean(losses)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else float("nan")
    return params, r_squared, jacobian_condition


def fit_scaling_law(
    compute: Sequence[float],
    losses: Sequence[float],
    sems: Optional[Sequence[float]] = None,
    *,
    bootstrap_samples: int = 200,
    random_seed: int = 2027,
) -> ScalingLawFit:
    c = np.asarray(compute, dtype=float)
    y = np.asarray(losses, dtype=float)
    s = np.zeros_like(y) if sems is None else np.asarray(sems, dtype=float)
    if c.ndim != 1 or y.ndim != 1 or s.ndim != 1 or not (len(c) == len(y) == len(s)):
        raise ValueError("compute, losses, and sems must be equal-length one-dimensional arrays")
    if len(c) < 4:
        raise ValueError("At least four observations are required")
    if np.any(~np.isfinite(c)) or np.any(c <= 0) or np.any(~np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("compute and losses must be finite and positive")
    if np.any(np.diff(c) <= 0):
        raise ValueError("compute must be strictly increasing")
    if np.any(~np.isfinite(s)) or np.any(s < 0):
        raise ValueError("sems must be finite and non-negative")

    params, r_squared, condition = _fit_once(c, y, s)
    reasons = []
    correlation = float(spearmanr(np.log(c), y).statistic)
    if not math.isfinite(correlation) or correlation > -0.6:
        reasons.append("loss is not consistently decreasing with compute")
    dynamic_range = float(np.max(y) - np.min(y))
    noise_floor = max(float(np.median(s)) * 3.0, float(np.median(y)) * 0.005)
    if dynamic_range <= noise_floor:
        reasons.append("loss improvement is too small relative to noise")
    if not math.isfinite(condition) or condition > 1e14:
        reasons.append("floor and exponent are not identifiable")
    if not math.isfinite(r_squared) or r_squared < 0.9:
        reasons.append("power law fit quality is below the calibration threshold")
    floor, amplitude, exponent = (float(value) for value in params)
    if floor >= float(np.min(y)) * 0.995:
        reasons.append("estimated loss floor is pinned to the smallest observation")
    if exponent <= 0.005 or exponent >= 2.95:
        reasons.append("exponent is pinned to an optimization boundary")

    parameter_samples = []
    if bootstrap_samples:
        rng = np.random.default_rng(random_seed)
        effective_sems = np.maximum(s, np.maximum(1e-6, y * 1e-3))
        for _ in range(bootstrap_samples):
            sampled = np.maximum(1e-12, y + rng.normal(0.0, effective_sems))
            try:
                boot_params, _, _ = _fit_once(c, sampled, s)
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            parameter_samples.append(tuple(float(value) for value in boot_params))
        if bootstrap_samples >= 20 and len(parameter_samples) < bootstrap_samples * 0.8:
            reasons.append("bootstrap fit stability is below 80%")
    floor_reason = "estimated loss floor is pinned to the smallest observation"
    short_range_reasons = [reason for reason in reasons if reason != floor_reason]
    floor_identifiable = floor_reason not in reasons
    return ScalingLawFit(
        loss_floor=floor,
        amplitude=amplitude,
        exponent=exponent,
        r_squared=r_squared,
        condition_number=condition,
        forecastable=not reasons,
        short_range_forecastable=not short_range_reasons,
        asymptotic_floor_identifiable=floor_identifiable,
        model_kind="floor_power_law" if floor_identifiable else "boundary_floor_power_law",
        refusal_reasons=tuple(reasons),
        parameter_samples=tuple(parameter_samples),
    )


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((observed - predicted) ** 2))
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    return 1.0 - residual / total if total > 0.0 else float("nan")


def _fit_pure_power(
    sizes: np.ndarray, losses: np.ndarray, sems: np.ndarray
) -> Dict[str, Any]:
    safe_sems = np.maximum(sems, np.maximum(1e-6, losses * 1e-4))

    def residuals(params: np.ndarray) -> np.ndarray:
        amplitude, exponent = np.exp(params)
        return (amplitude * sizes ** (-exponent) - losses) / safe_sems

    slope, intercept = np.polyfit(np.log(sizes), np.log(losses), 1)
    result = least_squares(
        residuals,
        np.array([intercept, math.log(max(1e-3, -slope))]),
        bounds=(np.array([-40.0, math.log(1e-3)]), np.array([60.0, math.log(3.0)])),
        max_nfev=20_000,
    )
    if not result.success:
        raise RuntimeError(f"pure-power optimization failed: {result.message}")
    amplitude, exponent = (float(value) for value in np.exp(result.x))
    predicted = amplitude * sizes ** (-exponent)
    return {
        "kind": "pure_power_law",
        "parameters": {"amplitude": amplitude, "exponent": exponent},
        "r_squared": _r_squared(losses, predicted),
        "parameter_count": 2,
        "predict": lambda target: amplitude * float(target) ** (-exponent),
    }


def _fit_floor_power_candidate(
    sizes: np.ndarray, losses: np.ndarray, sems: np.ndarray
) -> Dict[str, Any]:
    params, r_squared, condition = _fit_once(sizes, losses, sems)
    floor, amplitude, exponent = (float(value) for value in params)
    return {
        "kind": "floor_power_law",
        "parameters": {
            "loss_floor": floor,
            "amplitude": amplitude,
            "exponent": exponent,
        },
        "r_squared": r_squared,
        "condition_number": condition,
        "parameter_count": 3,
        "predict": lambda target: floor + amplitude * float(target) ** (-exponent),
    }


def _fit_broken_power(
    sizes: np.ndarray, losses: np.ndarray, sems: np.ndarray
) -> Dict[str, Any]:
    if len(sizes) < 6:
        raise ValueError("broken power law requires at least six observations")
    log_sizes = np.log(sizes)
    log_losses = np.log(losses)
    safe_log_sems = np.maximum(sems / losses, 1e-4)
    candidates: List[Dict[str, Any]] = []
    for break_index in range(2, len(sizes) - 2):
        log_break = float(log_sizes[break_index])
        left = log_sizes - log_break
        hinge = np.maximum(0.0, left)

        def residuals(params: np.ndarray) -> np.ndarray:
            log_at_break, slope_before, slope_change = params
            predicted = log_at_break + slope_before * left + slope_change * hinge
            return (predicted - log_losses) / safe_log_sems

        result = least_squares(
            residuals,
            np.array([log_losses[break_index], -0.2, 0.0]),
            bounds=(
                np.array([-20.0, -3.0, -3.0]),
                np.array([20.0, -1e-3, 3.0]),
            ),
            max_nfev=20_000,
        )
        if not result.success:
            continue
        log_at_break, slope_before, slope_change = (
            float(value) for value in result.x
        )
        slope_after = slope_before + slope_change
        if slope_after >= -1e-3 or slope_after < -3.0:
            continue
        predicted_log = (
            log_at_break + slope_before * left + slope_change * hinge
        )
        residual_sum = float(np.sum(((predicted_log - log_losses) / safe_log_sems) ** 2))
        parameter_count = 4  # Includes selection of the discrete breakpoint.
        aic = residual_sum + 2.0 * parameter_count
        candidates.append(
            {
                "log_at_break": log_at_break,
                "slope_before": slope_before,
                "slope_after": slope_after,
                "break_size": float(sizes[break_index]),
                "aic": aic,
                "predicted": np.exp(predicted_log),
                "parameter_count": parameter_count,
            }
        )
    if not candidates:
        raise RuntimeError("broken-power optimization found no decreasing fit")
    best = min(candidates, key=lambda row: float(row["aic"]))

    def predict(target: float) -> float:
        log_ratio = math.log(float(target) / float(best["break_size"]))
        slope = (
            float(best["slope_before"])
            if target <= float(best["break_size"])
            else float(best["slope_after"])
        )
        return math.exp(float(best["log_at_break"]) + slope * log_ratio)

    return {
        "kind": "broken_power_law",
        "parameters": {
            "loss_at_break": math.exp(float(best["log_at_break"])),
            "break_size": float(best["break_size"]),
            "exponent_before": -float(best["slope_before"]),
            "exponent_after": -float(best["slope_after"]),
        },
        "r_squared": _r_squared(losses, np.asarray(best["predicted"])),
        "parameter_count": int(best["parameter_count"]),
        "predict": predict,
    }


_ENSEMBLE_FITTERS: Mapping[
    str, Callable[[np.ndarray, np.ndarray, np.ndarray], Dict[str, Any]]
] = {
    "pure_power_law": _fit_pure_power,
    "floor_power_law": _fit_floor_power_candidate,
    "broken_power_law": _fit_broken_power,
}


def fit_scaling_ensemble(
    sizes: Sequence[float],
    losses: Sequence[float],
    sems: Optional[Sequence[float]],
    *,
    target_size: float,
    maximum_extrapolation_factor: float = 10.0,
    maximum_family_spread: float = 0.08,
    maximum_backtest_relative_error: float = 0.10,
    bootstrap_samples: int = 200,
    random_seed: int = 2027,
) -> Dict[str, Any]:
    """Fit competing laws and refuse forecasts that are not backtested.

    The interval is the union of family and bootstrap uncertainty. This is
    intentionally conservative: adding a flexible model can widen or withhold
    a forecast, never make an unsupported point estimate look more certain.
    """
    x = np.asarray(sizes, dtype=np.float64)
    y = np.asarray(losses, dtype=np.float64)
    s = np.zeros_like(y) if sems is None else np.asarray(sems, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or s.ndim != 1 or not (len(x) == len(y) == len(s)):
        raise ValueError("sizes, losses, and sems must be equal-length vectors")
    if len(x) < 5:
        raise ValueError("at least five observed scales are required")
    if np.any(~np.isfinite(x)) or np.any(x <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError("sizes must be finite, positive, and strictly increasing")
    if np.any(~np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("losses must be finite and positive")
    if np.any(~np.isfinite(s)) or np.any(s < 0):
        raise ValueError("sems must be finite and non-negative")
    if not math.isfinite(target_size) or target_size <= x[-1]:
        raise ValueError("target_size must be finite and larger than every observation")
    if maximum_extrapolation_factor <= 1.0:
        raise ValueError("maximum_extrapolation_factor must exceed one")

    reasons: List[str] = []
    extrapolation_factor = float(target_size / x[-1])
    if extrapolation_factor > maximum_extrapolation_factor:
        reasons.append(
            f"target extrapolation factor {extrapolation_factor:.3g} exceeds "
            f"the declared maximum {maximum_extrapolation_factor:.3g}"
        )
    candidates: List[Dict[str, Any]] = []
    for kind, fitter in _ENSEMBLE_FITTERS.items():
        try:
            fitted = fitter(x, y, s)
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        prediction = float(fitted.pop("predict")(target_size))
        if not math.isfinite(prediction) or prediction <= 0.0:
            continue
        fitted["target_prediction"] = prediction
        fitted["qualified"] = bool(
            math.isfinite(float(fitted["r_squared"]))
            and float(fitted["r_squared"]) >= 0.9
        )
        candidates.append(fitted)
    qualified = [row for row in candidates if row["qualified"]]
    if len(qualified) < 2:
        reasons.append("fewer than two scaling-law families pass the fit-quality gate")

    rolling_backtests: List[Dict[str, Any]] = []
    for heldout_index in range(4, len(x)):
        family_predictions = []
        for kind, fitter in _ENSEMBLE_FITTERS.items():
            try:
                fit = fitter(x[:heldout_index], y[:heldout_index], s[:heldout_index])
                prediction = float(fit["predict"](float(x[heldout_index])))
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            if math.isfinite(prediction) and prediction > 0.0:
                family_predictions.append({"kind": kind, "prediction": prediction})
        if not family_predictions:
            continue
        median_prediction = float(
            np.median([row["prediction"] for row in family_predictions])
        )
        relative_error = abs(median_prediction / float(y[heldout_index]) - 1.0)
        rolling_backtests.append(
            {
                "heldout_size": float(x[heldout_index]),
                "observed_loss": float(y[heldout_index]),
                "median_prediction": median_prediction,
                "relative_error": relative_error,
                "family_predictions": family_predictions,
                "passed": relative_error <= maximum_backtest_relative_error,
            }
        )
    if not rolling_backtests:
        reasons.append("no rolling upper-rung backtest could be evaluated")
    elif any(not row["passed"] for row in rolling_backtests):
        reasons.append("at least one rolling upper-rung backtest exceeds its error gate")

    point_predictions = [float(row["target_prediction"]) for row in qualified]
    if point_predictions:
        median_prediction = float(np.median(point_predictions))
        family_spread = (
            max(point_predictions) - min(point_predictions)
        ) / median_prediction
        if family_spread > maximum_family_spread:
            reasons.append(
                f"scaling-law family spread {family_spread:.3g} exceeds "
                f"the declared maximum {maximum_family_spread:.3g}"
            )
    else:
        median_prediction = float("nan")
        family_spread = float("inf")

    bootstrap_predictions: List[float] = []
    if qualified and bootstrap_samples > 0:
        rng = np.random.default_rng(random_seed)
        effective_sems = np.maximum(s, np.maximum(1e-6, y * 1e-3))
        qualified_kinds = [str(row["kind"]) for row in qualified]
        for _ in range(bootstrap_samples):
            sampled = np.maximum(1e-12, y + rng.normal(0.0, effective_sems))
            for kind in qualified_kinds:
                try:
                    fit = _ENSEMBLE_FITTERS[kind](x, sampled, s)
                    prediction = float(fit["predict"](target_size))
                except (RuntimeError, ValueError, FloatingPointError):
                    continue
                if math.isfinite(prediction) and prediction > 0.0:
                    bootstrap_predictions.append(prediction)
    interval_values = [*point_predictions, *bootstrap_predictions]
    prediction_interval = (
        [
            float(np.quantile(interval_values, 0.025)),
            float(np.quantile(interval_values, 0.975)),
        ]
        if interval_values
        else None
    )
    return {
        "schema_version": 1,
        "coordinate": "model_parameters_at_frozen_training_path",
        "target_size": float(target_size),
        "largest_observed_size": float(x[-1]),
        "extrapolation_factor": extrapolation_factor,
        "maximum_extrapolation_factor": maximum_extrapolation_factor,
        "candidate_fits": candidates,
        "qualified_family_count": len(qualified),
        "family_spread": family_spread,
        "maximum_family_spread": maximum_family_spread,
        "rolling_backtests": rolling_backtests,
        "maximum_backtest_relative_error": maximum_backtest_relative_error,
        "prediction": median_prediction if not reasons else None,
        "exploratory_prediction": median_prediction,
        "prediction_interval_95": prediction_interval,
        "bootstrap_prediction_count": len(bootstrap_predictions),
        "certified": not reasons,
        "refusal_reasons": reasons,
    }
