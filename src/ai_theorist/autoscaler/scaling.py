from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional, Sequence, Tuple

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
