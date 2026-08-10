from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CriticalBatchEstimate:
    estimator: str
    critical_batch_tokens: Optional[float]
    lower_batch_tokens: Optional[float]
    upper_batch_tokens: Optional[float]
    qualified: bool
    diagnostics: Dict[str, Any]
    refusal_reasons: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StepsToTargetObservation:
    batch_tokens: int
    optimizer_steps: int
    seed: int = 0

    def __post_init__(self) -> None:
        if self.batch_tokens <= 0 or self.optimizer_steps <= 0:
            raise ValueError("batch_tokens and optimizer_steps must be positive")


@dataclass(frozen=True)
class ContinuationObservation:
    batch_tokens: int
    validation_loss_before: float
    validation_loss_after: float
    tokens_processed: int
    seed: int = 0

    def __post_init__(self) -> None:
        if self.batch_tokens <= 0 or self.tokens_processed <= 0:
            raise ValueError("batch_tokens and tokens_processed must be positive")
        if not all(
            math.isfinite(value)
            for value in (self.validation_loss_before, self.validation_loss_after)
        ):
            raise ValueError("continuation losses must be finite")


def _refused(estimator: str, reasons: Iterable[str], diagnostics: Dict[str, Any]) -> CriticalBatchEstimate:
    return CriticalBatchEstimate(
        estimator,
        None,
        None,
        None,
        False,
        diagnostics,
        tuple(reasons),
    )


def estimate_steps_to_target_critical_batch(
    observations: Sequence[StepsToTargetObservation],
    *,
    overhead: float = 0.20,
    minimum_r_squared: float = 0.90,
    minimum_relative_dynamic_range: float = 0.05,
    bootstrap_samples: int = 400,
    seed: int = 17,
) -> CriticalBatchEstimate:
    """Fit S(B)=a+b/B and apply the Zhang et al. 20%-overhead definition."""
    if not 0.0 < overhead < 1.0:
        raise ValueError("overhead must lie in (0, 1)")
    if not 0.0 <= minimum_relative_dynamic_range < 1.0:
        raise ValueError("minimum_relative_dynamic_range must lie in [0, 1)")
    grouped: Dict[int, list[float]] = {}
    for row in observations:
        grouped.setdefault(row.batch_tokens, []).append(float(row.optimizer_steps))
    batches = np.asarray(sorted(grouped), dtype=np.float64)
    means = np.asarray([np.mean(grouped[int(batch)]) for batch in batches], dtype=np.float64)
    diagnostics: Dict[str, Any] = {
        "definition": f"{100 * overhead:.0f}% steps overhead relative to ideal linear scaling",
        "unique_batches": int(batches.size),
        "batch_tokens": batches.tolist(),
        "mean_steps_to_target": means.tolist(),
    }
    if batches.size < 4:
        return _refused(
            "steps_to_target",
            ("at least four unique batch sizes are required",),
            diagnostics,
        )
    design = np.column_stack((np.ones_like(batches), 1.0 / batches))
    coefficients, _, _, _ = np.linalg.lstsq(design, means, rcond=None)
    intercept, inverse_batch_coefficient = (float(value) for value in coefficients)
    fitted = design @ coefficients
    residual_sum = float(np.sum((means - fitted) ** 2))
    total_sum = float(np.sum((means - means.mean()) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0
    monotone_fraction = float(np.mean(np.diff(means) <= 0.0))
    dynamic_scale = max(
        float(np.max(np.abs(means))), np.finfo(np.float64).tiny
    )
    relative_dynamic_range = float((means.max() - means.min()) / dynamic_scale)
    diagnostics.update(
        {
            "fit": {"a": intercept, "b": inverse_batch_coefficient},
            "r_squared": r_squared,
            "monotone_fraction": monotone_fraction,
            "minimum_r_squared": minimum_r_squared,
            "relative_dynamic_range": relative_dynamic_range,
            "minimum_relative_dynamic_range": minimum_relative_dynamic_range,
        }
    )
    reasons = []
    if intercept <= 0.0 or inverse_batch_coefficient <= 0.0:
        reasons.append("the fitted a and b coefficients must both be positive")
    if r_squared < minimum_r_squared:
        reasons.append("the inverse-batch fit is not accurate enough")
    if relative_dynamic_range < minimum_relative_dynamic_range:
        reasons.append("the steps-to-target curve has insufficient dynamic range")
    if monotone_fraction < 2.0 / 3.0:
        reasons.append("steps to target are not sufficiently monotone in batch")
    if reasons:
        return _refused("steps_to_target", reasons, diagnostics)

    reference_batch = float(batches[0])
    critical = overhead * inverse_batch_coefficient / intercept + (1.0 + overhead) * reference_batch
    if not batches[0] < critical < batches[-1]:
        reasons.append("the estimated transition is not bracketed by the batch sweep")

    rng = np.random.default_rng(seed)
    bootstrap_values = []
    batch_keys = [int(batch) for batch in batches]
    for _ in range(max(0, bootstrap_samples)):
        sample_means = np.asarray(
            [rng.choice(grouped[batch], size=len(grouped[batch]), replace=True).mean() for batch in batch_keys]
        )
        sample_coefficients, _, _, _ = np.linalg.lstsq(design, sample_means, rcond=None)
        sample_a, sample_b = sample_coefficients
        if sample_a > 0.0 and sample_b > 0.0:
            bootstrap_values.append(
                overhead * float(sample_b) / float(sample_a)
                + (1.0 + overhead) * reference_batch
            )
    if len(bootstrap_values) >= 20:
        lower, upper = np.quantile(bootstrap_values, (0.05, 0.95))
    else:
        lower, upper = critical, critical
        diagnostics["bootstrap_warning"] = "too few valid bootstrap fits"
    diagnostics["bootstrap_valid_samples"] = len(bootstrap_values)
    return CriticalBatchEstimate(
        "steps_to_target",
        critical,
        float(lower),
        float(upper),
        not reasons,
        diagnostics,
        tuple(reasons),
    )


def estimate_direct_checkpoint_critical_batch(
    observations: Sequence[ContinuationObservation],
    *,
    overhead: float = 0.20,
) -> CriticalBatchEstimate:
    """Estimate the matched-token continuation point where progress ceases to transfer."""
    if not 0.0 < overhead < 1.0:
        raise ValueError("overhead must lie in (0, 1)")
    grouped: Dict[int, list[float]] = {}
    for row in observations:
        progress_per_token = (
            row.validation_loss_before - row.validation_loss_after
        ) / row.tokens_processed
        grouped.setdefault(row.batch_tokens, []).append(progress_per_token)
    batches = sorted(grouped)
    mean_progress = {batch: float(np.mean(grouped[batch])) for batch in batches}
    diagnostics: Dict[str, Any] = {
        "definition": "matched-token checkpoint continuation",
        "batch_tokens": batches,
        "mean_progress_per_token": [mean_progress[batch] for batch in batches],
    }
    if len(batches) < 4:
        return _refused(
            "direct_checkpoint",
            ("at least four unique continuation batches are required",),
            diagnostics,
        )
    reference_progress = mean_progress[batches[0]]
    if reference_progress <= 0.0:
        return _refused(
            "direct_checkpoint",
            ("the smallest-batch continuation made no positive progress",),
            diagnostics,
        )
    threshold = reference_progress / (1.0 + overhead)
    passing = [batch for batch in batches if mean_progress[batch] >= threshold]
    failing = [batch for batch in batches if mean_progress[batch] < threshold]
    diagnostics.update(
        {
            "reference_progress_per_token": reference_progress,
            "qualification_threshold": threshold,
            "passing_batches": passing,
            "failing_batches": failing,
        }
    )
    pass_after_failure = any(
        mean_progress[later] >= threshold
        for index, batch in enumerate(batches)
        if mean_progress[batch] < threshold
        for later in batches[index + 1 :]
    )
    if pass_after_failure:
        return _refused(
            "direct_checkpoint",
            ("continuation efficiency crosses the threshold more than once",),
            diagnostics,
        )
    if not passing:
        return _refused(
            "direct_checkpoint",
            ("no continuation batch meets the efficiency threshold",),
            diagnostics,
        )
    lower = float(max(passing))
    upper_candidates = [batch for batch in failing if batch > lower]
    if not upper_candidates:
        return CriticalBatchEstimate(
            "direct_checkpoint",
            lower,
            lower,
            None,
            False,
            diagnostics,
            ("the transition is not upper-bracketed",),
        )
    upper = float(min(upper_candidates))
    critical = math.sqrt(lower * upper)
    return CriticalBatchEstimate(
        "direct_checkpoint", critical, lower, upper, True, diagnostics
    )


def estimate_gradient_noise_critical_batch(
    microbatch_gradients: np.ndarray,
    *,
    microbatch_tokens: int = 1,
    bootstrap_samples: int = 400,
    seed: int = 23,
) -> CriticalBatchEstimate:
    """Estimate trace(Cov[g])/||E[g]||^2 in per-token gradient units."""
    gradients = np.asarray(microbatch_gradients, dtype=np.float64)
    if gradients.ndim < 2:
        raise ValueError("microbatch_gradients must have shape [samples, ...]")
    gradients = gradients.reshape(gradients.shape[0], -1)
    if microbatch_tokens <= 0:
        raise ValueError("microbatch_tokens must be positive")
    diagnostics: Dict[str, Any] = {
        "definition": "gradient-noise scale trace covariance / squared mean norm",
        "microbatch_count": int(gradients.shape[0]),
        "gradient_dimension": int(gradients.shape[1]),
        "microbatch_tokens": microbatch_tokens,
    }
    if gradients.shape[0] < 8:
        return _refused(
            "gradient_noise",
            ("at least eight independent microbatch gradients are required",),
            diagnostics,
        )
    mean = gradients.mean(axis=0)
    raw_mean_squared_norm = float(np.dot(mean, mean))
    centered = gradients - mean
    covariance_trace = float(np.sum(centered * centered) / (gradients.shape[0] - 1))
    # Remove the finite-sample contribution of gradient noise to ||sample mean||^2.
    signal = raw_mean_squared_norm - covariance_trace / gradients.shape[0]
    if signal <= np.finfo(np.float64).tiny or not math.isfinite(signal + covariance_trace):
        return _refused(
            "gradient_noise",
            ("the gradient signal is zero or non-finite",),
            diagnostics,
        )
    critical = microbatch_tokens * covariance_trace / signal
    diagnostics.update(
        {
            "raw_sample_mean_squared_norm": raw_mean_squared_norm,
            "debiased_mean_gradient_squared_norm": signal,
            "microbatch_covariance_trace": covariance_trace,
        }
    )
    rng = np.random.default_rng(seed)
    boot = []
    count = gradients.shape[0]
    for _ in range(max(0, bootstrap_samples)):
        sample = gradients[rng.integers(0, count, size=count)]
        sample_mean = sample.mean(axis=0)
        raw_sample_signal = float(np.dot(sample_mean, sample_mean))
        sample_centered = sample - sample_mean
        sample_trace = float(np.sum(sample_centered * sample_centered) / (count - 1))
        sample_signal = raw_sample_signal - sample_trace / count
        if sample_signal <= np.finfo(np.float64).tiny:
            continue
        boot.append(microbatch_tokens * sample_trace / sample_signal)
    if len(boot) < 20:
        return CriticalBatchEstimate(
            "gradient_noise",
            critical,
            None,
            None,
            False,
            diagnostics,
            ("too few valid bootstrap resamples",),
        )
    lower, upper = np.quantile(boot, (0.05, 0.95))
    diagnostics["bootstrap_valid_samples"] = len(boot)
    return CriticalBatchEstimate(
        "gradient_noise", critical, float(lower), float(upper), True, diagnostics
    )


def combine_critical_batch_estimates(
    estimates: Sequence[CriticalBatchEstimate],
    *,
    maximum_ratio: float = 2.0,
    minimum_estimators: int = 2,
) -> CriticalBatchEstimate:
    """Gate a consensus estimate without collapsing estimator definitions."""
    valid = [
        row
        for row in estimates
        if row.qualified and row.critical_batch_tokens is not None
    ]
    diagnostics: Dict[str, Any] = {
        "definition": "geometric consensus of separately qualified estimators",
        "component_estimates": [row.to_dict() for row in estimates],
        "maximum_allowed_ratio": maximum_ratio,
    }
    if len(valid) < minimum_estimators:
        return _refused(
            "consensus",
            (f"at least {minimum_estimators} qualified estimators are required",),
            diagnostics,
        )
    values = np.asarray([row.critical_batch_tokens for row in valid], dtype=np.float64)
    ratio = float(values.max() / values.min())
    diagnostics["cross_estimator_ratio"] = ratio
    if ratio > maximum_ratio:
        return _refused(
            "consensus",
            ("qualified estimators disagree beyond the allowed ratio",),
            diagnostics,
        )
    consensus = float(np.exp(np.mean(np.log(values))))
    lowers = [row.lower_batch_tokens for row in valid if row.lower_batch_tokens is not None]
    uppers = [row.upper_batch_tokens for row in valid if row.upper_batch_tokens is not None]
    return CriticalBatchEstimate(
        "consensus",
        consensus,
        min(lowers) if lowers else float(values.min()),
        max(uppers) if uppers else float(values.max()),
        True,
        diagnostics,
    )


def estimate_loss_optimal_batch(
    losses_by_batch: Dict[int, Sequence[float]],
) -> Dict[str, Any]:
    """Report B_opt separately from every critical-batch definition."""
    if not losses_by_batch:
        raise ValueError("losses_by_batch cannot be empty")
    rows = []
    for batch, losses in sorted(losses_by_batch.items()):
        values = np.asarray(losses, dtype=np.float64)
        if batch <= 0 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("batches and losses must be positive, non-empty, and finite")
        rows.append(
            {
                "batch_tokens": batch,
                "mean_loss": float(values.mean()),
                "sem_loss": float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0,
            }
        )
    best = min(rows, key=lambda row: (row["mean_loss"], row["batch_tokens"]))
    return {
        "definition": "loss-optimal batch after independent hyperparameter tuning",
        "optimal_batch_tokens": best["batch_tokens"],
        "rows": rows,
    }
