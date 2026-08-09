from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import mean, stdev
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .training import TrialResult


@dataclass(frozen=True)
class LearningRateSummary:
    normalized_learning_rate: float
    mean_final_validation_loss: float
    sem_final_validation_loss: float
    losses_by_seed: Dict[int, float]
    diverged_seeds: Tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TuningResult:
    selected_normalized_learning_rate: float
    numerical_best_normalized_learning_rate: float
    selection_rule: str
    summaries: Tuple[LearningRateSummary, ...]
    selected_index: int
    optimum_is_interior: bool
    expansion_rounds: int
    flat_minimum: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_trials(
    trials: Iterable[TrialResult], normalized_eta: float
) -> LearningRateSummary:
    selected = [
        trial for trial in trials
        if trial.normalized_learning_rate == normalized_eta
    ]
    if not selected:
        raise ValueError(f"No trials found for normalized eta {normalized_eta}")
    losses = {trial.seed: trial.final_validation_loss for trial in selected}
    finite = [loss for loss in losses.values() if math.isfinite(loss)]
    diverged = tuple(sorted(seed for seed, loss in losses.items() if not math.isfinite(loss)))
    if not finite or diverged:
        return LearningRateSummary(normalized_eta, float("inf"), float("inf"), losses, diverged)
    average = mean(finite)
    sem = stdev(finite) / math.sqrt(len(finite)) if len(finite) > 1 else 0.0
    return LearningRateSummary(normalized_eta, average, sem, losses, diverged)


def adaptive_tune(
    initial_rates: Sequence[float],
    seeds: Sequence[int],
    run_trial: Callable[[float, int], TrialResult],
    *,
    max_expansion_rounds: int,
    expansion_factor: float,
) -> Tuple[TuningResult, Tuple[TrialResult, ...]]:
    rates = sorted(set(float(rate) for rate in initial_rates))
    all_trials: List[TrialResult] = []
    evaluated = set()
    expansion_rounds = 0
    while True:
        for rate in rates:
            if rate in evaluated:
                continue
            for seed in seeds:
                all_trials.append(run_trial(rate, seed))
            evaluated.add(rate)
        summaries = tuple(summarize_trials(all_trials, rate) for rate in sorted(evaluated))
        finite_indices = [
            index for index, summary in enumerate(summaries)
            if math.isfinite(summary.mean_final_validation_loss)
        ]
        if not finite_indices:
            raise RuntimeError("Every learning-rate candidate diverged")
        selected_index = min(finite_indices, key=lambda index: summaries[index].mean_final_validation_loss)
        at_lower_edge = selected_index == 0
        at_upper_edge = selected_index == len(summaries) - 1
        if not (at_lower_edge or at_upper_edge) or expansion_rounds >= max_expansion_rounds:
            break
        if at_lower_edge:
            rates.append(summaries[0].normalized_learning_rate / expansion_factor)
        else:
            rates.append(summaries[-1].normalized_learning_rate * expansion_factor)
        rates.sort()
        expansion_rounds += 1

    numerical_best_index = selected_index
    numerical_best = summaries[numerical_best_index]
    one_sem_threshold = numerical_best.mean_final_validation_loss + numerical_best.sem_final_validation_loss
    statistically_tied = [
        index for index in finite_indices
        if summaries[index].mean_final_validation_loss <= one_sem_threshold
    ]
    # Prefer the smallest stable normalized eta among statistically
    # indistinguishable minima.
    # This is fixed before transfer and prevents noisy aggressive candidates from
    # winning by a numerically tiny reference-scale margin.
    selected_index = min(
        statistically_tied,
        key=lambda index: summaries[index].normalized_learning_rate,
    )
    best = summaries[selected_index]
    flat = len(statistically_tied) > 1
    result = TuningResult(
        selected_normalized_learning_rate=best.normalized_learning_rate,
        numerical_best_normalized_learning_rate=numerical_best.normalized_learning_rate,
        selection_rule="lowest_normalized_eta_within_one_sem_of_numerical_minimum",
        summaries=summaries,
        selected_index=selected_index,
        # The one-SEM rule may deliberately select a conservative interior LR,
        # but it must not disguise an unresolved numerical optimum at a search
        # boundary.
        optimum_is_interior=0 < numerical_best_index < len(summaries) - 1,
        expansion_rounds=expansion_rounds,
        flat_minimum=flat,
    )
    return result, tuple(all_trials)


def paired_mean_and_sem(candidate: MappingLike, reference: MappingLike) -> Tuple[float, float]:
    shared = sorted(set(candidate) & set(reference))
    if len(shared) < 2:
        raise ValueError("Paired comparisons require at least two shared seeds")
    differences = [candidate[seed] - reference[seed] for seed in shared]
    average = mean(differences)
    sem = stdev(differences) / math.sqrt(len(differences))
    return average, sem


MappingLike = Dict[int, float]


STANDARD_RESIDUAL_MLP = "standard_residual_mlp"
CHIZAT_MEAN_FIELD = "chizat_mean_field"
MOE_TABLE1_ADAM = "moe_table1_adam"
NUGPT_MID_ALIGNMENT = "nugpt_mid_alignment"


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def raw_learning_rate_from_normalized_eta(
    parameterization: str,
    optimizer_name: str,
    normalized_eta: float,
    *,
    width: int,
    depth: int = 1,
    alpha: float = 1.0,
) -> float:
    """Convert the transferable coordinate ``eta`` to an optimizer LR.

    ``eta`` is the quantity tuned and held fixed across scale.  The conversion
    is part of the parameterization, not an empirical power fitted to the
    width-wise argmin of a finite-horizon sweep.

    For the MVP's fan-in residual MLP, SGD uses the already validated muP-style
    ``eta / sqrt(width)`` rule and Adam is scale invariant.  For Chizat's
    mean-field block, plain SGD/GD uses Eq. (5), ``L M eta / alpha^2``.  Adam's
    scale-normalized update is width independent while its epsilon remains
    negligible; that case must be validated separately before certification.
    """
    eta = _positive_finite(normalized_eta, "normalized_eta")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
        raise ValueError("depth must be a positive integer")
    alpha = _positive_finite(alpha, "alpha")
    if optimizer_name not in {"sgd", "adam"}:
        raise ValueError(f"Unsupported optimizer transfer rule: {optimizer_name}")
    if parameterization == STANDARD_RESIDUAL_MLP:
        return eta if optimizer_name == "adam" else eta / math.sqrt(width)
    if parameterization == MOE_TABLE1_ADAM:
        if optimizer_name != "adam":
            raise ValueError("moe_table1_adam does not certify SGD")
        return eta
    if parameterization == NUGPT_MID_ALIGNMENT:
        if optimizer_name != "adam":
            raise ValueError("nugpt_mid_alignment currently certifies only Adam")
        return eta
    if parameterization == CHIZAT_MEAN_FIELD:
        return eta if optimizer_name == "adam" else depth * width * eta / alpha ** 2
    raise ValueError(f"Unsupported parameterization: {parameterization}")


def normalized_eta_from_raw_learning_rate(
    parameterization: str,
    optimizer_name: str,
    raw_learning_rate: float,
    *,
    width: int,
    depth: int = 1,
    alpha: float = 1.0,
) -> float:
    """Inverse of :func:`raw_learning_rate_from_normalized_eta`."""
    raw = _positive_finite(raw_learning_rate, "raw_learning_rate")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
        raise ValueError("depth must be a positive integer")
    alpha = _positive_finite(alpha, "alpha")
    if optimizer_name not in {"sgd", "adam"}:
        raise ValueError(f"Unsupported optimizer transfer rule: {optimizer_name}")
    if optimizer_name == "adam":
        return raw
    if parameterization == STANDARD_RESIDUAL_MLP:
        return raw * math.sqrt(width)
    if parameterization == CHIZAT_MEAN_FIELD:
        return raw * alpha ** 2 / (depth * width)
    raise ValueError(f"Unsupported parameterization: {parameterization}")


def optimizer_group_learning_rates_from_normalized_eta(
    parameterization: str,
    optimizer_name: str,
    normalized_eta: float,
    *,
    width: int,
    depth: int = 1,
    alpha: float = 1.0,
    expert_width: Optional[int] = None,
    reference_width: Optional[int] = None,
) -> Dict[str, float]:
    """Return every raw optimizer rate implied by one normalized coordinate."""
    base = raw_learning_rate_from_normalized_eta(
        parameterization,
        optimizer_name,
        normalized_eta,
        width=width,
        depth=depth,
        alpha=alpha,
    )
    if parameterization == NUGPT_MID_ALIGNMENT:
        if (
            isinstance(reference_width, bool)
            or not isinstance(reference_width, int)
            or reference_width <= 0
        ):
            raise ValueError("reference_width must be a positive integer for nuGPT")
        width_multiplier = width / reference_width
        hidden_rate = base * width_multiplier ** -0.75
        return {
            "nugpt_input": base * width_multiplier ** -0.5,
            "nugpt_hidden": hidden_rate,
            "nugpt_output": 0.5 * hidden_rate,
            "nugpt_rescalers": base,
        }
    if parameterization != MOE_TABLE1_ADAM:
        return {"all": base}
    if isinstance(expert_width, bool) or not isinstance(expert_width, int) or expert_width <= 0:
        raise ValueError("expert_width must be a positive integer")
    return {
        "adapters_and_norms": base,
        "readout_weight": base / width,
        "readout_bias": base,
        "moe_router": base / width,
        "moe_up": base / width,
        "moe_down": base / expert_width,
    }


@dataclass(frozen=True)
class FixedEtaTransferDiagnostics:
    """Finite-width convergence diagnostics at one fixed normalized eta."""

    dial_values: Tuple[float, ...]
    mean_losses: Tuple[float, ...]
    sem_losses: Tuple[float, ...]
    adjacent_absolute_gaps: Tuple[float, ...]
    adjacent_paired_gap_sems: Tuple[float, ...]
    absolute_spread: float
    log10_spread: float
    finite_size_exponent: float
    finite_size_intercept: float
    finite_size_slope: float
    finite_size_r_squared: float
    head_gap: float
    tail_gap: float
    noise_allowance: float
    settling: bool
    all_finite: bool
    accepted: bool
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def fixed_eta_transfer_diagnostics(
    dial_values: Sequence[float],
    losses_by_dial_seed: Sequence[Mapping[int, float]],
    *,
    settling_ratio: float = 1.3,
    finite_size_exponent: float = -0.5,
) -> FixedEtaTransferDiagnostics:
    """Test fixed-eta convergence without consulting a width-wise LR argmin.

    The leading finite-population correction is fitted against ``M^-1/2``.
    Acceptance requires finite runs and adjacent differences that settle toward
    the large-width end, up to a measured two-SEM noise allowance.  The
    log-loss spread is reported only as a descriptive statistic because it is
    ill-conditioned when losses approach zero.
    """
    dials = tuple(float(value) for value in dial_values)
    if not math.isfinite(finite_size_exponent) or finite_size_exponent >= 0.0:
        raise ValueError("finite_size_exponent must be finite and negative")
    if len(dials) < 3 or len(dials) != len(losses_by_dial_seed):
        raise ValueError("fixed-eta diagnostics require at least three aligned dial values")
    if any(not math.isfinite(value) or value <= 0.0 for value in dials):
        raise ValueError("dial values must be finite and positive")
    if any(right <= left for left, right in zip(dials, dials[1:])):
        raise ValueError("dial values must be strictly increasing")
    shared = set(losses_by_dial_seed[0])
    for row in losses_by_dial_seed[1:]:
        shared &= set(row)
    if len(shared) < 2:
        raise ValueError("fixed-eta diagnostics require at least two common seeds")
    seeds = sorted(shared)
    all_finite = all(
        math.isfinite(float(losses_by_dial_seed[index][seed]))
        for index in range(len(dials))
        for seed in seeds
    )
    means: List[float] = []
    sems: List[float] = []
    for row in losses_by_dial_seed:
        values = [float(row[seed]) for seed in seeds]
        if not all(math.isfinite(value) for value in values):
            means.append(float("inf"))
            sems.append(float("inf"))
            continue
        means.append(mean(values))
        sems.append(stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0)

    if not all_finite:
        return FixedEtaTransferDiagnostics(
            dials, tuple(means), tuple(sems), (), (), float("inf"), float("inf"),
            finite_size_exponent, float("nan"), float("nan"), float("nan"),
            float("inf"), float("inf"), float("inf"), False, False, False,
            "FAILS (non-finite fixed-eta trajectory)",
        )

    gaps = tuple(abs(right - left) for left, right in zip(means, means[1:]))
    paired_gap_sems = []
    for left, right in zip(losses_by_dial_seed, losses_by_dial_seed[1:]):
        differences = [float(right[seed]) - float(left[seed]) for seed in seeds]
        paired_gap_sems.append(
            stdev(differences) / math.sqrt(len(differences))
            if len(differences) > 1 else 0.0
        )
    spread = max(means) - min(means)
    positive_means = [value for value in means if value > 0.0]
    log_spread = (
        max(math.log10(value) for value in positive_means)
        - min(math.log10(value) for value in positive_means)
        if len(positive_means) == len(means)
        else float("inf")
    )
    x = [value ** finite_size_exponent for value in dials]
    x_bar, y_bar = mean(x), mean(means)
    x_var = sum((value - x_bar) ** 2 for value in x)
    slope = sum((a - x_bar) * (b - y_bar) for a, b in zip(x, means)) / x_var
    intercept = y_bar - slope * x_bar
    predictions = [intercept + slope * value for value in x]
    residual = sum((actual - predicted) ** 2 for actual, predicted in zip(means, predictions))
    total = sum((actual - y_bar) ** 2 for actual in means)
    r_squared = 1.0 if total <= 1e-30 else 1.0 - residual / total
    head_gap, tail_gap = gaps[0], gaps[-1]
    noise_allowance = 2.0 * max(paired_gap_sems)
    settling = tail_gap <= settling_ratio * head_gap + noise_allowance
    statistically_flat = spread <= noise_allowance
    accepted = all_finite and (settling or statistically_flat)
    status = "TRANSFERS (fixed eta; finite-width differences settle)" if accepted else (
        "SUSPECT (fixed-eta differences grow toward the largest widths)"
    )
    return FixedEtaTransferDiagnostics(
        dials, tuple(means), tuple(sems), gaps, tuple(paired_gap_sems),
        spread, log_spread, finite_size_exponent, intercept, slope,
        r_squared, head_gap, tail_gap, noise_allowance, settling, all_finite,
        accepted, status,
    )


def fixed_eta_noninferiority(
    transferred: MappingLike,
    conservative: MappingLike,
    *,
    relative_tolerance: float = 0.02,
) -> Dict[str, Any]:
    """Gate a fixed eta against a lower, conservative eta using paired seeds.

    A higher-rate probe that performs better is deliberately irrelevant here:
    it measures extra stability headroom, not failure of transfer.
    """
    if not transferred or not conservative:
        raise ValueError("noninferiority requires transferred and conservative trials")
    transferred_values = [float(value) for value in transferred.values()]
    conservative_values = [float(value) for value in conservative.values()]
    if not all(math.isfinite(value) for value in transferred_values):
        return {
            "accepted": False,
            "paired_loss_penalty": float("inf"),
            "paired_loss_penalty_sem": float("inf"),
            "tolerance": float("inf"),
        }
    finite_conservative = [value for value in conservative_values if math.isfinite(value)]
    if len(finite_conservative) != len(conservative_values):
        return {
            "accepted": True,
            "paired_loss_penalty": float("-inf"),
            "paired_loss_penalty_sem": float("inf"),
            "tolerance": float("inf"),
        }
    penalty, penalty_sem = paired_mean_and_sem(transferred, conservative)
    tolerance = max(2.0 * penalty_sem, relative_tolerance * mean(finite_conservative))
    return {
        "accepted": penalty <= tolerance,
        "paired_loss_penalty": penalty,
        "paired_loss_penalty_sem": penalty_sem,
        "tolerance": tolerance,
    }


def transfer_rule_name(
    optimizer_name: str, parameterization: str = STANDARD_RESIDUAL_MLP
) -> str:
    if parameterization == MOE_TABLE1_ADAM:
        if optimizer_name != "adam":
            raise ValueError("moe_table1_adam does not certify SGD")
        return "moe_table1_group_rates_from_normalized_eta"
    if parameterization == NUGPT_MID_ALIGNMENT:
        if optimizer_name != "adam":
            raise ValueError("nugpt_mid_alignment currently certifies only Adam")
        return "nugpt_mid_alignment_group_rates_with_post_step_sphere_projection"
    if optimizer_name == "adam":
        return "raw_lr_equals_normalized_eta"
    if optimizer_name == "sgd":
        return "raw_lr_equals_normalized_eta_over_sqrt_width"
    raise ValueError(f"Unsupported optimizer transfer rule: {optimizer_name}")
