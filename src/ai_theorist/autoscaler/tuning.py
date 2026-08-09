from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import mean, stdev
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from .training import TrialResult


@dataclass(frozen=True)
class LearningRateSummary:
    learning_rate: float
    mean_final_validation_loss: float
    sem_final_validation_loss: float
    losses_by_seed: Dict[int, float]
    diverged_seeds: Tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TuningResult:
    selected_learning_rate: float
    numerical_best_learning_rate: float
    selection_rule: str
    summaries: Tuple[LearningRateSummary, ...]
    selected_index: int
    optimum_is_interior: bool
    expansion_rounds: int
    flat_minimum: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_trials(trials: Iterable[TrialResult], learning_rate: float) -> LearningRateSummary:
    selected = [trial for trial in trials if trial.learning_rate == learning_rate]
    if not selected:
        raise ValueError(f"No trials found for learning rate {learning_rate}")
    losses = {trial.seed: trial.final_validation_loss for trial in selected}
    finite = [loss for loss in losses.values() if math.isfinite(loss)]
    diverged = tuple(sorted(seed for seed, loss in losses.items() if not math.isfinite(loss)))
    if not finite or diverged:
        return LearningRateSummary(learning_rate, float("inf"), float("inf"), losses, diverged)
    average = mean(finite)
    sem = stdev(finite) / math.sqrt(len(finite)) if len(finite) > 1 else 0.0
    return LearningRateSummary(learning_rate, average, sem, losses, diverged)


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
            rates.append(summaries[0].learning_rate / expansion_factor)
        else:
            rates.append(summaries[-1].learning_rate * expansion_factor)
        rates.sort()
        expansion_rounds += 1

    numerical_best_index = selected_index
    numerical_best = summaries[numerical_best_index]
    one_sem_threshold = numerical_best.mean_final_validation_loss + numerical_best.sem_final_validation_loss
    statistically_tied = [
        index for index in finite_indices
        if summaries[index].mean_final_validation_loss <= one_sem_threshold
    ]
    # Prefer the smallest stable LR among statistically indistinguishable minima.
    # This is fixed before transfer and prevents noisy aggressive candidates from
    # winning by a numerically tiny reference-scale margin.
    selected_index = min(statistically_tied, key=lambda index: summaries[index].learning_rate)
    best = summaries[selected_index]
    flat = len(statistically_tied) > 1
    result = TuningResult(
        selected_learning_rate=best.learning_rate,
        numerical_best_learning_rate=numerical_best.learning_rate,
        selection_rule="lowest_learning_rate_within_one_sem_of_numerical_minimum",
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


def transfer_learning_rate(
    optimizer_name: str,
    base_learning_rate: float,
    reference_width: int,
    target_width: int,
) -> float:
    if optimizer_name == "adam":
        return base_learning_rate
    if optimizer_name == "sgd":
        return base_learning_rate * math.sqrt(reference_width / target_width)
    raise ValueError(f"Unsupported optimizer transfer rule: {optimizer_name}")


def transfer_rule_name(optimizer_name: str) -> str:
    if optimizer_name == "adam":
        return "constant_global_learning_rate"
    if optimizer_name == "sgd":
        return "inverse_sqrt_width_learning_rate"
    raise ValueError(f"Unsupported optimizer transfer rule: {optimizer_name}")
