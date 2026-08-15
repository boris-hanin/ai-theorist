from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Sequence, Tuple

from .critical_batch import CriticalBatchEstimate


@dataclass(frozen=True)
class SchedulePoint:
    start_tokens: int
    learning_rate: float

    def __post_init__(self) -> None:
        if self.start_tokens < 0:
            raise ValueError("start_tokens must be non-negative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")


@dataclass(frozen=True)
class SeesawStage:
    start_tokens: int
    baseline_learning_rate: float
    learning_rate: float
    batch_tokens: int
    cumulative_batch_multiplier: float
    serial_step_multiplier: float


def compile_seesaw_schedule(
    baseline: Sequence[SchedulePoint],
    *,
    initial_batch_tokens: int,
    critical_batch_consensus: CriticalBatchEstimate,
    variance_dominated: bool,
    safety_fraction: float = 0.8,
    maximum_single_cut: float = 4.0,
) -> Dict[str, Any]:
    """Compile Seesaw stages only after the static critical-batch gate passes."""
    refusal_reasons = []
    if initial_batch_tokens <= 0:
        raise ValueError("initial_batch_tokens must be positive")
    if not 0.0 < safety_fraction <= 1.0:
        raise ValueError("safety_fraction must lie in (0, 1]")
    if maximum_single_cut <= 1.0:
        raise ValueError("maximum_single_cut must be > 1")
    if not baseline:
        raise ValueError("baseline schedule cannot be empty")
    ordered = tuple(sorted(baseline, key=lambda point: point.start_tokens))
    if tuple(baseline) != ordered or len({point.start_tokens for point in ordered}) != len(ordered):
        raise ValueError("baseline schedule must have unique, increasing token boundaries")
    if not critical_batch_consensus.qualified or critical_batch_consensus.critical_batch_tokens is None:
        refusal_reasons.append("critical-batch consensus has not qualified")
    if not variance_dominated:
        refusal_reasons.append("the late-training regime has not been shown to be variance dominated")
    if refusal_reasons:
        return {
            "qualified": False,
            "refusal_reasons": refusal_reasons,
            "stages": [],
            "negative_control": None,
        }

    batch_cap = int(
        math.floor(safety_fraction * critical_batch_consensus.critical_batch_tokens)
    )
    if initial_batch_tokens > batch_cap:
        return {
            "qualified": False,
            "refusal_reasons": ["initial batch already exceeds the critical-batch safety cap"],
            "batch_cap_tokens": batch_cap,
            "stages": [],
            "negative_control": None,
        }

    stages = [
        SeesawStage(
            ordered[0].start_tokens,
            ordered[0].learning_rate,
            ordered[0].learning_rate,
            initial_batch_tokens,
            1.0,
            1.0,
        )
    ]
    current_batch = initial_batch_tokens
    current_rate = ordered[0].learning_rate
    previous_baseline_rate = ordered[0].learning_rate
    stopped_at_cap = False
    for point in ordered[1:]:
        cut = previous_baseline_rate / point.learning_rate
        if cut < 1.0:
            raise ValueError("baseline learning rate must be non-increasing")
        if cut > maximum_single_cut:
            return {
                "qualified": False,
                "refusal_reasons": [
                    f"a single learning-rate cut ({cut:.3g}x) exceeds the staged safety limit"
                ],
                "batch_cap_tokens": batch_cap,
                "stages": [asdict(stage) for stage in stages],
                "negative_control": None,
            }
        proposed_batch = max(current_batch, int(round(current_batch * cut)))
        if proposed_batch > batch_cap:
            stopped_at_cap = True
            break
        current_batch = proposed_batch
        current_rate = current_rate / math.sqrt(cut)
        stages.append(
            SeesawStage(
                point.start_tokens,
                point.learning_rate,
                current_rate,
                current_batch,
                current_batch / initial_batch_tokens,
                initial_batch_tokens / current_batch,
            )
        )
        previous_baseline_rate = point.learning_rate

    # The deliberately aggressive control grows batch by cut^2 and leaves LR at
    # the baseline value.  It must never be silently used as a recommendation.
    negative_control = []
    control_batch = initial_batch_tokens
    for previous, point in zip(ordered, ordered[1:]):
        cut = previous.learning_rate / point.learning_rate
        control_batch = int(round(control_batch * cut * cut))
        negative_control.append(
            {
                "start_tokens": point.start_tokens,
                "learning_rate": point.learning_rate,
                "batch_tokens": control_batch,
                "intentionally_aggressive": True,
            }
        )
    return {
        "qualified": True,
        "refusal_reasons": [],
        "batch_cap_tokens": batch_cap,
        "stopped_at_cap": stopped_at_cap,
        "source_consensus": critical_batch_consensus.to_dict(),
        "stages": [asdict(stage) for stage in stages],
        "negative_control": negative_control,
    }
