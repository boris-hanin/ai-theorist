from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Mapping, Tuple


SCHEDULE_FAMILIES: Tuple[str, ...] = (
    "constant",
    "cosine_to_fraction",
    "linear_warmup_decay",
    "warmup_stable_decay",
)


def _fraction(value: Any, name: str, *, allow_one: bool = True) -> float:
    result = float(value)
    upper_ok = result <= 1.0 if allow_one else result < 1.0
    if not math.isfinite(result) or result < 0.0 or not upper_ok:
        bracket = "[0, 1]" if allow_one else "[0, 1)"
        raise ValueError(f"{name} must be in {bracket}")
    return result


@dataclass(frozen=True)
class LearningRateSchedule:
    """A normalized-time schedule whose peak is supplied by the optimizer contract."""

    family: str
    terminal_fraction: float = 0.0
    warmup_fraction: float = 0.0
    stable_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.family not in SCHEDULE_FAMILIES:
            raise ValueError(
                f"schedule family must be one of {', '.join(SCHEDULE_FAMILIES)}"
            )
        _fraction(self.terminal_fraction, "terminal_fraction")
        _fraction(self.warmup_fraction, "warmup_fraction", allow_one=False)
        _fraction(self.stable_fraction, "stable_fraction")
        if self.warmup_fraction + self.stable_fraction > 1.0:
            raise ValueError("warmup_fraction + stable_fraction cannot exceed one")
        if self.family == "constant" and (
            self.warmup_fraction or self.stable_fraction or self.terminal_fraction
        ):
            raise ValueError("constant schedule does not accept shape parameters")
        if self.family == "cosine_to_fraction" and (
            self.warmup_fraction or self.stable_fraction
        ):
            raise ValueError("cosine_to_fraction does not accept warmup/stable fractions")
        if self.family == "linear_warmup_decay" and self.stable_fraction:
            raise ValueError("linear_warmup_decay does not accept stable_fraction")

    @classmethod
    def from_payload(cls, payload: Any) -> "LearningRateSchedule":
        if isinstance(payload, str):
            if payload == "cosine_to_10_percent":
                return cls("cosine_to_fraction", terminal_fraction=0.1)
            if payload == "linear_warmup_decay_to_zero":
                return cls("linear_warmup_decay", warmup_fraction=0.1)
            if payload == "wsd":
                return cls(
                    "warmup_stable_decay",
                    warmup_fraction=0.02,
                    stable_fraction=0.78,
                )
            return cls(payload)
        if not isinstance(payload, Mapping):
            raise ValueError("learning-rate schedule must be a name or object")
        return cls(
            family=str(payload["family"]),
            terminal_fraction=float(payload.get("terminal_fraction", 0.0)),
            warmup_fraction=float(payload.get("warmup_fraction", 0.0)),
            stable_fraction=float(payload.get("stable_fraction", 0.0)),
        )

    @property
    def name(self) -> str:
        if self.family == "cosine_to_fraction":
            return f"cosine_to_{self.terminal_fraction:g}"
        if self.family == "linear_warmup_decay":
            return f"linear_warmup_{self.warmup_fraction:g}_decay_to_{self.terminal_fraction:g}"
        if self.family == "warmup_stable_decay":
            return (
                f"wsd_warmup_{self.warmup_fraction:g}_stable_{self.stable_fraction:g}"
                f"_to_{self.terminal_fraction:g}"
            )
        return "constant"

    def multiplier(self, step: int, total_steps: int) -> float:
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError("step must be a positive integer")
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        if step > total_steps:
            raise ValueError("step cannot exceed total_steps")
        if self.family == "constant" or total_steps == 1:
            return 1.0

        position = (step - 1) / (total_steps - 1)
        if self.family == "cosine_to_fraction":
            cosine = 0.5 * (1.0 + math.cos(math.pi * position))
            return self.terminal_fraction + (1.0 - self.terminal_fraction) * cosine

        if self.warmup_fraction > 0.0 and position < self.warmup_fraction:
            # The first update is deliberately nonzero even for very short runs.
            warmup_steps = max(1, int(math.ceil(self.warmup_fraction * total_steps)))
            return min(1.0, step / warmup_steps)

        decay_start = (
            self.warmup_fraction + self.stable_fraction
            if self.family == "warmup_stable_decay"
            else self.warmup_fraction
        )
        if position <= decay_start:
            return 1.0
        decay_position = (position - decay_start) / max(1e-12, 1.0 - decay_start)
        if self.family == "linear_warmup_decay":
            return 1.0 - (1.0 - self.terminal_fraction) * decay_position
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_position))
        return self.terminal_fraction + (1.0 - self.terminal_fraction) * cosine

    def audit(self, total_steps: int) -> Dict[str, Any]:
        values = [self.multiplier(step, total_steps) for step in range(1, total_steps + 1)]
        return {
            **asdict(self),
            "name": self.name,
            "total_steps": total_steps,
            "first_multiplier": values[0],
            "peak_multiplier": max(values),
            "last_multiplier": values[-1],
            "mean_multiplier": sum(values) / len(values),
            "integrated_multiplier_steps": sum(values),
        }
