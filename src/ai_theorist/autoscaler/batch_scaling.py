from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Callable, Dict, Mapping, Optional, Tuple


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _probability(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
    return result


@dataclass(frozen=True)
class OptimizerHyperparameters:
    """Optimizer coordinates stored independently of a framework implementation."""

    name: str
    learning_rate: float
    momentum: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.name not in {"sgd", "adam", "adamw"}:
            raise ValueError("optimizer must be sgd, adam, or adamw")
        _positive(self.learning_rate, "learning_rate")
        _probability(self.momentum, "momentum")
        _probability(self.beta1, "beta1")
        _probability(self.beta2, "beta2")
        _positive(self.epsilon, "epsilon")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransferContext:
    base_parameters: int
    target_parameters: int
    base_total_tokens: int
    target_total_tokens: int
    base_batch_tokens: int
    target_batch_tokens: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def parameter_multiplier(self) -> float:
        return self.target_parameters / self.base_parameters

    @property
    def horizon_multiplier(self) -> float:
        return self.target_total_tokens / self.base_total_tokens

    @property
    def batch_multiplier(self) -> float:
        return self.target_batch_tokens / self.base_batch_tokens

    @property
    def normalized_time_multiplier(self) -> float:
        """The joint batch/duration coordinate q = m_B / m_T."""
        return self.batch_multiplier / self.horizon_multiplier

    @property
    def base_tpp(self) -> float:
        return self.base_total_tokens / self.base_parameters

    @property
    def target_tpp(self) -> float:
        return self.target_total_tokens / self.target_parameters


@dataclass(frozen=True)
class TransferResult:
    rule: str
    source: OptimizerHyperparameters
    target: Optional[OptimizerHyperparameters]
    multipliers: Mapping[str, float]
    assumptions: Tuple[str, ...]
    valid: bool
    refusal_reasons: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule,
            "source": self.source.to_dict(),
            "target": self.target.to_dict() if self.target is not None else None,
            "multipliers": dict(self.multipliers),
            "assumptions": list(self.assumptions),
            "valid": self.valid,
            "refusal_reasons": list(self.refusal_reasons),
        }


@dataclass(frozen=True)
class TransferRule:
    name: str
    summary: str
    supported_optimizers: Tuple[str, ...]
    citation: str
    transform: Callable[[OptimizerHyperparameters, TransferContext, float], TransferResult]

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "summary": self.summary,
            "supported_optimizers": list(self.supported_optimizers),
            "citation": self.citation,
        }


def _invalid(
    rule: str,
    source: OptimizerHyperparameters,
    reason: str,
    assumptions: Tuple[str, ...],
) -> TransferResult:
    return TransferResult(rule, source, None, {}, assumptions, False, (reason,))


def _replace_optimizer(
    source: OptimizerHyperparameters,
    *,
    learning_rate_multiplier: float,
    epsilon_multiplier: float = 1.0,
    weight_decay_multiplier: float = 1.0,
    beta_gap_multiplier: Optional[float] = None,
    exact_beta_time_multiplier: Optional[float] = None,
) -> Optional[OptimizerHyperparameters]:
    beta1, beta2 = source.beta1, source.beta2
    if beta_gap_multiplier is not None:
        beta1 = 1.0 - beta_gap_multiplier * (1.0 - beta1)
        beta2 = 1.0 - beta_gap_multiplier * (1.0 - beta2)
    elif exact_beta_time_multiplier is not None:
        beta1 = source.beta1 ** exact_beta_time_multiplier
        beta2 = source.beta2 ** exact_beta_time_multiplier
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        return None
    return OptimizerHyperparameters(
        name=source.name,
        learning_rate=source.learning_rate * learning_rate_multiplier,
        momentum=source.momentum,
        beta1=beta1,
        beta2=beta2,
        epsilon=source.epsilon * epsilon_multiplier,
        weight_decay=source.weight_decay * weight_decay_multiplier,
    )


def _identity(
    source: OptimizerHyperparameters, context: TransferContext, _: float
) -> TransferResult:
    del context
    return TransferResult(
        "none",
        source,
        source,
        {"learning_rate": 1.0, "epsilon": 1.0, "weight_decay": 1.0},
        ("All optimizer coordinates are deliberately held fixed.",),
        True,
    )


def _sgd_linear(
    source: OptimizerHyperparameters, context: TransferContext, _: float
) -> TransferResult:
    assumptions = (
        "Only global batch changes; total token horizon is fixed.",
        "The target batch remains below the critical batch size.",
    )
    if source.name != "sgd":
        return _invalid("sgd_linear_batch", source, "rule requires SGD", assumptions)
    if not math.isclose(context.horizon_multiplier, 1.0):
        return _invalid(
            "sgd_linear_batch", source, "duration changes are outside this rule", assumptions
        )
    factor = context.batch_multiplier
    target = _replace_optimizer(source, learning_rate_multiplier=factor)
    return TransferResult(
        "sgd_linear_batch", source, target, {"learning_rate": factor}, assumptions, True
    )


def _adam_sde(
    source: OptimizerHyperparameters, context: TransferContext, _: float
) -> TransferResult:
    assumptions = (
        "Only global batch changes; total token horizon is fixed.",
        "The discrete optimizer remains in the SDE-valid regime.",
    )
    if source.name not in {"adam", "adamw"}:
        return _invalid("adam_sde_sqrt", source, "rule requires Adam or AdamW", assumptions)
    if not math.isclose(context.horizon_multiplier, 1.0):
        return _invalid(
            "adam_sde_sqrt", source, "duration changes require the joint rule", assumptions
        )
    factor = context.batch_multiplier
    target = _replace_optimizer(
        source,
        learning_rate_multiplier=math.sqrt(factor),
        epsilon_multiplier=1.0 / math.sqrt(factor),
        beta_gap_multiplier=factor,
    )
    if target is None:
        return _invalid(
            "adam_sde_sqrt",
            source,
            "scaled beta gap leaves [0, 1); use smaller stages",
            assumptions,
        )
    return TransferResult(
        "adam_sde_sqrt",
        source,
        target,
        {
            "learning_rate": math.sqrt(factor),
            "epsilon": 1.0 / math.sqrt(factor),
            "beta_gap": factor,
        },
        assumptions,
        True,
    )


def _complete_dp(
    source: OptimizerHyperparameters, context: TransferContext, _: float
) -> TransferResult:
    assumptions = (
        "Training is compared at matched normalized time.",
        "q = (target batch/base batch) / (target tokens/base tokens).",
        "PyTorch-style decoupled weight decay is used when the optimizer is AdamW.",
    )
    if source.name not in {"adam", "adamw"}:
        return _invalid("complete_dp_joint", source, "rule requires Adam or AdamW", assumptions)
    q = context.normalized_time_multiplier
    target = _replace_optimizer(
        source,
        learning_rate_multiplier=math.sqrt(q),
        epsilon_multiplier=1.0 / math.sqrt(q),
        weight_decay_multiplier=math.sqrt(q),
        beta_gap_multiplier=q,
    )
    if target is None:
        return _invalid(
            "complete_dp_joint",
            source,
            "scaled beta gap leaves [0, 1); use smaller stages",
            assumptions,
        )
    return TransferResult(
        "complete_dp_joint",
        source,
        target,
        {
            "learning_rate": math.sqrt(q),
            "epsilon": 1.0 / math.sqrt(q),
            "weight_decay": math.sqrt(q),
            "beta_gap": q,
        },
        assumptions,
        True,
    )


def _exact_token_half_life(
    source: OptimizerHyperparameters, context: TransferContext, _: float
) -> TransferResult:
    assumptions = (
        "Adam moment half-lives are preserved exactly in normalized-time units.",
        "Learning-rate and epsilon coordinates use the joint square-root rule.",
    )
    if source.name not in {"adam", "adamw"}:
        return _invalid("exact_token_half_life", source, "rule requires Adam or AdamW", assumptions)
    q = context.normalized_time_multiplier
    target = _replace_optimizer(
        source,
        learning_rate_multiplier=math.sqrt(q),
        epsilon_multiplier=1.0 / math.sqrt(q),
        weight_decay_multiplier=math.sqrt(q),
        exact_beta_time_multiplier=q,
    )
    return TransferResult(
        "exact_token_half_life",
        source,
        target,
        {
            "learning_rate": math.sqrt(q),
            "epsilon": 1.0 / math.sqrt(q),
            "weight_decay": math.sqrt(q),
            "beta_exponent": q,
        },
        assumptions,
        True,
    )


def _horizon_power(
    source: OptimizerHyperparameters, context: TransferContext, exponent: float
) -> TransferResult:
    assumptions = (
        "The supplied exponent was fitted without using the held-out target scale.",
        "Batch size is unchanged; this rule only transfers across duration.",
    )
    if not math.isclose(context.batch_multiplier, 1.0):
        return _invalid(
            "horizon_power_fit", source, "batch changes are outside this fitted rule", assumptions
        )
    if not math.isfinite(exponent) or exponent <= 0.0:
        return _invalid("horizon_power_fit", source, "exponent must be positive", assumptions)
    factor = context.horizon_multiplier ** (-exponent)
    target = _replace_optimizer(source, learning_rate_multiplier=factor)
    return TransferResult(
        "horizon_power_fit",
        source,
        target,
        {"learning_rate": factor, "horizon_exponent": exponent},
        assumptions,
        True,
    )


TRANSFER_RULES: Dict[str, TransferRule] = {
    rule.name: rule
    for rule in (
        TransferRule(
            "none",
            "Keep every optimizer coordinate fixed.",
            ("sgd", "adam", "adamw"),
            "negative control",
            _identity,
        ),
        TransferRule(
            "sgd_linear_batch",
            "Scale SGD learning rate linearly with batch at fixed duration.",
            ("sgd",),
            "Goyal et al. (2017)",
            _sgd_linear,
        ),
        TransferRule(
            "adam_sde_sqrt",
            "Adam SDE batch rule: sqrt LR, beta gaps, inverse-sqrt epsilon.",
            ("adam", "adamw"),
            "Malladi et al. (2022)",
            _adam_sde,
        ),
        TransferRule(
            "complete_dp_joint",
            "Joint batch-duration transform in q = m_B / m_T.",
            ("adam", "adamw"),
            "Complete(d)P (2025)",
            _complete_dp,
        ),
        TransferRule(
            "exact_token_half_life",
            "Exact-beta control preserving optimizer memory in normalized time.",
            ("adam", "adamw"),
            "internal mechanistic control",
            _exact_token_half_life,
        ),
        TransferRule(
            "horizon_power_fit",
            "Fit learning rate as a power law of token horizon.",
            ("sgd", "adam", "adamw"),
            "Bjorck et al. (2024)",
            _horizon_power,
        ),
    )
}


def transfer_rule_registry() -> Dict[str, Dict[str, Any]]:
    return {name: rule.describe() for name, rule in sorted(TRANSFER_RULES.items())}


def apply_transfer_rule(
    rule_name: str,
    source: OptimizerHyperparameters,
    context: TransferContext,
    *,
    horizon_exponent: float = 0.32,
) -> TransferResult:
    try:
        rule = TRANSFER_RULES[rule_name]
    except KeyError as exc:
        raise ValueError(f"unknown transfer rule: {rule_name}") from exc
    if source.name not in rule.supported_optimizers:
        return _invalid(
            rule_name,
            source,
            f"rule does not support {source.name}",
            (rule.summary,),
        )
    return rule.transform(source, context, horizon_exponent)


@dataclass(frozen=True)
class BatchRunRecord:
    """Canonical, framework-neutral record for one batch-scaling run."""

    run_id: str
    model_family: str
    optimizer: OptimizerHyperparameters
    seed: int
    parameter_count: int
    width: int
    depth: int
    total_tokens: int
    batch_tokens: int
    microbatch_tokens: int
    accumulation_steps: int
    data_parallel_replicas: int
    optimizer_steps: int
    nonpadding_tokens_seen: int
    learning_rate_schedule: str
    final_validation_loss: float
    estimated_flops: Optional[float] = None
    wall_time_seconds: Optional[float] = None
    target_loss_crossings: Mapping[str, Optional[int]] = field(default_factory=dict)
    validation_checkpoints: Tuple[Mapping[str, float], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported batch run record schema_version")
        if not self.run_id or not self.model_family or not self.learning_rate_schedule:
            raise ValueError("run_id, model_family, and learning_rate_schedule are required")
        for name in (
            "parameter_count",
            "width",
            "depth",
            "total_tokens",
            "batch_tokens",
            "microbatch_tokens",
            "accumulation_steps",
            "data_parallel_replicas",
            "optimizer_steps",
            "nonpadding_tokens_seen",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_tokens != (
            self.microbatch_tokens * self.accumulation_steps * self.data_parallel_replicas
        ):
            raise ValueError("batch_tokens must equal microbatch * accumulation * replicas")
        if not math.isfinite(self.final_validation_loss):
            raise ValueError("final_validation_loss must be finite")
        for name, value in (
            ("estimated_flops", self.estimated_flops),
            ("wall_time_seconds", self.wall_time_seconds),
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def tokens_per_parameter(self) -> float:
        return self.total_tokens / self.parameter_count

    def beta_half_life_tokens(self, beta: float) -> float:
        if beta == 0.0:
            return 0.0
        return self.batch_tokens * math.log(0.5) / math.log(beta)

    @property
    def optimizer_timescales(self) -> Dict[str, float]:
        if self.optimizer.name == "sgd":
            return {
                "momentum_half_life_tokens": self.beta_half_life_tokens(
                    self.optimizer.momentum
                )
            }
        return {
            "beta1_half_life_tokens": self.beta_half_life_tokens(self.optimizer.beta1),
            "beta2_half_life_tokens": self.beta_half_life_tokens(self.optimizer.beta2),
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["optimizer"] = self.optimizer.to_dict()
        payload["tokens_per_parameter"] = self.tokens_per_parameter
        payload["optimizer_timescales"] = self.optimizer_timescales
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BatchRunRecord":
        data = dict(payload)
        data.pop("tokens_per_parameter", None)
        data.pop("optimizer_timescales", None)
        data["optimizer"] = OptimizerHyperparameters(**dict(data["optimizer"]))
        data["validation_checkpoints"] = tuple(data.get("validation_checkpoints", ()))
        return cls(**data)
