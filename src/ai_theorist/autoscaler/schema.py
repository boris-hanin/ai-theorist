from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


class SpecError(ValueError):
    """Raised when a study specification is unsafe or internally inconsistent."""


def _strict_keys(data: Mapping[str, Any], allowed: Iterable[str], context: str) -> None:
    extras = sorted(set(data) - set(allowed))
    if extras:
        raise SpecError(f"Unknown {context} field(s): {', '.join(extras)}")


def _positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SpecError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_float(value: Any, name: str, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise SpecError(f"{name} must be finite{suffix}")
    return result


@dataclass(frozen=True)
class ArchitectureTemplate:
    block_type: str = "pre_norm_mlp"
    activation: str = "gelu"
    input_dim: int = 16
    output_dim: int = 1
    residual_multiplier: float = 1.0
    num_experts: int = 1
    active_experts: int = 1
    router_balance_rate: float = 0.1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArchitectureTemplate":
        _strict_keys(
            data,
            {
                "block_type",
                "activation",
                "input_dim",
                "output_dim",
                "residual_multiplier",
                "num_experts",
                "active_experts",
                "router_balance_rate",
            },
            "architecture",
        )
        obj = cls(**dict(data))
        if obj.block_type not in {"pre_norm_mlp", "pre_norm_moe"}:
            raise SpecError("block_type must be pre_norm_mlp or pre_norm_moe")
        if obj.activation not in {"relu", "gelu"}:
            raise SpecError("activation must be relu or gelu")
        _positive_int(obj.input_dim, "architecture.input_dim")
        _positive_int(obj.output_dim, "architecture.output_dim")
        multiplier = _finite_float(obj.residual_multiplier, "architecture.residual_multiplier", 0.0)
        if multiplier == 0.0:
            raise SpecError("architecture.residual_multiplier must be > 0")
        experts = _positive_int(obj.num_experts, "architecture.num_experts")
        active = _positive_int(obj.active_experts, "architecture.active_experts")
        if active > experts:
            raise SpecError("architecture.active_experts cannot exceed num_experts")
        balance_rate = _finite_float(
            obj.router_balance_rate, "architecture.router_balance_rate", 0.0
        )
        if obj.block_type == "pre_norm_mlp" and (
            experts != 1 or active != 1
        ):
            raise SpecError("MoE fields require block_type pre_norm_moe")
        if obj.block_type == "pre_norm_moe" and balance_rate == 0.0:
            raise SpecError("architecture.router_balance_rate must be > 0 for MoE")
        return obj


@dataclass(frozen=True)
class ScaleLevel:
    name: str
    width: int
    repeats: int
    expert_width: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], index: int) -> "ScaleLevel":
        _strict_keys(data, {"name", "width", "repeats", "expert_width"}, f"scales[{index}]")
        if not isinstance(data.get("name"), str) or not data["name"].strip():
            raise SpecError(f"scales[{index}].name must be a non-empty string")
        expert_width = data.get("expert_width")
        if expert_width is not None:
            expert_width = _positive_int(expert_width, f"scales[{index}].expert_width", 2)
        return cls(
            name=data["name"].strip(),
            width=_positive_int(data.get("width"), f"scales[{index}].width", 4),
            repeats=_positive_int(data.get("repeats"), f"scales[{index}].repeats"),
            expert_width=expert_width,
        )


@dataclass(frozen=True)
class OptimizerSpec:
    name: str = "adam"
    momentum: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OptimizerSpec":
        _strict_keys(data, {"name", "momentum", "beta1", "beta2", "epsilon"}, "optimizer")
        obj = cls(**dict(data))
        if obj.name not in {"sgd", "adam"}:
            raise SpecError("optimizer.name must be sgd or adam")
        momentum = _finite_float(obj.momentum, "optimizer.momentum", 0.0)
        if momentum >= 1.0:
            raise SpecError("optimizer.momentum must be < 1")
        for name, value in (("beta1", obj.beta1), ("beta2", obj.beta2)):
            checked = _finite_float(value, f"optimizer.{name}", 0.0)
            if checked >= 1.0:
                raise SpecError(f"optimizer.{name} must be < 1")
        if _finite_float(obj.epsilon, "optimizer.epsilon", 0.0) == 0.0:
            raise SpecError("optimizer.epsilon must be > 0")
        return obj


@dataclass(frozen=True)
class DatasetSpec:
    n_train: int = 1024
    n_validation: int = 256
    noise_std: float = 0.03
    seed: int = 1729

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetSpec":
        _strict_keys(data, {"n_train", "n_validation", "noise_std", "seed"}, "dataset")
        obj = cls(**dict(data))
        _positive_int(obj.n_train, "dataset.n_train", 8)
        _positive_int(obj.n_validation, "dataset.n_validation", 8)
        _finite_float(obj.noise_std, "dataset.noise_std", 0.0)
        _positive_int(obj.seed, "dataset.seed", 0)
        return obj


@dataclass(frozen=True)
class HorizonSpec:
    steps: int = 120
    batch_size: int = 64
    microbatch_size: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HorizonSpec":
        _strict_keys(data, {"steps", "batch_size", "microbatch_size"}, "horizon")
        obj = cls(**dict(data))
        _positive_int(obj.steps, "horizon.steps")
        _positive_int(obj.batch_size, "horizon.batch_size")
        if obj.microbatch_size is not None:
            _positive_int(obj.microbatch_size, "horizon.microbatch_size")
            if obj.microbatch_size > obj.batch_size or obj.batch_size % obj.microbatch_size:
                raise SpecError("horizon.microbatch_size must evenly divide batch_size")
        return obj


@dataclass(frozen=True)
class TuningSpec:
    normalized_learning_rates: Tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    max_expansion_rounds: int = 2
    expansion_factor: float = 3.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TuningSpec":
        _strict_keys(
            data,
            {"normalized_learning_rates", "max_expansion_rounds", "expansion_factor"},
            "tuning",
        )
        raw_rates = data.get("normalized_learning_rates", cls.normalized_learning_rates)
        if not isinstance(raw_rates, Sequence) or isinstance(raw_rates, (str, bytes)):
            raise SpecError("tuning.normalized_learning_rates must be a list")
        rates = tuple(sorted({
            _finite_float(v, "tuning.normalized_learning_rates[]", 0.0)
            for v in raw_rates
        }))
        if len(rates) < 3 or rates[0] <= 0.0:
            raise SpecError(
                "tuning.normalized_learning_rates must contain at least three unique positive values"
            )
        rounds = _positive_int(data.get("max_expansion_rounds", 2), "tuning.max_expansion_rounds", 0)
        factor = _finite_float(data.get("expansion_factor", 3.0), "tuning.expansion_factor", 1.0)
        if factor <= 1.0:
            raise SpecError("tuning.expansion_factor must be > 1")
        return cls(rates, rounds, factor)


@dataclass(frozen=True)
class ValidationSpec:
    transfer_probe_decades: float = 0.3
    run_negative_control: bool = True
    bootstrap_samples: int = 200
    routing_load_tolerance: float = 0.25

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationSpec":
        _strict_keys(
            data,
            {
                "transfer_probe_decades",
                "run_negative_control",
                "bootstrap_samples",
                "routing_load_tolerance",
            },
            "validation",
        )
        obj = cls(**dict(data))
        _finite_float(obj.transfer_probe_decades, "validation.transfer_probe_decades", 0.05)
        if not isinstance(obj.run_negative_control, bool):
            raise SpecError("validation.run_negative_control must be boolean")
        _positive_int(obj.bootstrap_samples, "validation.bootstrap_samples", 0)
        tolerance = _finite_float(
            obj.routing_load_tolerance, "validation.routing_load_tolerance", 0.0
        )
        if tolerance > 1.0:
            raise SpecError("validation.routing_load_tolerance must be <= 1")
        return obj


@dataclass(frozen=True)
class StudySpec:
    schema_version: int
    name: str
    architecture: ArchitectureTemplate
    optimizer: OptimizerSpec
    dataset: DatasetSpec
    horizon: HorizonSpec
    scales: Tuple[ScaleLevel, ...]
    tuning: TuningSpec
    validation: ValidationSpec
    seeds: Tuple[int, ...]
    holdout_count: int = 1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StudySpec":
        _strict_keys(
            data,
            {
                "schema_version", "name", "architecture", "optimizer", "dataset", "horizon",
                "scales", "tuning", "validation", "seeds", "holdout_count",
            },
            "study",
        )
        if data.get("schema_version") != 2:
            raise SpecError("schema_version must be 2")
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise SpecError("name must be a non-empty string")
        raw_scales = data.get("scales")
        if not isinstance(raw_scales, Sequence) or isinstance(raw_scales, (str, bytes)):
            raise SpecError("scales must be a list")
        scales = tuple(ScaleLevel.from_dict(item, i) for i, item in enumerate(raw_scales))
        if len(scales) < 5:
            raise SpecError("at least five scale levels are required for a fitted law plus holdout")
        names = [scale.name for scale in scales]
        if len(set(names)) != len(names):
            raise SpecError("scale names must be unique")
        seeds_raw = data.get("seeds")
        if not isinstance(seeds_raw, Sequence) or isinstance(seeds_raw, (str, bytes)):
            raise SpecError("seeds must be a list")
        seeds = tuple(_positive_int(seed, "seeds[]", 0) for seed in seeds_raw)
        if len(seeds) < 2 or len(set(seeds)) != len(seeds):
            raise SpecError("seeds must contain at least two unique non-negative integers")
        holdout_count = _positive_int(data.get("holdout_count", 1), "holdout_count")
        if len(scales) - holdout_count < 4:
            raise SpecError("at least four non-holdout scales are required")
        obj = cls(
            schema_version=2,
            name=name.strip(),
            architecture=ArchitectureTemplate.from_dict(data.get("architecture", {})),
            optimizer=OptimizerSpec.from_dict(data.get("optimizer", {})),
            dataset=DatasetSpec.from_dict(data.get("dataset", {})),
            horizon=HorizonSpec.from_dict(data.get("horizon", {})),
            scales=scales,
            tuning=TuningSpec.from_dict(data.get("tuning", {})),
            validation=ValidationSpec.from_dict(data.get("validation", {})),
            seeds=seeds,
            holdout_count=holdout_count,
        )
        if obj.architecture.block_type == "pre_norm_moe" and obj.optimizer.name != "adam":
            raise SpecError(
                "pre_norm_moe currently requires adam; SGD MoE transfer is not certified"
            )
        if obj.architecture.block_type == "pre_norm_moe":
            if any(scale.expert_width is None for scale in obj.scales):
                raise SpecError("every MoE scale requires expert_width M")
        elif any(scale.expert_width is not None for scale in obj.scales):
            raise SpecError("expert_width is only valid for pre_norm_moe")
        computes = [estimate_training_compute(obj, scale) for scale in scales]
        if any(right <= left for left, right in zip(computes, computes[1:])):
            raise SpecError("scales must be ordered by strictly increasing estimated compute")
        if obj.horizon.batch_size > obj.dataset.n_train:
            raise SpecError("horizon.batch_size cannot exceed dataset.n_train")
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode("utf-8")).hexdigest()


def parameter_count(spec: StudySpec, scale: ScaleLevel) -> int:
    width = scale.width
    arch = spec.architecture
    embed = arch.input_dim * width + width
    if arch.block_type == "pre_norm_mlp":
        block = 2 * width * width + 4 * width
    else:
        assert scale.expert_width is not None
        expert_width = scale.expert_width
        router = width * arch.num_experts + arch.num_experts
        experts = arch.num_experts * (
            expert_width * width
            + expert_width
            + width * expert_width
            + width
        )
        block = 2 * width + router + experts
    final_norm = 2 * width
    unembed = width * arch.output_dim + arch.output_dim
    return embed + scale.repeats * block + final_norm + unembed


def estimate_training_compute(spec: StudySpec, scale: ScaleLevel) -> int:
    # A stable proxy: forward + backward is approximately six parameter FLOPs/token.
    return 6 * parameter_count(spec, scale) * spec.horizon.batch_size * spec.horizon.steps


def compile_plan(spec: StudySpec) -> Dict[str, Any]:
    levels = []
    holdout_start = len(spec.scales) - spec.holdout_count
    for index, scale in enumerate(spec.scales):
        levels.append(
            {
                **asdict(scale),
                "parameter_count": parameter_count(spec, scale),
                "estimated_training_compute": estimate_training_compute(spec, scale),
                "role": "holdout" if index >= holdout_start else "fit",
            }
        )
    return {
        "schema_version": 2,
        "study_fingerprint": spec.fingerprint,
        "fixed_data_points": spec.dataset.n_train,
        "fixed_token_horizon": spec.horizon.steps * spec.horizon.batch_size,
        "optimizer": spec.optimizer.name,
        "tuned_hyperparameters": ["normalized_learning_rate_eta"],
        "learning_rate_grid_coordinate": "normalized_eta",
        "transfer_rule": (
            "moe_table1_group_rates_from_normalized_eta"
            if spec.architecture.block_type == "pre_norm_moe"
            else (
                "raw_lr_equals_normalized_eta"
                if spec.optimizer.name == "adam"
                else "raw_lr_equals_normalized_eta_over_sqrt_width"
            )
        ),
        "architecture_contract": (
            {
                "block_type": "pre_norm_moe",
                "expert_width": "declared independently at every scale",
                "num_experts": spec.architecture.num_experts,
                "active_experts": spec.architecture.active_experts,
                "active_fraction": (
                    spec.architecture.active_experts / spec.architecture.num_experts
                ),
                "sparsity_policy": "fixed_across_scale",
                "optimizer_parameterization": "moe_table1_adam",
                "recommended_joint_path": "L*M/D constant",
            }
            if spec.architecture.block_type == "pre_norm_moe"
            else {"block_type": "pre_norm_mlp"}
        ),
        "target_metric": "final_validation_loss",
        "levels": levels,
        "trial_budget_before_edge_expansion": (
            len(spec.tuning.normalized_learning_rates) * len(spec.seeds)
            # The selected reference trials already exist from tuning.
            + (len(spec.scales) - 1) * len(spec.seeds)
            # The center holdout probe already exists from transfer.
            + 2 * spec.holdout_count * len(spec.seeds)
            + (len(spec.seeds) if spec.validation.run_negative_control else 0)
        ),
    }


def default_study_spec(
    optimizer: str = "adam", quick: bool = False, block_type: str = "pre_norm_mlp"
) -> StudySpec:
    if optimizer not in {"sgd", "adam"}:
        raise SpecError("optimizer must be sgd or adam")
    if block_type not in {"pre_norm_mlp", "pre_norm_moe"}:
        raise SpecError("block_type must be pre_norm_mlp or pre_norm_moe")
    if block_type == "pre_norm_moe" and optimizer != "adam":
        raise SpecError("pre_norm_moe currently requires adam")
    if block_type == "pre_norm_moe":
        widths = (8, 18, 32, 50, 72) if quick else (18, 32, 72, 128, 200, 288)
        repeats = (2, 3, 4, 5, 6) if quick else (3, 4, 6, 8, 10, 12)
        expert_widths = (16, 24, 32, 40, 48) if quick else (24, 32, 48, 64, 80, 96)
    else:
        widths = (16, 24, 32, 48, 64) if quick else (32, 48, 64, 96, 128, 192)
        repeats = (1, 2, 3, 4, 6) if quick else (2, 3, 4, 6, 8, 12)
        expert_widths = (None,) * len(widths)
    # Preserve the previous raw-SGD search at the reference width while making
    # the public search coordinate the transferable eta = raw_lr * sqrt(M).
    fit_count = len(widths) - 1
    reference_width = widths[fit_count // 2]
    if block_type == "pre_norm_moe":
        rates = (3e-2, 1e-1, 3e-1, 1.0, 3.0)
    elif optimizer == "sgd":
        rates = tuple(
            rate * math.sqrt(reference_width)
            for rate in (3e-3, 1e-2, 3e-2, 1e-1, 3e-1)
        )
    else:
        rates = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    data = {
        "schema_version": 2,
        "name": f"{optimizer}-{'moe' if block_type == 'pre_norm_moe' else 'mlp'}-fixed-horizon",
        "architecture": (
            {
                "block_type": "pre_norm_moe",
                "activation": "gelu",
                "num_experts": 4,
                "active_experts": 1,
                "router_balance_rate": 0.1,
            }
            if block_type == "pre_norm_moe"
            else {"block_type": "pre_norm_mlp", "activation": "gelu"}
        ),
        "optimizer": {"name": optimizer},
        "dataset": {"n_train": 512 if quick else 4096, "n_validation": 256 if quick else 1024},
        "horizon": {"steps": 40 if quick else 400, "batch_size": 64},
        "scales": [
            {
                "name": f"S{i + 1}",
                "width": width,
                "repeats": depth,
                **({"expert_width": expert_widths[i]} if expert_widths[i] is not None else {}),
            }
            for i, (width, depth) in enumerate(zip(widths, repeats))
        ],
        "tuning": {
            "normalized_learning_rates": rates,
            "max_expansion_rounds": 1 if quick else 2,
        },
        "validation": {
            "transfer_probe_decades": 0.3,
            "run_negative_control": not quick,
            "bootstrap_samples": 40 if quick else 400,
            "routing_load_tolerance": 0.25,
        },
        "seeds": [11, 29] if quick else [11, 29, 47, 71],
        "holdout_count": 1,
    }
    return StudySpec.from_dict(data)
