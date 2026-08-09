from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import ResidualMLP, make_teacher_dataset
from .normalized_transformer import NormalizedTransformer, make_synthetic_markov_dataset
from .schema import ScaleLevel, StudySpec, parameter_count


@dataclass(frozen=True)
class TrialResult:
    scale: str
    width: int
    repeats: int
    seed: int
    normalized_learning_rate: float
    optimizer: str
    parameter_count: int
    steps_completed: int
    final_validation_loss: float
    train_loss_trace: List[Dict[str, float]]
    diverged: bool
    device: str
    duration_seconds: float = 0.0
    peak_memory_bytes: int = 0
    raw_learning_rate: Optional[float] = None
    raw_learning_rates: Optional[Dict[str, float]] = None
    expert_width: Optional[int] = None
    num_experts: Optional[int] = None
    active_experts: Optional[int] = None
    routing_loads: Optional[List[List[float]]] = None
    max_routing_load_imbalance: Optional[float] = None
    optimizer_parameterization: str = "declared"
    normalized_transformer_diagnostics: Optional[Dict[str, float]] = None
    learning_rate_schedule: str = "constant"
    n_train: int = 0
    n_validation: int = 0
    batch_size: int = 0
    microbatch_size: int = 0
    token_horizon: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_optimizer(model: torch.nn.Module, spec: StudySpec, learning_rate: float) -> torch.optim.Optimizer:
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    config = spec.optimizer
    if spec.architecture.block_type == "normalized_transformer":
        if config.name != "adam":
            raise ValueError("normalized_transformer currently supports only Adam")
        if not isinstance(model, NormalizedTransformer):
            raise TypeError("normalized_transformer optimizer requires its typed model")
        return torch.optim.Adam(
            model.optimizer_parameter_groups(learning_rate),
            lr=learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=0.0,
        )
    if spec.architecture.block_type == "pre_norm_moe":
        if config.name != "adam":
            raise ValueError("pre_norm_moe currently supports only Adam")
        if not hasattr(model, "optimizer_parameter_groups"):
            raise TypeError("MoE model does not expose optimizer parameter groups")
        groups = model.optimizer_parameter_groups(learning_rate)
        return torch.optim.Adam(
            groups,
            lr=learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
        )
    if config.name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=config.momentum)
    if config.name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
        )
    raise ValueError(f"Unsupported optimizer: {config.name}")


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def train_trial(
    spec: StudySpec,
    scale: ScaleLevel,
    normalized_learning_rate: float,
    seed: int,
    *,
    raw_learning_rate: Optional[float] = None,
    force_global_learning_rate: Optional[float] = None,
    device: str = "cpu",
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 0,
    stop_after_steps: Optional[int] = None,
    resume: bool = True,
) -> TrialResult:
    """Train at a normalized eta and record the parameterization's raw LR.

    Direct callers may omit ``raw_learning_rate``; in that compatibility mode
    the supplied normalized rate is used for both coordinates.  Autoscaler
    studies always pass both explicitly.
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started_at = time.monotonic()
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model: torch.nn.Module
    if spec.architecture.block_type == "normalized_transformer":
        model = NormalizedTransformer(
            spec.architecture,
            scale,
            parameterization=(
                "baseline_ngpt" if force_global_learning_rate is not None else "nugpt"
            ),
        ).to(device)
    else:
        model = ResidualMLP(spec.architecture, scale).to(device)
    optimizer_learning_rate = (
        normalized_learning_rate if raw_learning_rate is None else raw_learning_rate
    )
    if force_global_learning_rate is None:
        optimizer = make_optimizer(model, spec, optimizer_learning_rate)
        optimizer_parameterization = "declared"
    else:
        if force_global_learning_rate <= 0.0 or not math.isfinite(force_global_learning_rate):
            raise ValueError("force_global_learning_rate must be finite and positive")
        optimizer_learning_rate = force_global_learning_rate
        config = spec.optimizer
        if config.name == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=optimizer_learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=optimizer_learning_rate, momentum=config.momentum
            )
        optimizer_parameterization = "single_global_control"
    raw_learning_rates = {
        str(group.get("name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }
    peak_group_learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    if spec.architecture.block_type == "normalized_transformer":
        x_train, y_train, x_validation, y_validation = make_synthetic_markov_dataset(
            spec.architecture, spec.dataset, device=device
        )
    else:
        x_train, y_train, x_validation, y_validation = make_teacher_dataset(
            spec.architecture, spec.dataset, device=device
        )
    batch_generator = torch.Generator(device="cpu").manual_seed(100_003 + seed)
    step = 0
    trace: List[Dict[str, float]] = []
    if checkpoint_path is not None and resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        expected = {
            "study_fingerprint": spec.fingerprint,
            "scale": scale.name,
            "normalized_learning_rate": normalized_learning_rate,
            "raw_learning_rate": optimizer_learning_rate,
            "raw_learning_rates": raw_learning_rates,
            "optimizer_parameterization": optimizer_parameterization,
            "seed": seed,
        }
        actual = {key: checkpoint.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(f"Checkpoint metadata mismatch: expected {expected}, got {actual}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        batch_generator.set_state(checkpoint["batch_generator_state"])
        step = int(checkpoint["step"])
        trace = list(checkpoint["trace"])

    target_steps = spec.horizon.steps
    if stop_after_steps is not None:
        target_steps = min(target_steps, _validate_stop_step(stop_after_steps))
    trace_interval = max(1, spec.horizon.steps // 8)
    diverged = False
    schedule_multiplier = 1.0
    while step < target_steps:
        if spec.architecture.block_type == "normalized_transformer":
            schedule_position = step / max(1, spec.horizon.steps - 1)
            schedule_multiplier = 0.1 + 0.9 * 0.5 * (
                1.0 + math.cos(math.pi * schedule_position)
            )
            for group, peak_rate in zip(optimizer.param_groups, peak_group_learning_rates):
                group["lr"] = peak_rate * schedule_multiplier
        indices_cpu = torch.randint(
            0,
            spec.dataset.n_train,
            (spec.horizon.batch_size,),
            generator=batch_generator,
        )
        indices = indices_cpu.to(device) if device != "cpu" else indices_cpu
        optimizer.zero_grad(set_to_none=True)
        microbatch_size = spec.horizon.microbatch_size or spec.horizon.batch_size
        loss_value = torch.zeros((), device=device)
        finite_step = True
        routing_load_sums = None
        for start in range(0, spec.horizon.batch_size, microbatch_size):
            micro_indices = indices[start : start + microbatch_size]
            predictions = model(x_train[micro_indices])
            if spec.architecture.block_type == "normalized_transformer":
                micro_loss_sum = F.cross_entropy(
                    predictions.reshape(-1, spec.architecture.vocab_size),
                    y_train[micro_indices].reshape(-1),
                    reduction="sum",
                )
                loss_denominator = (
                    spec.horizon.batch_size * spec.architecture.context_length
                )
            else:
                micro_loss_sum = F.mse_loss(
                    predictions, y_train[micro_indices], reduction="sum"
                )
                loss_denominator = (
                    spec.horizon.batch_size * spec.architecture.output_dim
                )
            micro_loss = micro_loss_sum / loss_denominator
            if not torch.isfinite(micro_loss):
                finite_step = False
                break
            micro_loss.backward()
            loss_value = loss_value + micro_loss.detach()
            current_loads = (
                model.routing_loads() if isinstance(model, ResidualMLP) else None
            )
            if current_loads is not None:
                weight = len(micro_indices) / spec.horizon.batch_size
                if routing_load_sums is None:
                    routing_load_sums = [load * weight for load in current_loads]
                else:
                    for index, load in enumerate(current_loads):
                        routing_load_sums[index].add_(load, alpha=weight)
        if not finite_step:
            diverged = True
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        optimizer.step()
        if isinstance(model, NormalizedTransformer):
            model.project_normalized_weights()
        elif isinstance(model, ResidualMLP):
            model.update_router_balance(routing_load_sums)
        step += 1
        if step == 1 or step % trace_interval == 0 or step == target_steps:
            trace_row = {"step": float(step), "training_loss": float(loss_value.cpu())}
            if spec.architecture.block_type == "normalized_transformer":
                trace_row["peak_learning_rate_multiplier"] = schedule_multiplier
            trace.append(trace_row)
        if checkpoint_path is not None and checkpoint_every and (
            step % checkpoint_every == 0 or step == target_steps
        ):
            _atomic_torch_save(
                {
                    "study_fingerprint": spec.fingerprint,
                    "scale": scale.name,
                    "normalized_learning_rate": normalized_learning_rate,
                    "raw_learning_rate": optimizer_learning_rate,
                    "raw_learning_rates": raw_learning_rates,
                    "optimizer_parameterization": optimizer_parameterization,
                    "seed": seed,
                    "step": step,
                    "trace": trace,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "batch_generator_state": batch_generator.get_state(),
                },
                checkpoint_path,
            )

    model.eval()
    with torch.no_grad():
        validation_predictions = model(x_validation)
        if spec.architecture.block_type == "normalized_transformer":
            validation_loss_tensor = F.cross_entropy(
                validation_predictions.reshape(-1, spec.architecture.vocab_size),
                y_validation.reshape(-1),
            )
        else:
            validation_loss_tensor = F.mse_loss(validation_predictions, y_validation)
        validation_routing_loads = (
            model.routing_loads() if isinstance(model, ResidualMLP) else None
        )
        normalized_transformer_diagnostics = (
            model.sphere_diagnostics()
            if isinstance(model, NormalizedTransformer)
            else None
        )
    validation_loss = float(validation_loss_tensor.detach().cpu())
    diverged = diverged or not math.isfinite(validation_loss) or validation_loss > 1e8
    if diverged:
        validation_loss = float("inf")
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory_bytes = 0
    duration_seconds = time.monotonic() - started_at
    serialized_loads = (
        [[float(value) for value in load.cpu()] for load in validation_routing_loads]
        if validation_routing_loads is not None
        else None
    )
    if serialized_loads is None:
        max_load_imbalance = None
    else:
        target_load = spec.architecture.active_experts / spec.architecture.num_experts
        max_load_imbalance = max(
            abs(value - target_load) for load in serialized_loads for value in load
        )
    return TrialResult(
        scale=scale.name,
        width=scale.width,
        repeats=scale.repeats,
        seed=seed,
        normalized_learning_rate=normalized_learning_rate,
        optimizer=spec.optimizer.name,
        parameter_count=parameter_count(spec, scale),
        steps_completed=step,
        final_validation_loss=validation_loss,
        train_loss_trace=trace,
        diverged=diverged,
        device=device,
        duration_seconds=duration_seconds,
        peak_memory_bytes=peak_memory_bytes,
        raw_learning_rate=optimizer_learning_rate,
        raw_learning_rates=raw_learning_rates,
        expert_width=scale.expert_width,
        num_experts=(
            spec.architecture.num_experts
            if spec.architecture.block_type == "pre_norm_moe"
            else None
        ),
        active_experts=(
            spec.architecture.active_experts
            if spec.architecture.block_type == "pre_norm_moe"
            else None
        ),
        routing_loads=serialized_loads,
        max_routing_load_imbalance=max_load_imbalance,
        optimizer_parameterization=optimizer_parameterization,
        normalized_transformer_diagnostics=normalized_transformer_diagnostics,
        learning_rate_schedule=(
            "cosine_to_10_percent_without_warmup"
            if spec.architecture.block_type == "normalized_transformer"
            else "constant"
        ),
        n_train=spec.dataset.n_train,
        n_validation=spec.dataset.n_validation,
        batch_size=spec.horizon.batch_size,
        microbatch_size=spec.horizon.microbatch_size or spec.horizon.batch_size,
        token_horizon=(
            spec.horizon.steps
            * spec.horizon.batch_size
            * (
                spec.architecture.context_length
                if spec.architecture.block_type == "normalized_transformer"
                else 1
            )
        ),
    )


def _validate_stop_step(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("stop_after_steps must be a non-negative integer")
    return value
