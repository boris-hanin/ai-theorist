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

from .model import (
    ChizatResidualMLP,
    build_model,
    dataset_fingerprint,
    make_teacher_dataset,
)
from .muon import AuxAdamConfig, HybridMuonAdam, MuonConfig
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
    validation_loss_trace: List[Dict[str, float]]
    diverged: bool
    device: str
    duration_seconds: float = 0.0
    peak_memory_bytes: int = 0
    raw_learning_rate: Optional[float] = None
    raw_learning_rates: Optional[Dict[str, float]] = None
    expert_width: Optional[int] = None
    particle_width: Optional[int] = None
    num_experts: Optional[int] = None
    active_experts: Optional[int] = None
    routing_loads: Optional[List[List[float]]] = None
    max_routing_load_imbalance: Optional[float] = None
    optimizer_parameterization: str = "declared"
    dataset_kind: str = "sinusoid_quadratic"
    dataset_fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_optimizer(
    model: torch.nn.Module,
    spec: StudySpec,
    learning_rate: float,
    *,
    normalized_eta: Optional[float] = None,
    scale: Optional[ScaleLevel] = None,
    rate_rule: str = "declared",
):
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    config = spec.optimizer
    if spec.architecture.block_type == "chizat_mlp":
        if not isinstance(model, ChizatResidualMLP) or scale is None:
            raise TypeError("chizat_mlp optimizer construction requires its model and scale")
        from .tuning import CHIZAT_MEAN_FIELD, optimizer_group_learning_rates_from_normalized_eta

        eta = learning_rate if normalized_eta is None else normalized_eta
        rates = optimizer_group_learning_rates_from_normalized_eta(
            CHIZAT_MEAN_FIELD,
            config.name,
            eta,
            width=scale.width,
            depth=scale.repeats,
            particle_width=scale.particle_width,
            rule=rate_rule,
        )
        roles = model.semantic_parameter_roles()
        if config.name == "muon":
            return HybridMuonAdam(
                list(model.parameters()),
                roles,
                rates,
                muon_config=MuonConfig(
                    momentum=config.momentum,
                    nesterov=config.nesterov,
                    ns_steps=config.ns_steps,
                    epsilon=1e-7,
                    weight_decay=config.weight_decay,
                    adjustment=config.adjustment,
                ),
                auxiliary_config=AuxAdamConfig(
                    beta1=config.beta1,
                    beta2=config.beta2,
                    epsilon=config.epsilon,
                    weight_decay=config.weight_decay,
                ),
            )
        groups = model.optimizer_parameter_groups(rates)
        if config.name == "sgd":
            return torch.optim.SGD(
                groups,
                lr=eta,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        if config.name == "adam":
            return torch.optim.Adam(
                groups,
                lr=eta,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay,
            )
        raise ValueError(f"Unsupported Chizat optimizer: {config.name}")
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
            weight_decay=config.weight_decay,
        )
    if config.name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
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
    optimizer_rate_rule: str = "declared",
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
    model = build_model(spec.architecture, scale).to(device)
    optimizer_learning_rate = (
        normalized_learning_rate if raw_learning_rate is None else raw_learning_rate
    )
    if force_global_learning_rate is None:
        optimizer = make_optimizer(
            model,
            spec,
            optimizer_learning_rate,
            normalized_eta=normalized_learning_rate,
            scale=scale,
            rate_rule=optimizer_rate_rule,
        )
        optimizer_parameterization = optimizer_rate_rule
    else:
        if force_global_learning_rate <= 0.0 or not math.isfinite(force_global_learning_rate):
            raise ValueError("force_global_learning_rate must be finite and positive")
        optimizer_learning_rate = force_global_learning_rate
        config = spec.optimizer
        if config.name == "muon":
            raise ValueError("Muon single-global controls must use an explicit Chizat rate rule")
        if config.name == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=optimizer_learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=optimizer_learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        optimizer_parameterization = "single_global_control"
    raw_learning_rates = {
        str(group.get("name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }
    x_train, y_train, x_validation, y_validation = make_teacher_dataset(
        spec.architecture, spec.dataset, device=device
    )
    data_fingerprint = dataset_fingerprint(spec.architecture, spec.dataset)
    batch_generator = torch.Generator(device="cpu").manual_seed(100_003 + seed)
    step = 0
    trace: List[Dict[str, float]] = []
    with torch.no_grad():
        initial_validation_loss = float(
            F.mse_loss(model(x_validation), y_validation).detach().cpu()
        )
    validation_trace: List[Dict[str, float]] = [
        {"step": 0.0, "validation_loss": initial_validation_loss}
    ]
    if checkpoint_path is not None and resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        expected = {
            "study_fingerprint": spec.fingerprint,
            "scale": scale.name,
            "normalized_learning_rate": normalized_learning_rate,
            "raw_learning_rate": optimizer_learning_rate,
            "raw_learning_rates": raw_learning_rates,
            "optimizer_parameterization": optimizer_parameterization,
            "dataset_fingerprint": data_fingerprint,
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
        validation_trace = list(checkpoint["validation_trace"])

    target_steps = spec.horizon.steps
    if stop_after_steps is not None:
        target_steps = min(target_steps, _validate_stop_step(stop_after_steps))
    trace_interval = max(1, spec.horizon.steps // 8)
    diverged = False
    while step < target_steps:
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
            micro_loss_sum = F.mse_loss(
                predictions, y_train[micro_indices], reduction="sum"
            )
            micro_loss = micro_loss_sum / (spec.horizon.batch_size * spec.architecture.output_dim)
            if not torch.isfinite(micro_loss):
                finite_step = False
                break
            micro_loss.backward()
            loss_value = loss_value + micro_loss.detach()
            current_loads = model.routing_loads()
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
        model.update_router_balance(routing_load_sums)
        step += 1
        if step == 1 or step % trace_interval == 0 or step == target_steps:
            trace.append({"step": float(step), "training_loss": float(loss_value.cpu())})
            with torch.no_grad():
                checkpoint_validation_loss = float(
                    F.mse_loss(model(x_validation), y_validation).detach().cpu()
                )
            validation_trace.append(
                {"step": float(step), "validation_loss": checkpoint_validation_loss}
            )
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
                    "dataset_fingerprint": data_fingerprint,
                    "seed": seed,
                    "step": step,
                    "trace": trace,
                    "validation_trace": validation_trace,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "batch_generator_state": batch_generator.get_state(),
                },
                checkpoint_path,
            )

    model.eval()
    with torch.no_grad():
        validation_loss_tensor: Tensor = F.mse_loss(model(x_validation), y_validation)
        validation_routing_loads = model.routing_loads()
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
        validation_loss_trace=validation_trace,
        diverged=diverged,
        device=device,
        duration_seconds=duration_seconds,
        peak_memory_bytes=peak_memory_bytes,
        raw_learning_rate=optimizer_learning_rate,
        raw_learning_rates=raw_learning_rates,
        expert_width=scale.expert_width,
        particle_width=scale.particle_width,
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
        dataset_kind=spec.dataset.kind,
        dataset_fingerprint=data_fingerprint,
    )


def _validate_stop_step(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("stop_after_steps must be a non-negative integer")
    return value
