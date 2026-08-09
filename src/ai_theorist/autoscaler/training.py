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
from .schema import ScaleLevel, StudySpec, parameter_count


@dataclass(frozen=True)
class TrialResult:
    scale: str
    width: int
    repeats: int
    seed: int
    learning_rate: float
    optimizer: str
    parameter_count: int
    steps_completed: int
    final_validation_loss: float
    train_loss_trace: List[Dict[str, float]]
    diverged: bool
    device: str
    duration_seconds: float = 0.0
    peak_memory_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_optimizer(model: torch.nn.Module, spec: StudySpec, learning_rate: float) -> torch.optim.Optimizer:
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    config = spec.optimizer
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
    learning_rate: float,
    seed: int,
    *,
    device: str = "cpu",
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 0,
    stop_after_steps: Optional[int] = None,
    resume: bool = True,
) -> TrialResult:
    """Train to an exact step horizon, with bitwise-continuable local checkpoints."""
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
    model = ResidualMLP(spec.architecture, scale).to(device)
    optimizer = make_optimizer(model, spec, learning_rate)
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
            "learning_rate": learning_rate,
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
        for start in range(0, spec.horizon.batch_size, microbatch_size):
            micro_indices = indices[start : start + microbatch_size]
            micro_loss_sum = F.mse_loss(
                model(x_train[micro_indices]), y_train[micro_indices], reduction="sum"
            )
            micro_loss = micro_loss_sum / (spec.horizon.batch_size * spec.architecture.output_dim)
            if not torch.isfinite(micro_loss):
                finite_step = False
                break
            micro_loss.backward()
            loss_value = loss_value + micro_loss.detach()
        if not finite_step:
            diverged = True
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        optimizer.step()
        step += 1
        if step == 1 or step % trace_interval == 0 or step == target_steps:
            trace.append({"step": float(step), "training_loss": float(loss_value.cpu())})
        if checkpoint_path is not None and checkpoint_every and (
            step % checkpoint_every == 0 or step == target_steps
        ):
            _atomic_torch_save(
                {
                    "study_fingerprint": spec.fingerprint,
                    "scale": scale.name,
                    "learning_rate": learning_rate,
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
        validation_loss_tensor: Tensor = F.mse_loss(model(x_validation), y_validation)
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
    return TrialResult(
        scale=scale.name,
        width=scale.width,
        repeats=scale.repeats,
        seed=seed,
        learning_rate=learning_rate,
        optimizer=spec.optimizer.name,
        parameter_count=parameter_count(spec, scale),
        steps_completed=step,
        final_validation_loss=validation_loss,
        train_loss_trace=trace,
        diverged=diverged,
        device=device,
        duration_seconds=duration_seconds,
        peak_memory_bytes=peak_memory_bytes,
    )


def _validate_stop_step(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("stop_after_steps must be a non-negative integer")
    return value
