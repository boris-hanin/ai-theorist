from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from .batch_scaling import (
    BatchRunRecord,
    OptimizerHyperparameters,
    TransferContext,
    apply_transfer_rule,
    transfer_rule_registry,
)
from .critical_batch import (
    ContinuationObservation,
    CriticalBatchEstimate,
    StepsToTargetObservation,
    combine_critical_batch_estimates,
    estimate_direct_checkpoint_critical_batch,
    estimate_gradient_noise_critical_batch,
    estimate_loss_optimal_batch,
    estimate_steps_to_target_critical_batch,
)
from .lr_schedules import LearningRateSchedule
from .normalized_transformer import NormalizedTransformer, make_synthetic_markov_dataset
from .schema import ArchitectureTemplate, DatasetSpec, ScaleLevel
from .study import atomic_write_json


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


def _device_satisfies_request(actual: torch.device, requested: torch.device) -> bool:
    return actual.type == requested.type and (
        requested.index is None or actual.index == requested.index
    )


def _campaign_progress(
    callback: ProgressCallback,
    phase: str,
    completed: int,
    total: int,
    message: str,
) -> None:
    if callback is not None:
        callback(
            {
                "phase": phase,
                "completed": completed,
                "total": total,
                "message": message,
            }
        )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _float_list(value: Any, name: str) -> Tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a list")
    result = tuple(float(item) for item in value)
    if not result or any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise ValueError(f"{name} must contain positive finite numbers")
    return result


def _optimizer_from_payload(payload: Mapping[str, Any], learning_rate: float) -> OptimizerHyperparameters:
    return OptimizerHyperparameters(
        name=str(payload["name"]),
        learning_rate=learning_rate,
        momentum=float(payload.get("momentum", 0.0)),
        beta1=float(payload.get("beta1", 0.9)),
        beta2=float(payload.get("beta2", 0.999)),
        epsilon=float(payload.get("epsilon", 1e-8)),
        weight_decay=float(payload.get("weight_decay", 0.0)),
    )


def _quadratic_loss(theta: np.ndarray, spectrum: np.ndarray) -> float:
    return 0.5 * float(np.dot(theta * spectrum, theta))


def _run_quadratic_trial(
    *,
    spectrum: np.ndarray,
    optimizer: OptimizerHyperparameters,
    batch_tokens: int,
    maximum_steps: int,
    target_loss: float,
    noise_scale: float,
    seed: int,
    initial_theta: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    theta = (
        np.ones_like(spectrum) / math.sqrt(spectrum.size)
        if initial_theta is None
        else initial_theta.copy()
    )
    first_moment = np.zeros_like(theta)
    second_moment = np.zeros_like(theta)
    velocity = np.zeros_like(theta)
    crossing = None
    for step in range(1, maximum_steps + 1):
        gradient = spectrum * theta + rng.normal(
            scale=noise_scale / math.sqrt(batch_tokens), size=theta.shape
        )
        if optimizer.name == "sgd":
            velocity = optimizer.momentum * velocity + gradient
            theta -= optimizer.learning_rate * velocity
        else:
            first_moment = optimizer.beta1 * first_moment + (1.0 - optimizer.beta1) * gradient
            second_moment = (
                optimizer.beta2 * second_moment
                + (1.0 - optimizer.beta2) * gradient * gradient
            )
            first_hat = first_moment / (1.0 - optimizer.beta1**step)
            second_hat = second_moment / (1.0 - optimizer.beta2**step)
            theta -= optimizer.learning_rate * first_hat / (
                np.sqrt(second_hat) + optimizer.epsilon
            )
            if optimizer.name == "adamw":
                theta *= 1.0 - optimizer.learning_rate * optimizer.weight_decay
        loss = _quadratic_loss(theta, spectrum)
        if crossing is None and loss <= target_loss:
            crossing = step
    return {"loss": _quadratic_loss(theta, spectrum), "crossing_step": crossing, "theta": theta}


def run_quadratic_calibration(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Run a resumable-sized noisy-quadratic calibration before neural campaigns."""
    dimension = _positive_int(config.get("dimension", 64), "dimension")
    condition_number = float(config.get("condition_number", 30.0))
    noise_scale = float(config.get("noise_scale", 1.0))
    maximum_steps = _positive_int(config.get("maximum_steps", 400), "maximum_steps")
    target_loss = float(config.get("target_loss", 0.025))
    batches = tuple(_positive_int(int(value), "batch_tokens") for value in config.get("batch_tokens", [4, 8, 16, 32, 64, 128, 256]))
    seeds = tuple(int(value) for value in config.get("seeds", [11, 29, 47]))
    continuation_tokens = _positive_int(
        config.get("continuation_tokens", max(batches) * 8), "continuation_tokens"
    )
    if condition_number <= 1.0 or noise_scale <= 0.0 or target_loss <= 0.0:
        raise ValueError("condition_number, noise_scale, and target_loss must be positive")
    spectrum = np.geomspace(1.0, 1.0 / condition_number, dimension)
    records = []
    analyses: Dict[str, Any] = {}
    optimizer_payloads = config.get(
        "optimizers",
        [
            {"name": "sgd", "momentum": 0.0, "learning_rates": [0.05, 0.1, 0.2]},
            {
                "name": "adam",
                "beta1": 0.9,
                "beta2": 0.99,
                "epsilon": 1e-8,
                "learning_rates": [0.003, 0.01, 0.03],
            },
        ],
    )
    for optimizer_payload in optimizer_payloads:
        name = str(optimizer_payload["name"])
        learning_rates = _float_list(optimizer_payload["learning_rates"], "learning_rates")
        raw_trials = []
        for batch in batches:
            for learning_rate in learning_rates:
                optimizer = _optimizer_from_payload(optimizer_payload, learning_rate)
                for trial_seed in seeds:
                    started = time.monotonic()
                    result = _run_quadratic_trial(
                        spectrum=spectrum,
                        optimizer=optimizer,
                        batch_tokens=batch,
                        maximum_steps=maximum_steps,
                        target_loss=target_loss,
                        noise_scale=noise_scale,
                        seed=trial_seed,
                    )
                    duration = time.monotonic() - started
                    crossing = result["crossing_step"]
                    record = BatchRunRecord(
                        run_id=f"quadratic-{name}-b{batch}-lr{learning_rate:g}-s{trial_seed}",
                        model_family="paquette_noisy_quadratic",
                        optimizer=optimizer,
                        seed=trial_seed,
                        parameter_count=dimension,
                        width=dimension,
                        depth=1,
                        total_tokens=maximum_steps * batch,
                        batch_tokens=batch,
                        microbatch_tokens=batch,
                        accumulation_steps=1,
                        data_parallel_replicas=1,
                        optimizer_steps=maximum_steps,
                        nonpadding_tokens_seen=maximum_steps * batch,
                        learning_rate_schedule="constant",
                        final_validation_loss=float(result["loss"]),
                        estimated_flops=float(6 * dimension * maximum_steps * batch),
                        wall_time_seconds=duration,
                        target_loss_crossings={
                            f"{target_loss:g}": int(crossing) if crossing is not None else None
                        },
                        metadata={"condition_number": condition_number, "noise_scale": noise_scale},
                    )
                    records.append(record.to_dict())
                    raw_trials.append(
                        {
                            "batch": batch,
                            "learning_rate": learning_rate,
                            "seed": trial_seed,
                            "loss": float(result["loss"]),
                            "crossing": crossing,
                        }
                    )

        steps_rows = []
        losses_by_batch: Dict[int, List[float]] = {}
        for batch in batches:
            for trial_seed in seeds:
                choices = [
                    row for row in raw_trials if row["batch"] == batch and row["seed"] == trial_seed
                ]
                reached = [row for row in choices if row["crossing"] is not None]
                if reached:
                    best = min(reached, key=lambda row: (row["crossing"], row["loss"]))
                    steps_rows.append(
                        StepsToTargetObservation(batch, int(best["crossing"]), trial_seed)
                    )
                best_final = min(choices, key=lambda row: row["loss"])
                losses_by_batch.setdefault(batch, []).append(best_final["loss"])
        source_rate_rows = [
            (
                learning_rate,
                float(
                    np.mean(
                        [
                            row["loss"]
                            for row in raw_trials
                            if row["batch"] == batches[0]
                            and row["learning_rate"] == learning_rate
                        ]
                    )
                ),
            )
            for learning_rate in learning_rates
        ]
        best_rate_at_smallest = min(source_rate_rows, key=lambda item: item[1])[0]
        steps_estimate = estimate_steps_to_target_critical_batch(steps_rows)

        continuation_rows = []
        for batch in batches:
            steps = max(1, continuation_tokens // batch)
            for trial_seed in seeds:
                initial_theta = np.ones(dimension) / math.sqrt(dimension)
                before = _quadratic_loss(initial_theta, spectrum)
                after = _run_quadratic_trial(
                    spectrum=spectrum,
                    optimizer=_optimizer_from_payload(optimizer_payload, best_rate_at_smallest),
                    batch_tokens=batch,
                    maximum_steps=steps,
                    target_loss=0.0,
                    noise_scale=noise_scale,
                    seed=1000 + trial_seed,
                    initial_theta=initial_theta,
                )["loss"]
                continuation_rows.append(
                    ContinuationObservation(batch, before, float(after), steps * batch, trial_seed)
                )
        direct_estimate = estimate_direct_checkpoint_critical_batch(continuation_rows)
        rng = np.random.default_rng(901)
        microbatch = batches[0]
        mean_gradient = spectrum * (np.ones(dimension) / math.sqrt(dimension))
        gradient_samples = mean_gradient + rng.normal(
            scale=noise_scale / math.sqrt(microbatch), size=(128, dimension)
        )
        noise_estimate = estimate_gradient_noise_critical_batch(
            gradient_samples, microbatch_tokens=microbatch
        )
        consensus = combine_critical_batch_estimates(
            [steps_estimate, direct_estimate, noise_estimate]
        )

        transfer_target_batch = int(
            config.get("transfer_target_batch", batches[min(3, len(batches) - 1)])
        )
        if transfer_target_batch not in batches:
            raise ValueError("transfer_target_batch must be present in batch_tokens")
        default_transfer_tokens = maximum_steps * batches[0]
        default_transfer_tokens -= default_transfer_tokens % transfer_target_batch
        if default_transfer_tokens <= 0:
            default_transfer_tokens = math.lcm(batches[0], transfer_target_batch)
        transfer_tokens = int(config.get("transfer_tokens", default_transfer_tokens))
        if transfer_tokens <= 0 or transfer_tokens % batches[0] or transfer_tokens % transfer_target_batch:
            raise ValueError("transfer_tokens must be positive and divisible by source and target batches")
        transfer_steps = transfer_tokens // transfer_target_batch
        transfer_seed_offset = 5_000
        target_oracle_trials = []
        for learning_rate in learning_rates:
            target_optimizer = _optimizer_from_payload(optimizer_payload, learning_rate)
            for trial_seed in seeds:
                paired_seed = transfer_seed_offset + trial_seed
                result = _run_quadratic_trial(
                    spectrum=spectrum,
                    optimizer=target_optimizer,
                    batch_tokens=transfer_target_batch,
                    maximum_steps=transfer_steps,
                    target_loss=target_loss,
                    noise_scale=noise_scale,
                    seed=paired_seed,
                )
                target_oracle_trials.append(
                    {
                        "learning_rate": learning_rate,
                        "seed": trial_seed,
                        "loss": float(result["loss"]),
                    }
                )
                records.append(
                    BatchRunRecord(
                        run_id=(
                            f"quadratic-oracle-{name}-b{transfer_target_batch}"
                            f"-lr{learning_rate:g}-s{trial_seed}"
                        ),
                        model_family="paquette_noisy_quadratic_transfer",
                        optimizer=target_optimizer,
                        seed=trial_seed,
                        parameter_count=dimension,
                        width=dimension,
                        depth=1,
                        total_tokens=transfer_tokens,
                        batch_tokens=transfer_target_batch,
                        microbatch_tokens=transfer_target_batch,
                        accumulation_steps=1,
                        data_parallel_replicas=1,
                        optimizer_steps=transfer_steps,
                        nonpadding_tokens_seen=transfer_tokens,
                        learning_rate_schedule="constant",
                        final_validation_loss=float(result["loss"]),
                        estimated_flops=float(6 * dimension * transfer_tokens),
                        target_loss_crossings={
                            f"{target_loss:g}": result["crossing_step"]
                        },
                        metadata={"role": "independently_tuned_target_oracle"},
                    ).to_dict()
                )
        oracle_rate_rows = [
            {
                "learning_rate": learning_rate,
                "mean_loss": float(
                    np.mean(
                        [
                            row["loss"]
                            for row in target_oracle_trials
                            if row["learning_rate"] == learning_rate
                        ]
                    )
                ),
            }
            for learning_rate in learning_rates
        ]
        target_oracle = min(oracle_rate_rows, key=lambda row: row["mean_loss"])
        transfer_context = TransferContext(
            dimension,
            dimension,
            transfer_tokens,
            transfer_tokens,
            batches[0],
            transfer_target_batch,
        )
        candidate_rules = (
            ("none", "sgd_linear_batch")
            if name == "sgd"
            else ("none", "adam_sde_sqrt", "complete_dp_joint", "exact_token_half_life")
        )
        rule_rows = []
        source_optimizer = _optimizer_from_payload(
            optimizer_payload, best_rate_at_smallest
        )
        for rule_name in candidate_rules:
            rule_result = apply_transfer_rule(rule_name, source_optimizer, transfer_context)
            if not rule_result.valid or rule_result.target is None:
                rule_rows.append({**rule_result.to_dict(), "evaluated": False})
                continue
            paired_losses = []
            for trial_seed in seeds:
                result = _run_quadratic_trial(
                    spectrum=spectrum,
                    optimizer=rule_result.target,
                    batch_tokens=transfer_target_batch,
                    maximum_steps=transfer_steps,
                    target_loss=target_loss,
                    noise_scale=noise_scale,
                    seed=transfer_seed_offset + trial_seed,
                )
                paired_losses.append(float(result["loss"]))
                records.append(
                    BatchRunRecord(
                        run_id=(
                            f"quadratic-transfer-{rule_name}-{name}-b{transfer_target_batch}"
                            f"-s{trial_seed}"
                        ),
                        model_family="paquette_noisy_quadratic_transfer",
                        optimizer=rule_result.target,
                        seed=trial_seed,
                        parameter_count=dimension,
                        width=dimension,
                        depth=1,
                        total_tokens=transfer_tokens,
                        batch_tokens=transfer_target_batch,
                        microbatch_tokens=transfer_target_batch,
                        accumulation_steps=1,
                        data_parallel_replicas=1,
                        optimizer_steps=transfer_steps,
                        nonpadding_tokens_seen=transfer_tokens,
                        learning_rate_schedule="constant",
                        final_validation_loss=float(result["loss"]),
                        estimated_flops=float(6 * dimension * transfer_tokens),
                        target_loss_crossings={
                            f"{target_loss:g}": result["crossing_step"]
                        },
                        metadata={"role": "transfer_rule", "rule": rule_name},
                    ).to_dict()
                )
            mean_loss = float(np.mean(paired_losses))
            rule_rows.append(
                {
                    **rule_result.to_dict(),
                    "evaluated": True,
                    "mean_loss": mean_loss,
                    "oracle_mean_loss": target_oracle["mean_loss"],
                    "relative_regret": mean_loss / target_oracle["mean_loss"] - 1.0,
                }
            )
        analyses[name] = {
            "steps_to_target": steps_estimate.to_dict(),
            "direct_checkpoint": direct_estimate.to_dict(),
            "gradient_noise": noise_estimate.to_dict(),
            "consensus": consensus.to_dict(),
            "loss_optimal_batch": estimate_loss_optimal_batch(losses_by_batch),
            "source_learning_rate": best_rate_at_smallest,
            "transfer_rule_calibration": {
                "source_batch_tokens": batches[0],
                "target_batch_tokens": transfer_target_batch,
                "matched_total_tokens": transfer_tokens,
                "target_oracle": target_oracle,
                "target_oracle_grid": oracle_rate_rows,
                "rules": rule_rows,
            },
        }
    return {
        "schema_version": 1,
        "campaign": "paquette_noisy_quadratic_calibration",
        "config": dict(config),
        "transfer_rule_registry": transfer_rule_registry(),
        "records": records,
        "optimizer_analyses": analyses,
    }


def _validation_loss(
    model: NormalizedTransformer,
    x_validation: torch.Tensor,
    y_validation: torch.Tensor,
    vocab_size: int,
) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(x_validation)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y_validation.reshape(-1))
    model.train()
    return float(loss.detach().cpu())


def _make_torch_optimizer(
    model: NormalizedTransformer, hyperparameters: OptimizerHyperparameters
) -> torch.optim.Optimizer:
    if hyperparameters.name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=hyperparameters.learning_rate,
            momentum=hyperparameters.momentum,
            weight_decay=hyperparameters.weight_decay,
        )
    groups = model.optimizer_parameter_groups(
        hyperparameters.learning_rate,
        adam_epsilon=hyperparameters.epsilon,
    )
    optimizer_class = torch.optim.AdamW if hyperparameters.name == "adamw" else torch.optim.Adam
    return optimizer_class(
        groups,
        lr=hyperparameters.learning_rate,
        betas=(hyperparameters.beta1, hyperparameters.beta2),
        eps=hyperparameters.epsilon,
        weight_decay=hyperparameters.weight_decay,
    )


def run_transformer_batch_trial(
    *,
    architecture: ArchitectureTemplate,
    dataset: DatasetSpec,
    scale: ScaleLevel,
    optimizer: OptimizerHyperparameters,
    total_tokens: int,
    batch_examples: int,
    seed: int,
    target_validation_loss: Optional[float] = None,
    validation_interval: int = 1,
    learning_rate_schedule: Any = "constant",
    gradient_clip_norm: Optional[float] = 100.0,
    device: str = "cpu",
    initial_state: Optional[Mapping[str, torch.Tensor]] = None,
    prepared_dataset: Optional[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = None,
    prepared_dataset_metadata: Optional[Mapping[str, Any]] = None,
    dataset_identity: Optional[Mapping[str, Any]] = None,
    cache_directory: Optional[Path] = None,
    cache_key_suffix: str = "",
    cache_state: bool = False,
) -> Tuple[BatchRunRecord, Dict[str, Any]]:
    """Train one real normalized Transformer trial with either SGD or Adam."""
    if architecture.block_type != "normalized_transformer":
        raise ValueError("the batch census requires normalized_transformer")
    batch_examples = _positive_int(batch_examples, "batch_examples")
    total_tokens = _positive_int(total_tokens, "total_tokens")
    batch_tokens = batch_examples * architecture.context_length
    schedule = LearningRateSchedule.from_payload(learning_rate_schedule)
    if gradient_clip_norm is not None and (
        not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0.0
    ):
        raise ValueError("gradient_clip_norm must be positive and finite or null")
    if total_tokens % batch_tokens:
        raise ValueError("total_tokens must be divisible by batch_examples * context_length")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    identity = {
        "architecture": asdict(architecture),
        "dataset": (
            dict(dataset_identity) if dataset_identity is not None else asdict(dataset)
        ),
        "scale": asdict(scale),
        "optimizer": optimizer.to_dict(),
        "total_tokens": total_tokens,
        "batch_examples": batch_examples,
        "seed": seed,
        "target_validation_loss": target_validation_loss,
        "validation_interval": validation_interval,
        "learning_rate_schedule": asdict(schedule),
        "gradient_clip_norm": gradient_clip_norm,
        "cache_key_suffix": cache_key_suffix,
    }
    digest = sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    run_id = (
        f"transformer-{scale.name}-{optimizer.name}-b{batch_tokens}"
        f"-t{total_tokens}-lr{optimizer.learning_rate:g}-s{seed}-{digest}{cache_key_suffix}"
    )
    record_path = cache_directory / f"{run_id}.json" if cache_directory else None
    state_path = cache_directory / f"{run_id}.pt" if cache_directory else None
    cache_complete = (
        record_path is not None
        and record_path.exists()
        and (not cache_state or (state_path is not None and state_path.exists()))
    )
    if cache_complete:
        with record_path.open("r", encoding="utf-8") as handle:
            record = BatchRunRecord.from_dict(json.load(handle))
        if cache_state:
            assert state_path is not None
            cached = torch.load(state_path, map_location="cpu", weights_only=False)
            state = cached["state_dict"]
        else:
            state = {}
        return record, {
            "initial_validation_loss": float(record.validation_checkpoints[0]["validation_loss"]),
            "state_dict": state,
        }
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model = NormalizedTransformer(architecture, scale).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    torch_optimizer = _make_torch_optimizer(model, optimizer)
    peak_group_rates = [float(group["lr"]) for group in torch_optimizer.param_groups]
    peak_group_contract = [
        {
            "name": str(group.get("name", f"group_{index}")),
            "peak_learning_rate": float(group["lr"]),
            "epsilon": float(group.get("eps", optimizer.epsilon)),
            "learning_rate_formula": str(group.get("lr_formula", "unspecified")),
            "epsilon_formula": str(group.get("eps_formula", "unspecified")),
            "theory_contract_id": str(group.get("theory_contract_id", "unspecified")),
        }
        for index, group in enumerate(torch_optimizer.param_groups)
    ]
    if prepared_dataset is None:
        x_train, y_train, x_validation, y_validation = make_synthetic_markov_dataset(
            architecture, dataset, device=device
        )
    else:
        x_train, y_train, x_validation, y_validation = prepared_dataset
        expected_shapes = (
            (dataset.n_train, architecture.context_length),
            (dataset.n_train, architecture.context_length),
            (dataset.n_validation, architecture.context_length),
            (dataset.n_validation, architecture.context_length),
        )
        actual_shapes = tuple(tuple(tensor.shape) for tensor in prepared_dataset)
        if actual_shapes != expected_shapes:
            raise ValueError(
                "prepared_dataset shapes must match dataset counts and architecture context; "
                f"expected {expected_shapes}, got {actual_shapes}"
            )
        requested_device = torch.device(device)
        if any(
            not _device_satisfies_request(tensor.device, requested_device)
            for tensor in prepared_dataset
        ):
            raise ValueError("prepared_dataset tensors must already be on the requested device")
    generator = torch.Generator(device="cpu").manual_seed(100_003 + seed)
    steps = total_tokens // batch_tokens
    checkpoints = []
    schedule_trace = []
    crossing_step = None
    started = time.monotonic()
    initial_loss = _validation_loss(model, x_validation, y_validation, architecture.vocab_size)
    checkpoints.append({"step": 0.0, "tokens": 0.0, "validation_loss": initial_loss})
    for step in range(1, steps + 1):
        schedule_multiplier = schedule.multiplier(step, steps)
        for group, peak_rate in zip(torch_optimizer.param_groups, peak_group_rates):
            group["lr"] = peak_rate * schedule_multiplier
        indices_cpu = torch.randint(0, x_train.shape[0], (batch_examples,), generator=generator)
        indices = indices_cpu.to(device) if device != "cpu" else indices_cpu
        torch_optimizer.zero_grad(set_to_none=True)
        logits = model(x_train[indices])
        loss = F.cross_entropy(
            logits.reshape(-1, architecture.vocab_size), y_train[indices].reshape(-1)
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Transformer batch trial diverged")
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        torch_optimizer.step()
        model.project_normalized_weights()
        if step % validation_interval == 0 or step == steps:
            validation_loss = _validation_loss(
                model, x_validation, y_validation, architecture.vocab_size
            )
            checkpoints.append(
                {
                    "step": float(step),
                    "tokens": float(step * batch_tokens),
                    "validation_loss": validation_loss,
                }
            )
            schedule_trace.append(
                {
                    "step": float(step),
                    "tokens": float(step * batch_tokens),
                    "multiplier": schedule_multiplier,
                }
            )
            if (
                crossing_step is None
                and target_validation_loss is not None
                and validation_loss <= target_validation_loss
            ):
                crossing_step = step
    duration = time.monotonic() - started
    final_loss = float(checkpoints[-1]["validation_loss"])
    record = BatchRunRecord(
        run_id=run_id,
        model_family="normalized_transformer_batch_census",
        optimizer=optimizer,
        seed=seed,
        parameter_count=parameter_count,
        width=scale.width,
        depth=scale.repeats,
        total_tokens=total_tokens,
        batch_tokens=batch_tokens,
        microbatch_tokens=batch_tokens,
        accumulation_steps=1,
        data_parallel_replicas=1,
        optimizer_steps=steps,
        nonpadding_tokens_seen=total_tokens,
        learning_rate_schedule=schedule.name,
        final_validation_loss=final_loss,
        estimated_flops=float(6 * parameter_count * total_tokens),
        wall_time_seconds=duration,
        target_loss_crossings=(
            {f"{target_validation_loss:g}": crossing_step}
            if target_validation_loss is not None
            else {}
        ),
        validation_checkpoints=tuple(checkpoints),
        metadata={
            "batch_examples": batch_examples,
            "device": device,
            "unique_training_tokens": dataset.n_train * architecture.context_length,
            "presented_to_unique_token_ratio": (
                total_tokens / (dataset.n_train * architecture.context_length)
            ),
            "schedule": schedule.audit(steps),
            "schedule_trace": schedule_trace,
            "peak_parameter_group_learning_rates": peak_group_rates,
            "peak_parameter_group_contract": peak_group_contract,
            "gradient_clipping": (
                "none" if gradient_clip_norm is None else gradient_clip_norm
            ),
            "dataset": (
                dict(prepared_dataset_metadata)
                if prepared_dataset_metadata is not None
                else {
                    "kind": "synthetic_markov",
                    "task_type": dataset.task_type,
                    "sampling_seed": dataset.seed,
                }
            ),
        },
    )
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    if record_path is not None:
        cache_directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(record_path, record.to_dict())
    if cache_state and state_path is not None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{state_path.name}.", dir=cache_directory
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(
                {"initial_validation_loss": initial_loss, "state_dict": state}, temporary
            )
            os.replace(temporary, state_path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return record, {"initial_validation_loss": initial_loss, "state_dict": state}


def _measure_transformer_gradient_noise(
    architecture: ArchitectureTemplate,
    dataset: DatasetSpec,
    scale: ScaleLevel,
    *,
    seed: int,
    microbatch_examples: int,
    sample_count: int,
    device: str,
) -> Any:
    torch.manual_seed(seed)
    model = NormalizedTransformer(architecture, scale).to(device)
    x_train, y_train, _, _ = make_synthetic_markov_dataset(architecture, dataset, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed + 81_337)
    samples = []
    for _ in range(sample_count):
        indices_cpu = torch.randint(
            0, dataset.n_train, (microbatch_examples,), generator=generator
        )
        indices = indices_cpu.to(device) if device != "cpu" else indices_cpu
        model.zero_grad(set_to_none=True)
        logits = model(x_train[indices])
        loss = F.cross_entropy(
            logits.reshape(-1, architecture.vocab_size), y_train[indices].reshape(-1)
        )
        gradients = torch.autograd.grad(loss, tuple(model.parameters()))
        samples.append(
            torch.cat([gradient.detach().float().reshape(-1).cpu() for gradient in gradients]).numpy()
        )
    return estimate_gradient_noise_critical_batch(
        np.stack(samples),
        microbatch_tokens=microbatch_examples * architecture.context_length,
    )


def run_transformer_batch_census(
    config: Mapping[str, Any], *, device: str = "cpu", progress: ProgressCallback = None
) -> Dict[str, Any]:
    """Census critical batch across model size for both SGD and Adam."""
    architecture = ArchitectureTemplate.from_dict(dict(config["architecture"]))
    dataset = DatasetSpec.from_dict(dict(config["dataset"]))
    scales = tuple(
        ScaleLevel.from_dict(payload, index)
        for index, payload in enumerate(config["scales"])
    )
    batches = tuple(_positive_int(int(value), "batch_examples") for value in config["batch_examples"])
    total_tokens = _positive_int(config["total_tokens"], "total_tokens")
    target_loss = float(config["target_validation_loss"])
    validation_interval = _positive_int(config.get("validation_interval", 1), "validation_interval")
    seeds = tuple(int(value) for value in config.get("seeds", [11, 29]))
    gradient_sample_count = _positive_int(
        config.get("gradient_noise_samples", 8), "gradient_noise_samples"
    )
    cache_directory = (
        Path(config["cache_directory"]) if config.get("cache_directory") else None
    )
    planned_grid = sum(
        len(scales) * len(batches) * len(optimizer_payload["learning_rates"]) * len(seeds)
        for optimizer_payload in config["optimizers"]
    )
    completed_grid = 0
    _campaign_progress(progress, "preflight", 0, planned_grid, "Validated Transformer census")
    records: List[BatchRunRecord] = []
    analyses = []
    for scale in scales:
        gradient_cache_path = (
            cache_directory / f"gradient-noise-{scale.name}.json"
            if cache_directory is not None
            else None
        )
        if gradient_cache_path is not None and gradient_cache_path.exists():
            with gradient_cache_path.open("r", encoding="utf-8") as handle:
                noise_payload = json.load(handle)
            noise_payload["refusal_reasons"] = tuple(
                noise_payload.get("refusal_reasons", ())
            )
            scale_noise_estimate = CriticalBatchEstimate(**noise_payload)
        else:
            scale_noise_estimate = _measure_transformer_gradient_noise(
                architecture,
                dataset,
                scale,
                seed=seeds[0],
                microbatch_examples=batches[0],
                sample_count=gradient_sample_count,
                device=device,
            )
            if gradient_cache_path is not None:
                atomic_write_json(gradient_cache_path, scale_noise_estimate.to_dict())
        for optimizer_payload in config["optimizers"]:
            optimizer_name = str(optimizer_payload["name"])
            learning_rates = _float_list(optimizer_payload["learning_rates"], "learning_rates")
            current_records = []
            for batch_examples in batches:
                batch_tokens = batch_examples * architecture.context_length
                if total_tokens % batch_tokens:
                    raise ValueError(
                        f"total_tokens is not divisible by batch tokens {batch_tokens}"
                    )
                for learning_rate in learning_rates:
                    hyperparameters = _optimizer_from_payload(optimizer_payload, learning_rate)
                    for trial_seed in seeds:
                        record, _ = run_transformer_batch_trial(
                            architecture=architecture,
                            dataset=dataset,
                            scale=scale,
                            optimizer=hyperparameters,
                            total_tokens=total_tokens,
                            batch_examples=batch_examples,
                            seed=trial_seed,
                            target_validation_loss=target_loss,
                            validation_interval=validation_interval,
                            device=device,
                            cache_directory=cache_directory,
                        )
                        records.append(record)
                        current_records.append(record)
                        completed_grid += 1
                        _campaign_progress(
                            progress,
                            "training-grid",
                            completed_grid,
                            planned_grid,
                            f"{scale.name} · {optimizer_name} · batch {batch_examples}",
                        )
            steps_rows = []
            losses_by_batch: Dict[int, List[float]] = {}
            for batch_examples in batches:
                batch_tokens = batch_examples * architecture.context_length
                for trial_seed in seeds:
                    choices = [
                        record
                        for record in current_records
                        if record.batch_tokens == batch_tokens and record.seed == trial_seed
                    ]
                    crossing_key = f"{target_loss:g}"
                    reached = [
                        record
                        for record in choices
                        if record.target_loss_crossings[crossing_key] is not None
                    ]
                    if reached:
                        best = min(
                            reached,
                            key=lambda record: int(record.target_loss_crossings[crossing_key]),
                        )
                        steps_rows.append(
                            StepsToTargetObservation(
                                batch_tokens,
                                int(best.target_loss_crossings[crossing_key]),
                                trial_seed,
                            )
                        )
                    losses_by_batch.setdefault(batch_tokens, []).append(
                        min(record.final_validation_loss for record in choices)
                    )
            steps_estimate = estimate_steps_to_target_critical_batch(steps_rows)

            smallest_batch_tokens = batches[0] * architecture.context_length
            smallest = [
                record for record in current_records if record.batch_tokens == smallest_batch_tokens
            ]
            best_source = min(smallest, key=lambda record: record.final_validation_loss)
            continuation_tokens = int(config.get("continuation_tokens", total_tokens))
            continuation_rows = []
            checkpoint_tokens = int(config.get("checkpoint_tokens", total_tokens // 2))
            checkpoint_tokens -= checkpoint_tokens % smallest_batch_tokens
            if checkpoint_tokens <= 0:
                checkpoint_tokens = smallest_batch_tokens
            for trial_seed in seeds:
                _, checkpoint_extra = run_transformer_batch_trial(
                    architecture=architecture,
                    dataset=dataset,
                    scale=scale,
                    optimizer=best_source.optimizer,
                    total_tokens=checkpoint_tokens,
                    batch_examples=batches[0],
                    seed=80_000 + trial_seed,
                    validation_interval=max(1, checkpoint_tokens // smallest_batch_tokens),
                    device=device,
                    cache_directory=cache_directory,
                    cache_key_suffix="-cbs-checkpoint",
                    cache_state=True,
                )
                checkpoint_state = checkpoint_extra["state_dict"]
                for batch_examples in batches:
                    batch_tokens = batch_examples * architecture.context_length
                    matched_tokens = continuation_tokens - continuation_tokens % batch_tokens
                    if matched_tokens <= 0:
                        continue
                    record, extra = run_transformer_batch_trial(
                        architecture=architecture,
                        dataset=dataset,
                        scale=scale,
                        optimizer=best_source.optimizer,
                        total_tokens=matched_tokens,
                        batch_examples=batch_examples,
                        seed=90_000 + trial_seed,
                        validation_interval=max(1, matched_tokens // batch_tokens),
                        device=device,
                        initial_state=checkpoint_state,
                        cache_directory=cache_directory,
                        cache_key_suffix=f"-cbs-cont-from-{checkpoint_tokens}",
                    )
                    continuation_rows.append(
                        ContinuationObservation(
                            batch_tokens,
                            extra["initial_validation_loss"],
                            record.final_validation_loss,
                            matched_tokens,
                            trial_seed,
                        )
                    )
            direct_estimate = estimate_direct_checkpoint_critical_batch(continuation_rows)
            noise_estimate = scale_noise_estimate
            consensus = combine_critical_batch_estimates(
                [steps_estimate, direct_estimate, noise_estimate]
            )
            analyses.append(
                {
                    "scale": asdict(scale),
                    "parameter_count": current_records[0].parameter_count,
                    "optimizer": optimizer_name,
                    "steps_to_target": steps_estimate.to_dict(),
                    "direct_checkpoint": direct_estimate.to_dict(),
                    "gradient_noise": noise_estimate.to_dict(),
                    "consensus": consensus.to_dict(),
                    "loss_optimal_batch": estimate_loss_optimal_batch(losses_by_batch),
                }
            )
            _campaign_progress(
                progress,
                "critical-batch-analysis",
                completed_grid,
                planned_grid,
                f"Qualified estimators for {scale.name} · {optimizer_name}",
            )
    _campaign_progress(progress, "complete", planned_grid, planned_grid, "Batch census complete")
    return {
        "schema_version": 1,
        "campaign": "normalized_transformer_batch_census",
        "config": dict(config),
        "device": device,
        "records": [record.to_dict() for record in records],
        "scale_optimizer_analyses": analyses,
    }


def _nearest_power_of_two(value: float, minimum: int = 1) -> int:
    return max(minimum, 2 ** int(round(math.log2(max(1.0, value)))))


def run_constant_tpp_campaign(
    config: Mapping[str, Any], *, device: str = "cpu", progress: ProgressCallback = None
) -> Dict[str, Any]:
    """Tune fit scales, freeze rules, and evaluate only once on the largest scale."""
    architecture = ArchitectureTemplate.from_dict(dict(config["architecture"]))
    dataset = DatasetSpec.from_dict(dict(config["dataset"]))
    scales = tuple(
        ScaleLevel.from_dict(payload, index)
        for index, payload in enumerate(config["scales"])
    )
    if len(scales) < 3:
        raise ValueError("constant-TPP campaign requires at least three scales")
    optimizer_payload = dict(config["optimizer"])
    learning_rates = _float_list(optimizer_payload["learning_rates"], "learning_rates")
    seeds = tuple(int(value) for value in config.get("seeds", [11, 29]))
    tpp = float(config["tokens_per_parameter"])
    if not math.isfinite(tpp) or tpp <= 0.0:
        raise ValueError("tokens_per_parameter must be positive")
    base_batch_examples = _positive_int(config["base_batch_examples"], "base_batch_examples")
    batch_growth_exponent = float(config.get("batch_growth_exponent", 0.0))
    validation_interval = _positive_int(config.get("validation_interval", 8), "validation_interval")
    cache_directory = (
        Path(config["cache_directory"]) if config.get("cache_directory") else None
    )

    geometry = []
    for scale in scales:
        probe = NormalizedTransformer(architecture, scale)
        parameters = sum(parameter.numel() for parameter in probe.parameters())
        del probe
        if not geometry:
            base_parameters = parameters
        batch_examples = _nearest_power_of_two(
            base_batch_examples * (parameters / base_parameters) ** batch_growth_exponent
        )
        batch_tokens = batch_examples * architecture.context_length
        requested_tokens = max(batch_tokens, int(round(tpp * parameters)))
        total_tokens = max(batch_tokens, int(round(requested_tokens / batch_tokens)) * batch_tokens)
        geometry.append(
            {
                "scale": scale,
                "parameters": parameters,
                "batch_examples": batch_examples,
                "batch_tokens": batch_tokens,
                "total_tokens": total_tokens,
                "realized_tpp": total_tokens / parameters,
            }
        )
    tpp_spread = max(row["realized_tpp"] for row in geometry) / min(
        row["realized_tpp"] for row in geometry
    )
    maximum_tpp_spread = float(config.get("maximum_tpp_spread_ratio", 1.10))
    if maximum_tpp_spread < 1.0:
        raise ValueError("maximum_tpp_spread_ratio must be at least 1")
    if tpp_spread > maximum_tpp_spread:
        raise ValueError(
            f"rounded token horizons do not hold T/P constant: spread {tpp_spread:.4f} "
            f"exceeds {maximum_tpp_spread:.4f}"
        )
    progress_total = len(geometry) + 2
    _campaign_progress(progress, "fit-scales", 0, progress_total, "Tuning constant-T/P fit scales")

    oracle_records: List[BatchRunRecord] = []

    def tune_geometry_row(row: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[BatchRunRecord]]:
        scale_records = []
        for learning_rate in learning_rates:
            optimizer = _optimizer_from_payload(optimizer_payload, learning_rate)
            for trial_seed in seeds:
                record, _ = run_transformer_batch_trial(
                    architecture=architecture,
                    dataset=dataset,
                    scale=row["scale"],
                    optimizer=optimizer,
                    total_tokens=row["total_tokens"],
                    batch_examples=row["batch_examples"],
                    seed=trial_seed,
                    validation_interval=validation_interval,
                    device=device,
                    cache_directory=cache_directory,
                )
                scale_records.append(record)
        rate_rows = []
        for learning_rate in learning_rates:
            matching = [
                record
                for record in scale_records
                if record.optimizer.learning_rate == learning_rate
            ]
            rate_rows.append(
                {
                    "learning_rate": learning_rate,
                    "mean_loss": float(np.mean([record.final_validation_loss for record in matching])),
                }
            )
        best = min(rate_rows, key=lambda item: item["mean_loss"])
        best_index = learning_rates.index(best["learning_rate"])
        optimum = {
            **best,
            **{key: value for key, value in row.items() if key != "scale"},
            "scale": row["scale"].name,
            "optimum_is_interior": 0 < best_index < len(learning_rates) - 1,
            "rate_grid": rate_rows,
        }
        return optimum, scale_records

    fit_optima = []
    for row in geometry[:-1]:
        optimum, row_records = tune_geometry_row(row)
        fit_optima.append(optimum)
        oracle_records.extend(row_records)
        _campaign_progress(
            progress,
            "fit-scales",
            len(fit_optima),
            progress_total,
            f"Tuned {row['scale'].name}",
        )
    log_tokens = np.log([row["total_tokens"] for row in fit_optima])
    log_rates = np.log([row["learning_rate"] for row in fit_optima])
    if float(np.ptp(log_tokens)) == 0.0:
        raise ValueError("fit scales need at least two distinct rounded token horizons")
    fitted_slope = float(np.polyfit(log_tokens, log_rates, 1)[0])
    fitted_horizon_exponent = max(1e-6, -fitted_slope)
    source_best = fit_optima[0]
    fit_qualification = {
        "source_optimum_is_interior": source_best["optimum_is_interior"],
        "all_fit_optima_are_interior": all(
            row["optimum_is_interior"] for row in fit_optima
        ),
    }
    source_optimizer = _optimizer_from_payload(
        optimizer_payload, source_best["learning_rate"]
    )
    source_geometry = geometry[0]
    heldout_geometry = geometry[-1]
    context = TransferContext(
        source_geometry["parameters"],
        heldout_geometry["parameters"],
        source_geometry["total_tokens"],
        heldout_geometry["total_tokens"],
        source_geometry["batch_tokens"],
        heldout_geometry["batch_tokens"],
    )
    requested_rules = tuple(
        config.get(
            "transfer_rules",
            [
                "none",
                "adam_sde_sqrt",
                "complete_dp_joint",
                "exact_token_half_life",
                "horizon_power_fit",
            ],
        )
    )
    transfer_results = []
    transfer_records = []
    for rule_name in requested_rules:
        result = apply_transfer_rule(
            str(rule_name),
            source_optimizer,
            context,
            horizon_exponent=fitted_horizon_exponent,
        )
        if not result.valid or result.target is None:
            transfer_results.append({**result.to_dict(), "evaluated": False})
            continue
        rule_records = []
        for trial_seed in seeds:
            record, _ = run_transformer_batch_trial(
                architecture=architecture,
                dataset=dataset,
                scale=heldout_geometry["scale"],
                optimizer=result.target,
                total_tokens=heldout_geometry["total_tokens"],
                batch_examples=heldout_geometry["batch_examples"],
                seed=trial_seed,
                validation_interval=validation_interval,
                device=device,
                cache_directory=cache_directory,
            )
            rule_records.append(record)
            transfer_records.append(record)
        mean_loss = float(np.mean([record.final_validation_loss for record in rule_records]))
        transfer_results.append(
            {
                **result.to_dict(),
                "evaluated": True,
                "mean_heldout_loss": mean_loss,
                "recommendable": fit_qualification["source_optimum_is_interior"],
            }
        )
    _campaign_progress(
        progress,
        "heldout-transfer",
        len(geometry),
        progress_total,
        "Evaluated frozen rules on the held-out scale",
    )

    # The held-out oracle is deliberately executed after every frozen-rule
    # trial. It is used only to score regret, never to select or alter a rule.
    heldout_oracle, heldout_oracle_records = tune_geometry_row(heldout_geometry)
    oracle_records.extend(heldout_oracle_records)
    for row in transfer_results:
        if not row.get("evaluated"):
            continue
        mean_loss = row["mean_heldout_loss"]
        row["oracle_mean_heldout_loss"] = heldout_oracle["mean_loss"]
        row["absolute_regret"] = mean_loss - heldout_oracle["mean_loss"]
        row["relative_regret"] = mean_loss / heldout_oracle["mean_loss"] - 1.0
    _campaign_progress(
        progress,
        "complete",
        progress_total,
        progress_total,
        "Constant-T/P held-out campaign complete",
    )
    return {
        "schema_version": 1,
        "campaign": "constant_tokens_per_parameter_heldout_transfer",
        "device": device,
        "config": dict(config),
        "geometry": [
            {**{key: value for key, value in row.items() if key != "scale"}, "scale": asdict(row["scale"])}
            for row in geometry
        ],
        "tpp_spread_ratio": tpp_spread,
        "heldout_scale": scales[-1].name,
        "execution_order": [
            "fit_scale_oracles",
            "frozen_rule_heldout_trials",
            "heldout_oracle_for_regret_only",
        ],
        "fit_scale_optima": fit_optima,
        "fit_qualification": fit_qualification,
        "heldout_oracle": heldout_oracle,
        "fitted_horizon_exponent": fitted_horizon_exponent,
        "transfer_results": transfer_results,
        "oracle_records": [record.to_dict() for record in oracle_records],
        "transfer_records": [record.to_dict() for record in transfer_records],
    }
