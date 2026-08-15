from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from .critical_batch import (
    CriticalBatchEstimate,
    LocalBranchObservation,
    estimate_local_branched_critical_batch,
)
from .forecast_campaigns import (
    _autocast,
    _build_model_and_groups,
    _evaluate,
    _sample_rank_partitioned_batch,
    compile_real_text_scaling_plan,
    forecast_tokenized_text_spec,
)
from .lr_schedules import LearningRateSchedule
from .pretraining import (
    DistributedContext,
    PretrainingRuntimeSpec,
    TokenizedTextCorpus,
)
from .study import atomic_write_json


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]
CRITICAL_BATCH_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ForecastCriticalBatchTask:
    phase: str
    seed: int
    eta_multiplier: Optional[float] = None
    checkpoint_tokens: Optional[int] = None
    batch_examples: Optional[int] = None

    @property
    def task_id(self) -> str:
        fields = [self.phase, f"s{self.seed}"]
        if self.eta_multiplier is not None:
            fields.append(f"m{self.eta_multiplier:g}")
        if self.checkpoint_tokens is not None:
            fields.append(f"t{self.checkpoint_tokens}")
        if self.batch_examples is not None:
            fields.append(f"b{self.batch_examples}")
        return "-".join(fields)

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "task_id": self.task_id}


def _positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _sha256_digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def compile_forecast_critical_batch_plan(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Compile a leakage-safe local branched-training CBS campaign."""

    forecast_plan = compile_real_text_scaling_plan(config)
    raw = config.get("critical_batch")
    if not isinstance(raw, Mapping):
        raise ValueError("critical_batch must be an object")
    runtime = PretrainingRuntimeSpec.from_dict(config.get("runtime", {}))
    if runtime.distributed != "none" or runtime.num_processes != 1:
        raise ValueError("critical-batch workers must be independent one-GPU processes")
    context_length = int(config["architecture"]["context_length"])
    source_plan = _sha256_digest(
        raw.get("source_plan_fingerprint"),
        "critical_batch.source_plan_fingerprint",
    )
    if source_plan != forecast_plan["fingerprint"]:
        raise ValueError("critical-batch source plan does not match the bound forecast plan")
    source_selection = _sha256_digest(
        raw.get("source_selection_sha256"),
        "critical_batch.source_selection_sha256",
    )
    selected_eta = _positive_float(
        raw.get("selected_learning_rate"),
        "critical_batch.selected_learning_rate",
    )
    if selected_eta not in {float(value) for value in forecast_plan["learning_rates"]}:
        raise ValueError("selected critical-batch LR is not in the source tuning grid")
    selected_tau_raw = raw.get("selected_weight_decay_tau_ema")
    selected_tau = (
        None
        if selected_tau_raw is None
        else _positive_float(
            selected_tau_raw,
            "critical_batch.selected_weight_decay_tau_ema",
        )
    )
    if selected_tau is not None:
        tau_grid = {float(value) for value in forecast_plan.get("weight_decay_tau_ema_grid", ())}
        if selected_tau not in tau_grid:
            raise ValueError("selected tau_EMA is not in the source tuning grid")

    anchor_index = int(raw.get("anchor_scale_index", 0))
    scales = list(forecast_plan["scales"])
    if not 0 <= anchor_index < len(scales):
        raise ValueError("critical_batch.anchor_scale_index is out of range")
    reference_batch = _positive_int(
        int(raw.get("reference_batch_examples", config["batch_examples"])),
        "critical_batch.reference_batch_examples",
    )
    initial_batch = _positive_int(
        int(raw.get("initial_batch_examples", reference_batch)),
        "critical_batch.initial_batch_examples",
    )
    microbatch = _positive_int(
        int(raw.get("microbatch_examples", initial_batch)),
        "critical_batch.microbatch_examples",
    )
    batches = tuple(
        _positive_int(int(value), "critical_batch.batch_examples")
        for value in raw.get("batch_examples", ())
    )
    if (
        len(batches) < 4
        or tuple(sorted(set(batches))) != batches
        or batches[0] != initial_batch
    ):
        raise ValueError(
            "critical_batch.batch_examples must contain at least four unique "
            "increasing values beginning at initial_batch_examples"
        )
    if any(batch % microbatch for batch in batches):
        raise ValueError("every candidate batch must be divisible by microbatch_examples")
    pilot_batch = _positive_int(
        int(raw.get("pilot_batch_examples", reference_batch)),
        "critical_batch.pilot_batch_examples",
    )
    if pilot_batch % microbatch:
        raise ValueError("pilot_batch_examples must be divisible by microbatch_examples")
    checkpoints = tuple(
        _positive_int(int(value), "critical_batch.checkpoint_tokens")
        for value in raw.get("checkpoint_tokens", ())
    )
    if len(checkpoints) < 3 or tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError("at least three unique increasing checkpoint token counts are required")
    continuation_tokens = _positive_int(
        int(raw.get("continuation_tokens")),
        "critical_batch.continuation_tokens",
    )
    largest_batch_tokens = batches[-1] * context_length
    if any(value % largest_batch_tokens for value in checkpoints):
        raise ValueError("checkpoint tokens must align with the largest candidate batch")
    if continuation_tokens % largest_batch_tokens:
        raise ValueError("continuation tokens must align with the largest candidate batch")
    if continuation_tokens // largest_batch_tokens < 64:
        raise ValueError("the largest batch requires at least 64 continuation updates")
    pilot_tokens = _positive_int(
        int(raw.get("pilot_tokens", checkpoints[1])),
        "critical_batch.pilot_tokens",
    )
    if pilot_tokens % (pilot_batch * context_length):
        raise ValueError("pilot_tokens must align with the pilot batch")
    eta_multipliers = tuple(
        _positive_float(value, "critical_batch.eta_multipliers")
        for value in raw.get("eta_multipliers", ())
    )
    if len(eta_multipliers) < 5 or tuple(sorted(set(eta_multipliers))) != eta_multipliers:
        raise ValueError("eta_multipliers must contain at least five unique increasing values")
    seeds = tuple(int(value) for value in raw.get("seeds", config.get("seeds", ())))
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("critical-batch census requires at least three unique seeds")
    pilot_seed_count = _positive_int(
        int(raw.get("pilot_seed_count", 2)), "critical_batch.pilot_seed_count", 2
    )
    if pilot_seed_count > len(seeds):
        raise ValueError("pilot_seed_count cannot exceed the census seed count")
    loss_tolerance = _positive_float(
        raw.get("loss_tolerance", 0.01), "critical_batch.loss_tolerance"
    )
    safety_fraction = float(raw.get("safety_fraction", 0.8))
    if not math.isfinite(safety_fraction) or not 0.0 < safety_fraction <= 1.0:
        raise ValueError("critical_batch.safety_fraction must lie in (0, 1]")
    schedule = LearningRateSchedule.from_payload(
        raw.get(
            "schedule",
            {
                "family": "warmup_stable_decay",
                "warmup_fraction": 0.02,
                "stable_fraction": 0.78,
            },
        )
    )
    schedule_horizon = checkpoints[-1] + continuation_tokens
    pilot_trials = len(eta_multipliers) * pilot_seed_count
    baseline_trials = len(seeds)
    branch_trials = len(checkpoints) * len(batches) * len(seeds)
    payload = {
        "schema_version": CRITICAL_BATCH_SCHEMA_VERSION,
        "campaign": "forecast_critical_batch_census",
        "source_forecast_plan_fingerprint": source_plan,
        "source_selection_sha256": source_selection,
        "dataset_identity": forecast_plan["dataset_identity"],
        "architecture_contract": forecast_plan["architecture_contract"],
        "anchor_scale": dict(scales[anchor_index]),
        "selected_learning_rate": selected_eta,
        "selected_weight_decay_tau_ema": selected_tau,
        "reference_batch_examples": reference_batch,
        "initial_batch_examples": initial_batch,
        "microbatch_examples": microbatch,
        "pilot_batch_examples": pilot_batch,
        "batch_examples": list(batches),
        "checkpoint_tokens": list(checkpoints),
        "continuation_tokens": continuation_tokens,
        "schedule_horizon_tokens": schedule_horizon,
        "schedule": asdict(schedule),
        "eta_multipliers": list(eta_multipliers),
        "pilot_tokens": pilot_tokens,
        "pilot_seed_count": pilot_seed_count,
        "seeds": list(seeds),
        "loss_tolerance": loss_tolerance,
        "safety_fraction": safety_fraction,
        "pilot_trials": pilot_trials,
        "baseline_trials": baseline_trials,
        "branch_trials": branch_trials,
        "planned_grid_trials": pilot_trials + baseline_trials + branch_trials,
        "execution_order": [
            "bind_source_plan_and_selection",
            "tune_a_horizon_safe_reference_lr_on_fresh_seeds",
            "require_an_interior_finite_pilot_optimum",
            "train_fresh_small_batch_baselines_and_freeze_checkpoints",
            "run_matched_token_branches_with_adam_sqrt_lr_scaling",
            "require_each_local_cbs_transition_to_be_bracketed",
            "compile_power_of_two_batch_warmup_below_lower_cbs_bounds",
            "freeze_batch_and_per_group_lr_schedule_before_new_ladder",
        ],
    }
    payload["fingerprint"] = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def build_forecast_critical_batch_tasks(
    plan: Mapping[str, Any],
    *,
    phase: str,
) -> List[ForecastCriticalBatchTask]:
    seeds = [int(value) for value in plan["seeds"]]
    if phase == "pilot":
        return [
            ForecastCriticalBatchTask("pilot", seed, eta_multiplier=float(multiplier))
            for multiplier in plan["eta_multipliers"]
            for seed in seeds[: int(plan["pilot_seed_count"])]
        ]
    if phase == "baseline":
        return [ForecastCriticalBatchTask("baseline", seed) for seed in seeds]
    if phase == "branch":
        return [
            ForecastCriticalBatchTask(
                "branch",
                seed,
                checkpoint_tokens=int(checkpoint),
                batch_examples=int(batch),
            )
            for checkpoint in plan["checkpoint_tokens"]
            for batch in plan["batch_examples"]
            for seed in seeds
        ]
    raise ValueError("phase must be pilot, baseline, or branch")


def fit_critical_batch_growth(
    checkpoints: Sequence[Tuple[int, CriticalBatchEstimate]],
) -> Dict[str, Any]:
    qualified = [
        (tokens, estimate)
        for tokens, estimate in checkpoints
        if estimate.qualified and estimate.critical_batch_tokens is not None
    ]
    if len(qualified) < 3:
        return {
            "qualified": False,
            "refusal_reasons": ["at least three bracketed local CBS estimates are required"],
        }
    x = np.log(np.asarray([tokens for tokens, _ in qualified], dtype=np.float64))
    y = np.log(
        np.asarray(
            [estimate.critical_batch_tokens for _, estimate in qualified],
            dtype=np.float64,
        )
    )
    design = np.column_stack((np.ones_like(x), x))
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    total = float(np.sum((y - y.mean()) ** 2))
    residual = float(np.sum((y - fitted) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    intercept, exponent = (float(value) for value in coefficients)
    monotone = all(
        qualified[index][1].critical_batch_tokens
        <= qualified[index + 1][1].critical_batch_tokens
        for index in range(len(qualified) - 1)
    )
    reasons = []
    if not 0.0 <= exponent <= 1.0:
        reasons.append("the fitted CBS growth exponent must lie in [0, 1]")
    if r_squared < 0.8:
        reasons.append("the CBS power-law fit has R^2 below 0.8")
    if not monotone:
        reasons.append("local CBS must be monotone non-decreasing in token horizon")
    return {
        "qualified": not reasons,
        "refusal_reasons": reasons,
        "coefficient": math.exp(intercept),
        "token_exponent": exponent,
        "r_squared": r_squared,
        "monotone": monotone,
        "checkpoint_count": len(qualified),
    }


def compile_conservative_batch_warmup(
    *,
    checkpoints: Sequence[Tuple[int, CriticalBatchEstimate]],
    candidate_batch_examples: Sequence[int],
    context_length: int,
    initial_batch_examples: int,
    reference_batch_examples: int,
    safety_fraction: float = 0.8,
) -> Dict[str, Any]:
    """Use measured lower bounds only; never extrapolate a production batch."""

    if not 0.0 < safety_fraction <= 1.0:
        raise ValueError("safety_fraction must lie in (0, 1]")
    candidates = tuple(sorted(set(int(value) for value in candidate_batch_examples)))
    if not candidates or candidates[0] != initial_batch_examples:
        raise ValueError("candidate batches must begin at initial_batch_examples")
    stages = [
        {
            "start_tokens": 0,
            "batch_examples": initial_batch_examples,
            "batch_tokens": initial_batch_examples * context_length,
            "batch_multiplier_from_reference": (
                initial_batch_examples / reference_batch_examples
            ),
            "learning_rate_multiplier_from_reference": math.sqrt(
                initial_batch_examples / reference_batch_examples
            ),
            "evidence": "initial conservative batch",
        }
    ]
    reasons = []
    current = initial_batch_examples
    for checkpoint_tokens, estimate in sorted(checkpoints, key=lambda row: row[0]):
        if not estimate.qualified or estimate.lower_batch_tokens is None:
            reasons.append(f"checkpoint {checkpoint_tokens} lacks a bracketed CBS lower bound")
            continue
        safe_cap_tokens = safety_fraction * estimate.lower_batch_tokens
        eligible = [
            batch
            for batch in candidates
            if batch * context_length <= safe_cap_tokens
        ]
        if not eligible:
            reasons.append(
                f"checkpoint {checkpoint_tokens} places the initial batch above the safety cap"
            )
            continue
        selected = max(eligible)
        if selected > current:
            current = selected
            stages.append(
                {
                    "start_tokens": int(checkpoint_tokens),
                    "batch_examples": selected,
                    "batch_tokens": selected * context_length,
                    "batch_multiplier_from_reference": (
                        selected / reference_batch_examples
                    ),
                    "learning_rate_multiplier_from_reference": math.sqrt(
                        selected / reference_batch_examples
                    ),
                    "evidence": {
                        "estimator": estimate.estimator,
                        "critical_batch_lower_tokens": estimate.lower_batch_tokens,
                        "critical_batch_upper_tokens": estimate.upper_batch_tokens,
                        "safety_fraction": safety_fraction,
                    },
                }
            )
    return {
        "qualified": not reasons,
        "refusal_reasons": reasons,
        "batch_coordinate": "non-padding tokens per optimizer update",
        "learning_rate_rule": (
            "multiply every theory-defined parameter-group LR by sqrt(B/B_ref)"
        ),
        "weight_decay_rule": (
            "for finite tau_EMA, multiply decoupled weight decay by sqrt(B/B_ref) "
            "to preserve shrinkage per presented token"
        ),
        "uses_extrapolated_batch": False,
        "stages": stages,
    }


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _corpus(config: Mapping[str, Any], plan: Mapping[str, Any]) -> TokenizedTextCorpus:
    corpus = TokenizedTextCorpus(
        forecast_tokenized_text_spec(config),
        context_length=int(config["architecture"]["context_length"]),
        vocab_size=int(config["architecture"]["vocab_size"]),
    )
    if not corpus.tokenizer_is_pinned:
        raise ValueError("critical-batch census requires pinned tokenizer provenance")
    if corpus.identity_fingerprint != plan["dataset_identity"]["fingerprint"]:
        raise ValueError("critical-batch plan and token stream identity disagree")
    return corpus


def _runtime_for_batch(
    config: Mapping[str, Any], plan: Mapping[str, Any], batch_examples: int
) -> PretrainingRuntimeSpec:
    runtime = PretrainingRuntimeSpec.from_dict(config.get("runtime", {}))
    microbatch = int(plan["microbatch_examples"])
    if batch_examples % microbatch:
        raise ValueError("batch must be divisible by the census microbatch")
    return replace(runtime, gradient_accumulation_steps=batch_examples // microbatch)


def _restore_optimizer_hyperparameters(
    optimizer: torch.optim.Optimizer,
    desired_groups: Sequence[Mapping[str, Any]],
) -> None:
    for loaded, desired in zip(optimizer.param_groups, desired_groups):
        for key, value in desired.items():
            if key != "params":
                loaded[key] = value


def _train_assay(
    *,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    device: str,
    eta_reference: float,
    seed: int,
    batch_examples: int,
    stop_tokens: int,
    snapshot_path: Optional[Path] = None,
    checkpoint_paths: Optional[Mapping[int, Path]] = None,
) -> Dict[str, Any]:
    context_length = int(config["architecture"]["context_length"])
    batch_tokens = batch_examples * context_length
    reference_batch = int(plan["reference_batch_examples"])
    reference_batch_tokens = reference_batch * context_length
    schedule_horizon = int(plan["schedule_horizon_tokens"])
    if stop_tokens % batch_tokens:
        raise ValueError("assay stop_tokens must align with its batch")
    runtime = _runtime_for_batch(config, plan, batch_examples)
    context = DistributedContext(0, 1, 0, device)
    corpus = _corpus(config, plan)
    scale = dict(plan["anchor_scale"])
    scale["presented_tokens"] = schedule_horizon
    scale["optimizer_steps"] = schedule_horizon // reference_batch_tokens
    scale["tokens_per_parameter"] = schedule_horizon / int(scale["parameters"])
    eta_multiplier = math.sqrt(batch_examples / reference_batch)
    model, plain_model, optimizer, group_contract, group_audit = _build_model_and_groups(
        config=config,
        scale=scale,
        eta=eta_reference,
        weight_decay_tau_ema=plan.get("selected_weight_decay_tau_ema"),
        optimizer_mode="theory",
        runtime=runtime,
        context=context,
    )
    for group in optimizer.param_groups:
        group["lr"] = float(group["lr"]) * eta_multiplier
        group["weight_decay"] = float(group.get("weight_decay", 0.0)) * eta_multiplier
    peak_rates = [float(group["lr"]) for group in optimizer.param_groups]
    desired_groups = [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]
    generator = torch.Generator(device="cpu").manual_seed(100_003 + seed)
    start_tokens = 0
    if snapshot_path is not None:
        snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        if snapshot["plan_fingerprint"] != plan["fingerprint"]:
            raise ValueError("branch snapshot plan fingerprint mismatch")
        if int(snapshot["seed"]) != seed:
            raise ValueError("branch snapshot seed mismatch")
        plain_model.load_state_dict(snapshot["model_state_dict"])
        optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        _restore_optimizer_hyperparameters(optimizer, desired_groups)
        generator.set_state(snapshot["generator_state"])
        start_tokens = int(snapshot["tokens_seen"])
    if stop_tokens <= start_tokens or (stop_tokens - start_tokens) % batch_tokens:
        raise ValueError("assay continuation must contain a positive integral update count")
    schedule = LearningRateSchedule.from_payload(plan["schedule"])
    validation_examples = int(config.get("validation_examples", 256))
    validation_microbatch = int(config.get("validation_microbatch_examples", 4))
    initial_loss = _evaluate(
        model,
        corpus,
        vocab_size=int(config["architecture"]["vocab_size"]),
        validation_examples=validation_examples,
        validation_microbatch_examples=validation_microbatch,
        seed=seed,
        runtime=runtime,
        context=context,
    )
    local_microbatch = int(plan["microbatch_examples"])
    checkpoints = []
    started = time.monotonic()
    tokens_seen = start_tokens
    model.train()
    while tokens_seen < stop_tokens:
        multiplier = schedule.multiplier_for_token_update(
            tokens_before_update=tokens_seen,
            batch_tokens=batch_tokens,
            total_tokens=schedule_horizon,
        )
        for group, peak in zip(optimizer.param_groups, peak_rates):
            group["lr"] = peak * multiplier
        optimizer.zero_grad(set_to_none=True)
        for accumulation_index in range(runtime.gradient_accumulation_steps):
            inputs, targets = _sample_rank_partitioned_batch(
                corpus, "train", local_microbatch, generator, context
            )
            with _autocast(runtime, device):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, int(config["architecture"]["vocab_size"])),
                    targets.reshape(-1),
                ) / runtime.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise RuntimeError("critical-batch assay diverged")
            loss.backward()
        optimizer.step()
        tokens_seen += batch_tokens
        if checkpoint_paths and tokens_seen in checkpoint_paths:
            checkpoint_loss = _evaluate(
                model,
                corpus,
                vocab_size=int(config["architecture"]["vocab_size"]),
                validation_examples=validation_examples,
                validation_microbatch_examples=validation_microbatch,
                seed=seed,
                runtime=runtime,
                context=context,
            )
            path = checkpoint_paths[tokens_seen]
            _atomic_torch_save(
                {
                    "schema_version": CRITICAL_BATCH_SCHEMA_VERSION,
                    "plan_fingerprint": plan["fingerprint"],
                    "seed": seed,
                    "tokens_seen": tokens_seen,
                    "validation_loss": checkpoint_loss,
                    "model_state_dict": {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in plain_model.state_dict().items()
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "generator_state": generator.get_state(),
                },
                path,
            )
            checkpoints.append(
                {"tokens": tokens_seen, "validation_loss": checkpoint_loss, "path": str(path)}
            )
    final_loss = _evaluate(
        model,
        corpus,
        vocab_size=int(config["architecture"]["vocab_size"]),
        validation_examples=validation_examples,
        validation_microbatch_examples=validation_microbatch,
        seed=seed,
        runtime=runtime,
        context=context,
    )
    return {
        "schema_version": CRITICAL_BATCH_SCHEMA_VERSION,
        "plan_fingerprint": plan["fingerprint"],
        "seed": seed,
        "eta_reference": eta_reference,
        "eta_actual": eta_reference * eta_multiplier,
        "batch_examples": batch_examples,
        "batch_tokens": batch_tokens,
        "microbatch_examples": local_microbatch,
        "gradient_accumulation_steps": runtime.gradient_accumulation_steps,
        "start_tokens": start_tokens,
        "stop_tokens": stop_tokens,
        "initial_validation_loss": initial_loss,
        "final_validation_loss": final_loss,
        "duration_seconds": time.monotonic() - started,
        "peak_parameter_group_learning_rates": peak_rates,
        "peak_parameter_group_contract": group_contract,
        "optimizer_group_audit": group_audit,
        "checkpoints": checkpoints,
    }


def run_forecast_critical_batch_task(
    config: Mapping[str, Any],
    *,
    task: ForecastCriticalBatchTask,
    root: Path,
    device: str = "cuda",
    selected_eta_multiplier: Optional[float] = None,
) -> Dict[str, Any]:
    plan = compile_forecast_critical_batch_plan(config)
    task_root = root / task.phase / task.task_id
    result_path = task_root / "result.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text())
        if payload.get("plan_fingerprint") != plan["fingerprint"]:
            raise ValueError("cached critical-batch task has the wrong plan fingerprint")
        return payload
    task_root.mkdir(parents=True, exist_ok=True)
    if task.phase == "pilot":
        assert task.eta_multiplier is not None
        eta = float(plan["selected_learning_rate"]) * task.eta_multiplier
        try:
            result = _train_assay(
                config=config,
                plan=plan,
                device=device,
                eta_reference=eta,
                seed=task.seed,
                batch_examples=int(plan["pilot_batch_examples"]),
                stop_tokens=int(plan["pilot_tokens"]),
            )
            result["status"] = "completed"
        except RuntimeError as error:
            if "diverged" not in str(error):
                raise
            result = {
                "schema_version": CRITICAL_BATCH_SCHEMA_VERSION,
                "plan_fingerprint": plan["fingerprint"],
                "status": "diverged",
                "seed": task.seed,
                "eta_reference": eta,
                "batch_examples": int(plan["pilot_batch_examples"]),
                "batch_tokens": int(plan["pilot_batch_examples"])
                * int(config["architecture"]["context_length"]),
                "initial_validation_loss": 0.0,
                "final_validation_loss": 1e30,
                "error": str(error),
            }
        result["eta_multiplier"] = task.eta_multiplier
    elif task.phase == "baseline":
        if selected_eta_multiplier is None:
            raise ValueError("baseline phase requires the selected eta multiplier")
        eta = float(plan["selected_learning_rate"]) * selected_eta_multiplier
        checkpoint_paths = {
            int(tokens): task_root / f"checkpoint-{int(tokens)}.pt"
            for tokens in plan["checkpoint_tokens"]
        }
        completed_snapshots = [
            (tokens, path)
            for tokens, path in sorted(checkpoint_paths.items())
            if path.is_file() and tokens < int(plan["checkpoint_tokens"][-1])
        ]
        resume_snapshot = completed_snapshots[-1][1] if completed_snapshots else None
        pending_checkpoint_paths = {
            tokens: path
            for tokens, path in checkpoint_paths.items()
            if not path.is_file()
        }
        result = _train_assay(
            config=config,
            plan=plan,
            device=device,
            eta_reference=eta,
            seed=task.seed,
            batch_examples=int(plan["initial_batch_examples"]),
            stop_tokens=int(plan["checkpoint_tokens"][-1]),
            snapshot_path=resume_snapshot,
            checkpoint_paths=pending_checkpoint_paths,
        )
        result["resumed_from_snapshot"] = (
            None if resume_snapshot is None else str(resume_snapshot)
        )
        result["selected_eta_multiplier"] = selected_eta_multiplier
    elif task.phase == "branch":
        if selected_eta_multiplier is None:
            raise ValueError("branch phase requires the selected eta multiplier")
        assert task.checkpoint_tokens is not None and task.batch_examples is not None
        eta = float(plan["selected_learning_rate"]) * selected_eta_multiplier
        snapshot = (
            root
            / "baseline"
            / ForecastCriticalBatchTask("baseline", task.seed).task_id
            / f"checkpoint-{task.checkpoint_tokens}.pt"
        )
        if not snapshot.is_file():
            raise ValueError(f"missing immutable baseline snapshot: {snapshot}")
        result = _train_assay(
            config=config,
            plan=plan,
            device=device,
            eta_reference=eta,
            seed=task.seed,
            batch_examples=task.batch_examples,
            stop_tokens=task.checkpoint_tokens + int(plan["continuation_tokens"]),
            snapshot_path=snapshot,
        )
        result["checkpoint_tokens"] = task.checkpoint_tokens
        result["selected_eta_multiplier"] = selected_eta_multiplier
        result["source_snapshot_sha256"] = sha256(snapshot.read_bytes()).hexdigest()
        result["sampling_contract"] = (
            "identical_checkpoint_generator_state_and_equal_token_count"
        )
    else:
        raise ValueError("unknown critical-batch task phase")
    result["task"] = task.to_dict()
    atomic_write_json(result_path, result)
    return result


def select_forecast_cbs_pilot(
    config: Mapping[str, Any], root: Path, *, require_interior: bool = True
) -> Dict[str, Any]:
    plan = compile_forecast_critical_batch_plan(config)
    rows = []
    for multiplier in plan["eta_multipliers"]:
        records = []
        for seed in plan["seeds"][: int(plan["pilot_seed_count"])]:
            task = ForecastCriticalBatchTask("pilot", int(seed), float(multiplier))
            path = root / "pilot" / task.task_id / "result.json"
            if not path.is_file():
                raise ValueError(f"missing pilot task {task.task_id}")
            records.append(json.loads(path.read_text()))
        losses = [float(record["final_validation_loss"]) for record in records]
        rows.append(
            {
                "eta_multiplier": float(multiplier),
                "eta_reference": float(plan["selected_learning_rate"]) * float(multiplier),
                "mean_final_validation_loss": float(np.mean(losses)),
                "seed_losses": losses,
                "all_finite": all(math.isfinite(value) for value in losses),
                "all_improved": all(
                    float(record["final_validation_loss"])
                    < float(record["initial_validation_loss"])
                    for record in records
                ),
            }
        )
    eligible = [row for row in rows if row["all_finite"] and row["all_improved"]]
    if not eligible:
        raise ValueError("no pilot LR is finite and improving")
    selected = min(eligible, key=lambda row: row["mean_final_validation_loss"])
    index = list(plan["eta_multipliers"]).index(selected["eta_multiplier"])
    interior = 0 < index < len(plan["eta_multipliers"]) - 1
    result = {
        "schema_version": CRITICAL_BATCH_SCHEMA_VERSION,
        "plan_fingerprint": plan["fingerprint"],
        "selected_eta_multiplier": selected["eta_multiplier"],
        "selected_eta_reference": selected["eta_reference"],
        "optimum_is_interior": interior,
        "grid": rows,
    }
    if require_interior and not interior:
        raise ValueError("horizon-safe pilot LR optimum is on a grid boundary")
    atomic_write_json(root / "pilot-selection.json", result)
    return result


def aggregate_forecast_critical_batch(
    config: Mapping[str, Any], root: Path
) -> Dict[str, Any]:
    plan = compile_forecast_critical_batch_plan(config)
    selection = json.loads((root / "pilot-selection.json").read_text())
    if not selection.get("optimum_is_interior"):
        raise ValueError("critical-batch aggregation refuses a boundary pilot LR")
    checkpoints = []
    for checkpoint_tokens in plan["checkpoint_tokens"]:
        observations = []
        records_by_seed: Dict[int, List[Dict[str, Any]]] = {}
        for batch_examples in plan["batch_examples"]:
            for seed in plan["seeds"]:
                task = ForecastCriticalBatchTask(
                    "branch", int(seed), checkpoint_tokens=int(checkpoint_tokens), batch_examples=int(batch_examples)
                )
                path = root / "branch" / task.task_id / "result.json"
                if not path.is_file():
                    raise ValueError(f"missing branch task {task.task_id}")
                record = json.loads(path.read_text())
                if (
                    record.get("plan_fingerprint") != plan["fingerprint"]
                    or int(record["start_tokens"]) != int(checkpoint_tokens)
                    or int(record["stop_tokens"]) - int(record["start_tokens"])
                    != int(plan["continuation_tokens"])
                ):
                    raise ValueError(f"branch task {task.task_id} violates its token contract")
                records_by_seed.setdefault(int(seed), []).append(record)
                observations.append(
                    LocalBranchObservation(
                        batch_tokens=int(record["batch_tokens"]),
                        final_validation_loss=float(record["final_validation_loss"]),
                        seed=int(seed),
                    )
                )
        for seed, records in records_by_seed.items():
            initial_losses = [float(record["initial_validation_loss"]) for record in records]
            snapshot_hashes = {record["source_snapshot_sha256"] for record in records}
            if max(initial_losses) - min(initial_losses) > 1e-8:
                raise ValueError(
                    f"checkpoint {checkpoint_tokens} seed {seed} branches do not "
                    "share the same initial validation loss"
                )
            if len(snapshot_hashes) != 1:
                raise ValueError(
                    f"checkpoint {checkpoint_tokens} seed {seed} branches do not "
                    "share one immutable snapshot"
                )
        estimate = estimate_local_branched_critical_batch(
            observations,
            loss_tolerance=float(plan["loss_tolerance"]),
            minimum_seeds=len(plan["seeds"]),
        )
        checkpoints.append((int(checkpoint_tokens), estimate))
    growth = fit_critical_batch_growth(checkpoints)
    schedule = compile_conservative_batch_warmup(
        checkpoints=checkpoints,
        candidate_batch_examples=plan["batch_examples"],
        context_length=int(config["architecture"]["context_length"]),
        initial_batch_examples=int(plan["initial_batch_examples"]),
        reference_batch_examples=int(plan["reference_batch_examples"]),
        safety_fraction=float(plan["safety_fraction"]),
    )
    passed = bool(growth["qualified"] and schedule["qualified"])
    result = {
        "schema_version": CRITICAL_BATCH_SCHEMA_VERSION,
        "campaign": "forecast_critical_batch_census",
        "status": "completed" if passed else "failed",
        "plan": plan,
        "pilot_selection": selection,
        "local_estimates": [
            {"checkpoint_tokens": tokens, "estimate": estimate.to_dict()}
            for tokens, estimate in checkpoints
        ],
        "growth_fit": growth,
        "batch_warmup": schedule,
        "gates": {
            "pilot_lr_optimum_is_interior": bool(selection["optimum_is_interior"]),
            "every_local_cbs_is_bracketed": all(estimate.qualified for _, estimate in checkpoints),
            "cbs_growth_fit_qualified": bool(growth["qualified"]),
            "batch_warmup_qualified": bool(schedule["qualified"]),
            "no_extrapolated_batch_used": not schedule["uses_extrapolated_batch"],
        },
    }
    atomic_write_json(root / "result.json", result)
    if not passed:
        raise ValueError("critical-batch census failed its scientific gates")
    return result


def run_forecast_critical_batch_campaign(
    config: Mapping[str, Any],
    *,
    output_directory: Path,
    device: str = "cpu",
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    """Run the same gated campaign serially for a web job or local assay."""

    plan = compile_forecast_critical_batch_plan(config)
    total = int(plan["planned_grid_trials"])
    completed = 0

    def emit(phase: str, message: str) -> None:
        if progress is not None:
            progress(
                {
                    "phase": phase,
                    "completed": completed,
                    "total": total,
                    "message": message,
                }
            )

    output_directory.mkdir(parents=True, exist_ok=True)
    emit("pilot", "Calibrating a horizon-safe reference LR")
    for task in build_forecast_critical_batch_tasks(plan, phase="pilot"):
        run_forecast_critical_batch_task(
            config, task=task, root=output_directory, device=device
        )
        completed += 1
        emit("pilot", task.task_id)
    selection = select_forecast_cbs_pilot(config, output_directory)
    selected = float(selection["selected_eta_multiplier"])
    emit("baseline", "Training immutable small-batch checkpoints")
    for task in build_forecast_critical_batch_tasks(plan, phase="baseline"):
        run_forecast_critical_batch_task(
            config,
            task=task,
            root=output_directory,
            device=device,
            selected_eta_multiplier=selected,
        )
        completed += 1
        emit("baseline", task.task_id)
    emit("branch", "Running matched-token batch branches")
    for task in build_forecast_critical_batch_tasks(plan, phase="branch"):
        run_forecast_critical_batch_task(
            config,
            task=task,
            root=output_directory,
            device=device,
            selected_eta_multiplier=selected,
        )
        completed += 1
        emit("branch", task.task_id)
    result = aggregate_forecast_critical_batch(config, output_directory)
    emit("complete", "Critical-batch warmup qualified")
    return result
