#!/usr/bin/env python3
"""Fixed-eta transfer harness for interleaved Jiang MHSA + Chizat FFNs.

Shapes use ``label:L:M:D:dial``.  The primary joint path should hold
``rho = L*M/D`` constant; pure-axis and nonconstant-rho paths are valid
secondary audits.  This script implements the Adam coordinate preregistered in
``rounds/017-jiang-chizat-interleaved/prereg.md``.  It does not certify SGD.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_theorist.autoscaler.jiang_chizat import (  # noqa: E402
    JIANG_DENSE_REPORTED_LR_MULTIPLIERS,
    JIANG_COMPLETEP_ADAM_THEORY,
    JiangChizatReference,
    JiangChizatShape,
    JiangChizatTransformer,
)


RULES = (
    "primary",
    "fan_in_down",
    "omit_attention_width",
    "omit_ffn_hidden_width",
    "disable_attention",
)
JIANG_CHIZAT_LR_GROUPS = (
    "jiang_embeddings",
    "jiang_norms",
    "jiang_attention_qkv",
    "jiang_attention_output",
    "jiang_ffn_up",
    "jiang_ffn_down",
    "jiang_other_biases",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


@dataclass(frozen=True)
class Shape:
    label: str
    L: int
    M: int
    D: int
    dial: float

    @property
    def rho(self) -> float:
        return self.L * self.M / self.D

    def model_shape(self, head_dimension: int) -> JiangChizatShape:
        return JiangChizatShape(self.L, self.M, self.D, head_dimension)


@dataclass(frozen=True)
class Trial:
    label: str
    L: int
    M: int
    D: int
    dial: float
    rho: float
    seed: int
    normalized_eta: float
    learning_rate_multipliers: Dict[str, float]
    warmup_steps: int
    rule: str
    raw_learning_rates: Dict[str, float]
    adam_epsilons: Dict[str, float]
    validation_loss_checkpoints: Dict[int, float]
    attention_diagnostics: Dict[str, float]
    attention_movement: Dict[str, float]
    final_validation_loss: float
    diverged: bool
    optimizer_group_contract: Dict[str, object]


def parse_shape(text: str) -> Shape:
    parts = text.split(":")
    if len(parts) != 5:
        raise ValueError("shape must use label:L:M:D:dial")
    label = parts[0].strip()
    L, M, D = (int(value) for value in parts[1:4])
    dial = float(parts[4])
    if not label or min(L, M, D) <= 0 or not math.isfinite(dial) or dial <= 0:
        raise ValueError("shape label, L, M, D, and dial must be positive")
    return Shape(label, L, M, D, dial)


def validate_shapes(shapes: Sequence[Shape], head_dimension: int) -> Tuple[Shape, ...]:
    result = tuple(shapes)
    if len(result) < 4:
        raise ValueError("at least four shapes are required")
    if len({shape.label for shape in result}) != len(result):
        raise ValueError("shape labels must be unique")
    if any(right.dial <= left.dial for left, right in zip(result, result[1:])):
        raise ValueError("shape dials must be strictly increasing")
    for shape in result:
        shape.model_shape(head_dimension)
    return result


def synthetic_markov_data(
    *,
    vocab_size: int,
    context_length: int,
    n_train: int,
    n_validation: int,
    seed: int,
    noise_probability: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total = n_train + n_validation
    tokens = torch.empty(total, context_length + 1, dtype=torch.long)
    tokens[:, :2] = torch.randint(vocab_size, (total, 2), generator=generator)
    transition = torch.stack(
        [torch.randperm(vocab_size, generator=generator) for _ in range(4)]
    )
    for position in range(2, context_length + 1):
        state = tokens[:, position - 2].remainder(4)
        previous = tokens[:, position - 1]
        next_token = transition[state, previous]
        if noise_probability:
            replace = torch.rand(total, generator=generator) < noise_probability
            random_token = torch.randint(vocab_size, (total,), generator=generator)
            next_token = torch.where(replace, random_token, next_token)
        tokens[:, position] = next_token
    split = n_train
    return tuple(
        value.to(device)
        for value in (
            tokens[:split, :-1],
            tokens[:split, 1:],
            tokens[split:, :-1],
            tokens[split:, 1:],
        )
    )  # type: ignore[return-value]


@torch.no_grad()
def validation_loss(
    model: JiangChizatTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
) -> float:
    model.eval()
    loss_sum = 0.0
    token_count = 0
    for start in range(0, len(inputs), batch_size):
        batch_inputs = inputs[start : start + batch_size]
        batch_targets = targets[start : start + batch_size]
        logits = model(batch_inputs)
        loss_sum += float(
            F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                batch_targets.reshape(-1),
                reduction="sum",
            ).cpu()
        )
        token_count += batch_targets.numel()
    return loss_sum / token_count


@torch.no_grad()
def attention_snapshot(model: JiangChizatTransformer, tokens: torch.Tensor):
    model.eval()
    model(tokens)
    snapshots = []
    for block in model.blocks:
        logits = block.attention.last_attention_logits
        probabilities = block.attention.last_attention_probabilities
        if logits is None or probabilities is None:
            continue
        snapshots.append((logits.clone(), probabilities.clone()))
    return snapshots


def attention_movement(initial, final) -> Dict[str, float]:
    logit_square_sum = 0.0
    logit_count = 0
    mean_attention_square_sum = 0.0
    mean_attention_count = 0
    head_variances = []
    for (initial_logits, initial_probabilities), (final_logits, final_probabilities) in zip(
        initial, final
    ):
        finite = torch.isfinite(initial_logits) & torch.isfinite(final_logits)
        logit_difference = (final_logits - initial_logits)[finite].float()
        logit_square_sum += float(logit_difference.square().sum().cpu())
        logit_count += logit_difference.numel()
        initial_mean = initial_probabilities.float().mean(dim=1)
        final_mean = final_probabilities.float().mean(dim=1)
        mean_difference = final_mean - initial_mean
        mean_attention_square_sum += float(mean_difference.square().sum().cpu())
        mean_attention_count += mean_difference.numel()
        head_variances.append(float(final_probabilities.float().var(dim=1).mean().cpu()))
    return {
        "per_entry_attention_logit_delta_rms": math.sqrt(
            logit_square_sum / max(1, logit_count)
        ),
        "head_averaged_attention_delta_rms": math.sqrt(
            mean_attention_square_sum / max(1, mean_attention_count)
        ),
        "final_across_head_attention_variance": (
            sum(head_variances) / len(head_variances) if head_variances else float("nan")
        ),
    }


def run_trial(
    shape: Shape,
    *,
    reference: JiangChizatReference,
    head_dimension: int,
    vocab_size: int,
    context_length: int,
    n_train: int,
    n_validation: int,
    dataset_seed: int,
    eta: float,
    epsilon0: float,
    steps: int,
    batch_size: int,
    seed: int,
    rule: str,
    device: torch.device,
    learning_rate_multipliers: Mapping[str, float] | None = None,
    warmup_steps: int = 0,
) -> Trial:
    if rule not in RULES:
        raise ValueError(f"unknown rule: {rule}")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = JiangChizatTransformer(
        shape.model_shape(head_dimension),
        vocab_size=vocab_size,
        context_length=context_length,
        reference=reference,
        down_initialization="fan_in" if rule == "fan_in_down" else "mean_field",
        disable_attention=rule == "disable_attention",
    ).to(device)
    group_options = {
        "omit_attention_width_factor": rule == "omit_attention_width",
        "omit_ffn_hidden_width_factor": rule == "omit_ffn_hidden_width",
    }
    multipliers = dict(JIANG_DENSE_REPORTED_LR_MULTIPLIERS)
    multipliers.update(
        {
            name: float(value)
            for name, value in (learning_rate_multipliers or {}).items()
        }
    )
    groups = model.optimizer_parameter_groups(
        eta,
        epsilon0=epsilon0,
        learning_rate_multipliers=multipliers,
        **group_options,
    )
    optimizer_contract = model.optimizer_contract_audit(
        eta,
        epsilon0=epsilon0,
        learning_rate_multipliers=multipliers,
        **group_options,
    )
    optimizer = torch.optim.Adam(
        groups, lr=eta, betas=(0.9, 0.95), weight_decay=0.0
    )
    raw_rates = {str(group["name"]): float(group["lr"]) for group in groups}
    epsilons = {str(group["name"]): float(group["eps"]) for group in groups}
    x_train, y_train, x_validation, y_validation = synthetic_markov_data(
        vocab_size=vocab_size,
        context_length=context_length,
        n_train=n_train,
        n_validation=n_validation,
        seed=dataset_seed,
        noise_probability=0.03,
        device=device,
    )
    probe_tokens = x_validation[: min(4, n_validation)]
    initial_attention = attention_snapshot(model, probe_tokens)
    checkpoints = {0: validation_loss(model, x_validation, y_validation, batch_size=batch_size)}
    checkpoint_steps = {1, 2, steps}
    checkpoint_steps.update(max(1, round(steps * fraction / 8)) for fraction in range(1, 9))
    generator = torch.Generator(device="cpu").manual_seed(100_003 + seed)
    diverged = False
    if warmup_steps < 0 or warmup_steps > steps:
        raise ValueError("warmup_steps must lie between zero and steps")
    peak_rates = [float(group["lr"]) for group in optimizer.param_groups]
    model.train()
    for step in range(1, steps + 1):
        schedule_multiplier = min(1.0, (step + 1) / warmup_steps) if warmup_steps else 1.0
        for group, peak_rate in zip(optimizer.param_groups, peak_rates):
            group["lr"] = peak_rate * schedule_multiplier
        indices = torch.randint(0, n_train, (batch_size,), generator=generator).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train[indices])
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size), y_train[indices].reshape(-1)
        )
        if not torch.isfinite(loss):
            diverged = True
            break
        loss.backward()
        optimizer.step()
        if step in checkpoint_steps:
            value = validation_loss(
                model, x_validation, y_validation, batch_size=batch_size
            )
            checkpoints[step] = value
            if not math.isfinite(value) or value > 1e8:
                diverged = True
                break
            model.train()
    final_attention = attention_snapshot(model, probe_tokens)
    final_loss = checkpoints.get(steps, float("inf"))
    diverged = diverged or not math.isfinite(final_loss)
    return Trial(
        shape.label,
        shape.L,
        shape.M,
        shape.D,
        shape.dial,
        shape.rho,
        seed,
        eta,
        multipliers,
        warmup_steps,
        rule,
        raw_rates,
        epsilons,
        checkpoints,
        model.diagnostics(),
        attention_movement(initial_attention, final_attention),
        final_loss,
        diverged,
        optimizer_contract,
    )


def group_feature_velocity_audit(
    shape: Shape,
    *,
    reference: JiangChizatReference,
    head_dimension: int,
    vocab_size: int,
    context_length: int,
    n_train: int,
    n_validation: int,
    dataset_seed: int,
    eta: float,
    epsilon0: float,
    batch_size: int,
    seed: int,
    device: torch.device,
    learning_rate_multipliers: Mapping[str, float] | None = None,
) -> Dict[str, object]:
    x_train, y_train, x_validation, _ = synthetic_markov_data(
        vocab_size=vocab_size,
        context_length=context_length,
        n_train=n_train,
        n_validation=n_validation,
        seed=dataset_seed,
        noise_probability=0.03,
        device=device,
    )
    batch_inputs = x_train[:batch_size]
    batch_targets = y_train[:batch_size]
    probe = x_validation[: min(batch_size, n_validation)]
    velocities = {}
    for group_name in (
        "jiang_embeddings",
        "jiang_norms",
        "jiang_attention_qkv",
        "jiang_attention_output",
        "jiang_ffn_up",
        "jiang_ffn_down",
        "jiang_other_biases",
    ):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = JiangChizatTransformer(
            shape.model_shape(head_dimension),
            vocab_size=vocab_size,
            context_length=context_length,
            reference=reference,
        ).to(device)
        groups = model.optimizer_parameter_groups(
            eta,
            epsilon0=epsilon0,
            learning_rate_multipliers=learning_rate_multipliers,
        )
        group = next(item for item in groups if item["name"] == group_name)
        optimizer = torch.optim.Adam(
            [group], lr=float(group["lr"]), betas=(0.9, 0.95), weight_decay=0.0
        )
        model.eval()
        with torch.no_grad():
            before = model.forward_features(probe).float()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size), batch_targets.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            after = model.forward_features(probe).float()
        velocities[group_name] = float(((after - before).square().mean().sqrt() / eta).cpu())
    return {
        "shape": shape.label,
        "dial": shape.dial,
        "rho": shape.rho,
        "seed": seed,
        "normalized_eta": eta,
        "final_hidden_feature_velocity_rms_over_eta": velocities,
    }


def summarize_feature_velocity_audits(
    audits: Sequence[Mapping[str, object]],
    shapes: Sequence[Shape],
    seeds: Sequence[int],
    *,
    slope_tolerance: float = 0.15,
) -> Dict[str, object]:
    groups = (
        "jiang_embeddings",
        "jiang_norms",
        "jiang_attention_qkv",
        "jiang_attention_output",
        "jiang_ffn_up",
        "jiang_ffn_down",
        "jiang_other_biases",
    )
    rows = []
    for group in groups:
        means = []
        by_shape_seed = []
        for shape in shapes:
            values = {
                seed: float(
                    next(
                        row
                        for row in audits
                        if row["shape"] == shape.label and row["seed"] == seed
                    )["final_hidden_feature_velocity_rms_over_eta"][group]  # type: ignore[index]
                )
                for seed in seeds
            }
            by_shape_seed.append(values)
            means.append(sum(values.values()) / len(values))
        # The smallest point is excluded exactly as preregistered.
        selected_shapes = shapes[1:]
        selected_means = means[1:]
        finite_positive = all(math.isfinite(value) and value > 0.0 for value in selected_means)
        if finite_positive:
            x = [math.log(shape.dial) for shape in selected_shapes]
            y = [math.log(value) for value in selected_means]
            x_bar = sum(x) / len(x)
            y_bar = sum(y) / len(y)
            denominator = sum((value - x_bar) ** 2 for value in x)
            slope = (
                sum((a - x_bar) * (b - y_bar) for a, b in zip(x, y)) / denominator
                if denominator
                else float("nan")
            )
        else:
            slope = float("nan")
        rows.append(
            {
                "group": group,
                "mean_velocity_by_shape": means,
                "velocity_by_shape_seed": by_shape_seed,
                "log_velocity_vs_log_dial_slope_excluding_smallest": slope,
                "accepted": finite_positive and abs(slope) <= slope_tolerance,
            }
        )
    return {
        "criterion": "group-only final-hidden feature velocity RMS divided by eta",
        "slope_tolerance": slope_tolerance,
        "accepted": all(bool(row["accepted"]) for row in rows),
        "groups": rows,
    }


def selected(trials: Iterable[Trial], rule: str) -> List[Trial]:
    return [trial for trial in trials if trial.rule == rule]


def progress_report(
    trials: Sequence[Trial],
    shapes: Sequence[Shape],
    seeds: Sequence[int],
    *,
    rule: str,
    slope_tolerance: float = 0.30,
    minimum_progress: float = 1e-3,
) -> Dict[str, object]:
    rows = []
    chosen = selected(trials, rule)
    common_steps = sorted(set.intersection(*(set(trial.validation_loss_checkpoints) for trial in chosen)))
    for step in (value for value in common_steps if value > 0):
        mean_progress = []
        progress_by_shape_seed = []
        for shape in shapes:
            values = {}
            for seed in seeds:
                trial = next(
                    item for item in chosen if item.label == shape.label and item.seed == seed
                )
                initial = trial.validation_loss_checkpoints[0]
                current = trial.validation_loss_checkpoints[step]
                values[seed] = (initial - current) / max(abs(initial), 1e-12)
            progress_by_shape_seed.append(values)
            mean_progress.append(sum(values.values()) / len(values))
        nontrivial = all(
            math.isfinite(value) and value >= minimum_progress for value in mean_progress
        )
        if nontrivial:
            x = [math.log(shape.dial) for shape in shapes]
            y = [math.log(value) for value in mean_progress]
            x_bar = sum(x) / len(x)
            y_bar = sum(y) / len(y)
            slope = sum((a - x_bar) * (b - y_bar) for a, b in zip(x, y)) / sum(
                (value - x_bar) ** 2 for value in x
            )
        else:
            slope = float("nan")
        rows.append(
            {
                "step": step,
                "mean_fractional_progress": mean_progress,
                "progress_by_shape_seed": progress_by_shape_seed,
                "log_progress_vs_log_dial_slope": slope,
                "nontrivial": nontrivial,
                "scale_invariant": nontrivial and abs(slope) <= slope_tolerance,
            }
        )
    final = rows[-1]
    return {
        "rule": rule,
        "slope_tolerance": slope_tolerance,
        "minimum_progress": minimum_progress,
        "accepted": bool(final["scale_invariant"]),
        "final_log_progress_vs_log_dial_slope": final[
            "log_progress_vs_log_dial_slope"
        ],
        "checkpoints": rows,
    }


def build_report(
    trials: Sequence[Trial],
    shapes: Sequence[Shape],
    seeds: Sequence[int],
    *,
    reference: JiangChizatReference,
    eta: float,
    epsilon0: float,
    steps: int,
    rules: Sequence[str],
    feature_velocity_audit: Mapping[str, object],
) -> Dict[str, object]:
    primary = progress_report(trials, shapes, seeds, rule="primary")
    controls = {}
    for rule in rules:
        if rule == "primary":
            continue
        report = progress_report(trials, shapes, seeds, rule=rule)
        controls[rule] = {"rejected": not bool(report["accepted"]), "progress": report}
    return {
        "schema_version": 1,
        "experiment": "jiang_attention_chizat_ffn_interleaved_fixed_eta_transfer",
        "status": "completed",
        "host": socket.gethostname(),
        "completed_at": utc_now(),
        "parameterization": {
            "optimizer": "adam",
            "normalized_coordinate": "eta",
            "architecture": "pre-LN MHSA then dense mean-field FFN, each residual 1/L",
            "attention_logits": "QK^T/d_head",
            "ffn_down_initialization": "sqrt(D)/M",
            "reference": asdict(reference),
            "preferred_joint_invariant": "L*M/D",
        },
        "normalized_eta": eta,
        "epsilon0": epsilon0,
        "steps": steps,
        "seeds": list(seeds),
        "shapes": [{**asdict(shape), "rho": shape.rho} for shape in shapes],
        "transfer_verdict": {
            "accepted": bool(primary["accepted"]),
            "requires": ["nontrivial_progress", "absolute_log_progress_slope_at_most_0.30"],
        },
        "fixed_eta_progress": primary,
        "feature_velocity_audit": feature_velocity_audit,
        "negative_controls": controls,
        "trials": [asdict(trial) for trial in trials],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", type=parse_shape, required=True)
    parser.add_argument("--reference-L", type=int, required=True)
    parser.add_argument("--reference-M", type=int, required=True)
    parser.add_argument("--reference-D", type=int, required=True)
    parser.add_argument("--head-dimension", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-validation", type=int, default=128)
    parser.add_argument("--dataset-seed", type=int, default=1729)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--epsilon0", type=float, default=1e-12)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47, 71])
    parser.add_argument("--rules", nargs="+", choices=RULES, default=list(RULES))
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="run the preregistered group-only feature-velocity audit and stop",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shapes = validate_shapes(args.shapes, args.head_dimension)
    if "primary" not in args.rules:
        raise ValueError("rules must include primary")
    if sorted(set(args.seeds)) != args.seeds or len(args.seeds) < 2:
        raise ValueError("seeds must contain at least two unique increasing values")
    if min(args.steps, args.batch_size, args.n_train, args.n_validation) <= 0:
        raise ValueError("steps, batch size, and dataset sizes must be positive")
    if args.batch_size > args.n_train:
        raise ValueError("batch-size cannot exceed n-train")
    reference = JiangChizatReference(
        args.reference_L, args.reference_M, args.reference_D
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    print(
        json.dumps(
            {
                "phase": "theory-recall-before-trials",
                "theory": JIANG_COMPLETEP_ADAM_THEORY.to_dict(),
                "scope": "derived dense interleaving; not the sparse-MoE architecture",
                "optimizer": {
                    "betas": [0.9, 0.95],
                    "epsilon0": args.epsilon0,
                    "weight_decay": 0.0,
                    "gradient_clipping": "none",
                    "schedule": "linear warmup for first half then constant peak",
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    trials = []
    raw_audits = []
    for shape in shapes:
        for seed in args.seeds:
            print(
                json.dumps(
                    {"phase": "feature-velocity-audit", "shape": shape.label, "seed": seed},
                    sort_keys=True,
                ),
                flush=True,
            )
            raw_audits.append(
                group_feature_velocity_audit(
                    shape,
                    reference=reference,
                    head_dimension=args.head_dimension,
                    vocab_size=args.vocab_size,
                    context_length=args.context_length,
                    n_train=args.n_train,
                    n_validation=args.n_validation,
                    dataset_seed=args.dataset_seed,
                    eta=args.eta,
                    epsilon0=args.epsilon0,
                    batch_size=args.batch_size,
                    seed=seed,
                    device=device,
                )
            )
    feature_velocity_audit = summarize_feature_velocity_audits(
        raw_audits, shapes, args.seeds
    )
    if args.audit_only:
        audit_report = {
            "schema_version": 1,
            "experiment": "jiang_chizat_group_feature_velocity_audit",
            "status": "completed",
            "host": socket.gethostname(),
            "completed_at": utc_now(),
            "reference": asdict(reference),
            "normalized_eta": args.eta,
            "epsilon0": args.epsilon0,
            "seeds": list(args.seeds),
            "shapes": [{**asdict(shape), "rho": shape.rho} for shape in shapes],
            "feature_velocity_audit": feature_velocity_audit,
            "raw_audits": raw_audits,
        }
        atomic_write_json(args.output, audit_report)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "feature_velocity_audit_accepted": feature_velocity_audit[
                        "accepted"
                    ],
                },
                sort_keys=True,
            )
        )
        return
    for rule in args.rules:
        for shape in shapes:
            for seed in args.seeds:
                print(
                    json.dumps(
                        {
                            "rule": rule,
                            "shape": shape.label,
                            "seed": seed,
                            "status": "running",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                trials.append(
                    run_trial(
                        shape,
                        reference=reference,
                        head_dimension=args.head_dimension,
                        vocab_size=args.vocab_size,
                        context_length=args.context_length,
                        n_train=args.n_train,
                        n_validation=args.n_validation,
                        dataset_seed=args.dataset_seed,
                        eta=args.eta,
                        epsilon0=args.epsilon0,
                        steps=args.steps,
                        batch_size=args.batch_size,
                        seed=seed,
                        rule=rule,
                        device=device,
                        warmup_steps=args.steps // 2,
                    )
                )
    report = build_report(
        trials,
        shapes,
        args.seeds,
        reference=reference,
        eta=args.eta,
        epsilon0=args.epsilon0,
        steps=args.steps,
        rules=args.rules,
        feature_velocity_audit=feature_velocity_audit,
    )
    atomic_write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "transfer_accepted": report["transfer_verdict"]["accepted"],
                "final_slope": report["fixed_eta_progress"][
                    "final_log_progress_vs_log_dial_slope"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
