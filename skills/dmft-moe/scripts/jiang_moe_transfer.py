#!/usr/bin/env python3
"""Full Jiang et al. sparse-MoE Adam hyperparameter-transfer experiment.

Shapes use ``label:L:M:D:E:A:dial``.  The primary ladder must keep active
expert fraction A/E fixed.  The recommended joint ladder additionally keeps
L*M/D fixed while increasing model size.
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
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_theorist.autoscaler.jiang_moe import (  # noqa: E402
    JIANG_MOE_ADAM_THEORY,
    JIANG_MOE_REPORTED_LR_MULTIPLIERS,
    JiangMoEReference,
    JiangMoEShape,
    JiangMoETransformer,
)
from ai_theorist.autoscaler.lr_contract import raw_group_epsilons, raw_group_rates  # noqa: E402


PRIMARY_RULE = "table2"
CONTROL_RULES = (
    "global_lr_control",
    "omit_router_width",
    "omit_expert_down_ratio",
)
MOE_LR_GROUPS = (
    "jiang_moe_embeddings",
    "jiang_moe_norms",
    "jiang_moe_attention_qkv",
    "jiang_moe_attention_output",
    "jiang_moe_router",
    "jiang_moe_expert_up",
    "jiang_moe_expert_down",
    "jiang_moe_other_biases",
)


@dataclass(frozen=True)
class Shape:
    label: str
    L: int
    M: int
    D: int
    E: int
    A: int
    dial: float

    @property
    def kappa(self) -> float:
        return self.A / self.E

    @property
    def rho(self) -> float:
        return self.L * self.M / self.D

    def model_shape(self, head_dimension: int) -> JiangMoEShape:
        return JiangMoEShape(
            depth=self.L,
            residual_width=self.D,
            expert_width=self.M,
            head_dimension=head_dimension,
            num_experts=self.E,
            active_experts=self.A,
        )


@dataclass(frozen=True)
class Trial:
    label: str
    L: int
    M: int
    D: int
    E: int
    A: int
    dial: float
    kappa: float
    rho: float
    seed: int
    eta: float
    expert_bias_learning_rate: float
    learning_rate_multipliers: Dict[str, float]
    warmup_steps: int
    rule: str
    raw_learning_rates: Dict[str, float]
    adam_epsilons: Dict[str, float]
    initial_validation_loss: float
    final_validation_loss: float
    fractional_progress: float
    maximum_routing_load_deviation: float
    diverged: bool


def parse_shape(text: str) -> Shape:
    parts = text.split(":")
    if len(parts) != 7:
        raise ValueError("shape must use label:L:M:D:E:A:dial")
    label = parts[0].strip()
    L, M, D, E, A = (int(value) for value in parts[1:6])
    dial = float(parts[6])
    if not label or min(L, M, D, E, A) <= 0 or A > E or not math.isfinite(dial) or dial <= 0:
        raise ValueError("shape fields must be positive and A cannot exceed E")
    return Shape(label, L, M, D, E, A, dial)


def validate_shapes(shapes: Sequence[Shape], head_dimension: int) -> Tuple[Shape, ...]:
    rows = tuple(shapes)
    if len(rows) < 4:
        raise ValueError("at least four shapes are required")
    if len({row.label for row in rows}) != len(rows):
        raise ValueError("shape labels must be unique")
    if any(right.dial <= left.dial for left, right in zip(rows, rows[1:])):
        raise ValueError("shape dials must be strictly increasing")
    kappa = rows[0].kappa
    if any(not math.isclose(row.kappa, kappa) for row in rows):
        raise ValueError("Jiang MoE transfer requires fixed A/E sparsity")
    for row in rows:
        row.model_shape(head_dimension)
    return rows


def synthetic_markov_data(
    *,
    vocab_size: int,
    context_length: int,
    n_train: int,
    n_validation: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = n_train + n_validation
    tokens = torch.empty(count, context_length + 1, dtype=torch.long)
    tokens[:, :3] = torch.randint(vocab_size, (count, 3), generator=generator)
    transitions = torch.stack(
        [torch.randperm(vocab_size, generator=generator) for _ in range(8)]
    )
    for position in range(3, context_length + 1):
        state = (tokens[:, position - 3] + 3 * tokens[:, position - 2]).remainder(8)
        previous = tokens[:, position - 1]
        next_token = transitions[state, previous]
        noise = torch.rand(count, generator=generator) < 0.03
        random_token = torch.randint(vocab_size, (count,), generator=generator)
        tokens[:, position] = torch.where(noise, random_token, next_token)
    return tuple(
        value.to(device)
        for value in (
            tokens[:n_train, :-1],
            tokens[:n_train, 1:],
            tokens[n_train:, :-1],
            tokens[n_train:, 1:],
        )
    )  # type: ignore[return-value]


@torch.no_grad()
def validation_loss(
    model: JiangMoETransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, len(inputs), batch_size):
        batch_inputs = inputs[start : start + batch_size]
        batch_targets = targets[start : start + batch_size]
        logits = model(batch_inputs)
        total += float(
            F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                batch_targets.reshape(-1),
                reduction="sum",
            ).cpu()
        )
        count += batch_targets.numel()
    return total / count


def run_trial(
    shape: Shape,
    *,
    reference: JiangMoEReference,
    head_dimension: int,
    vocab_size: int,
    context_length: int,
    n_train: int,
    n_validation: int,
    dataset_seed: int,
    eta: float,
    epsilon0: float,
    expert_bias_learning_rate: float,
    steps: int,
    batch_size: int,
    seed: int,
    rule: str,
    device: torch.device,
    learning_rate_multipliers: Mapping[str, float] | None = None,
    warmup_steps: int = 0,
) -> Tuple[Trial, Dict[str, object]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = JiangMoETransformer(
        shape.model_shape(head_dimension),
        vocab_size=vocab_size,
        context_length=context_length,
        reference=reference,
    ).to(device)
    multipliers = dict(JIANG_MOE_REPORTED_LR_MULTIPLIERS)
    multipliers.update(
        {
            name: float(value)
            for name, value in (learning_rate_multipliers or {}).items()
        }
    )
    groups = model.optimizer_parameter_groups(
        eta,
        epsilon0=epsilon0,
        rule=rule,
        learning_rate_multipliers=multipliers,
    )
    audit = model.optimizer_contract_audit(
        eta,
        epsilon0=epsilon0,
        rule=rule,
        learning_rate_multipliers=multipliers,
    )
    optimizer = torch.optim.Adam(
        groups,
        lr=eta,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    x_train, y_train, x_validation, y_validation = synthetic_markov_data(
        vocab_size=vocab_size,
        context_length=context_length,
        n_train=n_train,
        n_validation=n_validation,
        seed=dataset_seed,
        device=device,
    )
    initial = validation_loss(model, x_validation, y_validation, batch_size)
    generator = torch.Generator(device="cpu").manual_seed(100_003 + seed)
    diverged = False
    if warmup_steps < 0 or warmup_steps > steps:
        raise ValueError("warmup_steps must lie between zero and steps")
    peak_rates = [float(group["lr"]) for group in optimizer.param_groups]
    for step in range(steps):
        schedule_multiplier = (
            min(1.0, (step + 1) / warmup_steps) if warmup_steps else 1.0
        )
        for group, peak_rate in zip(optimizer.param_groups, peak_rates):
            group["lr"] = peak_rate * schedule_multiplier
        indices = torch.randint(0, n_train, (batch_size,), generator=generator).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train[indices])
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y_train[indices].reshape(-1),
        )
        if not torch.isfinite(loss):
            diverged = True
            break
        loss.backward()
        optimizer.step()
        model.update_expert_biases(expert_bias_learning_rate)
    final = validation_loss(model, x_validation, y_validation, batch_size)
    routing = model.routing_diagnostics()
    diverged = diverged or not math.isfinite(final) or final > 1e8
    progress = (initial - final) / max(abs(initial), 1e-12)
    return (
        Trial(
            shape.label,
            shape.L,
            shape.M,
            shape.D,
            shape.E,
            shape.A,
            shape.dial,
            shape.kappa,
            shape.rho,
            seed,
            eta,
            expert_bias_learning_rate,
            multipliers,
            warmup_steps,
            rule,
            raw_group_rates(groups),
            raw_group_epsilons(groups),
            initial,
            final,
            progress,
            float(routing["maximum_absolute_load_deviation"]),
            diverged,
        ),
        {
            "optimizer": audit,
            "manual_expert_bias": model.manual_parameter_contract(
                expert_bias_learning_rate
            ),
            "schedule": {
                "kind": "linear_warmup_then_constant",
                "warmup_steps": warmup_steps,
                "constant_steps": steps - warmup_steps,
                "peak_learning_rates": raw_group_rates(groups),
            },
        },
    )


def _log_slope(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if not all(math.isfinite(value) and value > 0.0 for value in y_values):
        return float("nan")
    x = [math.log(value) for value in x_values]
    y = [math.log(value) for value in y_values]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    return sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / sum(
        (value - x_mean) ** 2 for value in x
    )


def fixed_eta_analysis(
    trials: Sequence[Trial],
    shapes: Sequence[Shape],
    seeds: Sequence[int],
    *,
    eta: float,
    rule: str,
) -> Dict[str, object]:
    rows = [
        [
            trial
            for trial in trials
            if trial.rule == rule
            and trial.label == shape.label
            and math.isclose(trial.eta, eta)
        ]
        for shape in shapes
    ]
    if any(len(row) != len(seeds) for row in rows):
        raise ValueError(f"incomplete fixed-eta factorial for {rule}")
    progress = [sum(trial.fractional_progress for trial in row) / len(row) for row in rows]
    losses = [sum(trial.final_validation_loss for trial in row) / len(row) for row in rows]
    routing = [
        sum(trial.maximum_routing_load_deviation for trial in row) / len(row)
        for row in rows
    ]
    slope = _log_slope([shape.dial for shape in shapes], progress)
    return {
        "rule": rule,
        "eta": eta,
        "mean_fractional_progress": progress,
        "mean_final_validation_loss": losses,
        "mean_maximum_routing_load_deviation": routing,
        "log_progress_vs_log_dial_slope": slope,
        "all_finite": all(not trial.diverged for row in rows for trial in row),
        "accepted": (
            all(not trial.diverged for row in rows for trial in row)
            and all(value >= 1e-3 for value in progress)
            and math.isfinite(slope)
            and abs(slope) <= 0.30
        ),
    }


def primary_analysis(
    trials: Sequence[Trial],
    shapes: Sequence[Shape],
    etas: Sequence[float],
    seeds: Sequence[int],
    reference_label: str,
    oracle_tolerance: float,
) -> Dict[str, object]:
    means = {}
    for shape in shapes:
        for eta in etas:
            rows = [
                trial.final_validation_loss
                for trial in trials
                if trial.rule == PRIMARY_RULE
                and trial.label == shape.label
                and math.isclose(trial.eta, eta)
            ]
            if len(rows) != len(seeds):
                raise ValueError("incomplete primary eta factorial")
            means[(shape.label, eta)] = sum(rows) / len(rows)
    best_eta = {
        shape.label: min(etas, key=lambda eta: means[(shape.label, eta)])
        for shape in shapes
    }
    reference_eta = best_eta[reference_label]
    reference_index = list(etas).index(reference_eta)
    offsets = {
        shape.label: math.log10(best_eta[shape.label] / reference_eta)
        for shape in shapes
    }
    fixed = fixed_eta_analysis(
        trials,
        shapes,
        seeds,
        eta=reference_eta,
        rule=PRIMARY_RULE,
    )
    oracle_losses = {
        shape.label: means[(shape.label, best_eta[shape.label])]
        for shape in shapes
    }
    fixed_losses = {
        shape.label: means[(shape.label, reference_eta)]
        for shape in shapes
    }
    oracle_ratios = {
        shape.label: fixed_losses[shape.label] / max(oracle_losses[shape.label], 1e-30)
        for shape in shapes
    }
    gates = {
        "reference_optimum_is_interior": 0 < reference_index < len(etas) - 1,
        "every_shape_has_a_complete_finite_oracle": all(
            math.isfinite(value) for value in oracle_losses.values()
        ),
        "fixed_reference_eta_dynamics_are_stable": bool(fixed["accepted"]),
        "fixed_reference_eta_near_shape_oracle": max(oracle_ratios.values()) <= oracle_tolerance,
    }
    return {
        "reference_shape": reference_label,
        "reference_eta": reference_eta,
        "reference_optimum_is_interior": 0 < reference_index < len(etas) - 1,
        "best_eta_by_shape": best_eta,
        "best_eta_offset_decades": offsets,
        "maximum_absolute_best_eta_offset_decades": max(
            abs(value) for value in offsets.values()
        ),
        "oracle_validation_loss_by_shape": oracle_losses,
        "fixed_reference_eta_validation_loss_by_shape": fixed_losses,
        "fixed_reference_eta_to_oracle_loss_ratio_by_shape": oracle_ratios,
        "maximum_fixed_reference_eta_to_oracle_loss_ratio": max(oracle_ratios.values()),
        "oracle_loss_ratio_tolerance": oracle_tolerance,
        "fixed_reference_eta": fixed,
        "diagnostics": {
            "exact_grid_argmin_drift_within_0.35_decades": (
                max(abs(value) for value in offsets.values()) <= 0.35
            ),
            "interpretation": (
                "exact discrete argmin drift is descriptive; the hard transfer gate is "
                "the fixed reference eta's loss ratio to each shape oracle"
            ),
        },
        "gates": gates,
        "accepted": all(gates.values()),
    }


def best_eta_for_reference(trials: Sequence[Trial], etas: Sequence[float]) -> float:
    expected_seeds = {trial.seed for trial in trials}

    def score(eta: float) -> float:
        rows = [
            trial
            for trial in trials
            if math.isclose(trial.eta, eta)
        ]
        if (
            {trial.seed for trial in rows} != expected_seeds
            or any(trial.diverged for trial in rows)
        ):
            return math.inf
        return sum(trial.final_validation_loss for trial in rows) / len(rows)

    return min(etas, key=score)


def best_group_multiplier(
    trials_by_multiplier: Mapping[float, Sequence[Trial]],
    *,
    minimum_relative_improvement: float,
) -> float:
    baseline = {trial.seed: trial for trial in trials_by_multiplier[1.0]}
    qualifying = [1.0]
    for relative_factor, rows in trials_by_multiplier.items():
        if relative_factor == 1.0:
            continue
        candidate = {trial.seed: trial for trial in rows}
        if set(candidate) != set(baseline):
            continue
        improvements = []
        for seed, base_trial in baseline.items():
            candidate_trial = candidate[seed]
            if base_trial.diverged or candidate_trial.diverged:
                improvements = []
                break
            improvements.append(
                (base_trial.final_validation_loss - candidate_trial.final_validation_loss)
                / max(abs(base_trial.final_validation_loss), 1e-30)
            )
        if (
            improvements
            and all(value > 0.0 for value in improvements)
            and sum(improvements) / len(improvements) >= minimum_relative_improvement
        ):
            qualifying.append(relative_factor)

    def score(relative_factor: float) -> float:
        finite = [
            trial.final_validation_loss
            for trial in trials_by_multiplier[relative_factor]
            if not trial.diverged
        ]
        return sum(finite) / len(finite) if finite else math.inf

    return min(qualifying, key=score)


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
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=parse_shape, nargs="+", required=True)
    parser.add_argument("--reference-shape", required=True)
    parser.add_argument("--head-dimension", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-validation", type=int, default=512)
    parser.add_argument("--dataset-seed", type=int, default=1729)
    parser.add_argument("--etas", type=float, nargs="+", required=True)
    parser.add_argument("--epsilon0", type=float, default=1e-12)
    parser.add_argument("--expert-bias-learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--multiplier-probes",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
        help="reference-only coordinate sweep for every optimizer group",
    )
    parser.add_argument(
        "--minimum-relative-multiplier-improvement",
        type=float,
        default=0.005,
    )
    parser.add_argument("--oracle-tolerance", type=float, default=1.10)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--controls", choices=CONTROL_RULES, nargs="+", default=list(CONTROL_RULES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shapes = validate_shapes(args.shapes, args.head_dimension)
    shape_by_label = {shape.label: shape for shape in shapes}
    if args.reference_shape not in shape_by_label:
        raise ValueError("reference shape must be in the shape ladder")
    reference_shape = shape_by_label[args.reference_shape]
    reference = JiangMoEReference(
        depth=reference_shape.L,
        residual_width=reference_shape.D,
        expert_width=reference_shape.M,
        num_experts=reference_shape.E,
        active_experts=reference_shape.A,
    )
    etas = tuple(args.etas)
    seeds = tuple(args.seeds)
    if len(etas) < 5 or any(value <= 0.0 for value in etas):
        raise ValueError("at least five positive eta probes are required")
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three unique paired seeds are required")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    print(
        json.dumps(
            {
                "phase": "theory-recall",
                "theory": JIANG_MOE_ADAM_THEORY.to_dict(),
                "protocol": {
                    "group_constants": "Appendix D.1 source constants, locally verified by relative coordinate probes at reference shape",
                    "reported_lr_multipliers": JIANG_MOE_REPORTED_LR_MULTIPLIERS,
                    "reported_init_multipliers": {
                        "attention_value": 1.0 / 16.0,
                        "expert_down": 1.0 / 4.0,
                    },
                    "schedule": "linear warmup for first half, then constant peak LR",
                    "gradient_clipping": "none",
                    "adam_betas": [0.9, 0.95],
                    "base_epsilon": args.epsilon0,
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if (
        not args.multiplier_probes
        or any(value <= 0.0 or not math.isfinite(value) for value in args.multiplier_probes)
        or 1.0 not in args.multiplier_probes
    ):
        raise ValueError("multiplier probes must be positive, finite, and include 1.0")
    if args.minimum_relative_multiplier_improvement < 0.0:
        raise ValueError("minimum relative multiplier improvement cannot be negative")
    if not math.isfinite(args.oracle_tolerance) or args.oracle_tolerance < 1.0:
        raise ValueError("oracle tolerance must be finite and at least one")
    warmup_steps = args.steps // 2
    source_multipliers = dict(JIANG_MOE_REPORTED_LR_MULTIPLIERS)
    calibration_trials: List[Trial] = []
    calibration_audits: Dict[str, object] = {}

    # Stage 1: tune the global coordinate with the authors' reported constants.
    global_screen: List[Trial] = []
    for eta in etas:
        for seed in seeds:
            trial, audit = run_trial(
                reference_shape,
                reference=reference,
                head_dimension=args.head_dimension,
                vocab_size=args.vocab_size,
                context_length=args.context_length,
                n_train=args.n_train,
                n_validation=args.n_validation,
                dataset_seed=args.dataset_seed,
                eta=eta,
                epsilon0=args.epsilon0,
                expert_bias_learning_rate=args.expert_bias_learning_rate,
                steps=args.steps,
                batch_size=args.batch_size,
                seed=seed,
                rule=PRIMARY_RULE,
                device=device,
                learning_rate_multipliers=source_multipliers,
                warmup_steps=warmup_steps,
            )
            global_screen.append(trial)
            calibration_trials.append(trial)
            calibration_audits.setdefault(f"global:eta={eta}", audit)
    calibration_eta = best_eta_for_reference(global_screen, etas)

    # Stage 2: tune each scale-independent group constant at the reference
    # shape.  The exponents themselves are never fitted.
    tuned_multipliers = dict(source_multipliers)
    multiplier_choices: Dict[str, Dict[str, float]] = {}
    for group_name in MOE_LR_GROUPS:
        candidates: Dict[float, List[Trial]] = {}
        base_multiplier = tuned_multipliers[group_name]
        for relative_factor in args.multiplier_probes:
            absolute_multiplier = base_multiplier * float(relative_factor)
            candidate_multipliers = {
                **tuned_multipliers,
                group_name: absolute_multiplier,
            }
            rows = []
            for seed in seeds:
                trial, audit = run_trial(
                    reference_shape,
                    reference=reference,
                    head_dimension=args.head_dimension,
                    vocab_size=args.vocab_size,
                    context_length=args.context_length,
                    n_train=args.n_train,
                    n_validation=args.n_validation,
                    dataset_seed=args.dataset_seed,
                    eta=calibration_eta,
                    epsilon0=args.epsilon0,
                    expert_bias_learning_rate=args.expert_bias_learning_rate,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    seed=seed,
                    rule=PRIMARY_RULE,
                    device=device,
                    learning_rate_multipliers=candidate_multipliers,
                    warmup_steps=warmup_steps,
                )
                rows.append(trial)
                calibration_trials.append(trial)
                calibration_audits.setdefault(
                    f"group={group_name}:relative_factor={relative_factor}:absolute={absolute_multiplier}", audit
                )
            candidates[float(relative_factor)] = rows
        selected_relative_factor = best_group_multiplier(
            candidates,
            minimum_relative_improvement=args.minimum_relative_multiplier_improvement,
        )
        selected_absolute_multiplier = base_multiplier * selected_relative_factor
        tuned_multipliers[group_name] = selected_absolute_multiplier
        multiplier_choices[group_name] = {
            "source": source_multipliers[group_name],
            "relative_factor": selected_relative_factor,
            "selected_absolute": selected_absolute_multiplier,
        }

    trials: List[Trial] = []
    audits: Dict[str, object] = {}
    for shape in shapes:
        for eta in etas:
            for seed in seeds:
                print(json.dumps({"phase": "primary-sweep", "shape": shape.label, "eta": eta, "seed": seed}, sort_keys=True), flush=True)
                trial, audit = run_trial(
                    shape,
                    reference=reference,
                    head_dimension=args.head_dimension,
                    vocab_size=args.vocab_size,
                    context_length=args.context_length,
                    n_train=args.n_train,
                    n_validation=args.n_validation,
                    dataset_seed=args.dataset_seed,
                    eta=eta,
                    epsilon0=args.epsilon0,
                    expert_bias_learning_rate=args.expert_bias_learning_rate,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    seed=seed,
                    rule=PRIMARY_RULE,
                    device=device,
                    learning_rate_multipliers=tuned_multipliers,
                    warmup_steps=warmup_steps,
                )
                trials.append(trial)
                audits.setdefault(f"{PRIMARY_RULE}:{shape.label}:eta={eta}", audit)
    primary = primary_analysis(
        trials,
        shapes,
        etas,
        seeds,
        args.reference_shape,
        args.oracle_tolerance,
    )
    selected_eta = float(primary["reference_eta"])
    controls = {}
    for rule in args.controls:
        for shape in shapes:
            for seed in seeds:
                print(json.dumps({"phase": "negative-control", "rule": rule, "shape": shape.label, "eta": selected_eta, "seed": seed}, sort_keys=True), flush=True)
                trial, audit = run_trial(
                    shape,
                    reference=reference,
                    head_dimension=args.head_dimension,
                    vocab_size=args.vocab_size,
                    context_length=args.context_length,
                    n_train=args.n_train,
                    n_validation=args.n_validation,
                    dataset_seed=args.dataset_seed,
                    eta=selected_eta,
                    epsilon0=args.epsilon0,
                    expert_bias_learning_rate=args.expert_bias_learning_rate,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    seed=seed,
                    rule=rule,
                    device=device,
                    learning_rate_multipliers=tuned_multipliers,
                    warmup_steps=warmup_steps,
                )
                trials.append(trial)
                audits.setdefault(f"{rule}:{shape.label}:eta={selected_eta}", audit)
        analysis = fixed_eta_analysis(
            trials,
            shapes,
            seeds,
            eta=selected_eta,
            rule=rule,
        )
        controls[rule] = {**analysis, "rejected": not bool(analysis["accepted"])}
    report = {
        "schema_version": 1,
        "experiment": "full_jiang_sparse_moe_adam_transfer",
        "status": "completed",
        "host": socket.gethostname(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "theory_recalled_before_trials": JIANG_MOE_ADAM_THEORY.to_dict(),
        "reference_group_calibration": {
            "method": "one ordered relative coordinate-sweep pass around the Appendix D.1 source constants at the reference shape",
            "global_eta_screen": list(etas),
            "selected_calibration_eta": calibration_eta,
            "relative_multiplier_probes": list(args.multiplier_probes),
            "minimum_relative_multiplier_improvement": args.minimum_relative_multiplier_improvement,
            "update_gate": "candidate must improve validation loss for every paired seed and exceed the mean relative-improvement threshold",
            "group_order": list(MOE_LR_GROUPS),
            "source_learning_rate_multipliers": source_multipliers,
            "selected_learning_rate_multiplier_details": multiplier_choices,
            "selected_learning_rate_multipliers": tuned_multipliers,
            "trials": [asdict(trial) for trial in calibration_trials],
            "optimizer_group_contract_audits": calibration_audits,
        },
        "architecture_contract": {
            "decoder": "tied token embedding/unembedding, learned absolute positions, pre-LN",
            "attention": "QK^T/d_head with fixed d_head",
            "residual_branches": "interleaved MHSA and MoE, each multiplied by 1/L",
            "router": "sigmoid gates, hard top-A selection, no gradient through selected set",
            "expert": "GELU one-hidden-layer MLP with Table-2 scaling and Appendix-D.1 down-init multiplier 1/4",
            "constant_scale_initialization": "attention V multiplier 1/16; expert down multiplier 1/4; all others one",
            "constant_scale_learning_rates": "attention QKV, router, and expert down multipliers 1/16; all others one before optional reference-only relative refinement",
            "sparsity": "A/E fixed across scale",
            "manual_expert_bias_update": "eta_bias constant across expert count at fixed A/E",
        },
        "fixed_variables": {
            "context_length": args.context_length,
            "n_train": args.n_train,
            "n_validation": args.n_validation,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "token_horizon": args.steps * args.batch_size * args.context_length,
            "dataset_seed": args.dataset_seed,
            "epsilon0": args.epsilon0,
            "expert_bias_learning_rate": args.expert_bias_learning_rate,
            "warmup_steps": warmup_steps,
            "constant_peak_steps": args.steps - warmup_steps,
            "gradient_clipping": "none",
            "adam_betas": [0.9, 0.95],
            "weight_decay": 0.0,
        },
        "reference": asdict(reference),
        "shapes": [{**asdict(shape), "kappa": shape.kappa, "rho": shape.rho} for shape in shapes],
        "etas": list(etas),
        "seeds": list(seeds),
        "optimizer_group_contract_audits": audits,
        "primary": primary,
        "negative_controls": controls,
        "verdict": {
            "primary_accepted": bool(primary["accepted"]),
            "transfer_certified": bool(primary["accepted"]),
            "all_negative_controls_rejected": all(bool(row["rejected"]) for row in controls.values()),
            "mechanism_discrimination_certified": bool(primary["accepted"])
            and bool(controls)
            and all(bool(row["rejected"]) for row in controls.values()),
            "certified": bool(primary["accepted"]),
        },
        "trials": [asdict(trial) for trial in trials],
    }
    atomic_write_json(args.output, report)
    print(json.dumps({"output": str(args.output), **report["verdict"]}, sort_keys=True))


if __name__ == "__main__":
    main()
