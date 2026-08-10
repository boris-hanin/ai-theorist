#!/usr/bin/env python3
"""Faithful joint-(L,M,D) transfer test for Chizat equation (22).

This is deliberately separate from the fixed-depth muP MLP harness.  It tests
the mean-ODE residual-particle architecture in Chizat (2025): fixed input and
output maps, critical MLU initialization, 1/(L M) residual branches, vanilla
full-batch GD, and independent normalized eta_u/eta_v coordinates whose raw
rates are multiplied by L M D.
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

from ai_theorist.autoscaler.chizat_resnet import (  # noqa: E402
    CHIZAT_2LP_GD_THEORY,
    Chizat2LPResNet,
    Chizat2LPShape,
    ChizatRateRule,
)
from ai_theorist.autoscaler.lr_contract import raw_group_rates  # noqa: E402


@dataclass(frozen=True)
class NamedShape:
    label: str
    shape: Chizat2LPShape


@dataclass(frozen=True)
class Trial:
    label: str
    L: int
    M: int
    D: int
    LM_over_D: float
    seed: int
    rule: str
    eta_u: float
    eta_v: float
    raw_learning_rates: Dict[str, float]
    initial_validation_loss: float
    final_training_loss: float
    final_validation_loss: float
    fractional_validation_progress: float
    u_relative_rms_movement: float
    v_relative_rms_movement: float
    diverged: bool


def parse_shapes(specification: str) -> List[NamedShape]:
    shapes: List[NamedShape] = []
    labels = set()
    for raw in specification.split(","):
        fields = raw.split(":")
        if len(fields) != 4:
            raise ValueError("each shape must be label:L:M:D")
        label, L, M, D = fields
        if not label or label in labels:
            raise ValueError("shape labels must be unique and nonempty")
        labels.add(label)
        shapes.append(
            NamedShape(
                label=label,
                shape=Chizat2LPShape(
                    depth=int(L), hidden_width=int(M), embedding_dimension=int(D)
                ),
            )
        )
    if not shapes:
        raise ValueError("at least one shape is required")
    return shapes


def parse_float_grid(specification: str) -> List[float]:
    values = [float(value) for value in specification.split(",")]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("eta grids must contain finite positive values")
    if len(set(values)) != len(values):
        raise ValueError("eta grid values must be unique")
    return values


def make_fixed_task(
    *,
    seed: int,
    n_train: int,
    n_validation: int,
    input_dimension: int,
    output_dimension: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A shape-independent nonlinear regression task."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    count = n_train + n_validation
    inputs = torch.randn(count, input_dimension, generator=generator, dtype=dtype)
    teacher_a = torch.randn(
        input_dimension, output_dimension, generator=generator, dtype=dtype
    ) / math.sqrt(input_dimension)
    teacher_b = torch.randn(
        input_dimension, output_dimension, generator=generator, dtype=dtype
    ) / math.sqrt(input_dimension)
    targets = torch.tanh(inputs @ teacher_a) + 0.35 * torch.sin(1.7 * inputs @ teacher_b)
    train_targets = targets[:n_train]
    center = train_targets.mean(dim=0, keepdim=True)
    scale = train_targets.std(dim=0, keepdim=True).clamp_min(torch.finfo(dtype).eps)
    targets = (targets - center) / scale
    return inputs[:n_train], targets[:n_train], inputs[n_train:], targets[n_train:]


def run_trial(
    *,
    named_shape: NamedShape,
    reference_shape: Chizat2LPShape,
    eta_u: float,
    eta_v: float,
    rule: ChizatRateRule,
    seed: int,
    steps: int,
    input_dimension: int,
    output_dimension: int,
    map_seed: int,
    task: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Trial, Dict[str, object]]:
    torch.manual_seed(seed)
    model = Chizat2LPResNet(
        named_shape.shape,
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        map_seed=map_seed,
        dtype=dtype,
    ).to(device)
    groups = model.optimizer_parameter_groups(
        eta_u=eta_u,
        eta_v=eta_v,
        rule=rule,
        reference_shape=reference_shape,
    )
    audit = model.optimizer_contract_audit(
        eta_u=eta_u,
        eta_v=eta_v,
        rule=rule,
        reference_shape=reference_shape,
    )
    optimizer = torch.optim.SGD(groups, lr=1.0, momentum=0.0, weight_decay=0.0)
    train_x, train_y, validation_x, validation_y = (tensor.to(device) for tensor in task)

    with torch.no_grad():
        initial_validation = float(F.mse_loss(model(validation_x), validation_y).item())
        initial_u = model.U.detach().clone()
        initial_v = model.V.detach().clone()
        initial_u_rms = float(initial_u.square().mean().sqrt().item())
        initial_v_rms = float(initial_v.square().mean().sqrt().item())

    diverged = False
    final_training = float("nan")
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        training_loss = F.mse_loss(model(train_x), train_y)
        if not torch.isfinite(training_loss):
            diverged = True
            break
        training_loss.backward()
        # Equation (23) is vanilla full-batch GD: no momentum, decay, schedule,
        # warmup, gradient clipping, or adaptive preconditioning is inserted.
        optimizer.step()
        final_training = float(training_loss.detach().item())
        if not torch.isfinite(model.U).all() or not torch.isfinite(model.V).all():
            diverged = True
            break

    with torch.no_grad():
        validation_loss = F.mse_loss(model(validation_x), validation_y)
        final_validation = float(validation_loss.item()) if torch.isfinite(validation_loss) else math.inf
        u_movement = float((model.U - initial_u).square().mean().sqrt().item()) / max(
            initial_u_rms, 1e-30
        )
        v_movement = float((model.V - initial_v).square().mean().sqrt().item()) / max(
            initial_v_rms, 1e-30
        )
    diverged = diverged or not math.isfinite(final_validation)
    progress = (initial_validation - final_validation) / max(initial_validation, 1e-30)
    shape = named_shape.shape
    return (
        Trial(
            label=named_shape.label,
            L=shape.depth,
            M=shape.hidden_width,
            D=shape.embedding_dimension,
            LM_over_D=shape.rho,
            seed=seed,
            rule=rule,
            eta_u=eta_u,
            eta_v=eta_v,
            raw_learning_rates=raw_group_rates(groups),
            initial_validation_loss=initial_validation,
            final_training_loss=final_training,
            final_validation_loss=final_validation,
            fractional_validation_progress=progress,
            u_relative_rms_movement=u_movement,
            v_relative_rms_movement=v_movement,
            diverged=diverged,
        ),
        audit,
    )


def mean_metric(records: Iterable[Trial], field: str) -> float:
    values = [float(getattr(record, field)) for record in records]
    return sum(values) / len(values) if values else math.inf


def select_best_pair(records: Sequence[Trial], label: str) -> Tuple[float, float]:
    candidates = sorted({(record.eta_u, record.eta_v) for record in records if record.label == label})
    if not candidates:
        raise ValueError(f"no primary records for shape {label}")

    expected_seeds = {record.seed for record in records if record.label == label}

    def complete_paired_score(pair: Tuple[float, float]) -> float:
        rows = [
            record
            for record in records
            if record.label == label
            and record.eta_u == pair[0]
            and record.eta_v == pair[1]
        ]
        if (
            {record.seed for record in rows} != expected_seeds
            or any(record.diverged for record in rows)
            or any(not math.isfinite(record.final_validation_loss) for record in rows)
        ):
            return math.inf
        return mean_metric(rows, "final_validation_loss")

    return min(
        candidates,
        key=complete_paired_score,
    )


def summarize(
    primary: Sequence[Trial],
    controls: Sequence[Trial],
    *,
    shapes: Sequence[NamedShape],
    reference_label: str,
    drift_tolerance_decades: float,
    oracle_tolerance: float,
    eta_us: Sequence[float],
    eta_vs: Sequence[float],
) -> Dict[str, object]:
    best_pairs = {shape.label: select_best_pair(primary, shape.label) for shape in shapes}
    reference_pair = best_pairs[reference_label]
    rows = []
    drifts = []
    oracle_ratios = []
    complete_reference_pair_by_shape = []
    for named_shape in shapes:
        pair = best_pairs[named_shape.label]
        drift = max(
            abs(math.log10(pair[0] / reference_pair[0])),
            abs(math.log10(pair[1] / reference_pair[1])),
        )
        drifts.append(drift)
        oracle_records = [
            record
            for record in primary
            if record.label == named_shape.label
            and record.eta_u == pair[0]
            and record.eta_v == pair[1]
            and not record.diverged
        ]
        fixed_records = [
            record
            for record in primary
            if record.label == named_shape.label
            and record.eta_u == reference_pair[0]
            and record.eta_v == reference_pair[1]
            and not record.diverged
        ]
        expected_seed_count = len(
            {record.seed for record in primary if record.label == named_shape.label}
        )
        complete_reference_pair_by_shape.append(
            len(fixed_records) == expected_seed_count
            and all(math.isfinite(record.final_validation_loss) for record in fixed_records)
        )
        oracle_loss = mean_metric(oracle_records, "final_validation_loss")
        fixed_loss = mean_metric(fixed_records, "final_validation_loss")
        ratio = fixed_loss / max(oracle_loss, 1e-30)
        oracle_ratios.append(ratio)
        rows.append(
            {
                "label": named_shape.label,
                "L": named_shape.shape.depth,
                "M": named_shape.shape.hidden_width,
                "D": named_shape.shape.embedding_dimension,
                "LM_over_D": named_shape.shape.rho,
                "shape_optimal_eta_u": pair[0],
                "shape_optimal_eta_v": pair[1],
                "optimum_drift_decades": drift,
                "oracle_validation_loss": oracle_loss,
                "reference_pair_validation_loss": fixed_loss,
                "reference_pair_to_oracle_loss_ratio": ratio,
                "reference_pair_fractional_progress": mean_metric(
                    fixed_records, "fractional_validation_progress"
                ),
            }
        )

    largest = shapes[-1].label
    correct_largest = [
        record
        for record in primary
        if record.label == largest
        and record.eta_u == reference_pair[0]
        and record.eta_v == reference_pair[1]
    ]
    control_rows = []
    for rule in ("omit_l", "omit_m", "omit_d", "constant_raw"):
        selected = [record for record in controls if record.label == largest and record.rule == rule]
        control_rows.append(
            {
                "rule": rule,
                "largest_shape_validation_loss": mean_metric(selected, "final_validation_loss"),
                "largest_shape_fractional_progress": mean_metric(
                    selected, "fractional_validation_progress"
                ),
            }
        )

    gates = {
        "every_shape_has_a_complete_finite_oracle": all(
            math.isfinite(row["oracle_validation_loss"]) for row in rows
        ),
        "reference_pair_complete_and_finite_at_every_shape": all(
            complete_reference_pair_by_shape
        ),
        "reference_optimum_interior_in_both_coordinates": (
            reference_pair[0] not in {min(eta_us), max(eta_us)}
            and reference_pair[1] not in {min(eta_vs), max(eta_vs)}
        ),
        "normalized_optimum_drift_within_tolerance": max(drifts) <= drift_tolerance_decades,
        "reference_pair_near_shape_oracle": max(oracle_ratios) <= oracle_tolerance,
        "reference_pair_makes_progress_at_every_shape": all(
            row["reference_pair_fractional_progress"] > 0 for row in rows
        ),
    }
    return {
        "reference_label": reference_label,
        "reference_eta_u": reference_pair[0],
        "reference_eta_v": reference_pair[1],
        "shape_rows": rows,
        "largest_shape_correct_validation_loss": mean_metric(
            correct_largest, "final_validation_loss"
        ),
        "largest_shape_correct_fractional_progress": mean_metric(
            correct_largest, "fractional_validation_progress"
        ),
        "negative_controls": control_rows,
        "exploratory_divergent_trial_count": sum(
            record.diverged for record in primary
        ),
        "gates": gates,
        "verdict": "PASS" if all(gates.values()) else "FAIL",
    }


def atomic_json_dump(payload: Mapping[str, object], output: Path) -> None:
    def json_safe(value: object) -> object:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", required=True, help="comma-separated label:L:M:D")
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--eta-us", default="0.03,0.1,0.3,1.0")
    parser.add_argument("--eta-vs", default="0.03,0.1,0.3,1.0")
    parser.add_argument("--seeds", default="11,29,47")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--n-validation", type=int, default=256)
    parser.add_argument("--input-dimension", type=int, default=8)
    parser.add_argument("--output-dimension", type=int, default=2)
    parser.add_argument("--task-seed", type=int, default=314159)
    parser.add_argument("--map-seed", type=int, default=202509)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--drift-tolerance-decades", type=float, default=0.55)
    parser.add_argument("--oracle-tolerance", type=float, default=1.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.steps <= 0 or min(args.n_train, args.n_validation) <= 0:
        raise ValueError("steps and dataset sizes must be positive")
    shapes = parse_shapes(args.shapes)
    by_label = {shape.label: shape for shape in shapes}
    if args.reference_label not in by_label:
        raise ValueError("reference label is not present in --shapes")
    for named_shape in shapes:
        if named_shape.shape.embedding_dimension > named_shape.shape.depth * named_shape.shape.hidden_width:
            raise ValueError(
                f"{named_shape.label} violates the documented D=O(LM) experimental domain"
            )
    eta_us = parse_float_grid(args.eta_us)
    eta_vs = parse_float_grid(args.eta_vs)
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if not seeds:
        raise ValueError("at least one seed is required")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    reference_shape = by_label[args.reference_label].shape
    task = make_fixed_task(
        seed=args.task_seed,
        n_train=args.n_train,
        n_validation=args.n_validation,
        input_dimension=args.input_dimension,
        output_dimension=args.output_dimension,
        dtype=dtype,
    )

    theory_recall = {
        "theory": CHIZAT_2LP_GD_THEORY.to_dict(),
        "forward_equations": {
            "embedding": "h_0 = W_in x; W_in fixed",
            "block": "h_l = h_(l-1) + (1/(L M)) sum_j v_(j,l) tanh(<u_(j,l), h_(l-1)>/D)",
            "readout": "f = W_out^T h_L / D; W_out fixed",
        },
        "initialization": "U_0 and V_0 iid entrywise N(0,D), i.e. std sqrt(D)",
        "optimizer": "vanilla full-batch GD, no clipping/momentum/decay/schedule/warmup",
        "group_rules": {
            "particle_u": "raw_lr_u = eta_u * L * M * D",
            "particle_v": "raw_lr_v = eta_v * L * M * D",
        },
        "tuning_protocol": "tune eta_u and eta_v independently at the reference shape, then keep both fixed",
        "scope_guard": "This is a mean-ODE residual-particle model, not muP MLP depth transfer.",
    }
    # This appears before the first model or optimizer is constructed, making
    # every job log self-contained and auditable.
    print(json.dumps({"theory_recall_before_trials": theory_recall}, sort_keys=True), flush=True)

    primary: List[Trial] = []
    audits: Dict[str, object] = {}
    for named_shape in shapes:
        for eta_u in eta_us:
            for eta_v in eta_vs:
                for seed in seeds:
                    trial, audit = run_trial(
                        named_shape=named_shape,
                        reference_shape=reference_shape,
                        eta_u=eta_u,
                        eta_v=eta_v,
                        rule="lmd",
                        seed=seed,
                        steps=args.steps,
                        input_dimension=args.input_dimension,
                        output_dimension=args.output_dimension,
                        map_seed=args.map_seed,
                        task=task,
                        device=device,
                        dtype=dtype,
                    )
                    primary.append(trial)
                    audit_key = f"{named_shape.label}:eta_u={eta_u}:eta_v={eta_v}"
                    audits.setdefault(audit_key, audit)
                    print(
                        json.dumps(
                            {
                                "trial": asdict(trial),
                                "progress": f"{len(primary)}/{len(shapes) * len(eta_us) * len(eta_vs) * len(seeds)}",
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    reference_pair = select_best_pair(primary, args.reference_label)
    controls: List[Trial] = []
    for named_shape in shapes:
        for rule in ("omit_l", "omit_m", "omit_d", "constant_raw"):
            for seed in seeds:
                trial, audit = run_trial(
                    named_shape=named_shape,
                    reference_shape=reference_shape,
                    eta_u=reference_pair[0],
                    eta_v=reference_pair[1],
                    rule=rule,
                    seed=seed,
                    steps=args.steps,
                    input_dimension=args.input_dimension,
                    output_dimension=args.output_dimension,
                    map_seed=args.map_seed,
                    task=task,
                    device=device,
                    dtype=dtype,
                )
                controls.append(trial)
                audits.setdefault(f"{named_shape.label}:{rule}", audit)

    summary = summarize(
        primary,
        controls,
        shapes=shapes,
        reference_label=args.reference_label,
        drift_tolerance_decades=args.drift_tolerance_decades,
        oracle_tolerance=args.oracle_tolerance,
        eta_us=eta_us,
        eta_vs=eta_vs,
    )
    payload: Dict[str, object] = {
        "experiment": "chizat_equation22_joint_L_M_D_per_group_lr_transfer",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "device": str(device),
        "dtype": args.dtype,
        "theory_recall_before_trials": theory_recall,
        "shape_contract": [
            {
                "label": named_shape.label,
                "L": named_shape.shape.depth,
                "M": named_shape.shape.hidden_width,
                "D": named_shape.shape.embedding_dimension,
                "LM_over_D": named_shape.shape.rho,
            }
            for named_shape in shapes
        ],
        "task_contract": {
            "kind": "fixed shape-independent nonlinear regression",
            "task_seed": args.task_seed,
            "map_seed": args.map_seed,
            "n_train": args.n_train,
            "n_validation": args.n_validation,
            "input_dimension": args.input_dimension,
            "output_dimension": args.output_dimension,
        },
        "protocol": {
            "eta_us": eta_us,
            "eta_vs": eta_vs,
            "seeds": seeds,
            "steps": args.steps,
            "full_batch": True,
            "gradient_clipping": False,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "schedule": "constant",
        },
        "optimizer_contract_audits": audits,
        "primary_trials": [asdict(trial) for trial in primary],
        "negative_control_trials": [asdict(trial) for trial in controls],
        "summary": summary,
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps({"output": str(args.output), "summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
