#!/usr/bin/env python3
"""Tune once and test fixed-eta Muon transfer in the end-to-end Chizat setup.

The primary rule uses RMS-matched Muon for residual-particle matrices and
auxiliary Adam for the trained embed/unembed boundaries:

    embed (Adam): eta
    U     (Muon): eta
    W     (Muon): sqrt(D) * eta
    unembed (Adam): eta / D

All rates above are raw optimizer learning rates.  Muon's internal
``match_rms_adamw`` shape adjustment is recorded separately.  The eta grid is
evaluated only at one declared reference shape; the one-SEM selection is then
held fixed over the L/M/D path.  Per-shape eta sweeps, when requested, are
diagnostic-only and cannot change the transfer verdict.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket
import sys
from statistics import mean, stdev
from typing import Dict, List, Mapping, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from chizat_lmd_transfer import (  # noqa: E402
    FixedTaskChizatNet,
    Shape,
    fixed_task_data,
    json_safe,
    parse_shape,
    progress_report,
    trajectory_report,
    validate_shapes,
)
from chizat_muon import (  # noqa: E402
    AuxAdamConfig,
    ChizatMuonAdam,
    MuonConfig,
    TRANSFER_RULES,
    chizat_muon_learning_rates,
    learning_rate_adjustment,
)


PRIMARY_RULE = "group_rms_D"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Trial:
    label: str
    L: int
    M: int
    D: int
    dial: float
    seed: int
    normalized_eta: float
    rule: str
    raw_learning_rates: Dict[str, float]
    effective_muon_multipliers: Dict[str, float]
    checkpoints: Dict[int, float]
    first_step_update_rms: Dict[str, float]
    first_step_update_to_weight_rms: Dict[str, float]
    final_loss: float
    diverged: bool


def _rms(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().square().mean().sqrt().cpu())


def _group_rms(
    before: Mapping[str, Sequence[torch.Tensor]],
    after: Mapping[str, Sequence[torch.Tensor]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    update_rms: Dict[str, float] = {}
    ratios: Dict[str, float] = {}
    for role in ("embed", "U", "W", "unembed"):
        updates = torch.cat(
            [
                (right.detach() - left).float().reshape(-1)
                for left, right in zip(before[role], after[role])
            ]
        )
        weights = torch.cat([value.detach().float().reshape(-1) for value in before[role]])
        update = _rms(updates)
        weight = _rms(weights)
        update_rms[role] = update
        ratios[role] = update / max(weight, 1e-30)
    return update_rms, ratios


def run_trial(
    shape: Shape,
    *,
    max_M: int,
    max_D: int,
    d0: int,
    P: int,
    eta: float,
    steps: int,
    seed: int,
    rule: str,
    device: torch.device,
    muon_config: MuonConfig,
    aux_config: AuxAdamConfig,
) -> Trial:
    net = FixedTaskChizatNet(
        shape,
        max_M=max_M,
        max_D=max_D,
        d0=d0,
        seed=seed,
        device=device,
    )
    X, y = fixed_task_data(d0, P, seed, device=device)
    for parameter in net.params():
        parameter.requires_grad_(True)
    rates = chizat_muon_learning_rates(
        rule, L=shape.L, M=shape.M, D=shape.D, eta=eta
    )
    optimizer = ChizatMuonAdam(
        net, rates, muon_config=muon_config, aux_config=aux_config
    )
    checkpoint_steps = {1, steps}
    checkpoint_steps.update(max(1, round(steps * fraction / 8.0)) for fraction in range(1, 9))
    with torch.no_grad():
        checkpoints: Dict[int, float] = {0: float(net.loss(X, y).cpu())}
    first_update: Dict[str, float] = {}
    first_ratio: Dict[str, float] = {}
    diverged = False
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = net.loss(X, y)
        if not torch.isfinite(loss):
            diverged = True
            break
        before = None
        if step == 1:
            before = {
                role: [parameter.detach().clone() for parameter in parameters]
                for role, parameters in net.parameter_groups().items()
            }
        loss.backward()
        optimizer.step()
        if before is not None:
            first_update, first_ratio = _group_rms(before, net.parameter_groups())
        if step in checkpoint_steps:
            with torch.no_grad():
                value = float(net.loss(X, y).cpu())
            checkpoints[step] = value
            if not math.isfinite(value) or value > 1e12:
                diverged = True
                break
    for checkpoint in checkpoint_steps:
        checkpoints.setdefault(checkpoint, float("inf"))
    final_loss = checkpoints[steps]
    diverged = diverged or not math.isfinite(final_loss)
    for parameter in net.params():
        parameter.requires_grad_(False)
    return Trial(
        label=shape.label,
        L=shape.L,
        M=shape.M,
        D=shape.D,
        dial=shape.dial,
        seed=seed,
        normalized_eta=eta,
        rule=rule,
        raw_learning_rates=rates,
        effective_muon_multipliers={
            "U": rates["U"]
            * learning_rate_adjustment(muon_config.adjustment, (shape.D, shape.M)),
            "W": rates["W"]
            * learning_rate_adjustment(muon_config.adjustment, (shape.M, shape.D)),
        },
        checkpoints=dict(sorted(checkpoints.items())),
        first_step_update_rms=first_update,
        first_step_update_to_weight_rms=first_ratio,
        final_loss=final_loss,
        diverged=diverged,
    )


def tuning_report(
    trials: Sequence[Trial],
    *,
    reference_shape: Shape,
    etas: Sequence[float],
    seeds: Sequence[int],
) -> Dict[str, object]:
    rows = []
    for eta in etas:
        selected = [
            trial
            for trial in trials
            if trial.rule == PRIMARY_RULE
            and trial.label == reference_shape.label
            and math.isclose(trial.normalized_eta, eta)
        ]
        if sorted(trial.seed for trial in selected) != list(seeds):
            raise ValueError(f"incomplete reference tuning row for eta={eta}")
        losses = [trial.final_loss for trial in selected]
        finite = all(math.isfinite(value) for value in losses)
        average = mean(losses) if finite else float("inf")
        sem = stdev(losses) / math.sqrt(len(losses)) if finite and len(losses) > 1 else (
            0.0 if finite else float("inf")
        )
        rows.append(
            {
                "normalized_eta": eta,
                "mean_final_loss": average,
                "sem_final_loss": sem,
                "losses_by_seed": {
                    trial.seed: trial.final_loss for trial in selected
                },
                "diverged_seeds": [trial.seed for trial in selected if trial.diverged],
            }
        )
    finite_indices = [
        index for index, row in enumerate(rows) if math.isfinite(float(row["mean_final_loss"]))
    ]
    if not finite_indices:
        raise RuntimeError("every reference-shape Muon eta diverged")
    numerical_index = min(finite_indices, key=lambda index: float(rows[index]["mean_final_loss"]))
    threshold = float(rows[numerical_index]["mean_final_loss"]) + float(
        rows[numerical_index]["sem_final_loss"]
    )
    tied = [
        index
        for index in finite_indices
        if float(rows[index]["mean_final_loss"]) <= threshold
    ]
    selected_index = min(tied, key=lambda index: float(rows[index]["normalized_eta"]))
    return {
        "reference_shape": reference_shape.label,
        "selection_rule": "lowest_eta_within_one_sem_of_numerical_minimum",
        "selected_normalized_eta": rows[selected_index]["normalized_eta"],
        "numerical_best_normalized_eta": rows[numerical_index]["normalized_eta"],
        "numerical_optimum_is_interior": 0 < numerical_index < len(rows) - 1,
        "flat_minimum": len(tied) > 1,
        "rows": rows,
    }


def update_scale_report(
    trials: Sequence[Trial], shapes: Sequence[Shape], seeds: Sequence[int], eta: float
) -> Dict[str, object]:
    rows = []
    for shape in shapes:
        selected = [
            trial
            for trial in trials
            if trial.rule == PRIMARY_RULE
            and trial.label == shape.label
            and math.isclose(trial.normalized_eta, eta)
        ]
        if sorted(trial.seed for trial in selected) != list(seeds):
            raise ValueError(f"incomplete update audit for {shape.label}")
        means = {
            role: mean(trial.first_step_update_rms[role] for trial in selected)
            for role in ("embed", "U", "W", "unembed")
        }
        expected_coordinates = {
            "embed": eta,
            "U": 0.2 * eta,
            "W": 0.2 * math.sqrt(shape.D) * eta,
            "unembed": eta / shape.D,
        }
        rows.append(
            {
                "shape": shape.label,
                "L": shape.L,
                "M": shape.M,
                "D": shape.D,
                "mean_first_step_update_rms": means,
                "normalized_update_rms": {
                    role: means[role] / expected_coordinates[role]
                    for role in means
                },
            }
        )
    normalized = [
        float(row["normalized_update_rms"][role])
        for row in rows
        for role in ("embed", "U", "W", "unembed")
    ]
    return {
        "purpose": "implementation_audit_not_transfer_verdict",
        "expected_raw_update_coordinates": {
            "embed": "eta",
            "U": "approximately 0.2*eta after RMS-matched Muon",
            "W": "approximately 0.2*sqrt(D)*eta after RMS-matched Muon",
            "unembed": "eta/D",
        },
        "finite": all(math.isfinite(value) for value in normalized),
        "rows": rows,
    }


def local_edge_report(
    trials: Sequence[Trial],
    shapes: Sequence[Shape],
    etas: Sequence[float],
) -> Dict[str, object]:
    rows = []
    for shape in shapes:
        losses = []
        for eta in etas:
            values = [
                trial.final_loss
                for trial in trials
                if trial.rule == PRIMARY_RULE
                and trial.label == shape.label
                and math.isclose(trial.normalized_eta, eta)
            ]
            losses.append(mean(values) if values and all(map(math.isfinite, values)) else float("inf"))
        finite = [index for index, value in enumerate(losses) if math.isfinite(value)]
        best = min(finite, key=lambda index: losses[index]) if finite else None
        rows.append(
            {
                "shape": shape.label,
                "mean_final_losses": losses,
                "local_best_normalized_eta": etas[best] if best is not None else None,
                "largest_finite_normalized_eta": etas[max(finite)] if finite else None,
                "optimum_is_interior": best is not None and 0 < best < len(etas) - 1,
            }
        )
    return {
        "purpose": "diagnostic_only_not_a_transfer_gate",
        "normalized_eta_grid": list(etas),
        "rows": rows,
    }


def paired_largest_shape_control(
    trials: Sequence[Trial],
    *,
    largest_shape: Shape,
    seeds: Sequence[int],
    eta: float,
    control_rule: str,
    relative_tolerance: float = 0.01,
) -> Dict[str, object]:
    """Reject a wrong rule that is resolvedly worse at the largest shape."""
    primary = {
        trial.seed: trial.final_loss
        for trial in trials
        if trial.rule == PRIMARY_RULE
        and trial.label == largest_shape.label
        and math.isclose(trial.normalized_eta, eta)
    }
    control = {
        trial.seed: trial.final_loss
        for trial in trials
        if trial.rule == control_rule
        and trial.label == largest_shape.label
        and math.isclose(trial.normalized_eta, eta)
    }
    if sorted(primary) != list(seeds) or sorted(control) != list(seeds):
        raise ValueError(f"incomplete paired largest-shape control for {control_rule}")
    if not all(math.isfinite(value) for value in primary.values()):
        raise ValueError("primary largest-shape trials must be finite")
    if not all(math.isfinite(value) for value in control.values()):
        return {
            "shape": largest_shape.label,
            "mean_control_minus_primary": float("inf"),
            "sem_control_minus_primary": float("inf"),
            "tolerance": float("inf"),
            "rejected": True,
            "reason": "control_nonfinite",
        }
    differences = [control[seed] - primary[seed] for seed in seeds]
    difference_mean = mean(differences)
    difference_sem = (
        stdev(differences) / math.sqrt(len(differences))
        if len(differences) > 1
        else 0.0
    )
    tolerance = max(
        2.0 * difference_sem,
        relative_tolerance * mean(primary.values()),
    )
    return {
        "shape": largest_shape.label,
        "mean_control_minus_primary": difference_mean,
        "sem_control_minus_primary": difference_sem,
        "tolerance": tolerance,
        "rejected": difference_mean > tolerance,
        "reason": "paired_largest_shape_final_loss",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", type=parse_shape, required=True)
    parser.add_argument("--etas", nargs="+", type=float, required=True)
    parser.add_argument("--max-expansion-rounds", type=int, default=2)
    parser.add_argument("--expansion-factor", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--d0", type=int, default=8)
    parser.add_argument("--P", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--reference-index", type=int, default=-1)
    parser.add_argument(
        "--rules", nargs="+", choices=TRANSFER_RULES, default=list(TRANSFER_RULES)
    )
    parser.add_argument(
        "--required-negative-controls",
        nargs="+",
        choices=TRANSFER_RULES,
        default=["wrong_W_D", "wrong_sgd_LMD", "wrong_constant_unembed"],
    )
    parser.add_argument("--finite-size-exponent", type=float, default=-0.5)
    parser.add_argument("--minimum-progress", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument(
        "--adjustment",
        choices=("match_rms_adamw", "original", "spectral_unclamped", "none"),
        default="match_rms_adamw",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--full-edge-diagnostic", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shapes = validate_shapes(args.shapes)
    etas = sorted(set(float(value) for value in args.etas))
    if len(etas) < 3 or any(not math.isfinite(value) or value <= 0.0 for value in etas):
        raise ValueError("etas must contain at least three unique positive finite values")
    if (
        isinstance(args.max_expansion_rounds, bool)
        or args.max_expansion_rounds < 0
        or not math.isfinite(args.expansion_factor)
        or args.expansion_factor <= 1.0
    ):
        raise ValueError("expansion rounds must be non-negative and factor must exceed one")
    if PRIMARY_RULE not in args.rules:
        raise ValueError("rules must include group_rms_D")
    if any(rule not in args.rules for rule in args.required_negative_controls):
        raise ValueError("every required negative control must be present in rules")
    if args.steps <= 0 or args.d0 <= 0 or args.P <= 1:
        raise ValueError("steps, d0, and P must be positive")
    if sorted(set(args.seeds)) != args.seeds or len(args.seeds) < 2:
        raise ValueError("seeds must contain at least two unique increasing values")
    reference_index = args.reference_index
    if reference_index == -1:
        reference_index = len(shapes) // 2
    if not 0 <= reference_index < len(shapes):
        raise ValueError("reference-index is outside the shape ladder")
    reference_shape = shapes[reference_index]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    muon_config = MuonConfig(
        momentum=args.momentum,
        nesterov=True,
        ns_steps=args.ns_steps,
        weight_decay=args.weight_decay,
        adjustment=args.adjustment,
    ).validate()
    aux_config = AuxAdamConfig(weight_decay=args.weight_decay).validate()
    max_M = max(shape.M for shape in shapes)
    max_D = max(shape.D for shape in shapes)
    cache: Dict[Tuple[str, str, float, int], Trial] = {}

    def run(shape: Shape, rule: str, eta: float, seed: int) -> Trial:
        key = (shape.label, rule, float(eta), int(seed))
        if key not in cache:
            cache[key] = run_trial(
                shape,
                max_M=max_M,
                max_D=max_D,
                d0=args.d0,
                P=args.P,
                eta=eta,
                steps=args.steps,
                seed=seed,
                rule=rule,
                device=device,
                muon_config=muon_config,
                aux_config=aux_config,
            )
        return cache[key]

    started_at = _utc_now()
    initial_etas = list(etas)
    expansion_rounds = 0
    while True:
        for eta in etas:
            for seed in args.seeds:
                run(reference_shape, PRIMARY_RULE, eta, seed)
        tuning = tuning_report(
            list(cache.values()),
            reference_shape=reference_shape,
            etas=etas,
            seeds=args.seeds,
        )
        if bool(tuning["numerical_optimum_is_interior"]):
            break
        if expansion_rounds >= args.max_expansion_rounds:
            break
        numerical_best = float(tuning["numerical_best_normalized_eta"])
        if math.isclose(numerical_best, etas[0]):
            etas = sorted({*etas, etas[0] / args.expansion_factor})
        elif math.isclose(numerical_best, etas[-1]):
            etas = sorted({*etas, etas[-1] * args.expansion_factor})
        else:
            break
        expansion_rounds += 1
    tuning["initial_normalized_eta_grid"] = initial_etas
    tuning["final_normalized_eta_grid"] = list(etas)
    tuning["expansion_rounds"] = expansion_rounds
    selected_eta = float(tuning["selected_normalized_eta"])
    for shape in shapes:
        for seed in args.seeds:
            run(shape, PRIMARY_RULE, selected_eta, seed)
    for rule in args.rules:
        if rule == PRIMARY_RULE:
            continue
        for shape in shapes:
            for seed in args.seeds:
                run(shape, rule, selected_eta, seed)
    if args.full_edge_diagnostic:
        for shape in shapes:
            for eta in etas:
                for seed in args.seeds:
                    run(shape, PRIMARY_RULE, eta, seed)

    selected_trials = [
        trial
        for trial in cache.values()
        if math.isclose(trial.normalized_eta, selected_eta)
    ]
    fixed_transfer = trajectory_report(
        selected_trials,
        shapes,
        args.seeds,
        rule=PRIMARY_RULE,
        finite_size_exponent=args.finite_size_exponent,
    )
    learning_progress = progress_report(
        selected_trials,
        shapes,
        args.seeds,
        rule=PRIMARY_RULE,
        minimum_progress=args.minimum_progress,
    )
    controls = {}
    for rule in args.rules:
        if rule == PRIMARY_RULE:
            continue
        trajectory = trajectory_report(
            selected_trials,
            shapes,
            args.seeds,
            rule=rule,
            finite_size_exponent=args.finite_size_exponent,
        )
        progress = progress_report(
            selected_trials,
            shapes,
            args.seeds,
            rule=rule,
            minimum_progress=args.minimum_progress,
        )
        paired_loss = paired_largest_shape_control(
            selected_trials,
            largest_shape=shapes[-1],
            seeds=args.seeds,
            eta=selected_eta,
            control_rule=rule,
        )
        controls[rule] = {
            "rejected": (
                not bool(trajectory["accepted"])
                or not bool(progress["accepted"])
                or bool(paired_loss["rejected"])
            ),
            "rejection_channels": {
                "trajectory": not bool(trajectory["accepted"]),
                "learning_progress": not bool(progress["accepted"]),
                "paired_largest_shape_loss": bool(paired_loss["rejected"]),
            },
            "trajectory": trajectory,
            "learning_progress": progress,
            "paired_largest_shape_loss": paired_loss,
        }
    controls_ok = all(
        controls[rule]["rejected"] for rule in args.required_negative_controls
    )
    accepted = (
        bool(tuning["numerical_optimum_is_interior"])
        and bool(fixed_transfer["accepted"])
        and bool(learning_progress["accepted"])
        and controls_ok
    )
    result = {
        "schema_version": 1,
        "experiment": "chizat_muon_joint_L_M_D_fixed_eta_transfer",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "host": socket.gethostname(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "optimizer_contract": {
            "block_optimizer": "Muon",
            "boundary_optimizer": "Adam",
            "routing": {
                "Muon": ["U", "W"],
                "Adam": ["embed", "unembed"],
            },
            "muon": muon_config.to_dict(),
            "auxiliary_adam": aux_config.to_dict(),
            "weight_decay_is_tuned": False,
        },
        "parameterization": {
            "normalized_coordinate": "eta",
            "primary_rule": PRIMARY_RULE,
            "raw_group_rates": {
                "embed": "eta",
                "U": "eta",
                "W": "sqrt(D)*eta",
                "unembed": "eta/D",
            },
            "boundary_initialization": {
                "embed": "N(0, 1/d0)",
                "unembed": "N(0, 1/D^2)",
                "biases": "absent",
            },
        },
        "shapes": [asdict(shape) for shape in shapes],
        "d0": args.d0,
        "P": args.P,
        "steps": args.steps,
        "seeds": args.seeds,
        "tuning": tuning,
        "transfer_verdict": {
            "accepted": accepted,
            "requires": [
                "interior_reference_eta_optimum",
                "fixed_eta_trajectory_settling",
                "nontrivial_scale_invariant_progress",
                "required_negative_controls_rejected",
            ],
            "required_negative_controls": args.required_negative_controls,
            "required_negative_controls_rejected": controls_ok,
        },
        "fixed_eta_trajectory": fixed_transfer,
        "learning_progress": learning_progress,
        "update_scale_audit": update_scale_report(
            selected_trials, shapes, args.seeds, selected_eta
        ),
        "negative_controls": controls,
        "edge_of_stability": (
            local_edge_report(list(cache.values()), shapes, etas)
            if args.full_edge_diagnostic
            else {
                "purpose": "diagnostic_only_not_a_transfer_gate",
                "status": "not_requested_except_reference_tuning_row",
            }
        ),
        "trials": [asdict(trial) for trial in cache.values()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_eta": selected_eta,
                "transfer_accepted": accepted,
                "controls": {
                    rule: payload["rejected"] for rule, payload in controls.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
