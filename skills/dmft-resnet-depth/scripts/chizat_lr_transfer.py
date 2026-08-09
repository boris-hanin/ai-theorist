#!/usr/bin/env python3
"""Canonical fixed-eta LR-transfer and stability-edge experiment.

This harness deliberately emits two independent claims:

* ``fixed_eta_transfer`` compares common-seed trajectories at one normalized
  eta and checks whether finite-width differences settle.
* ``edge_of_stability`` reports the per-dial finite-horizon argmin and largest
  finite probe, but is marked diagnostic-only and cannot change transfer.

Example (from the repository root):

    python skills/dmft-resnet-depth/scripts/chizat_lr_transfer.py \
      --axis width --dials 64 128 256 512 --fixed-depth 8 \
      --etas 50 63.0957 79.4328 100 125.893 158.489 199.526 251.189 \
      --transfer-eta 79.4328 --steps 80 --device cuda --output result.json
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
from typing import Dict, List, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_theorist.autoscaler.tuning import (  # noqa: E402
    CHIZAT_MEAN_FIELD,
    fixed_eta_transfer_diagnostics,
    raw_learning_rate_from_normalized_eta,
)
from mean_ode import MeanODENet, odedata  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class Trial:
    dial: int
    depth: int
    width: int
    seed: int
    normalized_eta: float
    raw_learning_rate: float
    rule: str
    checkpoints: Dict[int, float]
    final_loss: float
    diverged: bool


def raw_rate(rule: str, eta: float, *, depth: int, width: int, alpha: float) -> float:
    if rule == "correct_LM":
        return raw_learning_rate_from_normalized_eta(
            CHIZAT_MEAN_FIELD,
            "sgd",
            eta,
            width=width,
            depth=depth,
            alpha=alpha,
        )
    if rule == "omit_M":
        return depth * eta / alpha ** 2
    if rule == "omit_L":
        return width * eta / alpha ** 2
    raise ValueError(f"unknown LR rule: {rule}")


def run_trial(
    *,
    dial: int,
    depth: int,
    width: int,
    D: int,
    P: int,
    alpha: float,
    eta: float,
    steps: int,
    seed: int,
    rule: str,
    device: torch.device,
) -> Trial:
    net = MeanODENet(D, width, depth, alpha=alpha, seed=seed, dtype=torch.float64)
    net.U = [parameter.to(device) for parameter in net.U]
    net.W = [parameter.to(device) for parameter in net.W]
    X, Y = odedata(D, P, seed=seed, dtype=torch.float64)
    X, Y = X.to(device), Y.to(device)
    parameters = net.params()
    for parameter in parameters:
        parameter.requires_grad_(True)
    lr = raw_rate(rule, eta, depth=depth, width=width, alpha=alpha)
    checkpoint_steps = {1, steps}
    checkpoint_steps.update(
        max(1, round(steps * fraction / 8.0)) for fraction in range(1, 9)
    )
    with torch.no_grad():
        initial_loss = float(net.loss(X, Y).detach().cpu())
    checkpoints: Dict[int, float] = {0: initial_loss}
    diverged = False
    for step in range(1, steps + 1):
        loss = net.loss(X, Y)
        if not torch.isfinite(loss):
            diverged = True
            break
        gradients = torch.autograd.grad(loss, parameters)
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients):
                parameter -= lr * gradient
        if step in checkpoint_steps:
            with torch.no_grad():
                value = float(net.loss(X, Y).detach().cpu())
            checkpoints[step] = value
            if not math.isfinite(value) or value > 1e12:
                diverged = True
                break
    final_loss = checkpoints.get(steps, float("inf"))
    if not math.isfinite(final_loss):
        diverged = True
    for parameter in parameters:
        parameter.requires_grad_(False)
    return Trial(
        dial, depth, width, seed, eta, lr, rule, checkpoints, final_loss, diverged
    )


def fixed_eta_report(
    trials: Sequence[Trial], dials: Sequence[int], seeds: Sequence[int], eta: float,
    *, rule: str = "correct_LM", finite_size_exponent: float = -0.5,
) -> Dict[str, object]:
    selected = [
        trial for trial in trials
        if trial.rule == rule and math.isclose(trial.normalized_eta, eta)
    ]
    steps = sorted(set.intersection(*(set(trial.checkpoints) for trial in selected)))
    checkpoints = []
    for step in steps:
        rows = [
            {
                seed: next(
                    trial.checkpoints[step]
                    for trial in selected
                    if trial.dial == dial and trial.seed == seed
                )
                for seed in seeds
            }
            for dial in dials
        ]
        diagnostic = fixed_eta_transfer_diagnostics(
            dials, rows, finite_size_exponent=finite_size_exponent
        )
        checkpoints.append({"step": step, **diagnostic.to_dict()})
    eligible = [row for row in checkpoints if row["step"] > 1]
    return {
        "purpose": "primary_transfer_verdict",
        "rule": rule,
        "finite_size_exponent": finite_size_exponent,
        "normalized_eta": eta,
        "accepted": bool(eligible) and all(bool(row["accepted"]) for row in eligible),
        "checkpoints": checkpoints,
    }


def learning_progress_report(
    trials: Sequence[Trial], dials: Sequence[int], seeds: Sequence[int], eta: float,
    *, rule: str = "correct_LM", slope_tolerance: float = 0.3,
) -> Dict[str, object]:
    """Check that the amount learned has a nonzero, M^0 large-width limit.

    Absolute losses can look width-invariant when a wrong rule makes every
    update vanish.  Progress from the common step-0 baseline detects that
    degenerate no-learning limit and powers the omitted-M negative control.
    """
    selected = [
        trial for trial in trials
        if trial.rule == rule and math.isclose(trial.normalized_eta, eta)
    ]
    steps = sorted(set.intersection(*(set(trial.checkpoints) for trial in selected)))
    rows = []
    for step in (value for value in steps if value > 0):
        means = []
        by_dial_seed = []
        for dial in dials:
            values = {}
            for seed in seeds:
                trial = next(
                    item for item in selected if item.dial == dial and item.seed == seed
                )
                baseline = trial.checkpoints[0]
                values[seed] = (baseline - trial.checkpoints[step]) / max(abs(baseline), 1e-300)
            by_dial_seed.append(values)
            means.append(sum(values.values()) / len(values))
        positive = all(math.isfinite(value) and value > 0.0 for value in means)
        if positive:
            x = [math.log(value) for value in dials]
            y = [math.log(value) for value in means]
            x_bar, y_bar = sum(x) / len(x), sum(y) / len(y)
            slope = sum((a - x_bar) * (b - y_bar) for a, b in zip(x, y)) / sum(
                (a - x_bar) ** 2 for a in x
            )
        else:
            slope = float("nan")
        rows.append({
            "step": step,
            "mean_fractional_progress": means,
            "progress_by_dial_seed": by_dial_seed,
            "log_progress_vs_log_dial_slope": slope,
            "scale_invariant": positive and abs(slope) <= slope_tolerance,
        })
    final = rows[-1]
    return {
        "rule": rule,
        "normalized_eta": eta,
        "slope_tolerance": slope_tolerance,
        "accepted": bool(final["scale_invariant"]),
        "final_log_progress_vs_log_dial_slope": final[
            "log_progress_vs_log_dial_slope"
        ],
        "checkpoints": rows,
    }


def edge_report(trials: Sequence[Trial], dials: Sequence[int], etas: Sequence[float]) -> Dict[str, object]:
    rows = []
    for dial in dials:
        medians = []
        for eta in etas:
            values = [
                trial.final_loss for trial in trials
                if trial.rule == "correct_LM" and trial.dial == dial
                and math.isclose(trial.normalized_eta, eta)
            ]
            finite = [value for value in values if math.isfinite(value)]
            medians.append(
                float(torch.tensor(finite, dtype=torch.float64).median())
                if len(finite) == len(values) and values else float("inf")
            )
        finite_indices = [index for index, value in enumerate(medians) if math.isfinite(value)]
        best = min(finite_indices, key=lambda index: medians[index])
        rows.append(
            {
                "dial": dial,
                "local_best_normalized_eta": etas[best],
                "largest_finite_probe_normalized_eta": etas[max(finite_indices)],
                "optimum_is_interior": 0 < best < len(etas) - 1,
                "median_final_losses": medians,
            }
        )
    return {
        "purpose": "diagnostic_only_not_a_transfer_gate",
        "normalized_eta_grid": list(etas),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=("width", "depth"), required=True)
    parser.add_argument("--dials", type=int, nargs="+", required=True)
    parser.add_argument("--fixed-depth", type=int, default=8)
    parser.add_argument("--fixed-width", type=int, default=256)
    parser.add_argument("--etas", type=float, nargs="+", required=True)
    parser.add_argument("--transfer-eta", type=float, required=True)
    parser.add_argument("--D", type=int, default=32)
    parser.add_argument("--P", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--negative-control", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sorted(set(args.dials)) != args.dials or len(args.dials) < 3:
        raise ValueError("dials must contain at least three unique increasing values")
    etas = sorted(set(args.etas))
    if not any(math.isclose(args.transfer_eta, eta) for eta in etas):
        raise ValueError("transfer-eta must be present in the eta grid")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rules = ["correct_LM"]
    if args.negative_control:
        rules.append("omit_M" if args.axis == "width" else "omit_L")
    started_at = _utc_now()
    trials: List[Trial] = []
    for rule in rules:
        rule_etas = etas if rule == "correct_LM" else [args.transfer_eta]
        for dial in args.dials:
            depth, width = (
                (args.fixed_depth, dial) if args.axis == "width"
                else (dial, args.fixed_width)
            )
            for eta in rule_etas:
                for seed in args.seeds:
                    trials.append(
                        run_trial(
                            dial=dial, depth=depth, width=width, D=args.D, P=args.P,
                            alpha=args.alpha, eta=eta, steps=args.steps, seed=seed,
                            rule=rule, device=device,
                        )
                    )
    fixed_transfer = fixed_eta_report(
        trials,
        args.dials,
        args.seeds,
        args.transfer_eta,
        finite_size_exponent=-0.5 if args.axis == "width" else -1.0,
    )
    correct_progress = learning_progress_report(
        trials, args.dials, args.seeds, args.transfer_eta
    )
    result = {
        "schema_version": 1,
        "experiment": "chizat_fixed_eta_transfer_and_stability_edge",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "host": socket.gethostname(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "axis": args.axis,
        "dials": args.dials,
        "D": args.D,
        "P": args.P,
        "alpha": args.alpha,
        "steps": args.steps,
        "seeds": args.seeds,
        "raw_lr_rule": "L*M*normalized_eta/alpha^2",
        "transfer_verdict": {
            "accepted": bool(fixed_transfer["accepted"])
            and bool(correct_progress["accepted"]),
            "requires": ["fixed_eta_trajectory_settling", "nonzero_M0_learning_progress"],
        },
        "fixed_eta_transfer": fixed_transfer,
        "learning_progress": correct_progress,
        "edge_of_stability": edge_report(trials, args.dials, etas),
        "negative_control": None,
        "trials": [asdict(trial) for trial in trials if trial.rule == "correct_LM"],
    }
    if len(rules) > 1:
        control_report = fixed_eta_report(
            trials,
            args.dials,
            args.seeds,
            args.transfer_eta,
            rule=rules[-1],
            finite_size_exponent=-0.5 if args.axis == "width" else -1.0,
        )
        control_progress = learning_progress_report(
            trials, args.dials, args.seeds, args.transfer_eta, rule=rules[-1]
        )
        result["negative_control"] = {
            "rule": rules[-1],
            "rejected": not bool(control_progress["accepted"]),
            "fixed_eta_diagnostics": control_report,
            "learning_progress": control_progress,
            "trials": [asdict(trial) for trial in trials if trial.rule != "correct_LM"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "transfer_accepted": result["transfer_verdict"]["accepted"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
