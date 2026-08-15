#!/usr/bin/env python3
"""Joint L/M/D hyperparameter-transfer test for an end-to-end Chizat network.

The existing fixed-D Chizat coordinate uses one raw GD rate proportional to
``L*M``.  Once the representation dimension ``D`` changes, the two particle
groups no longer have the same Euclidean scaling.  For the fixed scalar task
used here, with a mean-field ``1/D`` readout, the group-natural candidate is

    lr_embed  = D*eta
    lr_U      = L*M*eta/D
    lr_W      = L*M*D*eta
    lr_unembed = eta/D

at ``alpha=1``.  The boundary rules follow their function-space kernels:
the fan-in embed has an ``O(D^-1)`` scalar-output kernel and the mean-field
unembed has an ``O(D)`` kernel.  Both boundary maps are trained by default.
The harness keeps the input data and scalar target identical across shapes,
couples initial conditions by slicing common maximum-size Gaussian arrays,
and keeps stability-edge diagnostics separate from the fixed-eta verdict.

Shape arguments use ``label:L:M:D:dial``.  ``dial`` must increase along the
path and is the coordinate used only for convergence/progress diagnostics.
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
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_theorist.autoscaler.tuning import fixed_eta_transfer_diagnostics  # noqa: E402


RULES = (
    "group_natural_LMD",
    "group_incoherent_LM_sqrtD",
    "single_LM_reference_D",
    "omit_L",
    "omit_M",
    "wrong_global_LMD",
    "freeze_embed",
    "freeze_unembed",
    "wrong_constant_boundaries",
)


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
class Shape:
    label: str
    L: int
    M: int
    D: int
    dial: float


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
    train_groups: str
    raw_learning_rates: Dict[str, float]
    checkpoints: Dict[int, float]
    final_loss: float
    diverged: bool


def parse_shape(text: str) -> Shape:
    parts = text.split(":")
    if len(parts) != 5:
        raise ValueError("shape must use label:L:M:D:dial")
    label = parts[0].strip()
    if not label:
        raise ValueError("shape label must be non-empty")
    L, M, D = (int(value) for value in parts[1:4])
    dial = float(parts[4])
    if min(L, M, D) <= 0 or not math.isfinite(dial) or dial <= 0.0:
        raise ValueError("shape L, M, D, and dial must be positive")
    return Shape(label, L, M, D, dial)


def validate_shapes(shapes: Sequence[Shape]) -> Tuple[Shape, ...]:
    result = tuple(shapes)
    if len(result) < 4:
        raise ValueError("at least four shapes are required")
    if len({shape.label for shape in result}) != len(result):
        raise ValueError("shape labels must be unique")
    if any(right.dial <= left.dial for left, right in zip(result, result[1:])):
        raise ValueError("shape dials must be strictly increasing")
    return result


def group_learning_rates(
    rule: str,
    shape: Shape,
    eta: float,
    *,
    reference_D: int,
    primary_rule: str = "group_natural_LMD",
) -> Dict[str, float]:
    if rule not in RULES:
        raise ValueError(f"unknown rule: {rule}")
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta must be finite and positive")
    L, M, D = shape.L, shape.M, shape.D
    coherent_u = L * M * eta / D
    coherent_w = L * M * D * eta
    incoherent_u = L * M * eta / math.sqrt(D)
    incoherent_w = L * M * math.sqrt(D) * eta
    if primary_rule == "group_natural_LMD":
        correct_u, correct_w = coherent_u, coherent_w
    elif primary_rule == "group_incoherent_LM_sqrtD":
        correct_u, correct_w = incoherent_u, incoherent_w
    else:
        raise ValueError("primary_rule must be a group-wise rule")
    if rule == "group_natural_LMD":
        u, w = coherent_u, coherent_w
    elif rule == "group_incoherent_LM_sqrtD":
        u, w = incoherent_u, incoherent_w
    elif rule == "single_LM_reference_D":
        u = w = L * M * reference_D * eta
    elif rule == "omit_L":
        u, w = correct_u / L, correct_w / L
    elif rule == "omit_M":
        u, w = correct_u / M, correct_w / M
    elif rule == "wrong_global_LMD":
        u = w = coherent_w
    else:
        # Boundary controls isolate embed/unembed scaling while leaving the
        # selected block parameterization untouched.
        u, w = correct_u, correct_w
    # E has entries O(d0^-1/2), while the scalar mean-field readout has
    # entries O(D^-1).  At initialization its backpropagated signal makes the
    # embed kernel O(D^-1); the readout kernel is O(D).  These rates therefore
    # keep both boundary-induced function updates O(eta) as D changes.
    embed = D * eta
    unembed = eta / D
    if rule == "freeze_embed":
        embed = 0.0
    elif rule == "freeze_unembed":
        unembed = 0.0
    elif rule == "wrong_constant_boundaries":
        embed = unembed = eta
    return {
        "embed": float(embed),
        "U": float(u),
        "W": float(w),
        "unembed": float(unembed),
    }


def _normal(shape: Sequence[int], seed: int, offset: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(10_000_019 * seed + offset)
    return torch.randn(*shape, generator=generator, dtype=dtype)


class FixedTaskChizatNet:
    """Chizat particles with nested, trainable embed and unembed maps.

    The embed is fan-in initialized, ``E_ad ~ N(0, 1/d0)``.  The scalar
    unembed is mean-field initialized, ``R_d ~ N(0, 1/D^2)``.  It therefore
    vanishes at initialization but can move at an O(1) function-space rate
    under the declared ``eta/D`` SGD learning rate.
    """

    def __init__(
        self,
        shape: Shape,
        *,
        max_M: int,
        max_D: int,
        d0: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.shape = shape
        self.device = device
        self.embed = (
            _normal((d0, max_D), seed, 101, dtype)[:, : shape.D].clone()
            / math.sqrt(d0)
        ).to(device)
        # Mean-field readout: init output vanishes as D^-1/2, while coherent
        # trained changes can remain O(1).
        self.unembed = (
            _normal((max_D, 1), seed, 103, dtype)[: shape.D].clone() / shape.D
        ).to(device)
        self.U = []
        self.W = []
        for layer in range(shape.L):
            U = _normal((max_D, max_M), seed, 1_000 + 2 * layer, dtype)
            W = _normal((max_M, max_D), seed, 1_001 + 2 * layer, dtype)
            self.U.append((U[: shape.D, : shape.M].clone() / math.sqrt(shape.D)).to(device))
            self.W.append(W[: shape.M, : shape.D].clone().to(device))

    def parameter_groups(self) -> Dict[str, List[torch.Tensor]]:
        return {
            "embed": [self.embed],
            "U": list(self.U),
            "W": list(self.W),
            "unembed": [self.unembed],
        }

    def named_parameters_for(self, train_groups: str = "all") -> List[Tuple[str, torch.Tensor]]:
        groups = self.parameter_groups()
        aliases = {
            "all": ("embed", "U", "W", "unembed"),
            "blocks": ("U", "W"),
            "both": ("U", "W"),  # historical name for block-only runs
            "boundaries": ("embed", "unembed"),
            "embed": ("embed",),
            "unembed": ("unembed",),
            "U": ("U",),
            "W": ("W",),
        }
        if train_groups not in aliases:
            raise ValueError(
                "train_groups must be all, blocks, boundaries, embed, unembed, U, or W"
            )
        return [
            (role, parameter)
            for role in aliases[train_groups]
            for parameter in groups[role]
        ]

    def params(self) -> List[torch.Tensor]:
        return [parameter for _, parameter in self.named_parameters_for("all")]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        h = X @ self.embed
        coefficient = 1.0 / (self.shape.L * self.shape.M)
        for U, W in zip(self.U, self.W):
            h = h + coefficient * (torch.tanh(h @ U) @ W)
        return (h @ self.unembed).squeeze(-1)

    def loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.forward(X) - y).pow(2).mean()


def fixed_task_data(
    d0: int,
    P: int,
    seed: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X = _normal((P, d0), seed, 17, dtype)
    teacher = _normal((d0,), seed, 19, dtype) / math.sqrt(d0)
    y = torch.tanh(X @ teacher)
    y = y / y.std().clamp_min(torch.finfo(dtype).eps)
    return X.to(device), y.to(device)


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
    train_groups: str,
    reference_D: int,
    primary_rule: str,
    device: torch.device,
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
    named_parameters = net.named_parameters_for(train_groups)
    parameters = [parameter for _, parameter in named_parameters]
    for parameter in parameters:
        parameter.requires_grad_(True)
    rates = group_learning_rates(
        rule,
        shape,
        eta,
        reference_D=reference_D,
        primary_rule=primary_rule,
    )
    checkpoint_steps = {1, steps}
    checkpoint_steps.update(max(1, round(steps * fraction / 8.0)) for fraction in range(1, 9))
    with torch.no_grad():
        checkpoints: Dict[int, float] = {0: float(net.loss(X, y).cpu())}
    diverged = False
    for step in range(1, steps + 1):
        loss = net.loss(X, y)
        if not torch.isfinite(loss):
            diverged = True
            break
        gradients = torch.autograd.grad(loss, parameters)
        with torch.no_grad():
            for (role, parameter), gradient in zip(named_parameters, gradients):
                parameter -= rates[role] * gradient
        if step in checkpoint_steps:
            with torch.no_grad():
                value = float(net.loss(X, y).cpu())
            checkpoints[step] = value
            if not math.isfinite(value) or value > 1e12:
                diverged = True
                break
    final_loss = checkpoints.get(steps, float("inf"))
    diverged = diverged or not math.isfinite(final_loss)
    for parameter in parameters:
        parameter.requires_grad_(False)
    return Trial(
        shape.label,
        shape.L,
        shape.M,
        shape.D,
        shape.dial,
        seed,
        eta,
        rule,
        train_groups,
        rates,
        checkpoints,
        final_loss,
        diverged,
    )


def _selected(trials: Iterable[Trial], rule: str) -> List[Trial]:
    return [trial for trial in trials if trial.rule == rule]


def progress_report(
    trials: Sequence[Trial],
    shapes: Sequence[Shape],
    seeds: Sequence[int],
    *,
    rule: str,
    slope_tolerance: float = 0.3,
    minimum_progress: float = 1e-3,
) -> Dict[str, object]:
    selected = _selected(trials, rule)
    common_steps = sorted(set.intersection(*(set(trial.checkpoints) for trial in selected)))
    rows = []
    for step in (value for value in common_steps if value > 0):
        means = []
        by_shape_seed = []
        for shape in shapes:
            values = {}
            for seed in seeds:
                trial = next(
                    item for item in selected if item.label == shape.label and item.seed == seed
                )
                initial = trial.checkpoints[0]
                values[seed] = (initial - trial.checkpoints[step]) / max(abs(initial), 1e-300)
            by_shape_seed.append(values)
            means.append(sum(values.values()) / len(values))
        nontrivial = all(
            math.isfinite(value) and value >= minimum_progress for value in means
        )
        if nontrivial:
            x = [math.log(shape.dial) for shape in shapes]
            y = [math.log(value) for value in means]
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
                "mean_fractional_progress": means,
                "progress_by_shape_seed": by_shape_seed,
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
        "final_log_progress_vs_log_dial_slope": final["log_progress_vs_log_dial_slope"],
        "checkpoints": rows,
    }


def trajectory_report(
    trials: Sequence[Trial],
    shapes: Sequence[Shape],
    seeds: Sequence[int],
    *,
    rule: str,
    finite_size_exponent: float,
) -> Dict[str, object]:
    selected = _selected(trials, rule)
    common_steps = sorted(set.intersection(*(set(trial.checkpoints) for trial in selected)))
    rows = []
    for step in common_steps:
        losses = []
        for shape in shapes:
            losses.append(
                {
                    seed: next(
                        item.checkpoints[step]
                        for item in selected
                        if item.label == shape.label and item.seed == seed
                    )
                    for seed in seeds
                }
            )
        diagnostics = fixed_eta_transfer_diagnostics(
            [shape.dial for shape in shapes],
            losses,
            finite_size_exponent=finite_size_exponent,
        )
        rows.append({"step": step, **diagnostics.to_dict()})
    eligible = [row for row in rows if row["step"] > 1]
    return {
        "rule": rule,
        "finite_size_exponent": finite_size_exponent,
        "accepted": bool(eligible) and all(bool(row["accepted"]) for row in eligible),
        "checkpoints": rows,
    }


def build_report(
    trials: Sequence[Trial],
    shapes: Sequence[Shape],
    seeds: Sequence[int],
    *,
    eta: float,
    steps: int,
    d0: int,
    P: int,
    reference_D: int,
    finite_size_exponent: float,
    rules: Sequence[str],
    train_groups: str,
    primary_rule: str,
    minimum_progress: float = 1e-3,
) -> Dict[str, object]:
    trained_boundary_roles = {
        "all": ["embed", "unembed"],
        "boundaries": ["embed", "unembed"],
        "embed": ["embed"],
        "unembed": ["unembed"],
    }.get(train_groups, [])
    primary_trajectory = trajectory_report(
        trials,
        shapes,
        seeds,
        rule=primary_rule,
        finite_size_exponent=finite_size_exponent,
    )
    primary_progress = progress_report(
        trials,
        shapes,
        seeds,
        rule=primary_rule,
        minimum_progress=minimum_progress,
    )
    controls = {}
    for rule in rules:
        if rule == primary_rule:
            continue
        trajectory = trajectory_report(
            trials,
            shapes,
            seeds,
            rule=rule,
            finite_size_exponent=finite_size_exponent,
        )
        progress = progress_report(
            trials, shapes, seeds, rule=rule, minimum_progress=minimum_progress
        )
        controls[rule] = {
            "rejected": not (bool(trajectory["accepted"]) and bool(progress["accepted"])),
            "trajectory": trajectory,
            "learning_progress": progress,
        }
    return {
        "schema_version": 1,
        "experiment": "chizat_joint_L_M_D_fixed_eta_transfer",
        "parameterization": {
            "normalized_coordinate": "eta",
            "primary_rule": primary_rule,
            "raw_group_rates": (
                {"U": "L*M*eta/D", "W": "L*M*D*eta"}
                if primary_rule == "group_natural_LMD"
                else {"U": "L*M*eta/sqrt(D)", "W": "L*M*sqrt(D)*eta"}
            ),
            "boundary_parameterization": {
                "embed_initialization": "N(0, 1/d0)",
                "unembed_initialization": "N(0, 1/D^2)",
                "embed_raw_lr": "D*eta",
                "unembed_raw_lr": "eta/D",
                "trained_roles": trained_boundary_roles,
                "biases": "absent",
            },
            "fixed_task": (
                "nested trainable embed and mean-field scalar unembed"
                if trained_boundary_roles == ["embed", "unembed"]
                else "nested fixed embed and mean-field scalar unembed"
            ),
        },
        "shapes": [asdict(shape) for shape in shapes],
        "d0": d0,
        "P": P,
        "steps": steps,
        "seeds": list(seeds),
        "normalized_eta": eta,
        "train_groups": train_groups,
        "reference_D_for_single_rate_control": reference_D,
        "transfer_verdict": {
            "accepted": bool(primary_trajectory["accepted"])
            and bool(primary_progress["accepted"]),
            "requires": [
                "fixed_eta_trajectory_settling",
                "nontrivial_scale_invariant_progress",
            ],
        },
        "fixed_eta_trajectory": primary_trajectory,
        "learning_progress": primary_progress,
        "negative_controls": controls,
        "trials": [asdict(trial) for trial in trials],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", type=parse_shape, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--d0", type=int, default=8)
    parser.add_argument("--P", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--rules", nargs="+", choices=RULES, default=list(RULES))
    parser.add_argument(
        "--primary-rule",
        choices=("group_natural_LMD", "group_incoherent_LM_sqrtD"),
        default="group_natural_LMD",
    )
    parser.add_argument(
        "--train-groups",
        choices=("all", "blocks", "boundaries", "embed", "unembed", "U", "W", "both"),
        default="all",
    )
    parser.add_argument("--reference-D", type=int, default=32)
    parser.add_argument("--finite-size-exponent", type=float, default=-0.5)
    parser.add_argument("--minimum-progress", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shapes = validate_shapes(args.shapes)
    if args.primary_rule not in args.rules:
        raise ValueError("rules must include primary-rule")
    if args.steps <= 0 or args.d0 <= 0 or args.P <= 1 or args.reference_D <= 0:
        raise ValueError("steps, d0, P, and reference-D must be positive")
    if not math.isfinite(args.minimum_progress) or args.minimum_progress <= 0.0:
        raise ValueError("minimum-progress must be finite and positive")
    if sorted(set(args.seeds)) != args.seeds or len(args.seeds) < 2:
        raise ValueError("seeds must contain at least two unique increasing values")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started_at = _utc_now()
    max_M = max(shape.M for shape in shapes)
    max_D = max(shape.D for shape in shapes)
    trials = []
    for rule in args.rules:
        for shape in shapes:
            for seed in args.seeds:
                trials.append(
                    run_trial(
                        shape,
                        max_M=max_M,
                        max_D=max_D,
                        d0=args.d0,
                        P=args.P,
                        eta=args.eta,
                        steps=args.steps,
                        seed=seed,
                        rule=rule,
                        train_groups=args.train_groups,
                        reference_D=args.reference_D,
                        primary_rule=args.primary_rule,
                        device=device,
                    )
                )
    result = build_report(
        trials,
        shapes,
        args.seeds,
        eta=args.eta,
        steps=args.steps,
        d0=args.d0,
        P=args.P,
        reference_D=args.reference_D,
        finite_size_exponent=args.finite_size_exponent,
        rules=args.rules,
        train_groups=args.train_groups,
        primary_rule=args.primary_rule,
        minimum_progress=args.minimum_progress,
    )
    result.update(
        {
            "started_at": started_at,
            "completed_at": _utc_now(),
            "host": socket.gethostname(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "transfer_accepted": result["transfer_verdict"]["accepted"],
                "controls": {
                    key: value["rejected"]
                    for key, value in result["negative_controls"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
