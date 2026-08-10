#!/usr/bin/env python3
"""Optimizer-specific muP width-transfer experiment for a residual MLP.

The experiment deliberately holds depth, data, batch size, and update horizon
fixed.  Width is the only scaling variable licensed by Tensor Programs V for
this harness.  Both the model parameterization (including the MuReadout
multiplier) and the optimizer groups change with width.
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
from typing import Dict, Iterable, List, Literal, Mapping, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_theorist.autoscaler.lr_contract import (  # noqa: E402
    LearningRateTheory,
    audit_optimizer_groups,
    raw_group_rates,
    theory_group,
)


OptimizerName = Literal["adam", "sgd"]
RuleName = Literal["mup", "global_lr_control"]


def theory_for(optimizer: OptimizerName) -> LearningRateTheory:
    return LearningRateTheory(
        contract_id=f"tensor-programs-v-mup-mlp-{optimizer}-v1",
        architecture="fixed-depth pre-LayerNorm residual MLP with a muP readout",
        optimizer=optimizer,
        source_title="Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer",
        source_url="https://arxiv.org/abs/2203.03466",
        source_version="arXiv:2203.03466v2 and microsoft/mup main optim.py",
        base_coordinate="eta tuned at reference width; raw group rates are optimizer-specific",
        applicability=(
            "width transfer only; fixed depth, dataset, batch size, horizon, optimizer "
            "hyperparameters, and residual multiplier"
        ),
    )


class MuPResidualBlock(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, width)
        self.down = nn.Linear(width, width)
        self.branch_scale = 1.0 / depth

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.branch_scale * self.down(F.gelu(self.up(self.norm(hidden))))


class MuPResidualMLP(nn.Module):
    """muP MLP with the exact 1/width MuReadout conversion."""

    def __init__(
        self,
        *,
        input_dimension: int,
        output_dimension: int,
        width: int,
        reference_width: int,
        depth: int,
    ) -> None:
        super().__init__()
        if min(input_dimension, output_dimension, width, reference_width, depth) <= 0:
            raise ValueError("all model dimensions must be positive")
        self.width = width
        self.reference_width = reference_width
        self.width_multiplier = width / reference_width
        self.depth = depth
        self.embed = nn.Linear(input_dimension, width)
        self.blocks = nn.ModuleList(MuPResidualBlock(width, depth) for _ in range(depth))
        self.final_norm = nn.LayerNorm(width)
        self.readout = nn.Linear(width, output_dimension)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0 / math.sqrt(self.embed.in_features))
        nn.init.zeros_(self.embed.bias)
        for block in self.blocks:
            nn.init.normal_(block.up.weight, mean=0.0, std=1.0 / math.sqrt(self.width))
            nn.init.zeros_(block.up.bias)
            nn.init.normal_(block.down.weight, mean=0.0, std=0.1 / math.sqrt(self.width))
            nn.init.zeros_(block.down.bias)
        # MuReadout converts the usual base-width initialization to constant
        # parameter variance, then divides its input by the width multiplier.
        # Zero initialization is the official coord-check recommendation and
        # removes the transient O(width^-1/2) output without changing the rules.
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_features(inputs)
        return self.readout(hidden / self.width_multiplier)

    def semantic_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        hidden_matrices: List[nn.Parameter] = []
        width_vectors: List[nn.Parameter] = [self.embed.bias]
        for block in self.blocks:
            hidden_matrices.extend([block.up.weight, block.down.weight])
            width_vectors.extend(
                [block.norm.weight, block.norm.bias, block.up.bias, block.down.bias]
            )
        width_vectors.extend([self.final_norm.weight, self.final_norm.bias])
        return {
            "mup_input_matrix": [self.embed.weight],
            "mup_hidden_matrices": hidden_matrices,
            "mup_width_vectors": width_vectors,
            "mup_output_weight": [self.readout.weight],
            "mup_output_bias": [self.readout.bias],
        }

    def optimizer_parameter_groups(
        self,
        eta: float,
        *,
        optimizer: OptimizerName,
        rule: RuleName = "mup",
    ) -> List[Dict[str, object]]:
        if not math.isfinite(eta) or eta <= 0.0:
            raise ValueError("eta must be finite and positive")
        if optimizer not in {"adam", "sgd"}:
            raise ValueError("optimizer must be adam or sgd")
        if rule not in {"mup", "global_lr_control"}:
            raise ValueError("unknown LR rule")
        multiplier = self.width_multiplier
        if optimizer == "adam":
            # MuAdam: two-infinite-dimension matrices use eta/m; tensors with
            # zero or one infinite dimension retain eta.
            predicted = {
                "mup_input_matrix": eta,
                "mup_hidden_matrices": eta / multiplier,
                "mup_width_vectors": eta,
                "mup_output_weight": eta,
                "mup_output_bias": eta,
            }
            formulas = {
                "mup_input_matrix": "eta",
                "mup_hidden_matrices": "eta * (width/reference_width)^(-1)",
                "mup_width_vectors": "eta",
                "mup_output_weight": "eta (MuReadout forward multiplier carries width scaling)",
                "mup_output_bias": "eta",
            }
        else:
            # MuSGD: one-infinite-dimension tensors use eta*m.  Square hidden
            # matrices have fan-in/fan-out multiplier ratio one and retain eta.
            predicted = {
                "mup_input_matrix": eta * multiplier,
                "mup_hidden_matrices": eta,
                "mup_width_vectors": eta * multiplier,
                "mup_output_weight": eta * multiplier,
                "mup_output_bias": eta,
            }
            formulas = {
                "mup_input_matrix": "eta * (width/reference_width)",
                "mup_hidden_matrices": "eta / (fan_in_multiplier/fan_out_multiplier) = eta",
                "mup_width_vectors": "eta * (width/reference_width)",
                "mup_output_weight": "eta * (width/reference_width)",
                "mup_output_bias": "eta",
            }
        if rule == "global_lr_control":
            predicted = {name: eta for name in predicted}
            formulas = {
                name: "eta (negative control: all optimizer-specific muP factors omitted)"
                for name in formulas
            }
        theory = theory_for(optimizer)
        semantic = self.semantic_parameter_groups()
        return [
            theory_group(
                name=name,
                params=semantic[name],
                lr=predicted[name],
                lr_formula=formulas[name],
                theory=theory,
                scale_factors={"width_ratio": multiplier, "depth_ratio": 1.0},
            )
            for name in semantic
        ]


@dataclass(frozen=True)
class Trial:
    optimizer: str
    rule: str
    width: int
    reference_width: int
    depth: int
    seed: int
    eta: float
    raw_learning_rates: Dict[str, float]
    initial_validation_loss: float
    final_validation_loss: float
    fractional_progress: float
    final_feature_rms: float
    diverged: bool


def teacher_data(
    *,
    input_dimension: int,
    n_train: int,
    n_validation: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    teacher_width = 128
    first = torch.randn(input_dimension, teacher_width, generator=generator) / math.sqrt(
        input_dimension
    )
    second = torch.randn(teacher_width, teacher_width, generator=generator) / math.sqrt(
        teacher_width
    )
    readout = torch.randn(teacher_width, 1, generator=generator) / math.sqrt(teacher_width)
    inputs = torch.randn(n_train + n_validation, input_dimension, generator=generator)
    hidden = torch.sin(inputs @ first) + 0.2 * torch.tanh(inputs @ first)
    hidden = torch.sin(hidden @ second) + 0.1 * hidden
    targets = hidden @ readout
    targets = (targets - targets.mean()) / targets.std().clamp_min(1e-6)
    return tuple(
        value.to(device)
        for value in (
            inputs[:n_train],
            targets[:n_train],
            inputs[n_train:],
            targets[n_train:],
        )
    )  # type: ignore[return-value]


@torch.no_grad()
def validation_metrics(
    model: MuPResidualMLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[float, float]:
    model.eval()
    predictions = model(inputs)
    loss = F.mse_loss(predictions, targets)
    features = model.forward_features(inputs)
    return float(loss.cpu()), float(features.float().square().mean().sqrt().cpu())


def run_trial(
    *,
    optimizer_name: OptimizerName,
    rule: RuleName,
    width: int,
    reference_width: int,
    depth: int,
    eta: float,
    steps: int,
    batch_size: int,
    seed: int,
    input_dimension: int,
    n_train: int,
    n_validation: int,
    dataset_seed: int,
    device: torch.device,
) -> Tuple[Trial, Dict[str, object]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = MuPResidualMLP(
        input_dimension=input_dimension,
        output_dimension=1,
        width=width,
        reference_width=reference_width,
        depth=depth,
    ).to(device)
    groups = model.optimizer_parameter_groups(eta, optimizer=optimizer_name, rule=rule)
    contract_audit = audit_optimizer_groups(model, groups, theory_for(optimizer_name))
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            groups,
            lr=eta,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )
        contract_audit["optimizer_options"] = {
            "name": "adam",
            "betas": [0.9, 0.95],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "schedule": "constant",
            "gradient_clipping": False,
        }
    else:
        optimizer = torch.optim.SGD(groups, lr=eta, momentum=0.0, weight_decay=0.0)
        contract_audit["optimizer_options"] = {
            "name": "sgd",
            "momentum": 0.0,
            "weight_decay": 0.0,
            "schedule": "constant",
            "gradient_clipping": False,
        }
    x_train, y_train, x_validation, y_validation = teacher_data(
        input_dimension=input_dimension,
        n_train=n_train,
        n_validation=n_validation,
        seed=dataset_seed,
        device=device,
    )
    initial_loss, _ = validation_metrics(model, x_validation, y_validation)
    generator = torch.Generator(device="cpu").manual_seed(100_003 + seed)
    diverged = False
    for _ in range(steps):
        indices = torch.randint(0, n_train, (batch_size,), generator=generator).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x_train[indices]), y_train[indices])
        if not torch.isfinite(loss):
            diverged = True
            break
        loss.backward()
        optimizer.step()
    final_loss, feature_rms = validation_metrics(model, x_validation, y_validation)
    diverged = diverged or not math.isfinite(final_loss) or final_loss > 1e8
    progress = (initial_loss - final_loss) / max(abs(initial_loss), 1e-12)
    return (
        Trial(
            optimizer_name,
            rule,
            width,
            reference_width,
            depth,
            seed,
            eta,
            raw_group_rates(groups),
            initial_loss,
            final_loss,
            progress,
            feature_rms,
            diverged,
        ),
        contract_audit,
    )


def _log_slope(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if not all(math.isfinite(value) and value > 0.0 for value in y_values):
        return float("nan")
    x = [math.log(value) for value in x_values]
    y = [math.log(value) for value in y_values]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    return sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / denominator


def analyze(
    trials: Sequence[Trial],
    *,
    widths: Sequence[int],
    etas: Sequence[float],
    seeds: Sequence[int],
    reference_width: int,
    rule: RuleName,
) -> Dict[str, object]:
    selected = [trial for trial in trials if trial.rule == rule]
    mean_losses: Dict[Tuple[int, float], float] = {}
    for width in widths:
        for eta in etas:
            rows = [
                trial.final_validation_loss
                for trial in selected
                if trial.width == width and math.isclose(trial.eta, eta)
            ]
            if len(rows) != len(seeds):
                raise ValueError("incomplete width/eta/seed factorial")
            mean_losses[(width, eta)] = sum(rows) / len(rows)
    best_eta_by_width = {
        width: min(etas, key=lambda eta: mean_losses[(width, eta)]) for width in widths
    }
    reference_eta = best_eta_by_width[reference_width]
    reference_index = list(etas).index(reference_eta)
    fixed_rows = [
        [
            trial
            for trial in selected
            if trial.width == width and math.isclose(trial.eta, reference_eta)
        ]
        for width in widths
    ]
    progress = [sum(row.fractional_progress for row in rows) / len(rows) for rows in fixed_rows]
    feature_rms = [sum(row.final_feature_rms for row in rows) / len(rows) for rows in fixed_rows]
    best_offsets = [math.log10(best_eta_by_width[width] / reference_eta) for width in widths]
    return {
        "rule": rule,
        "reference_width": reference_width,
        "reference_eta": reference_eta,
        "reference_optimum_is_interior": 0 < reference_index < len(etas) - 1,
        "best_eta_by_width": {str(width): best_eta_by_width[width] for width in widths},
        "best_eta_offset_decades": {str(width): value for width, value in zip(widths, best_offsets)},
        "maximum_absolute_best_eta_offset_decades": max(abs(value) for value in best_offsets),
        "fixed_reference_eta_mean_fractional_progress": progress,
        "fixed_reference_eta_log_progress_slope": _log_slope(widths, progress),
        "fixed_reference_eta_final_feature_rms": feature_rms,
        "fixed_reference_eta_log_feature_rms_slope": _log_slope(widths, feature_rms),
        "all_fixed_eta_trials_finite": all(not row.diverged for rows in fixed_rows for row in rows),
        "accepted": (
            0 < reference_index < len(etas) - 1
            and all(not row.diverged for rows in fixed_rows for row in rows)
            and all(value >= 1e-3 for value in progress)
            and abs(_log_slope(widths, progress)) <= 0.30
            and max(abs(value) for value in best_offsets) <= 0.35
        ),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), required=True)
    parser.add_argument("--widths", type=int, nargs="+", required=True)
    parser.add_argument("--reference-width", type=int, required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--etas", type=float, nargs="+", required=True)
    parser.add_argument("--rules", choices=("mup", "global_lr_control"), nargs="+", default=["mup", "global_lr_control"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47, 71])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-dimension", type=int, default=32)
    parser.add_argument("--n-train", type=int, default=8192)
    parser.add_argument("--n-validation", type=int, default=2048)
    parser.add_argument("--dataset-seed", type=int, default=1729)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    widths = tuple(args.widths)
    etas = tuple(args.etas)
    seeds = tuple(args.seeds)
    if len(widths) < 4 or len(set(widths)) != len(widths) or tuple(sorted(widths)) != widths:
        raise ValueError("widths must contain at least four unique increasing values")
    if args.reference_width not in widths:
        raise ValueError("reference width must be in the width ladder")
    if len(etas) < 5 or any(value <= 0.0 for value in etas):
        raise ValueError("etas must contain at least five positive values")
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three unique paired seeds are required")
    if args.batch_size > args.n_train:
        raise ValueError("batch size cannot exceed n_train")
    optimizer_name: OptimizerName = args.optimizer
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    theory = theory_for(optimizer_name)
    print(
        json.dumps(
            {
                "phase": "theory-recall",
                "theory": theory.to_dict(),
                "scope_guard": "width transfer only; depth is fixed for the entire campaign",
                "forward_parameterization": "MuReadout divides hidden features by width/reference_width",
                "optimizer_groups": "MuAdam and MuSGD use distinct tensor-type scaling rules",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    trials: List[Trial] = []
    audits: Dict[str, object] = {}
    for rule in args.rules:
        for width in widths:
            for eta in etas:
                for seed in seeds:
                    print(
                        json.dumps(
                            {"phase": "trial", "optimizer": optimizer_name, "rule": rule, "width": width, "eta": eta, "seed": seed},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    trial, audit = run_trial(
                        optimizer_name=optimizer_name,
                        rule=rule,
                        width=width,
                        reference_width=args.reference_width,
                        depth=args.depth,
                        eta=eta,
                        steps=args.steps,
                        batch_size=args.batch_size,
                        seed=seed,
                        input_dimension=args.input_dimension,
                        n_train=args.n_train,
                        n_validation=args.n_validation,
                        dataset_seed=args.dataset_seed,
                        device=device,
                    )
                    trials.append(trial)
                    audits.setdefault(f"{rule}:width={width}:eta={eta}", audit)
    analyses = {
        rule: analyze(
            trials,
            widths=widths,
            etas=etas,
            seeds=seeds,
            reference_width=args.reference_width,
            rule=rule,
        )
        for rule in args.rules
    }
    primary = analyses["mup"]
    control = analyses.get("global_lr_control")
    report = {
        "schema_version": 1,
        "experiment": "optimizer_specific_mup_residual_mlp_width_transfer",
        "status": "completed",
        "host": socket.gethostname(),
        "completed_at": utc_now(),
        "theory_recalled_before_trials": theory.to_dict(),
        "architecture_contract": {
            "scope": "fixed-depth residual MLP; no depth-transfer claim",
            "block": "pre-LayerNorm GELU two-linear residual branch with fixed 1/depth multiplier",
            "readout": "MuReadout input divided by width/reference_width",
            "initialization": "fan-in input/hidden matrices, zero MuReadout as official coord-check recommendation",
            "optimizer": (
                "MuAdam beta=(0.9,0.95), epsilon=1e-8"
                if optimizer_name == "adam"
                else "MuSGD momentum=0"
            ),
            "schedule": "constant",
            "gradient_clipping": "none",
        },
        "fixed_variables": {
            "depth": args.depth,
            "n_train": args.n_train,
            "n_validation": args.n_validation,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "dataset_seed": args.dataset_seed,
        },
        "widths": list(widths),
        "etas": list(etas),
        "seeds": list(seeds),
        "optimizer_group_contract_audits": audits,
        "analyses": analyses,
        "verdict": {
            "primary_accepted": bool(primary["accepted"]),
            "negative_control_rejected": (
                None if control is None else not bool(control["accepted"])
            ),
            "certified": bool(primary["accepted"])
            and control is not None
            and not bool(control["accepted"]),
        },
        "trials": [asdict(trial) for trial in trials],
    }
    atomic_write_json(args.output, report)
    print(json.dumps({"output": str(args.output), **report["verdict"]}, sort_keys=True))


if __name__ == "__main__":
    main()
