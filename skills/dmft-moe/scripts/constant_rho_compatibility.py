#!/usr/bin/env python3
"""Audit constant ``rho = L*M/D`` against the Jiang MoE parameterisation.

This is an algebra-first compatibility check, not a fitted scaling law. It
substitutes ``M = rho*D/L`` into every source-faithful initialization, Adam
learning-rate, and Adam-epsilon rule. An optional paired Monte Carlo check
measures the initial residual-stream variance in the reduced MoE model.

The convention under test is:

    raw expert-down std       = sqrt(D) / M
    residual branch factor    = 1 / L
    effective coefficient     = sqrt(D) / (L*M)

Putting ``sqrt(D)/(L*M)`` directly into the raw down matrix applies the depth
factor twice. ``double_depth`` is the preregistered negative control.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from statistics import fmean, stdev
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Shape:
    label: str
    L: int
    M: int
    D: int
    E: int
    A: int

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("shape label cannot be empty")
        if min(self.L, self.M, self.D, self.E, self.A) <= 0:
            raise ValueError("L, M, D, E, and A must all be positive")
        if self.A > self.E:
            raise ValueError("A cannot exceed E")

    @property
    def rho(self) -> float:
        return self.L * self.M / self.D

    @property
    def kappa(self) -> float:
        return self.A / self.E

    @property
    def alpha_ffn(self) -> float:
        return self.M / self.D

    @property
    def alpha_star(self) -> float:
        return self.D / (self.M * self.E * self.L)

    @property
    def stream_init_variance_proxy(self) -> float:
        # Constants from the gate distribution are intentionally omitted.
        return self.D / (self.L * self.A * self.M)


def parse_shape(value: str) -> Shape:
    parts = value.split(":")
    if len(parts) != 6:
        raise ValueError("shape must be label:L:M:D:E:A")
    return Shape(parts[0], *(int(item) for item in parts[1:]))


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1e-300)


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2 or any(x <= 0 for x in xs) or any(y <= 0 for y in ys):
        raise ValueError("positive x/y sequences of equal length are required")
    lx = [math.log(value) for value in xs]
    ly = [math.log(value) for value in ys]
    xbar, ybar = fmean(lx), fmean(ly)
    denominator = sum((value - xbar) ** 2 for value in lx)
    if denominator == 0:
        raise ValueError("x values must not all be equal")
    return sum((x - xbar) * (y - ybar) for x, y in zip(lx, ly)) / denominator


def _regime(alpha_ffn: float) -> str:
    if alpha_ffn >= 16.0:
        return "measured_clean_asymptotic"
    if alpha_ffn >= 1.8:
        return "coherent_term_dominant_but_crossover_visible"
    return "crossover_or_below; do_not_call_theory_certified"


def derive(reference: Shape, shapes: Iterable[Shape]) -> dict[str, object]:
    rows = tuple(shapes)
    if len(rows) < 2:
        raise ValueError("at least two shapes are required")
    if reference not in rows:
        rows = (reference, *rows)
    if len({shape.label for shape in rows}) != len(rows):
        raise ValueError("shape labels must be unique")

    output_rows: list[dict[str, object]] = []
    for shape in rows:
        rL = shape.L / reference.L
        rM = shape.M / reference.M
        rD = shape.D / reference.D
        rE = shape.E / reference.E
        rA = shape.A / reference.A
        ralpha = shape.alpha_ffn / reference.alpha_ffn

        # Table 2 / Scaling Rule substitutions relative to the tuned reference.
        # Source-specific numerical constants cancel in these ratios.
        init = {
            "embedding": 1.0,
            "effective_tied_unembedding": rD ** -1.0,
            "attention_qko": rD ** -0.5,
            "attention_value": rD ** -0.5,
            "router_gamma1": rD ** -1.0,
            "expert_up": rD ** -0.5,
            "expert_down_raw": rD ** -0.5 * ralpha ** -1.0,
        }
        init["expert_down_after_residual"] = init["expert_down_raw"] / rL
        init["expert_down_after_residual_constant_rho_prediction"] = rD ** -0.5
        init["double_depth_after_residual"] = init["expert_down_raw"] / (rL * rL)

        learning_rates = {
            "embeddings": 1.0,
            "norms": 1.0,
            "attention_qkv": rD ** -1.0,
            "attention_output": rD ** -1.0,
            "router": rD ** -1.0,
            "expert_up": rD ** -1.0,
            "expert_down": rM ** -1.0,
            "other_biases": 1.0,
            "manual_expert_bias": 1.0,
        }
        learning_rates["expert_down_after_residual"] = learning_rates["expert_down"] / rL
        learning_rates["expert_down_after_residual_constant_rho_prediction"] = rD ** -1.0

        adam_epsilons = {
            "embeddings": rD ** -1.0,
            "norms": 1.0,
            "attention_qkv": rD ** -1.0 * rL ** -1.0,
            "attention_output": rD ** -1.0 * rL ** -1.0,
            "router": rD ** -1.0 * rL ** -1.0,
            "expert_up": rM ** -1.0 * rL ** -1.0,
            "expert_down": rD * rM ** -2.0 * rL ** -1.0,
            "other_biases": rL ** -1.0,
        }

        absolute = {
            "expert_down_raw_std_without_source_constant": math.sqrt(shape.D) / shape.M,
            "expert_down_after_residual": math.sqrt(shape.D) / (shape.L * shape.M),
            "expert_down_after_residual_constant_rho": 1.0 / (shape.rho * math.sqrt(shape.D)),
            "expert_down_lr_without_source_constant": 1.0 / shape.M,
            "expert_down_lr_after_residual": 1.0 / (shape.L * shape.M),
            "expert_down_lr_after_residual_constant_rho": 1.0 / (shape.rho * shape.D),
            "adam_epsilon_expert_up_without_source_constant": 1.0 / (shape.M * shape.L),
            "adam_epsilon_expert_down_without_source_constant": shape.D / (shape.M**2 * shape.L),
            "alpha_star_constant_rho": 1.0 / (shape.rho * shape.E),
            "stream_variance_constant_rho": 1.0 / (shape.rho * shape.A),
        }

        output_rows.append(
            {
                **asdict(shape),
                "rho": shape.rho,
                "kappa": shape.kappa,
                "alpha_ffn": shape.alpha_ffn,
                "alpha_ffn_regime": _regime(shape.alpha_ffn),
                "alpha_star": shape.alpha_star,
                "stream_init_variance_proxy": shape.stream_init_variance_proxy,
                "ratios_to_reference": {"L": rL, "M": rM, "D": rD, "E": rE, "A": rA},
                "initialization_ratios": init,
                "learning_rate_ratios": learning_rates,
                "adam_epsilon_ratios": adam_epsilons,
                "absolute_scale_identities": absolute,
            }
        )

    rho0, kappa0 = rows[0].rho, rows[0].kappa
    constant_rho_error = max(_relative_error(shape.rho, rho0) for shape in rows)
    constant_kappa_error = max(_relative_error(shape.kappa, kappa0) for shape in rows)
    effective_init_errors = [
        _relative_error(
            float(row["initialization_ratios"]["expert_down_after_residual"]),
            float(row["initialization_ratios"]["expert_down_after_residual_constant_rho_prediction"]),
        )
        for row in output_rows
    ]
    effective_lr_errors = [
        _relative_error(
            float(row["learning_rate_ratios"]["expert_down_after_residual"]),
            float(row["learning_rate_ratios"]["expert_down_after_residual_constant_rho_prediction"]),
        )
        for row in output_rows
    ]
    alpha_stars = [shape.alpha_star for shape in rows]
    stream_vars = [shape.stream_init_variance_proxy for shape in rows]
    same_alpha_star = max(alpha_stars) / min(alpha_stars) - 1.0
    same_stream_variance = max(stream_vars) / min(stream_vars) - 1.0
    depths = [float(shape.L) for shape in rows]
    double_depth_theory = [
        shape.stream_init_variance_proxy / (shape.L / reference.L) ** 2 for shape in rows
    ]

    checks = {
        "constant_rho_relative_error": constant_rho_error,
        "constant_kappa_relative_error": constant_kappa_error,
        "effective_down_init_identity_max_relative_error": max(effective_init_errors),
        "effective_down_lr_identity_max_relative_error": max(effective_lr_errors),
        "alpha_star_spread_fraction": same_alpha_star,
        "stream_init_variance_proxy_spread_fraction": same_stream_variance,
        "constant_rho_stream_variance_slope_in_L": _slope(depths, stream_vars),
        "double_depth_stream_variance_slope_in_L": _slope(depths, double_depth_theory),
    }
    compatible = (
        constant_rho_error < 1e-12
        and constant_kappa_error < 1e-12
        and max(effective_init_errors) < 1e-12
        and max(effective_lr_errors) < 1e-12
    )
    limit_statement = (
        "same finite-alpha_star neural-SDE sector"
        if same_alpha_star < 1e-12 and alpha_stars[0] > 0
        else "alpha_star changes; not the same Jiang limiting process"
    )
    return {
        "schema_version": 1,
        "scope": (
            "source-parameterisation and structural mean-field compatibility; "
            "not a full MoE DMFT solution or a language-loss prediction"
        ),
        "verdict": {
            "constant_rho_is_parameterisation_compatible": compatible,
            "sqrtD_over_LM_is": "effective residualized down scale, not raw down-matrix std",
            "limit_sector": limit_statement,
            "universal_neural_ode_requires": "E -> infinity with A/E fixed (or another path making alpha_star -> 0)",
            "depth_caveat": (
                "at fixed rho, alpha_ffn=rho/L; unbounded depth eventually enters the "
                "measured crossover and is not a clean alpha_ffn-transfer regime"
            ),
        },
        "checks": checks,
        "rows": output_rows,
    }


def monte_carlo(shapes: Sequence[Shape], *, seeds: int, input_dimension: int, tokens: int) -> dict[str, object]:
    if seeds < 2:
        raise ValueError("Monte Carlo requires at least two seeds")
    import torch

    from moe import MoENet, data

    torch.set_num_threads(1)
    values: dict[str, dict[str, object]] = {}
    for shape in shapes:
        correct: list[float] = []
        double_depth: list[float] = []
        for seed in range(seeds):
            X, _ = data(input_dimension, tokens, seed=91_000 + seed, dtype=torch.float32)
            net = MoENet(
                n=shape.D,
                L=shape.L,
                E=shape.E,
                kappa=shape.kappa,
                alpha_ffn=shape.alpha_ffn,
                D=input_dimension,
                gamma=None,
                b_std=1.0,
                seed=31_000 + seed,
                dtype=torch.float32,
            )
            correct.append(net.stream_init_var(X))
            with torch.no_grad():
                for weight in net.Wd:
                    weight.div_(shape.L)
            double_depth.append(net.stream_init_var(X))
        values[shape.label] = {
            "L": shape.L,
            "correct_mean": fmean(correct),
            "correct_sem": stdev(correct) / math.sqrt(seeds),
            "double_depth_mean": fmean(double_depth),
            "double_depth_sem": stdev(double_depth) / math.sqrt(seeds),
            "paired_values": [
                {"correct": left, "double_depth": right}
                for left, right in zip(correct, double_depth)
            ],
        }
    ordered = [values[shape.label] for shape in shapes]
    return {
        "seeds": seeds,
        "input_dimension": input_dimension,
        "tokens": tokens,
        "rows": values,
        "correct_variance_slope_in_L": _slope(
            [float(row["L"]) for row in ordered],
            [float(row["correct_mean"]) for row in ordered],
        ),
        "double_depth_variance_slope_in_L": _slope(
            [float(row["L"]) for row in ordered],
            [float(row["double_depth_mean"]) for row in ordered],
        ),
    }


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        help="label:L:M:D:E:A; repeat in increasing depth order",
    )
    parser.add_argument("--reference-label", default="l2")
    parser.add_argument("--monte-carlo-seeds", type=int, default=0)
    parser.add_argument("--input-dimension", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    shapes = tuple(
        args.shape
        or (
            Shape("l2", 2, 256, 16, 4, 1),
            Shape("l4", 4, 256, 32, 4, 1),
            Shape("l8", 8, 256, 64, 4, 1),
            Shape("l16", 16, 256, 128, 4, 1),
        )
    )
    references = [shape for shape in shapes if shape.label == args.reference_label]
    if len(references) != 1:
        raise ValueError("reference-label must select exactly one shape")
    payload = derive(references[0], shapes)
    if args.monte_carlo_seeds:
        payload["monte_carlo"] = monte_carlo(
            shapes,
            seeds=args.monte_carlo_seeds,
            input_dimension=args.input_dimension,
            tokens=args.tokens,
        )
    if args.output:
        _atomic_write(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
