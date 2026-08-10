#!/usr/bin/env python3
"""Reference-tuned transfer for the dense Jiang-attention/Chizat-FFN hybrid.

This is a derived architecture, not the sparse MoE from Jiang et al.  Its
contract is nevertheless complete: GPT-style boundaries, pre-LN, QK^T/d_head,
1/L branches, mean-field FFN initialization, every Adam LR/epsilon group,
reference-tuned scale-independent group constants, and the fixed-token
warmup/constant schedule are all explicit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket
import sys
from typing import Dict, List, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai_theorist.autoscaler.jiang_chizat import (  # noqa: E402
    JIANG_DENSE_REPORTED_LR_MULTIPLIERS,
    JIANG_COMPLETEP_ADAM_THEORY,
    JiangChizatReference,
)
from jiang_chizat_transfer import (  # noqa: E402
    JIANG_CHIZAT_LR_GROUPS,
    RULES,
    Shape,
    Trial,
    atomic_write_json,
    group_feature_velocity_audit,
    parse_shape,
    run_trial,
    summarize_feature_velocity_audits,
    validate_shapes,
)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.inf


def _log_slope(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if not all(math.isfinite(value) and value > 0.0 for value in y_values):
        return float("nan")
    x = [math.log(value) for value in x_values]
    y = [math.log(value) for value in y_values]
    x_mean = _mean(x)
    y_mean = _mean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    return (
        sum((left - x_mean) * (right - y_mean) for left, right in zip(x, y))
        / denominator
        if denominator
        else float("nan")
    )


def best_eta(trials: Sequence[Trial], etas: Sequence[float]) -> float:
    expected_seeds = {trial.seed for trial in trials}

    def score(eta: float) -> float:
        rows = [
            trial
            for trial in trials
            if math.isclose(trial.normalized_eta, eta)
        ]
        if (
            {trial.seed for trial in rows} != expected_seeds
            or any(trial.diverged for trial in rows)
        ):
            return math.inf
        return _mean([trial.final_validation_loss for trial in rows])

    return min(etas, key=score)


def best_multiplier(
    trials: Mapping[float, Sequence[Trial]],
    *,
    minimum_relative_improvement: float,
) -> float:
    """Keep factor one unless a candidate improves every paired seed."""
    baseline = {trial.seed: trial for trial in trials[1.0]}
    qualifying = [1.0]
    for relative_factor, rows in trials.items():
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
            and _mean(improvements) >= minimum_relative_improvement
        ):
            qualifying.append(relative_factor)
    return min(
        qualifying,
        key=lambda factor: _mean(
            [trial.final_validation_loss for trial in trials[factor] if not trial.diverged]
        ),
    )


def fixed_eta_analysis(
    trials: Sequence[Trial],
    shapes: Sequence[Shape],
    seeds: Sequence[int],
    *,
    eta: float,
    rule: str,
) -> Dict[str, object]:
    progress = []
    losses = []
    rows = []
    for shape in shapes:
        selected = [
            trial
            for trial in trials
            if trial.label == shape.label
            and trial.rule == rule
            and math.isclose(trial.normalized_eta, eta)
        ]
        if len(selected) != len(seeds):
            raise ValueError(f"incomplete fixed-eta factorial for {rule}:{shape.label}")
        shape_progress = [
            (trial.validation_loss_checkpoints[0] - trial.final_validation_loss)
            / max(abs(trial.validation_loss_checkpoints[0]), 1e-30)
            for trial in selected
        ]
        progress.append(_mean(shape_progress))
        losses.append(_mean([trial.final_validation_loss for trial in selected]))
        rows.append(
            {
                "label": shape.label,
                "mean_fractional_progress": progress[-1],
                "mean_final_validation_loss": losses[-1],
                "all_finite": all(not trial.diverged for trial in selected),
            }
        )
    slope = _log_slope([shape.dial for shape in shapes], progress)
    accepted = (
        all(bool(row["all_finite"]) for row in rows)
        and all(value >= 1e-3 for value in progress)
        and math.isfinite(slope)
        and abs(slope) <= 0.30
    )
    return {
        "rule": rule,
        "eta": eta,
        "rows": rows,
        "log_progress_vs_log_dial_slope": slope,
        "accepted": accepted,
    }


def primary_analysis(
    trials: Sequence[Trial],
    shapes: Sequence[Shape],
    etas: Sequence[float],
    seeds: Sequence[int],
    reference_label: str,
    oracle_tolerance: float,
) -> Dict[str, object]:
    best_by_shape = {}
    means_by_shape: Dict[str, Dict[float, float]] = {}
    for shape in shapes:
        means = {}
        for eta in etas:
            selected = [
                trial.final_validation_loss
                for trial in trials
                if trial.label == shape.label
                and trial.rule == "primary"
                and math.isclose(trial.normalized_eta, eta)
                and not trial.diverged
            ]
            if len(selected) != len(seeds):
                means[eta] = math.inf
            else:
                means[eta] = _mean(selected)
        means_by_shape[shape.label] = means
        best_by_shape[shape.label] = min(etas, key=lambda eta: means[eta])
    reference_eta = best_by_shape[reference_label]
    offsets = {
        label: math.log10(value / reference_eta)
        for label, value in best_by_shape.items()
    }
    fixed = fixed_eta_analysis(
        trials,
        shapes,
        seeds,
        eta=reference_eta,
        rule="primary",
    )
    reference_index = list(etas).index(reference_eta)
    oracle_losses = {
        shape.label: means_by_shape[shape.label][best_by_shape[shape.label]]
        for shape in shapes
    }
    fixed_losses = {
        shape.label: means_by_shape[shape.label][reference_eta]
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
        "best_eta_by_shape": best_by_shape,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", type=parse_shape, required=True)
    parser.add_argument("--reference-shape", required=True)
    parser.add_argument("--etas", nargs="+", type=float, required=True)
    parser.add_argument("--multiplier-probes", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--minimum-relative-multiplier-improvement", type=float, default=0.005)
    parser.add_argument("--oracle-tolerance", type=float, default=1.10)
    parser.add_argument("--head-dimension", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-validation", type=int, default=512)
    parser.add_argument("--dataset-seed", type=int, default=1729)
    parser.add_argument("--epsilon0", type=float, default=1e-12)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument(
        "--controls",
        nargs="+",
        choices=RULES[1:],
        default=["fan_in_down", "omit_attention_width", "omit_ffn_hidden_width", "disable_attention"],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shapes = validate_shapes(args.shapes, args.head_dimension)
    by_label = {shape.label: shape for shape in shapes}
    if args.reference_shape not in by_label:
        raise ValueError("reference shape must occur in --shapes")
    etas = tuple(sorted(set(args.etas)))
    seeds = tuple(args.seeds)
    probes = tuple(sorted(set(args.multiplier_probes)))
    if len(etas) < 5 or any(value <= 0.0 for value in etas):
        raise ValueError("at least five positive eta probes are required")
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three unique paired seeds are required")
    if not probes or 1.0 not in probes or any(value <= 0.0 for value in probes):
        raise ValueError("multiplier probes must be positive and include 1.0")
    if args.minimum_relative_multiplier_improvement < 0.0:
        raise ValueError("minimum relative multiplier improvement cannot be negative")
    if not math.isfinite(args.oracle_tolerance) or args.oracle_tolerance < 1.0:
        raise ValueError("oracle tolerance must be finite and at least one")
    if min(args.steps, args.batch_size, args.n_train, args.n_validation) <= 0:
        raise ValueError("steps, batch size, and dataset sizes must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    reference_shape = by_label[args.reference_shape]
    reference = JiangChizatReference(
        reference_shape.L,
        reference_shape.M,
        reference_shape.D,
    )
    warmup_steps = args.steps // 2
    print(
        json.dumps(
            {
                "phase": "theory-recall-before-trials",
                "theory": JIANG_COMPLETEP_ADAM_THEORY.to_dict(),
                "architecture_scope": "derived dense interleaving; not sparse MoE",
                "group_rules": {
                    "embeddings": "lr c_embed*eta; eps epsilon0*(D/D0)^-1",
                    "norms": "lr c_norm*eta; eps epsilon0",
                    "attention_qkv": "lr c_qkv*eta*(D/D0)^-1 with source c_qkv=1/16; eps epsilon0*(D/D0)^-1*(L/L0)^-1",
                    "attention_output": "lr c_out*eta*(D/D0)^-1 with source c_out=1; same epsilon rule",
                    "ffn_up": "lr c_up*eta*(D/D0)^-1; eps epsilon0*(M/M0)^-1*(L/L0)^-1",
                    "ffn_down": "lr c_down*eta*(M/M0)^-1 with source c_down=1/16; eps epsilon0*(D/D0)*(M/M0)^-2*(L/L0)^-1",
                    "other_biases": "lr eta; eps epsilon0*(L/L0)^-1",
                },
                "reported_initialization_constants": {
                    "attention_value": "1/16",
                    "ffn_down": "1/4",
                },
                "protocol": "source constants, locally verified by relative coordinate probes; 50% linear warmup then constant; no clipping",
            },
            sort_keys=True,
        ),
        flush=True,
    )

    def execute(
        shape: Shape,
        eta: float,
        seed: int,
        *,
        rule: str,
        multipliers: Mapping[str, float],
    ) -> Trial:
        return run_trial(
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
            steps=args.steps,
            batch_size=args.batch_size,
            seed=seed,
            rule=rule,
            device=device,
            learning_rate_multipliers=multipliers,
            warmup_steps=warmup_steps,
        )

    source_multipliers = dict(JIANG_DENSE_REPORTED_LR_MULTIPLIERS)
    calibration_trials: List[Trial] = []
    initial_screen = [
        execute(reference_shape, eta, seed, rule="primary", multipliers=source_multipliers)
        for eta in etas
        for seed in seeds
    ]
    calibration_trials.extend(initial_screen)
    calibration_eta = best_eta(initial_screen, etas)
    tuned_multipliers = dict(source_multipliers)
    multiplier_choices: Dict[str, Dict[str, float]] = {}
    for group_name in JIANG_CHIZAT_LR_GROUPS:
        candidates = {}
        base_multiplier = tuned_multipliers[group_name]
        for relative_factor in probes:
            absolute_multiplier = base_multiplier * relative_factor
            candidate = {**tuned_multipliers, group_name: absolute_multiplier}
            rows = [
                execute(reference_shape, calibration_eta, seed, rule="primary", multipliers=candidate)
                for seed in seeds
            ]
            calibration_trials.extend(rows)
            candidates[relative_factor] = rows
        selected_relative_factor = best_multiplier(
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

    primary_trials = [
        execute(shape, eta, seed, rule="primary", multipliers=tuned_multipliers)
        for shape in shapes
        for eta in etas
        for seed in seeds
    ]
    primary = primary_analysis(
        primary_trials,
        shapes,
        etas,
        seeds,
        args.reference_shape,
        args.oracle_tolerance,
    )
    selected_eta = float(primary["reference_eta"])
    control_trials = [
        execute(shape, selected_eta, seed, rule=rule, multipliers=tuned_multipliers)
        for rule in args.controls
        for shape in shapes
        for seed in seeds
    ]
    controls = {
        rule: fixed_eta_analysis(
            control_trials,
            shapes,
            seeds,
            eta=selected_eta,
            rule=rule,
        )
        for rule in args.controls
    }
    feature_audits = [
        group_feature_velocity_audit(
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
            batch_size=args.batch_size,
            seed=seed,
            device=device,
            learning_rate_multipliers=tuned_multipliers,
        )
        for shape in shapes
        for seed in seeds
    ]
    feature_summary = summarize_feature_velocity_audits(feature_audits, shapes, seeds)
    report = {
        "schema_version": 2,
        "experiment": "reference_tuned_jiang_attention_chizat_ffn_transfer",
        "status": "completed",
        "host": socket.gethostname(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "theory_recalled_before_trials": JIANG_COMPLETEP_ADAM_THEORY.to_dict(),
        "architecture_contract": {
            "scope": "derived dense interleaving, not sparse MoE",
            "boundaries": "tied token embedding/unembedding, learned absolute positions, final affine LayerNorm",
            "block": "pre-LN MHSA then dense GELU FFN; both residual branches scaled 1/L",
            "attention": "QKV/O with biases in the Table-2 bias group; QK^T/d_head; causal; fixed d_head; V init multiplier 1/16",
            "ffn": "up/down biases in the Table-2 bias group; up std D^-1/2; down std (1/4)*sqrt(D)/M",
            "layer_norm": "affine PyTorch LayerNorm, epsilon=1e-5",
            "optimizer": "Adam beta=(0.9,0.95), Table-2 group epsilons, zero weight decay",
            "schedule": "linear warmup for first half, constant peak for second half",
            "gradient_clipping": "none",
        },
        "reference": asdict(reference),
        "reference_group_calibration": {
            "method": "one ordered coordinate-sweep pass",
            "initial_global_eta_screen": list(etas),
            "selected_calibration_eta": calibration_eta,
            "relative_multiplier_probes": list(probes),
            "minimum_relative_multiplier_improvement": args.minimum_relative_multiplier_improvement,
            "update_gate": "candidate must improve validation loss for every paired seed and exceed the mean relative-improvement threshold",
            "group_order": list(JIANG_CHIZAT_LR_GROUPS),
            "source_learning_rate_multipliers": source_multipliers,
            "selected_learning_rate_multiplier_details": multiplier_choices,
            "selected_learning_rate_multipliers": tuned_multipliers,
            "trials": [asdict(trial) for trial in calibration_trials],
        },
        "fixed_variables": {
            "context_length": args.context_length,
            "n_train": args.n_train,
            "n_validation": args.n_validation,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "token_horizon": args.steps * args.batch_size * args.context_length,
            "warmup_steps": warmup_steps,
            "epsilon0": args.epsilon0,
            "dataset_seed": args.dataset_seed,
        },
        "shapes": [{**asdict(shape), "rho_LM_over_D": shape.rho} for shape in shapes],
        "etas": list(etas),
        "seeds": list(seeds),
        "primary": primary,
        "feature_velocity_audit": {
            "summary": feature_summary,
            "raw": feature_audits,
        },
        "negative_controls": {
            rule: {**analysis, "rejected": not bool(analysis["accepted"])}
            for rule, analysis in controls.items()
        },
        "trials": [asdict(trial) for trial in primary_trials + control_trials],
    }
    report["verdict"] = {
        "primary_accepted": bool(primary["accepted"]),
        "transfer_certified": bool(primary["accepted"]),
        "feature_velocity_accepted": bool(feature_summary["accepted"]),
        "all_controls_rejected": all(not bool(row["accepted"]) for row in controls.values()),
        "mechanism_discrimination_certified": bool(primary["accepted"])
        and bool(feature_summary["accepted"])
        and all(not bool(row["accepted"]) for row in controls.values()),
        "certified": bool(primary["accepted"]),
    }
    atomic_write_json(args.output, report)
    print(json.dumps({"output": str(args.output), **report["verdict"]}, sort_keys=True))


if __name__ == "__main__":
    main()
