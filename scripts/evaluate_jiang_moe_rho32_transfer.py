#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable, Mapping

from ai_theorist.autoscaler.study import atomic_write_json


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain an object")
    return value


def _records(root: Path) -> list[Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for path in root.glob("*/shard-*/trials/*.json"):
        payload = _load(path)
        if "run_id" not in payload or "metadata" not in payload:
            continue
        rows[str(payload["run_id"])] = payload
    return list(rows.values())


def _mean_sem(values: Iterable[float]) -> tuple[float, float]:
    rows = tuple(float(value) for value in values)
    if not rows:
        raise ValueError("cannot summarize an empty sequence")
    return fmean(rows), stdev(rows) / math.sqrt(len(rows)) if len(rows) > 1 else 0.0


def _slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2 or any(x <= 0 for x in xs) or any(y <= 0 for y in ys):
        return math.nan
    lx, ly = [math.log(x) for x in xs], [math.log(y) for y in ys]
    xbar, ybar = fmean(lx), fmean(ly)
    denominator = sum((x - xbar) ** 2 for x in lx)
    return sum((x - xbar) * (y - ybar) for x, y in zip(lx, ly)) / denominator


def evaluate(
    plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    records: list[Mapping[str, Any]],
    *,
    minimum_progress: float = 0.001,
    maximum_progress_slope: float = 0.30,
    minimum_control_degradation: float = 0.005,
) -> dict[str, Any]:
    selected_eta = float(selection["selected_learning_rate"])
    seeds = [int(seed) for seed in plan["seeds"]]
    scales = [dict(row) for row in plan["scales"]]
    reference_index = int(plan["architecture_contract"]["reference_scale_index"])
    reference_name = str(scales[reference_index]["name"])

    theory_by_scale: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    wrong_by_scale: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        metadata = dict(record["metadata"])
        scale = dict(metadata["scale"])
        mode = str(metadata["optimizer_mode"])
        eta = float(dict(record["optimizer"])["learning_rate"])
        if not math.isclose(eta, selected_eta, rel_tol=0.0, abs_tol=0.0):
            continue
        target = theory_by_scale if mode == "theory" else wrong_by_scale if mode == "wrong_global" else None
        if target is not None:
            target[str(scale["name"])].append(record)

    summaries: list[dict[str, Any]] = []
    all_theory_records: list[Mapping[str, Any]] = []
    expected_seed_set = set(seeds)
    complete_factorial = True
    common_checkpoint_steps: list[int] | None = None
    checkpoint_steps_match = True
    for scale in scales:
        name = str(scale["name"])
        rows = theory_by_scale[name]
        all_theory_records.extend(rows)
        row_seeds = {int(row["seed"]) for row in rows}
        if len(rows) != len(seeds) or row_seeds != expected_seed_set:
            complete_factorial = False
        progresses: list[float] = []
        final_losses: list[float] = []
        routing_deviations: list[float] = []
        trajectories: list[list[dict[str, float]]] = []
        for row in rows:
            checkpoints = [dict(item) for item in row["validation_checkpoints"]]
            steps = [int(item["step"]) for item in checkpoints]
            if common_checkpoint_steps is None:
                common_checkpoint_steps = steps
            elif steps != common_checkpoint_steps:
                checkpoint_steps_match = False
            initial = float(checkpoints[0]["validation_loss"])
            final = float(row["final_validation_loss"])
            progresses.append((initial - final) / max(abs(initial), 1e-30))
            final_losses.append(final)
            diagnostics = dict(dict(row["metadata"])["diagnostics"])
            routing_deviations.append(
                float(diagnostics["maximum_absolute_load_deviation"])
            )
            trajectories.append(
                [
                    {
                        "step": float(item["step"]),
                        "tokens": float(item["tokens"]),
                        "validation_loss": float(item["validation_loss"]),
                    }
                    for item in checkpoints
                ]
            )
        if rows:
            progress_mean, progress_sem = _mean_sem(progresses)
            loss_mean, loss_sem = _mean_sem(final_losses)
            routing_mean, routing_sem = _mean_sem(routing_deviations)
        else:
            progress_mean = progress_sem = loss_mean = loss_sem = routing_mean = routing_sem = math.nan
        summaries.append(
            {
                "scale": name,
                "depth": int(scale["depth"]),
                "D": int(scale["width"]),
                "M": int(scale["hidden_width"]),
                "alpha_ffn": float(scale["hidden_width"]) / float(scale["width"]),
                "rho": float(scale["rho_lm_over_d"]),
                "active_non_embedding_parameters": int(
                    scale["active_non_embedding_parameters"]
                ),
                "seeds": sorted(row_seeds),
                "mean_fractional_progress": progress_mean,
                "sem_fractional_progress": progress_sem,
                "mean_final_validation_loss": loss_mean,
                "sem_final_validation_loss": loss_sem,
                "mean_maximum_routing_load_deviation": routing_mean,
                "sem_maximum_routing_load_deviation": routing_sem,
                "trajectories": trajectories,
            }
        )

    progress_slope = _slope(
        [float(row["active_non_embedding_parameters"]) for row in summaries],
        [float(row["mean_fractional_progress"]) for row in summaries],
    )
    finite = all(
        math.isfinite(float(row["final_validation_loss"]))
        and all(
            math.isfinite(float(point["validation_loss"]))
            for point in row["validation_checkpoints"]
        )
        for row in all_theory_records
    )
    optimizer_audits_complete = all(
        dict(dict(row["metadata"])["optimizer_group_audit"])["complete"] is True
        and dict(dict(row["metadata"])["optimizer_group_audit"])["disjoint"] is True
        and len(dict(dict(row["metadata"])["optimizer_group_audit"])["groups"]) == 8
        for row in all_theory_records
    )
    routing_non_degenerate = all(
        float(dict(dict(row["metadata"])["diagnostics"])["maximum_absolute_expert_bias"])
        > 0.0
        and all(
            int(count) > 0
            for count in dict(dict(row["metadata"])["diagnostics"])[
                "routing_token_counts"
            ]
        )
        for row in all_theory_records
    )

    endpoint_name = str(scales[-1]["name"])
    wrong_rows = wrong_by_scale[endpoint_name]
    correct_endpoint = [
        float(row["final_validation_loss"])
        for row in theory_by_scale[endpoint_name]
    ]
    wrong_losses = [float(row["final_validation_loss"]) for row in wrong_rows]
    control_complete = (
        len(wrong_rows) == len(seeds)
        and {int(row["seed"]) for row in wrong_rows} == expected_seed_set
    )
    if correct_endpoint and wrong_losses:
        correct_mean, correct_sem = _mean_sem(correct_endpoint)
        wrong_mean, wrong_sem = _mean_sem(wrong_losses)
        control_degradation = wrong_mean / correct_mean - 1.0
    else:
        correct_mean = correct_sem = wrong_mean = wrong_sem = control_degradation = math.nan
    negative_control = {
        "endpoint_scale": endpoint_name,
        "correct_mean_loss": correct_mean,
        "correct_sem_loss": correct_sem,
        "wrong_global_mean_loss": wrong_mean,
        "wrong_global_sem_loss": wrong_sem,
        "relative_degradation": control_degradation,
        "minimum_required_degradation": minimum_control_degradation,
        "complete": control_complete,
        "passed": control_complete
        and math.isfinite(control_degradation)
        and control_degradation >= minimum_control_degradation,
    }

    alpha_stars = [
        float(row["width"])
        / (
            float(row["hidden_width"])
            * float(row["num_experts"])
            * float(row["depth"])
        )
        for row in scales
    ]
    gates = {
        "reference_eta_is_interior": selection.get(
            "learning_rate_optimum_is_interior"
        )
        is True,
        "complete_three_seed_fixed_eta_factorial": complete_factorial,
        "every_trajectory_is_finite": finite,
        "checkpoint_steps_match": checkpoint_steps_match,
        "every_shape_makes_minimum_progress": all(
            float(row["mean_fractional_progress"]) >= minimum_progress
            for row in summaries
        ),
        "fixed_eta_progress_slope_within_bar": math.isfinite(progress_slope)
        and abs(progress_slope) <= maximum_progress_slope,
        "every_optimizer_contract_has_eight_complete_disjoint_groups": optimizer_audits_complete,
        "routing_is_non_degenerate": routing_non_degenerate,
        "wrong_global_lr_control_is_worse": negative_control["passed"],
        "rho_is_exactly_32": all(float(row["rho_lm_over_d"]) == 32.0 for row in scales),
        "alpha_star_is_fixed_one_over_128": all(
            value == 1.0 / 128.0 for value in alpha_stars
        ),
        "deepest_alpha_ffn_remains_above_crossover": (
            float(scales[-1]["hidden_width"]) / float(scales[-1]["width"])
        )
        > 1.8,
    }
    return {
        "schema_version": 1,
        "status": "passed" if all(gates.values()) else "failed",
        "scientific_status": "short_horizon_three_seed_fixed_eta_transfer_pilot",
        "claim_restriction": (
            "No local nonreference retuning was run. This tests fixed-eta dynamics, "
            "not proximity to each finite-shape stability edge or token-horizon transfer."
        ),
        "selected_reference_eta": selected_eta,
        "reference_scale": reference_name,
        "minimum_fractional_progress": minimum_progress,
        "maximum_absolute_log_progress_slope": maximum_progress_slope,
        "log_progress_vs_log_active_nonembedding_parameter_slope": progress_slope,
        "scales": summaries,
        "negative_control": negative_control,
        "gates": gates,
        "accepted": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("fleet_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        _load(args.plan),
        _load(args.selection),
        _records(args.fleet_root),
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["accepted"]:
        failed = [name for name, passed in result["gates"].items() if not passed]
        raise SystemExit("rho32 MoE fixed-eta transfer failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
