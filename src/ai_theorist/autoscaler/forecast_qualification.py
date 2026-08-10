from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Dict, Mapping, Sequence, Tuple


def _logical_key(record: Mapping[str, Any]) -> Tuple[str, str, float, int]:
    metadata = record.get("metadata", {})
    scale = metadata.get("scale", {})
    optimizer = record.get("optimizer", {})
    return (
        str(scale.get("name")),
        str(metadata.get("optimizer_mode")),
        float(optimizer.get("learning_rate")),
        int(record.get("seed")),
    )


def _logical_records(
    records: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str, float, int], Mapping[str, Any]]:
    grouped: Dict[Tuple[str, str, float, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_logical_key(record)].append(record)
    result = {}
    for key, matches in grouped.items():
        losses = {float(row["final_validation_loss"]) for row in matches}
        run_ids = {str(row["run_id"]) for row in matches}
        if len(losses) != 1 or len(run_ids) != 1:
            raise ValueError(f"conflicting duplicate logical record: {key}")
        result[key] = matches[0]
    return result


def _group_contract(record: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    groups = record.get("metadata", {}).get("peak_parameter_group_contract", [])
    return {str(group["name"]): dict(group) for group in groups}


def compare_forecast_topologies(
    single_result: Mapping[str, Any],
    ddp_result: Mapping[str, Any],
    *,
    maximum_absolute_loss_delta: float = 1e-3,
) -> Dict[str, Any]:
    """Prove that two-GPU DDP preserves the one-GPU forecast experiment."""

    if not math.isfinite(maximum_absolute_loss_delta) or maximum_absolute_loss_delta <= 0:
        raise ValueError("maximum_absolute_loss_delta must be finite and positive")
    errors = []
    for label, result in (("single", single_result), ("ddp", ddp_result)):
        if result.get("status") != "completed":
            errors.append(f"{label} result is not completed")
        if result.get("campaign") != "real_text_scaling_ladder":
            errors.append(f"{label} result is not a forecast ladder")
    for field in ("dataset", "architecture_contract"):
        if single_result.get(field) != ddp_result.get(field):
            errors.append(f"{field} changed with topology")
    single_selected = single_result.get("reference_tuning", {}).get(
        "selected_learning_rate"
    )
    ddp_selected = ddp_result.get("reference_tuning", {}).get(
        "selected_learning_rate"
    )
    if single_selected != ddp_selected:
        errors.append("reference learning-rate selection changed with topology")

    single_records = _logical_records(single_result.get("records", []))
    ddp_records = _logical_records(ddp_result.get("records", []))
    if set(single_records) != set(ddp_records):
        missing_ddp = sorted(set(single_records) - set(ddp_records))
        missing_single = sorted(set(ddp_records) - set(single_records))
        errors.append(
            "logical trial sets differ: "
            f"missing_ddp={missing_ddp}, missing_single={missing_single}"
        )

    comparisons = []
    maximum_observed_delta = 0.0
    for key in sorted(set(single_records) & set(ddp_records)):
        single = single_records[key]
        ddp = ddp_records[key]
        for field in (
            "parameter_count",
            "optimizer_steps",
            "total_tokens",
            "batch_tokens",
            "accumulation_steps",
            "learning_rate_schedule",
        ):
            if single.get(field) != ddp.get(field):
                errors.append(f"{key}: {field} changed with topology")
        if int(single.get("data_parallel_replicas", 0)) != 1:
            errors.append(f"{key}: single result does not report one replica")
        if int(ddp.get("data_parallel_replicas", 0)) != 2:
            errors.append(f"{key}: DDP result does not report two replicas")
        if single.get("metadata", {}).get("sampling_contract") != (
            "replicated_global_draw_rank_partition_v1"
        ) or ddp.get("metadata", {}).get("sampling_contract") != (
            "replicated_global_draw_rank_partition_v1"
        ):
            errors.append(f"{key}: topology-preserving sampling contract is missing")

        single_groups = _group_contract(single)
        ddp_groups = _group_contract(ddp)
        if single_groups != ddp_groups:
            errors.append(f"{key}: per-parameter optimizer contract changed")
        single_checkpoints = list(single.get("validation_checkpoints", []))
        ddp_checkpoints = list(ddp.get("validation_checkpoints", []))
        if len(single_checkpoints) != len(ddp_checkpoints):
            errors.append(f"{key}: validation checkpoint count changed")
        checkpoint_deltas = []
        for left, right in zip(single_checkpoints, ddp_checkpoints):
            if left.get("step") != right.get("step") or left.get("tokens") != right.get(
                "tokens"
            ):
                errors.append(f"{key}: validation checkpoint coordinates changed")
                continue
            checkpoint_deltas.append(
                abs(float(left["validation_loss"]) - float(right["validation_loss"]))
            )
        final_delta = abs(
            float(single["final_validation_loss"])
            - float(ddp["final_validation_loss"])
        )
        observed = max([final_delta, *checkpoint_deltas])
        maximum_observed_delta = max(maximum_observed_delta, observed)
        if observed > maximum_absolute_loss_delta:
            errors.append(
                f"{key}: loss delta {observed:.6g} exceeds "
                f"{maximum_absolute_loss_delta:.6g}"
            )
        comparisons.append(
            {
                "scale": key[0],
                "optimizer_mode": key[1],
                "learning_rate": key[2],
                "seed": key[3],
                "maximum_loss_delta": observed,
                "single_wall_seconds": float(single["wall_time_seconds"]),
                "ddp_wall_seconds": float(ddp["wall_time_seconds"]),
                "speedup": (
                    float(single["wall_time_seconds"])
                    / float(ddp["wall_time_seconds"])
                    if float(ddp["wall_time_seconds"]) > 0
                    else None
                ),
            }
        )
    return {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "maximum_absolute_loss_delta": maximum_absolute_loss_delta,
        "maximum_observed_loss_delta": maximum_observed_delta,
        "selected_learning_rate": single_selected,
        "logical_trials_compared": len(comparisons),
        "comparisons": comparisons,
        "errors": errors,
    }
