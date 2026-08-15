#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ai_theorist.autoscaler.scaling import fit_scaling_ensemble
from ai_theorist.autoscaler.study import atomic_write_json


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _fingerprint(value: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _selected_reference_record(
    tune_root: Path, selected_eta: float
) -> tuple[Mapping[str, Any], Path]:
    matches: list[tuple[Mapping[str, Any], Path]] = []
    for path in tune_root.glob("shard-*/trials/*.json"):
        row = _load(path)
        if (
            row.get("metadata", {}).get("scale", {}).get("name") == "S1"
            and row.get("metadata", {}).get("optimizer_mode") == "theory"
            and math.isclose(
                float(row.get("optimizer", {}).get("learning_rate", math.nan)),
                selected_eta,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            matches.append((row, path))
    if len(matches) != 1:
        raise ValueError("selected S1 tuning record is missing or ambiguous")
    return matches[0]


def _ddp_record(ladder_root: Path, name: str) -> tuple[Mapping[str, Any], Path]:
    path = ladder_root / name / "ladder-shard-000.json"
    shard = _load(path)
    records = list(shard.get("records", ()))
    if shard.get("status") != "completed" or len(records) != 1:
        raise ValueError(f"{name} DDP shard is incomplete or ambiguous")
    row = records[0]
    if row.get("metadata", {}).get("scale", {}).get("name") != name:
        raise ValueError(f"{name} DDP shard contains the wrong scale")
    return row, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "frozen-prediction.json"

    preregistration_path = args.root / "preregistration.json"
    plan_path = args.root / "plan-single.json"
    config_path = args.root / "config-single.json"
    selection_path = args.root / "reference-selection.json"
    preregistration = _load(preregistration_path)
    plan = _load(plan_path)
    config = _load(config_path)
    selection = _load(selection_path)
    if preregistration.get("status") != "preregistered":
        raise ValueError("campaign is not preregistered")
    if selection.get("learning_rate_optimum_is_interior") is not True:
        raise ValueError("reference eta is not interior")
    scales = [dict(row) for row in plan["scales"]]
    target_name = str(scales[-1]["name"])
    if (
        args.root / "ladder" / target_name / "ladder-shard-000.json"
    ).exists():
        raise ValueError(
            f"refusing to freeze a prediction after the {target_name} reveal"
        )

    selected_eta = float(selection["selected_learning_rate"])
    records: dict[str, Mapping[str, Any]] = {}
    sources: dict[str, str] = {}
    reference, reference_path = _selected_reference_record(
        args.root / "tune", selected_eta
    )
    records["S1"] = reference
    sources["S1"] = _sha(reference_path)
    for scale in scales[1:-1]:
        name = str(scale["name"])
        record, path = _ddp_record(args.root / "ladder", name)
        records[name] = record
        sources[name] = _sha(path)

    fit_scales = scales[:-1]
    sizes: list[float] = []
    losses: list[float] = []
    rows: list[dict[str, Any]] = []
    for scale in fit_scales:
        name = str(scale["name"])
        record = records[name]
        loss = float(record["final_validation_loss"])
        if not math.isfinite(loss) or loss <= 0.0:
            raise ValueError(f"{name} loss is non-finite")
        if int(record["nonpadding_tokens_seen"]) != int(scale["presented_tokens"]):
            raise ValueError(f"{name} token budget mismatch")
        if float(record["optimizer"]["learning_rate"]) != selected_eta:
            raise ValueError(f"{name} does not use the selected eta")
        sizes.append(float(scale["active_non_embedding_parameters"]))
        losses.append(loss)
        rows.append(
            {
                "scale": name,
                "active_non_embedding_parameters": int(
                    scale["active_non_embedding_parameters"]
                ),
                "active_parameters": int(scale["active_parameters"]),
                "total_parameters": int(scale["parameters"]),
                "presented_tokens": int(scale["presented_tokens"]),
                "validation_loss": loss,
                "data_parallel_replicas": int(record["data_parallel_replicas"]),
            }
        )

    target = dict(scales[-1])
    ladder = config["ladder"]
    fit = fit_scaling_ensemble(
        sizes,
        losses,
        [0.0] * len(losses),
        target_size=float(target["active_non_embedding_parameters"]),
        maximum_extrapolation_factor=float(ladder["maximum_extrapolation_factor"]),
        maximum_family_spread=float(ladder["maximum_family_spread"]),
        maximum_backtest_relative_error=float(
            ladder["maximum_backtest_relative_error"]
        ),
        bootstrap_samples=400,
    )
    payload = {
        "schema_version": 1,
        "status": f"frozen_before_{target_name}_reveal",
        "certified_forecast": bool(fit["certified"]),
        "scientific_status": (
            f"preregistered_single_seed_{target_name}_holdout_prediction"
        ),
        "preregistration_sha256": _sha(preregistration_path),
        "plan_sha256": _sha(plan_path),
        "config_sha256": _sha(config_path),
        "selection_sha256": _sha(selection_path),
        "selected_learning_rate": selected_eta,
        "fit_parameter_axis": "active_non_embedding_parameters",
        "fit_rows": rows,
        "source_shard_sha256": sources,
        "target": target,
        "prediction": fit,
    }
    payload["fingerprint"] = _fingerprint(payload)
    atomic_write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
