"""Strict expansion and sequential execution of optimizer-by-dataset matrices."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .schema import SpecError, StudySpec
from .study import atomic_write_json, run_study


def _strict_keys(data: Mapping[str, Any], allowed: Sequence[str], context: str) -> None:
    extras = sorted(set(data) - set(allowed))
    if extras:
        raise SpecError(f"Unknown {context} field(s): {', '.join(extras)}")


def _named_entries(
    entries: Any, *, context: str, payload_key: str, extra_keys: Sequence[str] = ()
) -> List[Mapping[str, Any]]:
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
        raise SpecError(f"{context} must be a non-empty list")
    result = []
    seen = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            raise SpecError(f"{context}[{index}] must be an object")
        _strict_keys(raw, ("id", payload_key, *extra_keys), f"{context}[{index}]")
        identifier = raw.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise SpecError(f"{context}[{index}].id must be a non-empty string")
        if identifier in seen:
            raise SpecError(f"duplicate {context} id: {identifier}")
        payload = raw.get(payload_key)
        if not isinstance(payload, Mapping):
            raise SpecError(f"{context}[{index}].{payload_key} must be an object")
        seen.add(identifier)
        result.append(raw)
    return result


def expand_validation_matrix(payload: Mapping[str, Any]) -> Tuple[Tuple[str, StudySpec], ...]:
    _strict_keys(
        payload,
        ("matrix_schema_version", "name", "base_spec", "optimizers", "datasets"),
        "matrix",
    )
    if payload.get("matrix_schema_version") != 1:
        raise SpecError("matrix_schema_version must be 1")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SpecError("matrix.name must be a non-empty string")
    base = payload.get("base_spec")
    if not isinstance(base, Mapping):
        raise SpecError("matrix.base_spec must be an object")
    optimizers = _named_entries(
        payload.get("optimizers"),
        context="matrix.optimizers",
        payload_key="optimizer",
        extra_keys=("tuning",),
    )
    datasets = _named_entries(
        payload.get("datasets"), context="matrix.datasets", payload_key="dataset"
    )

    cells = []
    for optimizer_entry in optimizers:
        for dataset_entry in datasets:
            data = copy.deepcopy(dict(base))
            optimizer_id = str(optimizer_entry["id"])
            dataset_id = str(dataset_entry["id"])
            data["name"] = f"{name}-{optimizer_id}-{dataset_id}"
            data["optimizer"] = dict(optimizer_entry["optimizer"])
            data["dataset"] = {
                **dict(data.get("dataset", {})),
                **dict(dataset_entry["dataset"]),
            }
            if "tuning" in optimizer_entry:
                if not isinstance(optimizer_entry["tuning"], Mapping):
                    raise SpecError(
                        f"matrix optimizer {optimizer_id} tuning must be an object"
                    )
                data["tuning"] = dict(optimizer_entry["tuning"])
            spec = StudySpec.from_dict(data)
            cells.append((f"{optimizer_id}__{dataset_id}", spec))
    return tuple(cells)


def compile_validation_matrix(payload: Mapping[str, Any]) -> Dict[str, Any]:
    cells = expand_validation_matrix(payload)
    return {
        "matrix_schema_version": 1,
        "name": payload["name"],
        "cell_count": len(cells),
        "execution_policy": "sequential_no_gpu_overlap",
        "cells": [
            {
                "cell": cell,
                "study_name": spec.name,
                "study_fingerprint": spec.fingerprint,
                "optimizer": spec.optimizer.name,
                "dataset": spec.dataset.kind,
            }
            for cell, spec in cells
        ],
    }


def run_validation_matrix(
    payload: Mapping[str, Any], *, output_dir: Path, device: str = "cpu"
) -> Dict[str, Any]:
    """Run cells sequentially so two campaigns never contend for one accelerator."""
    cells = expand_validation_matrix(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    compiled = compile_validation_matrix(payload)
    atomic_write_json(output_dir / "matrix-manifest.json", compiled)
    summaries = []
    for cell, spec in cells:
        result = run_study(spec, device=device, output_dir=output_dir / cell)
        summaries.append(
            {
                "cell": cell,
                "study_fingerprint": spec.fingerprint,
                "optimizer": spec.optimizer.name,
                "dataset": spec.dataset.kind,
                "forecastable": result["forecastable"],
                "selected_normalized_eta": result["tuning"][
                    "selected_normalized_learning_rate"
                ],
                "transfer_accepted": all(
                    check["accepted"] for check in result["transfer_checks"]
                ),
                "negative_control_rejected": (
                    result["negative_control"] is None
                    or bool(result["negative_control"]["rejected"])
                ),
                "heldout_accepted": all(
                    check["accepted"] for check in result["holdout_calibration"]
                ),
                "refusal_reasons": result["refusal_reasons"],
            }
        )
        atomic_write_json(
            output_dir / "matrix-summary.json",
            {
                **compiled,
                "device": device,
                "completed_cells": len(summaries),
                "summaries": summaries,
            },
        )
    return {
        **compiled,
        "device": device,
        "completed_cells": len(summaries),
        "summaries": summaries,
    }
