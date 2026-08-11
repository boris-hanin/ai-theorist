#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ai_theorist.autoscaler.study import atomic_write_json


def _object(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _read(path: Path, name: str) -> Dict[str, Any]:
    return _object(json.loads(path.read_text()), name)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _rows(selection: Mapping[str, Any], source: str) -> List[Dict[str, Any]]:
    raw = selection.get("grid")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{source} selection grid must be an array")
    rows = []
    for value in raw:
        row = _object(value, f"{source} grid row")
        losses = [float(item) for item in row["seed_losses"]]
        if len(losses) != 3 or not all(math.isfinite(item) for item in losses):
            raise ValueError(f"{source} grid row must contain three finite seed losses")
        tau = row.get("weight_decay_tau_ema")
        rows.append(
            {
                "source": source,
                "learning_rate": float(row["learning_rate"]),
                "weight_decay_tau_ema": None if tau is None else float(tau),
                "mean_validation_loss": float(row["mean_validation_loss"]),
                "sem_validation_loss": float(row["sem_validation_loss"]),
                "seed_losses": losses,
            }
        )
    return rows


def _deduplicate(
    rows: Sequence[Mapping[str, Any]], *, maximum_overlap_delta: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    unique: Dict[Tuple[float, Optional[float]], Dict[str, Any]] = {}
    overlaps = []
    for value in rows:
        row = dict(value)
        key = (row["learning_rate"], row["weight_decay_tau_ema"])
        previous = unique.get(key)
        if previous is None:
            unique[key] = row
            continue
        deltas = [
            abs(float(left) - float(right))
            for left, right in zip(previous["seed_losses"], row["seed_losses"])
        ]
        overlap = {
            "learning_rate": key[0],
            "weight_decay_tau_ema": key[1],
            "sources": [previous["source"], row["source"]],
            "maximum_seed_loss_delta": max(deltas),
            "passed": max(deltas) <= maximum_overlap_delta,
        }
        overlaps.append(overlap)
        if not overlap["passed"]:
            raise ValueError(
                "duplicated adaptive tuning cell failed reproducibility gate: "
                + json.dumps(overlap, sort_keys=True)
            )
        if row["mean_validation_loss"] < previous["mean_validation_loss"]:
            unique[key] = row
    return list(unique.values()), overlaps


def _decision(
    rows: Sequence[Mapping[str, Any]],
    *,
    finite_sources: Sequence[str],
    zero_source: str,
) -> Dict[str, Any]:
    selected = dict(min(rows, key=lambda row: float(row["mean_validation_loss"])))
    rates = sorted({float(row["learning_rate"]) for row in rows})
    rate_index = rates.index(float(selected["learning_rate"]))
    eta_interior = 0 < rate_index < len(rates) - 1
    tau = selected["weight_decay_tau_ema"]
    if tau is None:
        mode = "zero"
        tau_resolved = selected["source"] == zero_source
        finite_tau_interior = None
    else:
        mode = "finite_tau_ema"
        finite_grid = sorted(
            {
                float(row["weight_decay_tau_ema"])
                for row in rows
                if row["weight_decay_tau_ema"] is not None
            }
        )
        tau_index = finite_grid.index(float(tau))
        finite_tau_interior = 0 < tau_index < len(finite_grid) - 1
        tau_resolved = selected["source"] in set(finite_sources)
    gates = {
        "learning_rate_optimum_is_interior": eta_interior,
        "selected_source_matches_decay_mode": tau_resolved,
        "finite_tau_optimum_is_interior_or_zero_selected": (
            mode == "zero" or finite_tau_interior is True
        ),
    }
    return {
        "selected_source": selected["source"],
        "selected_learning_rate": selected["learning_rate"],
        "selected_weight_decay_mode": mode,
        "selected_weight_decay_tau_ema": tau,
        "mean_validation_loss": selected["mean_validation_loss"],
        "sem_validation_loss": selected["sem_validation_loss"],
        "seed_losses": selected["seed_losses"],
        "finite_tau_optimum_is_interior": finite_tau_interior,
        "gates": gates,
        "candidate_grid": sorted(
            [dict(row) for row in rows],
            key=lambda row: (
                row["learning_rate"],
                float("inf")
                if row["weight_decay_tau_ema"] is None
                else row["weight_decay_tau_ema"],
            ),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select finite tau or exact zero decay after the adaptive extension."
    )
    parser.add_argument("preregistration", type=Path)
    parser.add_argument("original_jiang_selection", type=Path)
    parser.add_argument("expanded_jiang_selection", type=Path)
    parser.add_argument("zero_jiang_selection", type=Path)
    parser.add_argument("original_completep_selection", type=Path)
    parser.add_argument("zero_completep_selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prereg = _read(args.preregistration, "adaptive preregistration")
    inputs = {
        "original_jiang": args.original_jiang_selection,
        "expanded_jiang": args.expanded_jiang_selection,
        "zero_jiang": args.zero_jiang_selection,
        "original_completep": args.original_completep_selection,
        "zero_completep": args.zero_completep_selection,
    }
    selections = {
        name: _read(path, f"{name} selection") for name, path in inputs.items()
    }
    fingerprint_gates = {
        f"{name}_plan_matches_preregistration": (
            selections[name]["plan_fingerprint"]
            == prereg["plans"][name]["plan_fingerprint"]
        )
        for name in inputs
    }

    jiang_rows, overlaps = _deduplicate(
        [
            row
            for name in ("original_jiang", "expanded_jiang", "zero_jiang")
            for row in _rows(selections[name], name)
        ],
        maximum_overlap_delta=0.005,
    )
    completep_rows, completep_overlaps = _deduplicate(
        [
            row
            for name in ("original_completep", "zero_completep")
            for row in _rows(selections[name], name)
        ],
        maximum_overlap_delta=0.005,
    )
    jiang = _decision(
        jiang_rows,
        finite_sources=("original_jiang", "expanded_jiang"),
        zero_source="zero_jiang",
    )
    completep = _decision(
        completep_rows,
        finite_sources=("original_completep",),
        zero_source="zero_completep",
    )

    def source_selection_matches(decision: Mapping[str, Any]) -> bool:
        source = str(decision["selected_source"])
        source_selection = selections[source]
        return (
            float(source_selection["selected_learning_rate"])
            == float(decision["selected_learning_rate"])
            and source_selection.get("selected_weight_decay_tau_ema")
            == decision.get("selected_weight_decay_tau_ema")
        )

    gates = {
        "adaptive_preregistration_valid": (
            prereg.get("status") == "preregistered_adaptive_extension"
            and all(_object(prereg.get("gates"), "preregistration gates").values())
        ),
        **fingerprint_gates,
        "jiang_overlap_reproduced": bool(overlaps)
        and all(row["passed"] for row in overlaps),
        "completep_has_no_unexpected_overlap": not completep_overlaps,
        "jiang_decision_passed": all(jiang["gates"].values()),
        "completep_decision_passed": all(completep["gates"].values()),
        "jiang_selected_source_has_same_local_optimum": source_selection_matches(
            jiang
        ),
        "completep_selected_source_has_same_local_optimum": source_selection_matches(
            completep
        ),
    }
    passed = all(gates.values())
    payload = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "claim_scope": prereg["claim_scope"],
        "preregistration_sha256": _sha256(args.preregistration),
        "selection_inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "jiang": jiang,
        "completep": completep,
        "overlap_reproducibility": overlaps,
        "gates": gates,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        failed_gates = [name for name, value in gates.items() if not value]
        raise SystemExit("adaptive selection failed gates: " + ", ".join(failed_gates))


if __name__ == "__main__":
    main()
