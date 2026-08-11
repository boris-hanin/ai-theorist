#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ai_theorist.autoscaler.study import atomic_write_json


def _object(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _read(path: Path, name: str) -> Dict[str, Any]:
    return _object(json.loads(path.read_text()), name)


def _rows(selection: Mapping[str, Any], source: str) -> List[Dict[str, Any]]:
    grid = selection.get("grid")
    if not isinstance(grid, Sequence) or isinstance(grid, (str, bytes)):
        raise ValueError(f"{source} selection grid must be an array")
    rows: List[Dict[str, Any]] = []
    for value in grid:
        row = _object(value, f"{source} grid row")
        losses = [float(item) for item in row["seed_losses"]]
        if len(losses) != 3 or not all(math.isfinite(item) for item in losses):
            raise ValueError(f"{source} grid row needs three finite seed losses")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select rho=32 AdamW finite tau_EMA or exact zero decay."
    )
    parser.add_argument("preregistration", type=Path)
    parser.add_argument("base_selection", type=Path)
    parser.add_argument("expanded_selection", type=Path)
    parser.add_argument("zero_selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prereg = _read(args.preregistration, "preregistration")
    base = _read(args.base_selection, "base finite selection")
    expanded = _read(args.expanded_selection, "expanded finite selection")
    zero = _read(args.zero_selection, "zero selection")
    rows = (
        _rows(base, "jiang_base_finite_tau")
        + _rows(expanded, "jiang_expanded_finite_tau")
        + _rows(zero, "jiang_zero")
    )
    selected = min(rows, key=lambda row: float(row["mean_validation_loss"]))
    rates = sorted({float(row["learning_rate"]) for row in rows})
    rate_index = rates.index(float(selected["learning_rate"]))
    eta_interior = 0 < rate_index < len(rates) - 1
    tau: Optional[float] = selected["weight_decay_tau_ema"]
    if tau is None:
        mode = "zero"
        finite_tau_interior = None
    else:
        mode = "finite_tau_ema"
        taus = sorted(
            {
                float(row["weight_decay_tau_ema"])
                for row in rows
                if row["weight_decay_tau_ema"] is not None
            }
        )
        tau_index = taus.index(tau)
        finite_tau_interior = 0 < tau_index < len(taus) - 1
    if tau is None:
        expected_source = "jiang_zero"
        source_selection = zero
    elif tau <= 0.5628:
        expected_source = "jiang_base_finite_tau"
        source_selection = base
    else:
        expected_source = "jiang_expanded_finite_tau"
        source_selection = expanded
    gates = {
        "rho32_preregistration_passed": (
            prereg.get("status") == "preregistered_rho32_correction"
            and all(_object(prereg.get("gates"), "preregistration gates").values())
        ),
        "base_plan_matches_preregistration": (
            base.get("plan_fingerprint")
            == prereg["plans"]["jiang_base_finite_tau"]["plan_fingerprint"]
        ),
        "expanded_plan_matches_preregistration": (
            expanded.get("plan_fingerprint")
            == prereg["plans"]["jiang_expanded_finite_tau"]["plan_fingerprint"]
        ),
        "zero_plan_matches_preregistration": (
            zero.get("plan_fingerprint")
            == prereg["plans"]["jiang_zero"]["plan_fingerprint"]
        ),
        "selected_source_matches_decay_mode": selected["source"] == expected_source,
        "selected_source_has_same_local_optimum": (
            float(source_selection["selected_learning_rate"])
            == float(selected["learning_rate"])
            and source_selection.get("selected_weight_decay_tau_ema") == tau
        ),
        "learning_rate_optimum_is_interior": eta_interior,
        "finite_tau_optimum_is_interior_or_zero_selected": (
            tau is None or finite_tau_interior is True
        ),
    }
    passed = all(gates.values())
    payload = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "plan_fingerprint": source_selection["plan_fingerprint"],
        "selected_source": selected["source"],
        "selected_learning_rate": selected["learning_rate"],
        "selected_weight_decay_mode": mode,
        "selected_weight_decay_tau_ema": tau,
        "mean_validation_loss": selected["mean_validation_loss"],
        "sem_validation_loss": selected["sem_validation_loss"],
        "seed_losses": selected["seed_losses"],
        "learning_rate_optimum_is_interior": eta_interior,
        "finite_tau_optimum_is_interior": finite_tau_interior,
        "optimum_is_interior": eta_interior and (
            tau is None or finite_tau_interior is True
        ),
        "grid": sorted(
            rows,
            key=lambda row: (
                float(row["learning_rate"]),
                math.inf
                if row["weight_decay_tau_ema"] is None
                else float(row["weight_decay_tau_ema"]),
            ),
        ),
        "gates": gates,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit("rho=32 selection failed gates: " + ", ".join(failed))


if __name__ == "__main__":
    main()
