#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ai_theorist.autoscaler.study import atomic_write_json


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preregistration", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prereg = _load(args.preregistration)
    selection = _load(args.selection)
    if prereg.get("status") != "preregistered":
        raise ValueError("shape ablation was not preregistered")
    primary = prereg["primary_estimand"]
    frozen_eta = float(primary["frozen_learning_rate"])
    grid = selection["grid"]
    cell = next(
        row for row in grid if math.isclose(float(row["learning_rate"]), frozen_eta)
    )
    selected_eta = float(selection["selected_learning_rate"])
    selected_cell = next(
        row
        for row in grid
        if math.isclose(float(row["learning_rate"]), selected_eta)
    )
    baseline_loss = float(primary["baseline_mean_validation_loss"])
    deep_loss = float(cell["mean_validation_loss"])
    payload = {
        "schema_version": 1,
        "status": "completed",
        "scientific_status": prereg["scientific_status"],
        "primary_transfer_test": {
            "learning_rate": frozen_eta,
            "baseline_8_layer_mean_validation_loss": baseline_loss,
            "deep_16_layer_mean_validation_loss": deep_loss,
            "loss_delta_deep_minus_baseline": deep_loss - baseline_loss,
            "deep_seed_losses": cell["seed_losses"],
            "improved": deep_loss < baseline_loss,
        },
        "learning_rate_diagnostic": {
            "selected_learning_rate": selected_eta,
            "selected_mean_validation_loss": selected_cell["mean_validation_loss"],
            "selected_seed_losses": selected_cell["seed_losses"],
            "optimum_is_interior": selection["optimum_is_interior"],
            "frozen_eta_is_selected": math.isclose(selected_eta, frozen_eta),
            "grid": grid,
        },
        "target_geometry": prereg["target_geometry"],
        "plan_fingerprint": prereg["plan_fingerprint"],
        "dataset_fingerprint": prereg["dataset_fingerprint"],
        "tokenizer_fingerprint": prereg["tokenizer_fingerprint"],
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
