#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from ai_theorist.autoscaler.study import atomic_write_json


JIANG_EXPANDED_TAU_GRID = [0.5628, 1.1256, 2.2512, 4.5024, 9.0048]


def _object(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return deepcopy(dict(value))


def _read(path: Path, name: str) -> Dict[str, Any]:
    return _object(json.loads(path.read_text()), name)


def _zero_decay(config: Mapping[str, Any], label: str) -> Dict[str, Any]:
    result = deepcopy(dict(config))
    optimizer = _object(result.get("optimizer"), f"{label} optimizer")
    if optimizer.get("name") != "adamw":
        raise ValueError(f"{label} zero-decay control requires AdamW")
    optimizer.pop("weight_decay_tau_ema", None)
    optimizer.pop("weight_decay_tau_ema_grid", None)
    optimizer["weight_decay"] = 0.0
    result["optimizer"] = optimizer
    result["adaptive_extension_label"] = f"{label}_exact_zero_weight_decay"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the adaptive finite-tau expansion and exact zero-decay controls."
    )
    parser.add_argument("jiang_config", type=Path)
    parser.add_argument("completep_config", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    jiang = _read(args.jiang_config, "Jiang config")
    completep = _read(args.completep_config, "CompleteP config")
    jiang_optimizer = _object(jiang.get("optimizer"), "Jiang optimizer")
    original_tau_grid = jiang_optimizer.get("weight_decay_tau_ema_grid")
    if original_tau_grid != [0.035175, 0.07035, 0.1407, 0.2814, 0.5628]:
        raise ValueError("Jiang base config does not have the reviewed tau_EMA grid")

    expanded = deepcopy(jiang)
    expanded_optimizer = _object(expanded["optimizer"], "expanded Jiang optimizer")
    expanded_optimizer["weight_decay_tau_ema_grid"] = JIANG_EXPANDED_TAU_GRID
    expanded["optimizer"] = expanded_optimizer
    expanded["adaptive_extension_label"] = "jiang_finite_tau_boundary_expansion"

    outputs = {
        "jiang-expanded.json": expanded,
        "jiang-zero.json": _zero_decay(jiang, "jiang"),
        "completep-zero.json": _zero_decay(completep, "completep"),
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for filename, payload in outputs.items():
        atomic_write_json(args.output_directory / filename, payload)
    print(
        json.dumps(
            {
                "status": "prepared",
                "output_directory": str(args.output_directory),
                "files": sorted(outputs),
                "jiang_expanded_tau_grid": JIANG_EXPANDED_TAU_GRID,
                "zero_decay_representation": "AdamW weight_decay=0.0 (tau_EMA=infinity)",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
