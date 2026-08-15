#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from ai_theorist.autoscaler.study import atomic_write_json


EXPECTED_TAU_GRID = [0.035175, 0.07035, 0.1407, 0.2814, 0.5628]


def _object(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return deepcopy(dict(value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive the exact-zero AdamW arm from the reviewed rho=32 config."
    )
    parser.add_argument("finite_config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    finite = _object(json.loads(args.finite_config.read_text()), "finite config")
    architecture = _object(finite.get("architecture"), "architecture")
    ladder = _object(finite.get("ladder"), "ladder")
    optimizer = _object(finite.get("optimizer"), "optimizer")
    reference = (
        architecture.get("reference_depth"),
        architecture.get("reference_hidden_width"),
        architecture.get("reference_residual_width"),
    )
    if architecture.get("block_type") != "jiang_chizat_transformer":
        raise ValueError("rho=32 zero-decay derivation requires Jiang-Chizat")
    if reference != (2, 1024, 64) or float(ladder.get("rho_lm_over_d")) != 32.0:
        raise ValueError("reviewed rho=32 reference must be (L0,M0,D0)=(2,1024,64)")
    if optimizer.get("name") != "adamw":
        raise ValueError("zero-decay arm requires AdamW")
    if optimizer.get("weight_decay_tau_ema_grid") != EXPECTED_TAU_GRID:
        raise ValueError("finite tau_EMA grid differs from the reviewed broad grid")

    zero = deepcopy(finite)
    zero_optimizer = _object(zero["optimizer"], "zero optimizer")
    zero_optimizer.pop("weight_decay_tau_ema_grid")
    zero_optimizer.pop("weight_decay_tau_ema", None)
    zero_optimizer["weight_decay"] = 0.0
    zero["optimizer"] = zero_optimizer
    zero["campaign_label"] = "jiang_rho32_exact_zero_weight_decay"
    atomic_write_json(args.output, zero)
    print(
        json.dumps(
            {
                "status": "prepared",
                "source": str(args.finite_config),
                "output": str(args.output),
                "rho_lm_over_d": 32.0,
                "reference": {"L0": 2, "M0": 1024, "D0": 64},
                "weight_decay": 0.0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
