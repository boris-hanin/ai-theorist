#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

from ai_theorist.autoscaler.study import atomic_write_json


def _object(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _read(path: Path, name: str) -> Dict[str, Any]:
    return _object(json.loads(path.read_text()), name)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _selection(result: Mapping[str, Any]) -> Dict[str, Any]:
    return _object(
        result.get("reference_tuning", result.get("reference_selection")),
        "result reference selection",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package the corrected rho=32 Jiang and rerun CompleteP ladders."
    )
    parser.add_argument("preregistration", type=Path)
    parser.add_argument("jiang_decision", type=Path)
    parser.add_argument("jiang_result", type=Path)
    parser.add_argument("completep_result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prereg = _read(args.preregistration, "preregistration")
    decision = _read(args.jiang_decision, "Jiang decision")
    jiang = _read(args.jiang_result, "Jiang result")
    completep = _read(args.completep_result, "CompleteP result")
    jiang_selection = _selection(jiang)
    completep_selection = _selection(completep)
    jiang_target = _object(jiang["scales"][-1], "Jiang target")
    completep_target = _object(completep["target"], "CompleteP target")
    expected_jiang_plan = prereg["plans"][decision["selected_source"]][
        "plan_fingerprint"
    ]
    jiang_loss = float(jiang_target["mean_validation_loss"])
    completep_loss = float(completep_target["mean_validation_loss"])
    completep_frozen = prereg["completep_selection"]
    gates = {
        "preregistration_passed": (
            prereg.get("status") == "preregistered_rho32_correction"
            and all(_object(prereg.get("gates"), "preregistration gates").values())
        ),
        "jiang_selection_passed": (
            decision.get("status") == "passed"
            and all(_object(decision.get("gates"), "decision gates").values())
        ),
        "jiang_result_completed": jiang.get("status") == "completed",
        "completep_result_passed": completep.get("status") == "passed",
        "jiang_plan_matches_selected_arm": (
            jiang.get("plan_fingerprint") == expected_jiang_plan
        ),
        "completep_plan_matches_preregistration": (
            completep.get("plan_fingerprint")
            == prereg["plans"]["completep"]["plan_fingerprint"]
        ),
        "jiang_selection_matches_decision": (
            float(jiang_selection["selected_learning_rate"])
            == float(decision["selected_learning_rate"])
            and jiang_selection.get("selected_weight_decay_tau_ema")
            == decision.get("selected_weight_decay_tau_ema")
        ),
        "completep_selection_matches_frozen_evidence": (
            float(completep_selection["selected_learning_rate"])
            == float(completep_frozen["selected_learning_rate"])
            and completep_selection.get("selected_weight_decay_tau_ema")
            == completep_frozen.get("selected_weight_decay_tau_ema")
        ),
        "same_dataset_fingerprint": (
            jiang["dataset"]["fingerprint"]
            == completep["dataset"]["fingerprint"]
            == prereg["dataset_fingerprint"]
        ),
        "same_tokenizer_fingerprint": (
            jiang["dataset"]["tokenizer_fingerprint"]
            == completep["dataset"]["tokenizer_fingerprint"]
            == prereg["tokenizer_fingerprint"]
        ),
        "both_targets_have_three_finite_losses": (
            len(jiang_target["seed_losses"])
            == len(completep_target["seed_losses"])
            == 3
            and math.isfinite(jiang_loss)
            and math.isfinite(completep_loss)
        ),
        "both_targets_use_same_tpp": abs(
            float(jiang_target["tokens_per_parameter"])
            - float(completep_target["tokens_per_parameter"])
        ) <= 0.001,
    }
    passed = all(gates.values())
    payload = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "claim_scope": prereg["claim_scope"],
        "source_hashes": {
            "preregistration": _sha256(args.preregistration),
            "jiang_decision": _sha256(args.jiang_decision),
            "jiang_result": _sha256(args.jiang_result),
            "completep_result": _sha256(args.completep_result),
        },
        "jiang_rho32": {
            "rho_lm_over_d": 32.0,
            "parameters": int(jiang_target["parameters"]),
            "mean_validation_loss": jiang_loss,
            "seed_losses": jiang_target["seed_losses"],
            "selected_learning_rate": decision["selected_learning_rate"],
            "selected_weight_decay_mode": decision[
                "selected_weight_decay_mode"
            ],
            "selected_weight_decay_tau_ema": decision.get(
                "selected_weight_decay_tau_ema"
            ),
        },
        "completep": {
            "parameters": int(completep_target["parameters"]),
            "mean_validation_loss": completep_loss,
            "seed_losses": completep_target["seed_losses"],
            "selected_learning_rate": completep_selection[
                "selected_learning_rate"
            ],
            "selected_weight_decay_tau_ema": completep_selection.get(
                "selected_weight_decay_tau_ema"
            ),
        },
        "comparison_note": (
            "The endpoint parameter counts differ because exact rho=32 plus 64-wide "
            "attention heads discretizes the Jiang ladder; this is a paired evidence "
            "package, not a parameter-matched architecture ranking."
        ),
        "gates": gates,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit("rho=32 pair failed gates: " + ", ".join(failed))


if __name__ == "__main__":
    main()
