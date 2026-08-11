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
        description="Evaluate the adaptive finite-tau/zero-decay 100M pair."
    )
    parser.add_argument("preregistration", type=Path)
    parser.add_argument("decision", type=Path)
    parser.add_argument("jiang_result", type=Path)
    parser.add_argument("completep_result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prereg = _read(args.preregistration, "adaptive preregistration")
    decision = _read(args.decision, "adaptive decision")
    jiang = _read(args.jiang_result, "Jiang result")
    completep = _read(args.completep_result, "CompleteP result")
    jiang_selection = _selection(jiang)
    completep_selection = _selection(completep)
    jiang_target = _object(jiang["scales"][-1], "Jiang target")
    completep_target = _object(completep.get("target"), "CompleteP target")

    def selection_matches(name: str, observed: Mapping[str, Any]) -> bool:
        expected = decision[name]
        return (
            float(observed["selected_learning_rate"])
            == float(expected["selected_learning_rate"])
            and observed.get("selected_weight_decay_tau_ema")
            == expected.get("selected_weight_decay_tau_ema")
        )

    expected_jiang_plan = prereg["plans"][
        decision["jiang"]["selected_source"]
    ]["plan_fingerprint"]
    expected_completep_plan = prereg["plans"][
        decision["completep"]["selected_source"]
    ]["plan_fingerprint"]
    jiang_loss = float(jiang_target["mean_validation_loss"])
    completep_loss = float(completep_target["mean_validation_loss"])
    gates = {
        "adaptive_preregistration_valid": (
            prereg.get("status") == "preregistered_adaptive_extension"
            and all(_object(prereg.get("gates"), "preregistration gates").values())
        ),
        "adaptive_decision_passed": decision.get("status") == "passed"
        and all(_object(decision.get("gates"), "decision gates").values()),
        "jiang_result_completed": jiang.get("status") == "completed",
        "completep_result_passed": completep.get("status") == "passed",
        "jiang_plan_matches_selected_source": (
            jiang.get("plan_fingerprint") == expected_jiang_plan
        ),
        "completep_plan_matches_selected_source": (
            completep.get("plan_fingerprint") == expected_completep_plan
        ),
        "jiang_selection_matches_decision": selection_matches(
            "jiang", jiang_selection
        ),
        "completep_selection_matches_decision": selection_matches(
            "completep", completep_selection
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
        "matched_parameter_count_within_one_percent": abs(
            int(completep_target["parameters"])
            / int(jiang_target["parameters"])
            - 1.0
        )
        <= 0.01,
        "same_tokens_per_parameter": abs(
            float(completep_target["tokens_per_parameter"])
            - float(jiang_target["tokens_per_parameter"])
        )
        <= 0.001,
        "finite_target_losses": math.isfinite(jiang_loss)
        and math.isfinite(completep_loss),
    }
    passed = all(gates.values())
    payload = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "claim_scope": prereg["claim_scope"],
        "source_hashes": {
            "preregistration": _sha256(args.preregistration),
            "decision": _sha256(args.decision),
            "jiang_result": _sha256(args.jiang_result),
            "completep_result": _sha256(args.completep_result),
        },
        "jiang": {
            "selected_weight_decay_mode": decision["jiang"][
                "selected_weight_decay_mode"
            ],
            "selected_learning_rate": jiang_selection["selected_learning_rate"],
            "selected_weight_decay_tau_ema": jiang_selection.get(
                "selected_weight_decay_tau_ema"
            ),
            "parameters": int(jiang_target["parameters"]),
            "mean_validation_loss": jiang_loss,
            "seed_losses": jiang_target["seed_losses"],
        },
        "completep": {
            "selected_weight_decay_mode": decision["completep"][
                "selected_weight_decay_mode"
            ],
            "selected_learning_rate": completep_selection[
                "selected_learning_rate"
            ],
            "selected_weight_decay_tau_ema": completep_selection.get(
                "selected_weight_decay_tau_ema"
            ),
            "parameters": int(completep_target["parameters"]),
            "mean_validation_loss": completep_loss,
            "seed_losses": completep_target["seed_losses"],
        },
        "comparison": {
            "validation_loss_completep_minus_jiang": completep_loss - jiang_loss,
            "perplexity_ratio_completep_over_jiang": math.exp(
                completep_loss - jiang_loss
            ),
        },
        "gates": gates,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        failed_gates = [name for name, value in gates.items() if not value]
        raise SystemExit("adaptive pair failed gates: " + ", ".join(failed_gates))


if __name__ == "__main__":
    main()
