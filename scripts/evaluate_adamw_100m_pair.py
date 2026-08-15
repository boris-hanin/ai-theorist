#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ai_theorist.autoscaler.study import atomic_write_json


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _selection(result: Mapping[str, Any]) -> Mapping[str, Any]:
    value = result.get("reference_tuning", result.get("reference_selection"))
    return _object(value, "reference selection")


def _jiang_target(result: Mapping[str, Any]) -> Mapping[str, Any]:
    scales = result.get("scales")
    if not isinstance(scales, list) or not scales:
        raise ValueError("Jiang result has no scales")
    return _object(scales[-1], "Jiang target")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen matched 100M AdamW architecture pair."
    )
    parser.add_argument("preregistration", type=Path)
    parser.add_argument("jiang_result", type=Path)
    parser.add_argument("completep_result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prereg = _object(json.loads(args.preregistration.read_text()), "preregistration")
    jiang = _object(json.loads(args.jiang_result.read_text()), "Jiang result")
    completep = _object(
        json.loads(args.completep_result.read_text()), "CompleteP result"
    )
    jiang_target = _jiang_target(jiang)
    completep_target = _object(completep.get("target"), "CompleteP target")
    jiang_selection = _selection(jiang)
    completep_selection = _selection(completep)
    jiang_loss = float(jiang_target["mean_validation_loss"])
    completep_loss = float(completep_target["mean_validation_loss"])

    gates = {
        "preregistration_passed": (
            prereg.get("status") == "preregistered"
            and all(_object(prereg.get("gates"), "preregistration gates").values())
        ),
        "jiang_result_completed": jiang.get("status") == "completed",
        "completep_result_passed": completep.get("status") == "passed",
        "jiang_plan_matches_preregistration": (
            jiang.get("plan_fingerprint")
            == prereg["jiang"]["plan_fingerprint"]
        ),
        "completep_plan_matches_preregistration": (
            completep.get("plan_fingerprint")
            == prereg["completep"]["plan_fingerprint"]
        ),
        "dataset_fingerprints_match": (
            jiang["dataset"]["fingerprint"]
            == completep["dataset"]["fingerprint"]
            == prereg["dataset_fingerprint"]
        ),
        "tokenizer_fingerprints_match": (
            jiang["dataset"]["tokenizer_fingerprint"]
            == completep["dataset"]["tokenizer_fingerprint"]
            == prereg["tokenizer_fingerprint"]
        ),
        "both_eta_tau_optima_are_interior": (
            jiang_selection.get("optimum_is_interior") is True
            and completep_selection.get("optimum_is_interior") is True
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
        "same_seed_count": (
            len(completep_target["seed_losses"])
            == len(jiang_target["seed_losses"])
            == len(prereg["seeds"])
        ),
        "finite_losses": math.isfinite(jiang_loss) and math.isfinite(completep_loss),
    }
    passed = all(gates.values())
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "claim_scope": prereg["claim_scope"],
        "preregistration_sha256": _sha256(args.preregistration),
        "source_results": {
            "jiang": {
                "path": str(args.jiang_result),
                "sha256": _sha256(args.jiang_result),
            },
            "completep": {
                "path": str(args.completep_result),
                "sha256": _sha256(args.completep_result),
            },
        },
        "jiang": {
            "parameters": int(jiang_target["parameters"]),
            "mean_validation_loss": jiang_loss,
            "seed_losses": jiang_target["seed_losses"],
            "selected_learning_rate": jiang_selection["selected_learning_rate"],
            "selected_weight_decay_tau_ema": jiang_selection[
                "selected_weight_decay_tau_ema"
            ],
        },
        "completep": {
            "parameters": int(completep_target["parameters"]),
            "mean_validation_loss": completep_loss,
            "seed_losses": completep_target["seed_losses"],
            "selected_learning_rate": completep_selection[
                "selected_learning_rate"
            ],
            "selected_weight_decay_tau_ema": completep_selection[
                "selected_weight_decay_tau_ema"
            ],
        },
        "comparison": {
            "validation_loss_completep_minus_jiang": completep_loss - jiang_loss,
            "perplexity_ratio_completep_over_jiang": math.exp(
                completep_loss - jiang_loss
            ),
            "parameter_ratio_completep_over_jiang": (
                int(completep_target["parameters"])
                / int(jiang_target["parameters"])
            ),
        },
        "gates": gates,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        failed_gates = [name for name, value in gates.items() if not value]
        raise SystemExit("matched pair failed gates: " + ", ".join(failed_gates))


if __name__ == "__main__":
    main()
