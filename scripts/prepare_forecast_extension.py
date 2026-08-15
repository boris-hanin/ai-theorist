#!/usr/bin/env python3
"""Bind and preregister one frozen-rule upper-rung forecast extension."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Mapping

from ai_theorist.autoscaler.forecast_campaigns import (
    bind_real_text_scaling_config,
    compile_real_text_scaling_plan,
)
from ai_theorist.autoscaler.scaling import fit_scaling_ensemble
from ai_theorist.autoscaler.study import atomic_write_json
from ai_theorist.autoscaler.tokenization import load_token_stream_manifest


def _read_object(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validation_contract(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "packing": manifest["packing"],
        "split": manifest["splits"]["validation"],
        "tokenizer_fingerprint": manifest["tokenizer_fingerprint"],
    }


def _require_equal(label: str, current: Any, expected: Any) -> None:
    if current != expected:
        raise ValueError(f"extension prefix lock failed for {label}")


def _verify_prefix_lock(
    *,
    parent_config: Mapping[str, Any],
    parent_plan: Mapping[str, Any],
    parent_result: Mapping[str, Any],
    parent_selection: Mapping[str, Any],
    extension_config: Mapping[str, Any],
    extension_plan: Mapping[str, Any],
    parent_manifest: Mapping[str, Any],
    extension_manifest: Mapping[str, Any],
) -> None:
    contract = extension_plan["extension_contract"]
    _require_equal(
        "parent plan fingerprint",
        contract["parent_plan_fingerprint"],
        parent_plan["fingerprint"],
    )
    _require_equal(
        "parent result plan fingerprint",
        parent_result["plan_fingerprint"],
        parent_plan["fingerprint"],
    )
    _require_equal(
        "parent dataset fingerprint",
        contract["parent_dataset_fingerprint"],
        parent_plan["dataset_identity"]["fingerprint"],
    )
    _require_equal(
        "selected learning rate",
        float(contract["selected_learning_rate"]),
        float(parent_selection["selected_learning_rate"]),
    )
    if not bool(parent_selection.get("optimum_is_interior")):
        raise ValueError("parent learning-rate selection is not interior")

    for key in (
        "architecture",
        "optimizer",
        "schedule",
        "batch_examples",
        "validation_examples",
        "validation_microbatch_examples",
        "validation_interval_steps",
        "runtime",
    ):
        _require_equal(key, extension_config[key], parent_config[key])
    _require_equal(
        "architecture contract",
        extension_plan["architecture_contract"],
        parent_plan["architecture_contract"],
    )
    parent_scales = list(parent_plan["scales"])
    extension_scales = list(extension_plan["scales"])
    if len(extension_scales) != len(parent_scales) + 1:
        raise ValueError("extension plan must add exactly one scale")
    geometry_keys = {
        "name",
        "block_type",
        "target_parameters",
        "parameters",
        "depth",
        "width",
        "hidden_width",
        "num_heads",
        "presented_tokens",
        "optimizer_steps",
        "tokens_per_parameter",
        "rho_lm_over_d",
        "rho_relative_error",
    }
    for index, parent_scale in enumerate(parent_scales):
        extension_scale = extension_scales[index]
        _require_equal(
            f"scale {parent_scale['name']} geometry",
            {key: extension_scale[key] for key in geometry_keys},
            {key: parent_scale[key] for key in geometry_keys},
        )
    target = extension_scales[-1]
    _require_equal("target scale", target["name"], contract["target_scale"])
    _require_equal(
        "target parameter count",
        int(target["parameters"]),
        int(contract["expected_target_parameters"]),
    )
    if not bool(target["heldout"]):
        raise ValueError("extension target scale must remain held out")
    if float(target["rho_relative_error"]) != 0.0:
        raise ValueError("extension target must preserve L*M/D exactly")
    if int(extension_plan["dataset_identity"]["training_tokens"]) < int(
        target["presented_tokens"]
    ):
        raise ValueError("extension token stream is smaller than the training horizon")

    _require_equal(
        "tokenizer fingerprint",
        extension_manifest["tokenizer_fingerprint"],
        parent_manifest["tokenizer_fingerprint"],
    )
    _require_equal(
        "validation token contract",
        _validation_contract(extension_manifest),
        _validation_contract(parent_manifest),
    )


def _frozen_prediction(
    parent_result: Mapping[str, Any],
    parent_config: Mapping[str, Any],
    target: Mapping[str, Any],
) -> Dict[str, Any]:
    rows = list(parent_result["scales"])
    ladder = parent_config["ladder"]
    axis = str(parent_result.get("fit_parameter_axis", "parameters"))
    if axis not in {"parameters", "non_embedding_parameters"}:
        raise ValueError("parent fit parameter axis is unsupported")
    return fit_scaling_ensemble(
        [row[axis] for row in rows],
        [row["mean_validation_loss"] for row in rows],
        [row["sem_validation_loss"] for row in rows],
        target_size=float(target[axis]),
        maximum_extrapolation_factor=float(
            ladder.get("maximum_extrapolation_factor", 10.0)
        ),
        maximum_family_spread=float(ladder.get("maximum_family_spread", 0.08)),
        maximum_backtest_relative_error=float(
            ladder.get("maximum_backtest_relative_error", 0.10)
        ),
        bootstrap_samples=int(parent_config.get("bootstrap_samples", 200)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("parent_run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    template = _read_object(args.template)
    parent_plan = _read_object(args.parent_run / "plan.json")
    parent_config = _read_object(args.parent_run / "bound-config.json")
    parent_result_path = args.parent_run / "aggregate" / "result.json"
    parent_result = _read_object(parent_result_path)
    parent_selection = _read_object(args.parent_run / "reference-selection.json")
    contract = template.get("extension_contract")
    if not isinstance(contract, dict):
        raise ValueError("extension template is missing extension_contract")
    _require_equal(
        "parent aggregate SHA-256",
        _hash_file(parent_result_path),
        contract["parent_aggregate_sha256"],
    )

    bound_config, binding = bind_real_text_scaling_config(template, args.manifest)
    plan = compile_real_text_scaling_plan(bound_config)
    parent_manifest = load_token_stream_manifest(
        Path(parent_config["dataset"]["token_stream_manifest_path"]),
        verify_files=True,
    )
    extension_manifest = load_token_stream_manifest(args.manifest, verify_files=True)
    _verify_prefix_lock(
        parent_config=parent_config,
        parent_plan=parent_plan,
        parent_result=parent_result,
        parent_selection=parent_selection,
        extension_config=bound_config,
        extension_plan=plan,
        parent_manifest=parent_manifest,
        extension_manifest=extension_manifest,
    )
    target = plan["scales"][-1]
    prediction = _frozen_prediction(parent_result, parent_config, target)
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    preregistration_core: Dict[str, Any] = {
        "schema_version": 1,
        "status": "preregistered",
        "code_commit": code_commit,
        "parent_plan_fingerprint": parent_plan["fingerprint"],
        "parent_aggregate_sha256": _hash_file(parent_result_path),
        "parent_dataset_fingerprint": parent_plan["dataset_identity"]["fingerprint"],
        "extension_plan_fingerprint": plan["fingerprint"],
        "extension_dataset_fingerprint": plan["dataset_identity"]["fingerprint"],
        "validation_contract_sha256": _fingerprint(
            _validation_contract(extension_manifest)
        ),
        "selected_learning_rate": float(contract["selected_learning_rate"]),
        "seed": int(contract["target_seed"]),
        "target": target,
        "frozen_prediction": prediction,
        "fit_parameter_axis": parent_result.get(
            "fit_parameter_axis", "parameters"
        ),
        "evaluation_contract": {
            "primary_metric": "final_validation_loss",
            "report_relative_error_to_exploratory_prediction": True,
            "report_95_percent_interval_coverage": True,
            "no_retuning_after_reveal": True,
            "wrong_lr_control": False,
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    prereg_path = args.output / "preregistration.json"
    if prereg_path.is_file():
        preregistration = _read_object(prereg_path)
        existing_core = dict(preregistration)
        existing_fingerprint = existing_core.pop("fingerprint", None)
        existing_core.pop("created_at", None)
        unsigned = dict(preregistration)
        unsigned.pop("fingerprint", None)
        if (
            existing_core != preregistration_core
            or existing_fingerprint != _fingerprint(unsigned)
        ):
            raise ValueError("an incompatible extension preregistration already exists")
    else:
        preregistration = {
            **preregistration_core,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        preregistration["fingerprint"] = _fingerprint(preregistration)
        atomic_write_json(prereg_path, preregistration)
    atomic_write_json(args.output / "bound-config.json", bound_config)
    atomic_write_json(args.output / "binding.json", binding)
    atomic_write_json(args.output / "plan.json", plan)
    print(
        json.dumps(
            {
                "status": "preregistered",
                "output": str(args.output.resolve()),
                "plan_fingerprint": plan["fingerprint"],
                "dataset_fingerprint": plan["dataset_identity"]["fingerprint"],
                "validation_contract_sha256": preregistration[
                    "validation_contract_sha256"
                ],
                "target": target,
                "frozen_exploratory_prediction": prediction[
                    "exploratory_prediction"
                ],
                "prediction_interval_95": prediction["prediction_interval_95"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
