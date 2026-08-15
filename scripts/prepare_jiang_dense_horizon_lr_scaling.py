#!/usr/bin/env python3
"""Preregister target-horizon T^-1/3 learning-rate transfer runs."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

from ai_theorist.autoscaler.study import atomic_write_json
from ai_theorist.autoscaler.tokenization import token_stream_identity

from prepare_jiang_dense_horizon_scaling import (
    EXPECTED_1B_GEOMETRY,
    EXPECTED_300M_GEOMETRY,
    EXPECTED_TOKENIZER_FINGERPRINT,
    _compile_horizon,
    _load,
    _record,
)


HORIZON_EXPONENT = -1.0 / 3.0


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scaled_eta(source_eta: float, target_tpp: float) -> float:
    return source_eta * (target_tpp / 10.0) ** HORIZON_EXPONENT


def _validation_contract(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"sha256": str(row["sha256"]), "tokens": int(row["tokens"])}
        for row in manifest["splits"]["validation"]["shards"]
    ]


def _training_contract(manifest: Mapping[str, Any]) -> list[tuple[str, int]]:
    return [
        (str(row["sha256"]), int(row["tokens"]))
        for row in manifest["splits"]["train"]["shards"]
    ]


def _optimizer_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    optimizer = config["optimizer"]
    return {
        "name": optimizer["name"],
        "beta1": float(optimizer["beta1"]),
        "beta2": float(optimizer["beta2"]),
        "epsilon": float(optimizer["epsilon"]),
        "weight_decay": float(optimizer["weight_decay"]),
        "fused": bool(optimizer["fused"]),
        "learning_rate_multipliers": {
            str(key): float(value)
            for key, value in optimizer["learning_rate_multipliers"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_300m_root", type=Path)
    parser.add_argument("source_1b_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("failed_constant_lr_root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    revealed = list(args.output_root.glob("**/ladder-shard-*.json"))
    if revealed:
        raise ValueError("target-horizon outcomes already exist; preregistration is frozen")

    manifest_path = args.manifest.resolve()
    receipt_path = args.receipt.resolve()
    identity = token_stream_identity(
        manifest_path,
        verification_receipt_path=receipt_path,
    )
    receipt_payload = _load(receipt_path)
    if identity["tokenizer_fingerprint"] != EXPECTED_TOKENIZER_FINGERPRINT:
        raise ValueError("the pinned Mistral tokenizer fingerprint changed")

    source_300_config_path = args.source_300m_root / "ten-x" / "config.json"
    source_300_result_path = args.source_300m_root / "result.json"
    source_1b_config_path = args.source_1b_root / "config.json"
    source_1b_result_path = args.source_1b_root / "result.json"
    source_300_config = _load(source_300_config_path)
    source_1b_config = _load(source_1b_config_path)
    source_300_record = _record(_load(source_300_result_path), "300m")
    source_1b_record = _record(_load(source_1b_result_path), "1b")
    source_300_eta = float(source_300_record["optimizer"]["learning_rate"])
    source_1b_eta = float(source_1b_record["optimizer"]["learning_rate"])

    specs = (
        (
            "dense-300m-20tpp",
            "dense_300m_20tpp",
            source_300_config,
            source_300_record,
            20.0,
            [10.0, 15.0, 20.0],
            EXPECTED_300M_GEOMETRY,
            _scaled_eta(source_300_eta, 20.0),
        ),
        (
            "dense-300m-40tpp",
            "dense_300m_40tpp",
            source_300_config,
            source_300_record,
            40.0,
            [10.0, 15.0, 20.0, 30.0, 40.0],
            EXPECTED_300M_GEOMETRY,
            _scaled_eta(source_300_eta, 40.0),
        ),
        (
            "dense-1b-20tpp",
            "dense_1b_20tpp",
            source_1b_config,
            source_1b_record,
            20.0,
            [10.0, 15.0, 20.0],
            EXPECTED_1B_GEOMETRY,
            _scaled_eta(source_1b_eta, 20.0),
        ),
    )
    campaigns: dict[str, dict[str, Any]] = {}
    compiled: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for (
        label,
        key,
        source_config,
        source_record,
        target_tpp,
        retained_tpp,
        geometry,
        selected_eta,
    ) in specs:
        config, plan, campaign = _compile_horizon(
            source_config=source_config,
            source_record=source_record,
            manifest=manifest_path,
            receipt=receipt_path,
            total_tpp=target_tpp,
            retained_tpp=retained_tpp,
            expected_geometry=geometry,
            selected_learning_rate=selected_eta,
        )
        campaign.update(
            {
                "target_tokens_per_parameter": target_tpp,
                "horizon_learning_rate_exponent": HORIZON_EXPONENT,
                "horizon_learning_rate_factor": (target_tpp / 10.0)
                ** HORIZON_EXPONENT,
                "learning_rate_rule": "eta_T = eta_10 * (T / 10)^(-1/3)",
                "optimizer_contract": _optimizer_contract(config),
                "plan_fingerprint": plan["fingerprint"],
            }
        )
        campaigns[key] = campaign
        compiled.append((label, config, plan))

    args.output_root.mkdir(parents=True, exist_ok=True)
    for label, config, plan in compiled:
        target = args.output_root / label
        target.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target / "config.json", config)
        atomic_write_json(target / "plan.json", plan)

    repo_root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    new_manifest = _load(manifest_path)
    old_300_manifest = _load(
        Path(source_300_config["dataset"]["token_stream_manifest_path"])
    )
    old_1b_manifest = _load(
        Path(source_1b_config["dataset"]["token_stream_manifest_path"])
    )
    new_train = _training_contract(new_manifest)
    old_300_train = _training_contract(old_300_manifest)
    old_1b_train = _training_contract(old_1b_manifest)
    failed_receipt = args.failed_constant_lr_root / "halt-receipt.json"
    failed_pre = args.failed_constant_lr_root / "preregistration.json"
    if not failed_receipt.is_file() or not failed_pre.is_file():
        raise ValueError("the constant-LR failure control is not durably bound")

    expected_etas = {
        "dense_300m_20tpp": _scaled_eta(source_300_eta, 20.0),
        "dense_300m_40tpp": _scaled_eta(source_300_eta, 40.0),
        "dense_1b_20tpp": _scaled_eta(source_1b_eta, 20.0),
    }
    gates = {
        "verified_immutable_token_stream": receipt_payload.get("status") == "passed",
        "same_pinned_mistral_tokenizer": identity["tokenizer_fingerprint"]
        == EXPECTED_TOKENIZER_FINGERPRINT,
        "identical_validation_contract": (
            _validation_contract(new_manifest)
            == _validation_contract(old_300_manifest)
            == _validation_contract(old_1b_manifest)
        ),
        "old_training_streams_are_exact_prefixes": (
            new_train[: len(old_300_train)] == old_300_train
            and new_train[: len(old_1b_train)] == old_1b_train
        ),
        "enough_unique_tokens_for_every_run": int(identity["training_tokens"])
        >= max(int(value["target"]["presented_tokens"]) for value in campaigns.values()),
        "exact_horizon_learning_rate_rule": all(
            math.isclose(
                float(campaigns[key]["selected_learning_rate"]),
                eta,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for key, eta in expected_etas.items()
        ),
        "all_parameter_group_multipliers_preserved": (
            campaigns["dense_300m_20tpp"]["optimizer_contract"]
            ["learning_rate_multipliers"]
            == campaigns["dense_300m_40tpp"]["optimizer_contract"]
            ["learning_rate_multipliers"]
            == _optimizer_contract(source_300_config)["learning_rate_multipliers"]
            and campaigns["dense_1b_20tpp"]["optimizer_contract"]
            ["learning_rate_multipliers"]
            == _optimizer_contract(source_1b_config)["learning_rate_multipliers"]
        ),
        "adamw_constants_and_zero_decay_preserved": all(
            value["optimizer_contract"]["name"] == "adamw"
            and value["optimizer_contract"]["beta1"] == 0.9
            and value["optimizer_contract"]["beta2"] == 0.95
            and value["optimizer_contract"]["epsilon"] == 1e-12
            and value["optimizer_contract"]["weight_decay"] == 0.0
            for value in campaigns.values()
        ),
        "warmup_fixed_in_presented_token_coordinates": all(
            value["schedule_contract"]["fixed_in_presented_token_coordinates"]
            and value["schedule_contract"]["source_warmup_steps"]
            == value["schedule_contract"]["new_warmup_steps"]
            for value in campaigns.values()
        ),
        "constant_lr_failure_control_bound": _load(failed_receipt).get("status")
        == "halted_after_observed_long_horizon_instability",
        "new_outcomes_unseen": not revealed,
    }
    if not all(gates.values()):
        raise ValueError(
            "horizon LR preregistration failed: "
            + ", ".join(name for name, passed in gates.items() if not passed)
        )

    preregistration = {
        "schema_version": 1,
        "status": "preregistered",
        "scientific_status": (
            "preregistered exploratory single-seed target-horizon learning-rate "
            "transfer test; eta scales as T^-1/3 and every architecture-defined "
            "parameter-group multiplier remains frozen"
        ),
        "repo_commit": commit,
        "dataset_identity": identity,
        "horizon_learning_rate_rule": {
            "formula": "eta_T = eta_10 * (T / 10)^(-1/3)",
            "exponent": HORIZON_EXPONENT,
            "reference_tokens_per_parameter": 10.0,
            "reference_300m_eta": source_300_eta,
            "reference_1b_eta": source_1b_eta,
        },
        "sources": {
            "300m_config_sha256": _sha256(source_300_config_path),
            "300m_result_sha256": _sha256(source_300_result_path),
            "1b_config_sha256": _sha256(source_1b_config_path),
            "1b_result_sha256": _sha256(source_1b_result_path),
            "constant_lr_failure_preregistration_sha256": _sha256(failed_pre),
            "constant_lr_failure_halt_receipt_sha256": _sha256(failed_receipt),
        },
        "campaigns": campaigns,
        "adaptive_execution_gates": {
            "after_300m_20tpp": (
                "all retained losses finite; no post-10-TPP loss exceeds 1.10x "
                "the 10-TPP loss; 20-TPP loss is below 10-TPP loss"
            ),
            "after_300m_40tpp": (
                "all retained losses finite; no post-10-TPP loss exceeds 1.10x "
                "the 10-TPP loss; 40-TPP loss is below both 10- and 20-TPP losses"
            ),
            "after_1b_20tpp": (
                "all retained losses finite; no post-10-TPP loss exceeds 1.10x "
                "the 10-TPP loss; 20-TPP loss is below 10-TPP loss"
            ),
        },
        "execution_order": [
            "fresh_300m_20tpp_from_initialization",
            "apply_preregistered_300m_20tpp_stability_gate",
            "fresh_300m_40tpp_from_initialization",
            "apply_preregistered_300m_40tpp_stability_gate",
            "fresh_1b_20tpp_from_initialization",
            "apply_preregistered_1b_20tpp_stability_gate",
        ],
        "gates": gates,
    }
    atomic_write_json(args.output_root / "preregistration.json", preregistration)
    print(json.dumps(preregistration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
