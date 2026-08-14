#!/usr/bin/env python3
"""Compile the retained-checkpoint 300M/40-TPP and 1B/20-TPP runs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

from ai_theorist.autoscaler.forecast_campaigns import (
    compile_real_text_scaling_plan,
)
from ai_theorist.autoscaler.study import atomic_write_json
from ai_theorist.autoscaler.tokenization import token_stream_identity


EXPECTED_TOKENIZER_FINGERPRINT = (
    "d52f662783555cbf11f6a0cd8af35016652cda033389db471813c7d30f6958c5"
)
EXPECTED_300M_GEOMETRY = (299_177_600, 245_929_600, 8, 1_600, 6_400)
EXPECTED_1B_GEOMETRY = (1_008_531_456, 906_295_296, 8, 3_072, 12_288)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _record(source_result: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    if label == "300m":
        if source_result.get("status") != "completed_topology_unqualified":
            raise ValueError("the preserved 300M/10-TPP source result is incomplete")
        record = source_result.get("ten_x", {}).get("record")
    else:
        if source_result.get("status") != "completed":
            raise ValueError("the preserved 1B/10-TPP source result is incomplete")
        record = source_result.get("record")
    if not isinstance(record, Mapping):
        raise ValueError(f"the {label} source record is missing")
    return record


def _compile_horizon(
    *,
    source_config: Mapping[str, Any],
    source_record: Mapping[str, Any],
    manifest: Path,
    receipt: Path,
    total_tpp: float,
    retained_tpp: list[float],
    expected_geometry: tuple[int, int, int, int, int],
    selected_learning_rate: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = deepcopy(dict(source_config))
    config["dataset"]["token_stream_manifest_path"] = str(manifest.resolve())
    config["dataset"]["token_stream_verification_receipt_path"] = str(
        receipt.resolve()
    )
    config["run_profile"] = "extension"
    config["seeds"] = [int(source_record["seed"])]
    config["exploratory_single_seed"] = True
    config["bootstrap_samples"] = 0
    config["run_negative_control"] = False
    ladder = config["ladder"]
    ladder.pop("optimizer_steps", None)
    ladder["tokens_per_parameter"] = total_tpp
    ladder["target_forecasts"] = []
    ladder["maximum_repetition_ratio"] = 1.0
    runtime = config["runtime"]
    runtime.update(
        {
            "precision": "bf16",
            "attention_backend": "flash",
            "distributed": "ddp",
            "num_processes": 8,
            "gradient_accumulation_steps": 32,
            "activation_checkpointing": False,
            "checkpoint_interval_steps": 0,
            "checkpoint_interval_seconds": 900,
            "resume": True,
            "retained_checkpoint_tokens_per_parameter": retained_tpp,
        }
    )
    source_eta = float(source_record["optimizer"]["learning_rate"])
    if source_eta not in {
        float(value) for value in config["optimizer"]["learning_rates"]
    }:
        raise ValueError("source learning rate is absent from the horizon config")
    selected_eta = (
        source_eta
        if selected_learning_rate is None
        else float(selected_learning_rate)
    )
    if not math.isfinite(selected_eta) or selected_eta <= 0.0:
        raise ValueError("selected horizon learning rate must be positive and finite")
    config["optimizer"]["learning_rates"] = [selected_eta]
    if float(source_record["optimizer"]["weight_decay"]) != 0.0:
        raise ValueError("dense horizon extension requires the frozen zero decay")

    source_steps = int(source_record["optimizer_steps"])
    source_warmup_steps = int(math.ceil(0.5 * source_steps))
    config["schedule"] = {
        "family": "linear_warmup_constant",
        "warmup_fraction": 0.1,
    }
    provisional = compile_real_text_scaling_plan(config)
    target_steps = int(provisional["scales"][-1]["optimizer_steps"])
    warmup_fraction = source_warmup_steps / target_steps
    config["schedule"] = {
        "family": "linear_warmup_constant",
        "warmup_fraction": warmup_fraction,
    }
    config["validation_interval_steps"] = max(1, target_steps // 8)
    plan = compile_real_text_scaling_plan(config)
    target = plan["scales"][-1]
    geometry = (
        int(target["parameters"]),
        int(target["non_embedding_parameters"]),
        int(target["depth"]),
        int(target["width"]),
        int(target["hidden_width"]),
    )
    if geometry != expected_geometry:
        raise ValueError(f"compiled geometry {geometry} != {expected_geometry}")
    if not math.isclose(float(target["rho_lm_over_d"]), 32.0):
        raise ValueError("dense horizon geometry is not exact rho=32")
    if math.ceil(warmup_fraction * target_steps) != source_warmup_steps:
        raise ValueError("fixed source warmup did not survive horizon compilation")
    checkpoints = plan["retained_checkpoint_contract"]["scales"][target["name"]]
    if [row["requested_tokens_per_parameter"] for row in checkpoints] != retained_tpp:
        raise ValueError("retained horizon contract changed during compilation")
    if int(target["presented_tokens"]) > int(
        plan["dataset_identity"]["training_tokens"]
    ):
        raise ValueError("horizon requires corpus repetition")
    schedule_contract = {
        "source_schedule": source_record["learning_rate_schedule"],
        "source_optimizer_steps": source_steps,
        "source_warmup_steps": source_warmup_steps,
        "source_warmup_tokens": source_warmup_steps
        * int(source_record["batch_tokens"]),
        "new_optimizer_steps": target_steps,
        "new_warmup_fraction": warmup_fraction,
        "new_warmup_steps": math.ceil(warmup_fraction * target_steps),
        "fixed_in_presented_token_coordinates": True,
    }
    return config, plan, {
        "target": target,
        "selected_learning_rate": selected_eta,
        "source_learning_rate": source_eta,
        "seed": int(source_record["seed"]),
        "task_id": (
            f"ladder-{target['name']}-theory-eta{selected_eta:g}-"
            f"seed{int(source_record['seed'])}"
        ),
        "source_10tpp_loss": float(source_record["final_validation_loss"]),
        "source_dataset_fingerprint": source_record["metadata"][
            "dataset_fingerprint"
        ],
        "schedule_contract": schedule_contract,
        "retained_checkpoints": checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_300m_root", type=Path)
    parser.add_argument("source_1b_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    revealed = list(args.output_root.glob("**/ladder-shard-*.json"))
    if revealed:
        raise ValueError("dense horizon outcomes already exist; preregistration is frozen")
    identity = token_stream_identity(
        args.manifest.resolve(),
        verification_receipt_path=args.receipt.resolve(),
    )
    receipt_payload = _load(args.receipt.resolve())
    if identity["tokenizer_fingerprint"] != EXPECTED_TOKENIZER_FINGERPRINT:
        raise ValueError("the pinned Mistral tokenizer fingerprint changed")

    source_300_config_path = args.source_300m_root / "ten-x" / "config.json"
    source_300_result_path = args.source_300m_root / "result.json"
    source_1b_config_path = args.source_1b_root / "config.json"
    source_1b_result_path = args.source_1b_root / "result.json"
    source_300_result = _load(source_300_result_path)
    source_1b_result = _load(source_1b_result_path)
    source_300_record = _record(source_300_result, "300m")
    source_1b_record = _record(source_1b_result, "1b")

    config_300, plan_300, pre_300 = _compile_horizon(
        source_config=_load(source_300_config_path),
        source_record=source_300_record,
        manifest=args.manifest,
        receipt=args.receipt,
        total_tpp=40.0,
        retained_tpp=[10.0, 20.0, 40.0],
        expected_geometry=EXPECTED_300M_GEOMETRY,
    )
    config_1b, plan_1b, pre_1b = _compile_horizon(
        source_config=_load(source_1b_config_path),
        source_record=source_1b_record,
        manifest=args.manifest,
        receipt=args.receipt,
        total_tpp=20.0,
        retained_tpp=[10.0, 20.0],
        expected_geometry=EXPECTED_1B_GEOMETRY,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, config, plan in (
        ("dense-300m-40tpp", config_300, plan_300),
        ("dense-1b-20tpp", config_1b, plan_1b),
    ):
        target = args.output_root / name
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
    gates = {
        "verified_immutable_token_stream": receipt_payload.get("status") == "passed",
        "same_pinned_mistral_tokenizer": (
            identity["tokenizer_fingerprint"] == EXPECTED_TOKENIZER_FINGERPRINT
        ),
        "enough_unique_tokens_for_every_run": (
            int(identity["training_tokens"])
            >= max(
                int(plan_300["scales"][-1]["presented_tokens"]),
                int(plan_1b["scales"][-1]["presented_tokens"]),
            )
        ),
        "exact_300m_rho32_geometry": (
            tuple(
                int(plan_300["scales"][-1][key])
                for key in (
                    "parameters",
                    "non_embedding_parameters",
                    "depth",
                    "width",
                    "hidden_width",
                )
            )
            == EXPECTED_300M_GEOMETRY
        ),
        "exact_1b_rho32_geometry": (
            tuple(
                int(plan_1b["scales"][-1][key])
                for key in (
                    "parameters",
                    "non_embedding_parameters",
                    "depth",
                    "width",
                    "hidden_width",
                )
            )
            == EXPECTED_1B_GEOMETRY
        ),
        "full_state_10_20_40tpp_retention": (
            [
                row["requested_tokens_per_parameter"]
                for row in pre_300["retained_checkpoints"]
            ]
            == [10.0, 20.0, 40.0]
            and [
                row["requested_tokens_per_parameter"]
                for row in pre_1b["retained_checkpoints"]
            ]
            == [10.0, 20.0]
        ),
        "source_warmup_fixed_in_token_coordinates": (
            pre_300["schedule_contract"]["fixed_in_presented_token_coordinates"]
            and pre_1b["schedule_contract"]["fixed_in_presented_token_coordinates"]
        ),
        "outcomes_unseen": not revealed,
    }
    if not all(gates.values()):
        raise ValueError(
            "dense horizon preregistration failed: "
            + ", ".join(name for name, passed in gates.items() if not passed)
        )
    preregistration = {
        "schema_version": 1,
        "status": "preregistered",
        "scientific_status": (
            "exploratory_single_seed_dense_token_horizon_extension; source 300M "
            "topology qualification limitation remains binding"
        ),
        "repo_commit": commit,
        "dataset_identity": identity,
        "sources": {
            "300m_config_sha256": _sha256(source_300_config_path),
            "300m_result_sha256": _sha256(source_300_result_path),
            "1b_config_sha256": _sha256(source_1b_config_path),
            "1b_result_sha256": _sha256(source_1b_result_path),
        },
        "campaigns": {
            "dense_300m_40tpp": {
                **pre_300,
                "plan_fingerprint": plan_300["fingerprint"],
            },
            "dense_1b_20tpp": {
                **pre_1b,
                "plan_fingerprint": plan_1b["fingerprint"],
            },
        },
        "execution_order": [
            "run_300m_to_40tpp_and_reveal_10_20_40_milestones",
            "verify_every_retained_full_state_checkpoint",
            "run_1b_to_20tpp_and_reveal_10_20_milestones",
            "compare_source_and_new_10tpp_losses_before_interpreting_horizon_gains",
        ],
        "gates": gates,
    }
    atomic_write_json(args.output_root / "preregistration.json", preregistration)
    print(json.dumps(preregistration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
