#!/usr/bin/env python3
"""Verify retained states and summarize dense 10/20/40-TPP horizon losses."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from ai_theorist.autoscaler.study import atomic_write_json


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _one_record(path: Path) -> Mapping[str, Any]:
    shard = _load(path)
    records = shard.get("records", [])
    if shard.get("status") != "completed" or len(records) != 1:
        raise ValueError(f"{path} is not one completed trial")
    return records[0]


def _evaluate_record(
    record: Mapping[str, Any],
    campaign: Mapping[str, Any],
) -> dict[str, Any]:
    expected_target = campaign["target"]
    if int(record["parameter_count"]) != int(expected_target["parameters"]):
        raise ValueError("record geometry differs from preregistration")
    losses_by_step = {
        int(row["step"]): float(row["validation_loss"])
        for row in record["validation_checkpoints"]
    }
    retained_rows = record["metadata"].get("retained_checkpoints", [])
    expected_rows = campaign["retained_checkpoints"]
    if len(retained_rows) != len(expected_rows):
        raise ValueError("record retained-checkpoint count changed")
    horizons = []
    for observed, expected in zip(retained_rows, expected_rows):
        if int(observed["optimizer_step"]) != int(expected["optimizer_step"]):
            raise ValueError("retained checkpoint step changed")
        step = int(observed["optimizer_step"])
        if step not in losses_by_step:
            raise ValueError("retained horizon has no matching validation loss")
        base = Path(str(observed["base_path"]))
        checkpoint_path = base.with_suffix(".pt")
        if not checkpoint_path.is_file():
            raise ValueError(f"retained checkpoint is missing: {checkpoint_path}")
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        if (
            int(payload.get("schema_version", -1)) != 2
            or int(payload.get("step", -1)) != step
            or "model_state_dict" not in payload
            or "optimizer_state_dict" not in payload
            or "generator_state" not in payload
        ):
            raise ValueError(f"retained checkpoint is incomplete: {checkpoint_path}")
        if int(payload["extra"]["tokens_seen"]) != int(
            observed["presented_tokens"]
        ):
            raise ValueError("retained checkpoint token coordinate changed")
        horizons.append(
            {
                "requested_tokens_per_parameter": float(
                    observed["requested_tokens_per_parameter"]
                ),
                "effective_tokens_per_parameter": float(
                    observed["effective_tokens_per_parameter"]
                ),
                "optimizer_step": step,
                "presented_tokens": int(observed["presented_tokens"]),
                "validation_loss": losses_by_step[step],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "checkpoint_sha256": _sha256(checkpoint_path),
                "full_model_optimizer_generator_state_verified": True,
            }
        )
        del payload
    return {
        "parameter_count": int(record["parameter_count"]),
        "non_embedding_parameters": int(
            record["metadata"]["scale"]["non_embedding_parameters"]
        ),
        "selected_learning_rate": float(record["optimizer"]["learning_rate"]),
        "seed": int(record["seed"]),
        "total_tokens": int(record["total_tokens"]),
        "final_validation_loss": float(record["final_validation_loss"]),
        "wall_time_seconds": float(record["wall_time_seconds"]),
        "peak_memory_bytes": int(record["metadata"]["peak_memory_bytes"]),
        "horizons": horizons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preregistration", type=Path)
    parser.add_argument("shard_300m", type=Path)
    parser.add_argument("shard_1b", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pre = _load(args.preregistration)
    if pre.get("status") != "preregistered" or not all(pre["gates"].values()):
        raise ValueError("dense horizon preregistration is invalid")
    result_300 = _evaluate_record(
        _one_record(args.shard_300m),
        pre["campaigns"]["dense_300m_40tpp"],
    )
    result_1b = _evaluate_record(
        _one_record(args.shard_1b),
        pre["campaigns"]["dense_1b_20tpp"],
    )
    source_300_loss = float(
        pre["campaigns"]["dense_300m_40tpp"]["source_10tpp_loss"]
    )
    source_1b_loss = float(
        pre["campaigns"]["dense_1b_20tpp"]["source_10tpp_loss"]
    )
    new_300_10 = float(result_300["horizons"][0]["validation_loss"])
    new_1b_10 = float(result_1b["horizons"][0]["validation_loss"])
    source_comparison = {
        "300m": {
            "source_10tpp_loss": source_300_loss,
            "new_10tpp_loss": new_300_10,
            "relative_delta": new_300_10 / source_300_loss - 1.0,
            "strict_reproduction_gate": False,
            "reason": "the new stream is a larger immutable continuation",
        },
        "1b": {
            "source_10tpp_loss": source_1b_loss,
            "new_10tpp_loss": new_1b_10,
            "absolute_delta": abs(new_1b_10 - source_1b_loss),
            "relative_delta": new_1b_10 / source_1b_loss - 1.0,
            "strict_reproduction_gate": False,
            "reason": "the new stream appends unique training documents while preserving validation",
        },
    }
    gates = {
        "preregistration_valid": all(pre["gates"].values()),
        "all_five_retained_states_verified": (
            len(result_300["horizons"]) == 3
            and len(result_1b["horizons"]) == 2
            and all(
                row["full_model_optimizer_generator_state_verified"]
                for result in (result_300, result_1b)
                for row in result["horizons"]
            )
        ),
        "finite_losses": all(
            math.isfinite(float(row["validation_loss"]))
            for result in (result_300, result_1b)
            for row in result["horizons"]
        ),
        "fresh_10tpp_baselines_recorded_before_longer_horizons": (
            result_300["horizons"][0]["requested_tokens_per_parameter"] == 10.0
            and result_1b["horizons"][0]["requested_tokens_per_parameter"] == 10.0
        ),
        "zero_corpus_repetition": (
            result_300["total_tokens"]
            <= int(pre["dataset_identity"]["training_tokens"])
            and result_1b["total_tokens"]
            <= int(pre["dataset_identity"]["training_tokens"])
        ),
    }
    payload = {
        "schema_version": 1,
        "status": "completed" if all(gates.values()) else "failed",
        "scientific_status": pre["scientific_status"],
        "preregistration_sha256": _sha256(args.preregistration),
        "shard_300m_sha256": _sha256(args.shard_300m),
        "shard_1b_sha256": _sha256(args.shard_1b),
        "dense_300m_40tpp": result_300,
        "dense_1b_20tpp": result_1b,
        "source_10tpp_comparison": source_comparison,
        "gates": gates,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
