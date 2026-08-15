#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a completed Jiang-AdamW 100M result as the baseline for the "
            "matched CompleteP comparison."
        )
    )
    parser.add_argument("template", type=Path)
    parser.add_argument("jiang_result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    template = _object(json.loads(args.template.read_text(encoding="utf-8")), "template")
    result_bytes = args.jiang_result.read_bytes()
    result = _object(json.loads(result_bytes), "jiang result")
    if result.get("status") != "completed":
        raise ValueError("Jiang result is not complete")
    architecture = _object(result.get("architecture_contract"), "architecture_contract")
    if architecture.get("parameterization") != "jiang_completep_adamw":
        raise ValueError("baseline is not the Jiang CompleteP AdamW parameterization")
    tuning = _object(result.get("reference_tuning"), "reference_tuning")
    if not tuning.get("learning_rate_optimum_is_interior"):
        raise ValueError("Jiang reference eta optimum is not interior")
    if not tuning.get("weight_decay_optimum_is_interior"):
        raise ValueError("Jiang reference tau_EMA optimum is not interior")
    if tuning.get("selected_weight_decay_tau_ema") is None:
        raise ValueError("Jiang result did not freeze a tuned tau_EMA")

    scales = result.get("scales")
    if not isinstance(scales, list) or not scales:
        raise ValueError("Jiang result has no scale ladder")
    target = _object(scales[-1], "largest Jiang scale")
    parameters = int(target.get("parameters", 0))
    if not 99_000_000 <= parameters <= 101_000_000:
        raise ValueError("largest Jiang scale is not a matched 100M model")
    seed_losses = [float(value) for value in target.get("seed_losses", ())]
    template_seeds = [int(value) for value in template.get("seeds", ())]
    if len(seed_losses) != len(template_seeds) or not all(
        math.isfinite(value) for value in seed_losses
    ):
        raise ValueError("Jiang 100M seed losses are incomplete or non-finite")
    mean_loss = float(target.get("mean_validation_loss", math.nan))
    if not math.isfinite(mean_loss):
        raise ValueError("Jiang 100M mean validation loss is non-finite")

    target_name = target.get("name")
    target_records = [
        _object(record, "record")
        for record in result.get("records", ())
        if _object(record, "record").get("metadata", {}).get("scale", {}).get("name")
        == target_name
        and _object(record, "record").get("metadata", {}).get("optimizer_mode")
        == "theory"
    ]
    if len(target_records) != len(template_seeds):
        raise ValueError("Jiang 100M record count does not match the declared seeds")
    for record in target_records:
        optimizer = _object(record.get("optimizer"), "record.optimizer")
        metadata = _object(record.get("metadata"), "record.metadata")
        audit = _object(metadata.get("optimizer_group_audit"), "optimizer_group_audit")
        theory = _object(audit.get("theory"), "optimizer_group_audit.theory")
        if optimizer.get("name") != "adamw" or theory.get("optimizer") != "adamw":
            raise ValueError("Jiang target record did not use the AdamW contract")
        if metadata.get("weight_decay_tau_ema") != tuning[
            "selected_weight_decay_tau_ema"
        ]:
            raise ValueError("Jiang target record used a different tau_EMA")

    dataset = _object(result.get("dataset"), "dataset")
    prepared = json.loads(json.dumps(template))
    prepared["comparison_contract"] = {
        "baseline_plan_fingerprint": str(result["plan_fingerprint"]),
        "baseline_aggregate_sha256": sha256(result_bytes).hexdigest(),
        "baseline_dataset_fingerprint": str(dataset["fingerprint"]),
        "baseline_tokenizer_fingerprint": str(dataset["tokenizer_fingerprint"]),
        "baseline_architecture": "jiang_chizat_transformer",
        "baseline_parameters": parameters,
        "baseline_mean_validation_loss": mean_loss,
        "baseline_seed_losses": seed_losses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(prepared, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
