#!/usr/bin/env python3
"""Verify a larger token stream preserves a declared prefix and validation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from ai_theorist.autoscaler.study import atomic_write_json


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (int(row["tokens"]), int(row["bytes"]), str(row["sha256"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_manifest", type=Path)
    parser.add_argument("continuation_manifest", type=Path)
    parser.add_argument("--required-prefix-tokens", type=int, required=True)
    parser.add_argument("--minimum-training-tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.required_prefix_tokens <= 0 or args.minimum_training_tokens <= 0:
        raise ValueError("token requirements must be positive")

    base = _load(args.base_manifest)
    continuation = _load(args.continuation_manifest)
    base_train = list(base["splits"]["train"]["shards"])
    continuation_train = list(continuation["splits"]["train"]["shards"])
    continuation_tokens = int(continuation["splits"]["train"]["tokens"])
    cumulative = 0
    compared = []
    for index, base_shard in enumerate(base_train):
        if index >= len(continuation_train):
            raise ValueError("continuation ends before the required base prefix")
        continuation_shard = continuation_train[index]
        if _signature(base_shard) != _signature(continuation_shard):
            raise ValueError(f"training token shard {index} is not prefix-identical")
        cumulative += int(base_shard["tokens"])
        compared.append(index)
        if cumulative >= args.required_prefix_tokens:
            break
    if cumulative < args.required_prefix_tokens:
        raise ValueError("base stream itself is shorter than the required prefix")

    base_validation = [
        _signature(row) for row in base["splits"]["validation"]["shards"]
    ]
    continuation_validation = [
        _signature(row)
        for row in continuation["splits"]["validation"]["shards"]
    ]
    gates = {
        "same_tokenizer": continuation["tokenizer_fingerprint"]
        == base["tokenizer_fingerprint"],
        "same_vocabulary": int(continuation["vocab_size"]) == int(base["vocab_size"]),
        "same_packing_contract": continuation["packing"] == base["packing"],
        "prefix_identical_through_calibration_horizon": cumulative
        >= args.required_prefix_tokens,
        "validation_shards_identical": continuation_validation == base_validation,
        "continuation_large_enough_for_1b": continuation_tokens
        >= args.minimum_training_tokens,
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError("token-stream continuation failed: " + ", ".join(failed))
    payload = {
        "schema_version": 1,
        "status": "passed",
        "base_fingerprint": base["fingerprint"],
        "continuation_fingerprint": continuation["fingerprint"],
        "required_prefix_tokens": args.required_prefix_tokens,
        "verified_prefix_tokens": cumulative,
        "compared_train_shard_indices": compared,
        "minimum_training_tokens": args.minimum_training_tokens,
        "continuation_training_tokens": continuation_tokens,
        "validation_tokens": int(base["splits"]["validation"]["tokens"]),
        "gates": gates,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
