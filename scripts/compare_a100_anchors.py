#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def throughput(result: dict[str, Any]) -> float:
    tokens = sum(int(trial["token_horizon"]) for trial in result["trials"])
    seconds = sum(float(trial["duration_seconds"]) for trial in result["trials"])
    return tokens / seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate matched A100 qualification anchors")
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--maximum-relative-loss-delta", type=float, default=0.005)
    parser.add_argument("--minimum-throughput-ratio", type=float, default=0.70)
    args = parser.parse_args()

    first = load(args.first)
    second = load(args.second)
    reasons: list[str] = []
    if first.get("study_fingerprint") != second.get("study_fingerprint"):
        reasons.append("study fingerprints differ")
    if first.get("status") != "completed" or second.get("status") != "completed":
        reasons.append("one or both anchors did not complete")

    first_rows = {row["scale"]: row for row in first.get("scale_results", [])}
    second_rows = {row["scale"]: row for row in second.get("scale_results", [])}
    if first_rows.keys() != second_rows.keys():
        reasons.append("completed scale sets differ")
    relative_deltas: dict[str, float] = {}
    for scale in sorted(first_rows.keys() & second_rows.keys()):
        left = float(first_rows[scale]["mean_final_validation_loss"])
        right = float(second_rows[scale]["mean_final_validation_loss"])
        relative_deltas[scale] = abs(left - right) / max(abs(left), abs(right), 1e-12)
    maximum_delta = max(relative_deltas.values(), default=math.inf)
    if maximum_delta > args.maximum_relative_loss_delta:
        reasons.append(
            f"maximum relative loss delta {maximum_delta:.6g} exceeds "
            f"{args.maximum_relative_loss_delta:.6g}"
        )

    if any(bool(trial.get("diverged")) for trial in first.get("trials", [])):
        reasons.append("first anchor contains a diverged trial")
    if any(bool(trial.get("diverged")) for trial in second.get("trials", [])):
        reasons.append("second anchor contains a diverged trial")
    if not first.get("normalization_quality", {}).get("accepted", False):
        reasons.append("first anchor failed normalization invariants")
    if not second.get("normalization_quality", {}).get("accepted", False):
        reasons.append("second anchor failed normalization invariants")

    first_throughput = throughput(first)
    second_throughput = throughput(second)
    throughput_ratio = min(first_throughput, second_throughput) / max(
        first_throughput, second_throughput, 1e-12
    )
    if throughput_ratio < args.minimum_throughput_ratio:
        reasons.append(
            f"throughput ratio {throughput_ratio:.3f} is below "
            f"{args.minimum_throughput_ratio:.3f}"
        )

    payload = {
        "accepted": not reasons,
        "reasons": reasons,
        "study_fingerprint": first.get("study_fingerprint"),
        "relative_loss_deltas": relative_deltas,
        "maximum_relative_loss_delta": maximum_delta,
        "throughput_tokens_per_second": [first_throughput, second_throughput],
        "throughput_ratio": throughput_ratio,
        "peak_memory_bytes": [
            max((int(t["peak_memory_bytes"]) for t in first.get("trials", [])), default=0),
            max((int(t["peak_memory_bytes"]) for t in second.get("trials", [])), default=0),
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["accepted"] else 1)


if __name__ == "__main__":
    main()
