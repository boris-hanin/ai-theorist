#!/usr/bin/env python3
"""Aggregate the gated T^-1/3 dense horizon phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_theorist.autoscaler.study import atomic_write_json

from evaluate_jiang_dense_horizon_scaling import _load, _sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preregistration", type=Path)
    parser.add_argument("phase_300m_20tpp", type=Path)
    parser.add_argument("phase_300m_40tpp", type=Path)
    parser.add_argument("phase_1b_20tpp", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    preregistration = _load(args.preregistration)
    expected = {
        "dense_300m_20tpp": args.phase_300m_20tpp,
        "dense_300m_40tpp": args.phase_300m_40tpp,
        "dense_1b_20tpp": args.phase_1b_20tpp,
    }
    phases = {key: _load(path) for key, path in expected.items()}
    gates = {
        "preregistration_valid": preregistration.get("status")
        == "preregistered"
        and all(preregistration["gates"].values()),
        "all_adaptive_phase_gates_passed": all(
            value.get("status") == "passed" and all(value["gates"].values())
            for value in phases.values()
        ),
        "phase_identities_match": all(
            value.get("campaign_key") == key for key, value in phases.items()
        ),
    }
    payload = {
        "schema_version": 1,
        "status": "completed" if all(gates.values()) else "failed",
        "scientific_status": preregistration["scientific_status"],
        "preregistration_sha256": _sha256(args.preregistration),
        "phase_sha256": {
            key: _sha256(path) for key, path in expected.items()
        },
        "phases": phases,
        "gates": gates,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
