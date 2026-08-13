#!/usr/bin/env python3
"""Repair explicitly named replicated-DDP rank files from one coherent rank."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict

import torch


def _header(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "schema_version": payload.get("schema_version"),
        "identity_fingerprint": payload.get("identity_fingerprint"),
        "world_size": payload.get("world_size"),
        "rank": payload.get("rank"),
        "step": payload.get("step"),
        "tokens_seen": payload.get("extra", {}).get("tokens_seen"),
    }


def _fingerprint(payload: Dict[str, Any]) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_save(payload: Dict[str, Any], path: Path) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", type=Path)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--source-rank", type=int, default=0)
    parser.add_argument("--target-rank", type=int, action="append", required=True)
    parser.add_argument("--audit-directory", type=Path, required=True)
    parser.add_argument("--replicated-ddp", action="store_true", required=True)
    args = parser.parse_args()

    pattern = f"{args.base_path.name}.rank-*-of-*.pt"
    paths = sorted(args.base_path.parent.glob(pattern))
    if not paths:
        raise ValueError("no distributed checkpoint rank files found")
    before = [_header(path) for path in paths]
    world_sizes = {int(row["world_size"]) for row in before}
    identities = {str(row["identity_fingerprint"]) for row in before}
    ranks = {int(row["rank"]) for row in before}
    if world_sizes != {len(paths)} or len(identities) != 1:
        raise ValueError("checkpoint topology or identity is inconsistent")
    if ranks != set(range(len(paths))):
        raise ValueError("checkpoint rank files are incomplete")
    by_rank = {int(row["rank"]): row for row in before}
    targets = set(args.target_rank)
    if args.source_rank in targets or not targets:
        raise ValueError("source rank and target ranks must be disjoint")
    if set(by_rank) - targets and any(
        int(by_rank[rank]["step"]) != args.expected_step
        for rank in set(by_rank) - targets
    ):
        raise ValueError("a non-target checkpoint is not at the expected step")
    if any(int(by_rank[rank]["step"]) == args.expected_step for rank in targets):
        raise ValueError("a requested repair target is already coherent")

    source_path = Path(by_rank[args.source_rank]["path"])
    source = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    if int(source["step"]) != args.expected_step:
        raise ValueError("source checkpoint is not at the expected step")

    args.audit_directory.mkdir(parents=True, exist_ok=False)
    archived = []
    for rank in sorted(targets):
        target_path = Path(by_rank[rank]["path"])
        archive_path = args.audit_directory / target_path.name
        os.link(target_path, archive_path)
        archived.append(_header(archive_path))
        repaired = dict(source)
        repaired["rank"] = rank
        _atomic_save(repaired, target_path)

    after = [_header(path) for path in paths]
    if any(int(row["step"]) != args.expected_step for row in after):
        raise RuntimeError("checkpoint repair did not produce one coherent step")
    if {int(row["rank"]) for row in after} != set(range(len(paths))):
        raise RuntimeError("checkpoint repair changed rank coverage")
    audit: Dict[str, Any] = {
        "schema_version": 1,
        "status": "repaired",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recovery_contract": "replicated DDP model and optimizer state",
        "expected_step": args.expected_step,
        "source_rank": args.source_rank,
        "target_ranks": sorted(targets),
        "before": before,
        "archived_originals": archived,
        "after": after,
    }
    audit["fingerprint"] = _fingerprint(audit)
    (args.audit_directory / "recovery.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
