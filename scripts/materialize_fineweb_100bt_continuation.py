#!/usr/bin/env python3
"""Build a deduplicated FineWeb-Edu continuation with an exact old-token prefix.

The original calibration stream came from the nested sample-10BT release.  That
release is too small for a unique 1B-at-10-TPP run.  This builder preserves every
old token shard byte-for-byte, reuses the remaining disjoint sample-10BT rows,
and appends sample-100BT documents only when their FineWeb IDs were not already
used by training or validation.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

import pyarrow.parquet as parquet

from ai_theorist.autoscaler import tokenization
from ai_theorist.autoscaler.public_corpora import (
    PUBLIC_CORPUS_CATALOG,
    _download_parquet_file,
    _hash_file,
    _parquet_inventory,
    _source_revision,
)
from ai_theorist.autoscaler.study import atomic_write_json
from ai_theorist.autoscaler.tokenization import (
    TOKEN_STREAM_FORMAT,
    TOKEN_STREAM_MANIFEST_SCHEMA_VERSION,
    TOKEN_STREAM_PACKING_CONTRACT,
    load_token_stream_manifest,
    load_tokenizer_manifest,
    resolve_pinned_tokenizer,
)


SOURCE_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
PARQUET_REVISION = "92cece42bcce787ee4af4619ab449fe48d86230d"
BASE_DATASET_FINGERPRINT = (
    "1b854ee220230e0421acd8312d313a72d396de2234474ec20f63ba1ce4f1d703"
)
TOKENIZER_FINGERPRINT = (
    "d52f662783555cbf11f6a0cd8af35016652cda033389db471813c7d30f6958c5"
)
REQUIRED_TRAIN_TOKENS = 10_085_203_968
DEFAULT_EXTRA_TEXT_BYTES = 12 * 1024**3
SHARD_TOKEN_LIMIT = 16_777_216
DOCUMENT_BATCH_SIZE = 512


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), sort_keys=True), flush=True)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            yield value


def _load_used_ids(paths: list[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        before = len(result)
        for index, row in enumerate(_iter_jsonl(path), start=1):
            source_id = row.get("source_id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(f"{path} contains a missing FineWeb source_id")
            result.add(source_id)
            if index % 250_000 == 0:
                _emit(
                    {
                        "phase": "indexing_used_ids",
                        "path": str(path),
                        "documents_read": index,
                        "unique_ids": len(result),
                    }
                )
        _emit(
            {
                "phase": "indexed_used_ids",
                "path": str(path),
                "new_unique_ids": len(result) - before,
                "unique_ids": len(result),
            }
        )
    return result


def _sample_100bt_inventory() -> dict[str, Any]:
    catalog = {
        **PUBLIC_CORPUS_CATALOG["fineweb_edu"],
        "name": "FineWeb-Edu sample-100BT continuation",
        "config": "sample-100BT",
    }
    if _source_revision(str(catalog["dataset"]), SOURCE_REVISION) != SOURCE_REVISION:
        raise RuntimeError("FineWeb-Edu source revision changed")
    if (
        _source_revision(str(catalog["dataset"]), PARQUET_REVISION)
        != PARQUET_REVISION
    ):
        raise RuntimeError("FineWeb-Edu parquet revision changed")
    return _parquet_inventory(
        catalog, source_split="train", parquet_revision=PARQUET_REVISION
    )


def initialize(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    base_stream = load_token_stream_manifest(args.base_stream, verify_files=True)
    base_tokenizer = load_tokenizer_manifest(args.base_tokenizer, verify_assets=True)
    if base_stream["fingerprint"] != BASE_DATASET_FINGERPRINT:
        raise ValueError("base FineWeb stream fingerprint changed")
    if base_tokenizer["fingerprint"] != TOKENIZER_FINGERPRINT:
        raise ValueError("base Mistral tokenizer fingerprint changed")
    tokenizer_directory = args.output_root / "tokenizer"
    if not tokenizer_directory.exists():
        shutil.copytree(
            args.base_tokenizer.parent,
            tokenizer_directory,
            copy_function=os.link,
        )
    resolved = resolve_pinned_tokenizer("mistral_7b_v03", tokenizer_directory)
    if resolved.manifest["fingerprint"] != TOKENIZER_FINGERPRINT:
        raise ValueError("resolved continuation tokenizer changed")
    secondary_checkpoint = _load(args.secondary_checkpoint)
    if int(secondary_checkpoint["file_position"]) != args.secondary.stat().st_size:
        raise ValueError("sample-10BT secondary file is not at its fsynced checkpoint")
    if int(secondary_checkpoint["next_source_row"]) != 9_672_101:
        raise ValueError("sample-10BT secondary did not reach the immutable inventory end")
    atomic_write_json(
        args.output_root / "initialization.json",
        {
            "schema_version": 1,
            "status": "passed",
            "base_stream": str(args.base_stream.resolve()),
            "base_stream_fingerprint": base_stream["fingerprint"],
            "base_train_tokens": base_stream["splits"]["train"]["tokens"],
            "base_tokenizer": str(args.base_tokenizer.resolve()),
            "tokenizer_fingerprint": resolved.manifest["fingerprint"],
            "secondary_path": str(args.secondary.resolve()),
            "secondary_checkpoint": str(args.secondary_checkpoint.resolve()),
            "secondary_documents": secondary_checkpoint["document_count"],
            "secondary_text_bytes": secondary_checkpoint["text_bytes"],
            "required_train_tokens": REQUIRED_TRAIN_TOKENS,
            "extra_text_bytes": args.extra_text_bytes,
        },
    )
    _emit({"phase": "initialized", "output_root": str(args.output_root)})


def materialize_extra(args: argparse.Namespace) -> None:
    result_path = args.output_root / "extra-materialization.json"
    extra_path = args.output_root / "extra-100bt.jsonl"
    if result_path.is_file() and extra_path.is_file():
        result = _load(result_path)
        if (
            result.get("status") == "complete"
            and int(result["file_bytes"]) == extra_path.stat().st_size
            and result["file_sha256"] == _hash_file(extra_path)
        ):
            _emit({"phase": "extra_materialization_cached", **result})
            return

    inventory = _sample_100bt_inventory()
    partial = args.output_root / ".extra-100bt.jsonl.partial"
    checkpoint_path = args.output_root / ".extra-100bt.jsonl.partial.json"
    contract = {
        "schema_version": 1,
        "source_revision": SOURCE_REVISION,
        "parquet_revision": PARQUET_REVISION,
        "inventory_fingerprint": inventory["fingerprint"],
        "target_text_bytes": args.extra_text_bytes,
        "source_batch_rows": args.source_batch_rows,
        "excluded_paths": [
            str(path.resolve())
            for path in (args.base_train, args.base_validation, args.secondary)
        ],
    }
    checkpoint: dict[str, Any] | None = None
    if checkpoint_path.is_file() and partial.is_file():
        candidate = _load(checkpoint_path)
        if all(candidate.get(key) == value for key, value in contract.items()):
            if int(candidate["file_position"]) <= partial.stat().st_size:
                checkpoint = candidate
    if checkpoint is None:
        text_bytes = 0
        documents = 0
        duplicates_skipped = 0
        next_source_row = 0
        source_files: list[dict[str, Any]] = []
        handle = partial.open("wb")
    else:
        text_bytes = int(checkpoint["text_bytes"])
        documents = int(checkpoint["documents"])
        duplicates_skipped = int(checkpoint["duplicates_skipped"])
        next_source_row = int(checkpoint["next_source_row"])
        source_files = list(checkpoint.get("source_files", []))
        handle = partial.open("r+b")
        handle.truncate(int(checkpoint["file_position"]))
        handle.seek(0, os.SEEK_END)
    used_ids = _load_used_ids([args.base_train, args.base_validation, args.secondary])
    if partial.is_file() and partial.stat().st_size:
        used_ids.update(
            str(row["source_id"])
            for row in _iter_jsonl(partial)
            if isinstance(row.get("source_id"), str)
        )
    known_urls = {str(row.get("url")) for row in source_files}
    catalog = {**PUBLIC_CORPUS_CATALOG["fineweb_edu"], "config": "sample-100BT"}
    cache = args.output_root / "source-parquet-100bt"
    global_row = 0
    with handle:
        for entry in inventory["files"]:
            local = _download_parquet_file(entry, cache)
            parquet_file = parquet.ParquetFile(local["path"])
            file_rows = int(parquet_file.metadata.num_rows)
            file_end = global_row + file_rows
            if file_end <= next_source_row:
                global_row = file_end
                continue
            if local["url"] not in known_urls:
                source_files.append(local)
                known_urls.add(local["url"])
            batch_start = global_row
            for batch in parquet_file.iter_batches(
                batch_size=args.source_batch_rows,
                columns=[str(catalog["text_field"]), str(catalog["id_field"])],
            ):
                batch_rows = len(batch)
                batch_end = batch_start + batch_rows
                if batch_end <= next_source_row:
                    batch_start = batch_end
                    continue
                values = batch.to_pydict()
                for local_index, text in enumerate(values[str(catalog["text_field"])]):
                    row_index = batch_start + local_index
                    if row_index < next_source_row:
                        continue
                    source_id = values[str(catalog["id_field"])][local_index]
                    if not isinstance(source_id, str) or not source_id:
                        raise ValueError("sample-100BT row has no stable FineWeb ID")
                    if source_id in used_ids:
                        duplicates_skipped += 1
                        continue
                    if not isinstance(text, str) or not text:
                        continue
                    encoded = text.encode("utf-8")
                    handle.write(
                        (
                            json.dumps(
                                {
                                    "text": text,
                                    "source_row": row_index,
                                    "source_id": source_id,
                                    "source_config": "sample-100BT",
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        ).encode("utf-8")
                    )
                    used_ids.add(source_id)
                    text_bytes += len(encoded)
                    documents += 1
                    if text_bytes >= args.extra_text_bytes:
                        break
                next_source_row = batch_end
                handle.flush()
                os.fsync(handle.fileno())
                atomic_write_json(
                    checkpoint_path,
                    {
                        **contract,
                        "next_source_row": next_source_row,
                        "text_bytes": text_bytes,
                        "documents": documents,
                        "duplicates_skipped": duplicates_skipped,
                        "file_position": handle.tell(),
                        "source_files": source_files,
                    },
                )
                _emit(
                    {
                        "phase": "materializing_extra_100bt",
                        "text_bytes": text_bytes,
                        "target_text_bytes": args.extra_text_bytes,
                        "documents": documents,
                        "duplicates_skipped": duplicates_skipped,
                        "source_row": next_source_row,
                    }
                )
                if text_bytes >= args.extra_text_bytes:
                    break
                batch_start = batch_end
            global_row = file_end
            if text_bytes >= args.extra_text_bytes:
                break
        if text_bytes < args.extra_text_bytes:
            raise RuntimeError("sample-100BT inventory ended before the extra target")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, extra_path)
    checkpoint_path.unlink(missing_ok=True)
    inventory_after = _sample_100bt_inventory()
    if inventory_after["fingerprint"] != inventory["fingerprint"]:
        raise RuntimeError("sample-100BT inventory changed during materialization")
    result = {
        "schema_version": 1,
        "status": "complete",
        "path": str(extra_path.resolve()),
        "documents": documents,
        "text_bytes": text_bytes,
        "file_bytes": extra_path.stat().st_size,
        "file_sha256": _hash_file(extra_path),
        "duplicates_skipped": duplicates_skipped,
        "last_source_row": next_source_row - 1,
        "source_revision": SOURCE_REVISION,
        "parquet_revision": PARQUET_REVISION,
        "inventory_fingerprint": inventory["fingerprint"],
        "source_parquet_files": source_files,
    }
    atomic_write_json(result_path, result)
    _emit({"phase": "extra_materialization_complete", **result})


def _tokenize_one(
    *,
    name: str,
    source_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    output = output_root / f"{name}-tokens"
    output.mkdir(parents=True, exist_ok=True)
    # The resumable sample-10BT artifact intentionally retains its `.partial`
    # suffix.  The shared tokenizer treats only paths ending in `.jsonl` as
    # record streams; passing the artifact directly would therefore read the
    # entire multi-gigabyte JSONL file as one plain-text document.  Bind the
    # exact same inode at a canonical JSONL path so format dispatch is explicit
    # without copying or transforming any corpus bytes.
    tokenization_source = source_path
    if source_path.suffix.lower() != ".jsonl":
        bound_sources = output_root / "bound-sources"
        bound_sources.mkdir(parents=True, exist_ok=True)
        canonical = bound_sources / f"{name}.jsonl"
        if canonical.exists():
            source_stat = source_path.stat()
            canonical_stat = canonical.stat()
            if (
                source_stat.st_dev != canonical_stat.st_dev
                or source_stat.st_ino != canonical_stat.st_ino
            ):
                raise ValueError(
                    f"bound JSONL source {canonical} is not the exact source inode"
                )
        else:
            os.link(source_path, canonical)
        tokenization_source = canonical
    result_path = output / "result.json"
    if result_path.is_file():
        result = _load(result_path)
        if result.get("source_sha256") == _hash_file(tokenization_source):
            valid = True
            for shard in result.get("shards", []):
                path = output / str(shard["path"])
                valid &= path.is_file() and path.stat().st_size == int(shard["bytes"])
            if valid:
                _emit({"phase": f"{name}_tokenization_cached", **result})
                return result
    tokenizer_directory = output_root / "tokenizer"
    resolved = resolve_pinned_tokenizer("mistral_7b_v03", tokenizer_directory)

    def progress(event: dict[str, Any]) -> None:
        _emit({"segment": name, **event})

    source_sha = _hash_file(tokenization_source)
    result = tokenization._materialize_token_split(
        split=name,
        source_path=tokenization_source,
        output_directory=output,
        text_field="text",
        tokenizer=resolved,
        shard_token_limit=SHARD_TOKEN_LIMIT,
        progress=progress,
        source_fingerprint=source_sha,
        document_batch_size=DOCUMENT_BATCH_SIZE,
    )
    atomic_write_json(result_path, result)
    _emit({"phase": f"{name}_tokenization_complete", **result})
    return result


def tokenize_segment(args: argparse.Namespace) -> None:
    if args.segment == "secondary":
        source = args.secondary
    elif args.segment == "extra":
        source = args.output_root / "extra-100bt.jsonl"
    else:
        raise ValueError("segment must be secondary or extra")
    if not source.is_file():
        raise FileNotFoundError(source)
    _tokenize_one(name=args.segment, source_path=source, output_root=args.output_root)


def _link_shards(
    *,
    source_directory: Path,
    source_rows: list[Mapping[str, Any]],
    destination: Path,
    split: str,
    start_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, source_row in enumerate(source_rows):
        name = f"{split}-{start_index + offset:05d}.bin"
        source = source_directory / str(source_row["path"])
        target = destination / name
        os.link(source, target)
        rows.append({**dict(source_row), "path": name})
    return rows


def assemble(args: argparse.Namespace) -> None:
    base = load_token_stream_manifest(args.base_stream, verify_files=True)
    secondary = _load(args.output_root / "secondary-tokens" / "result.json")
    extra = _load(args.output_root / "extra-tokens" / "result.json")
    tokenizer_manifest = load_tokenizer_manifest(
        args.output_root / "tokenizer" / "manifest.json", verify_assets=True
    )
    if tokenizer_manifest["fingerprint"] != TOKENIZER_FINGERPRINT:
        raise ValueError("continuation tokenizer fingerprint changed")
    final = args.output_root / "token-streams"
    temporary = args.output_root / ".token-streams.assembling"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    train_rows: list[dict[str, Any]] = []
    train_rows.extend(
        _link_shards(
            source_directory=args.base_stream.parent,
            source_rows=base["splits"]["train"]["shards"],
            destination=temporary,
            split="train",
            start_index=0,
        )
    )
    train_rows.extend(
        _link_shards(
            source_directory=args.output_root / "secondary-tokens",
            source_rows=secondary["shards"],
            destination=temporary,
            split="train",
            start_index=len(train_rows),
        )
    )
    train_rows.extend(
        _link_shards(
            source_directory=args.output_root / "extra-tokens",
            source_rows=extra["shards"],
            destination=temporary,
            split="train",
            start_index=len(train_rows),
        )
    )
    validation_rows = _link_shards(
        source_directory=args.base_stream.parent,
        source_rows=base["splits"]["validation"]["shards"],
        destination=temporary,
        split="validation",
        start_index=0,
    )
    train_tokens = sum(int(row["tokens"]) for row in train_rows)
    if train_tokens < REQUIRED_TRAIN_TOKENS:
        raise RuntimeError(
            f"continuation has {train_tokens} tokens, below {REQUIRED_TRAIN_TOKENS}"
        )
    splits = {
        "train": {
            "documents": int(base["splits"]["train"]["documents"])
            + int(secondary["documents"])
            + int(extra["documents"]),
            "tokens": train_tokens,
            "source_sha256": sha256(
                json.dumps(
                    [
                        base["splits"]["train"]["source_sha256"],
                        secondary["source_sha256"],
                        extra["source_sha256"],
                    ],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "shards": train_rows,
        },
        "validation": {
            **base["splits"]["validation"],
            "shards": validation_rows,
        },
    }
    content_fingerprint = tokenization._fingerprint(
        {
            split: [
                {"sha256": row["sha256"], "tokens": row["tokens"]}
                for row in metadata["shards"]
            ]
            for split, metadata in splits.items()
        }
    )
    manifest = {
        "schema_version": TOKEN_STREAM_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "format": TOKEN_STREAM_FORMAT,
        "dtype": "uint32_le",
        "tokenizer_id": "mistral_7b_v03",
        "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
        "tokenizer_manifest_path": "../tokenizer/manifest.json",
        "vocab_size": 32768,
        "packing": {
            "contract": TOKEN_STREAM_PACKING_CONTRACT,
            "add_special_tokens": False,
            "append_document_separator": True,
            "document_separator_token_id": 2,
            "cross_document_attention": True,
            "cross_shard_windows": False,
            "shard_token_limit": SHARD_TOKEN_LIMIT,
        },
        "content_fingerprint": content_fingerprint,
        "splits": splits,
    }
    manifest["fingerprint"] = tokenization._fingerprint(manifest)
    atomic_write_json(temporary / "manifest.json", manifest)
    if final.exists():
        previous = load_token_stream_manifest(final / "manifest.json", verify_files=True)
        if previous["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("an existing continuation manifest has different content")
        shutil.rmtree(temporary)
    else:
        os.replace(temporary, final)
    verified = load_token_stream_manifest(final / "manifest.json", verify_files=True)
    if [row["sha256"] for row in verified["splits"]["train"]["shards"][: len(base["splits"]["train"]["shards"])]] != [
        row["sha256"] for row in base["splits"]["train"]["shards"]
    ]:
        raise ValueError("assembled continuation does not preserve the base shard prefix")
    provenance = {
        "schema_version": 1,
        "status": "complete",
        "scientific_status": "exploratory_fineweb_edu_sample10bt_prefix_plus_deduplicated_sample100bt_continuation",
        "base_stream_fingerprint": base["fingerprint"],
        "continuation_stream_fingerprint": verified["fingerprint"],
        "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
        "source_revision": SOURCE_REVISION,
        "parquet_revision": PARQUET_REVISION,
        "base_train_tokens": base["splits"]["train"]["tokens"],
        "continuation_train_tokens": verified["splits"]["train"]["tokens"],
        "required_train_tokens": REQUIRED_TRAIN_TOKENS,
        "secondary": secondary,
        "extra": _load(args.output_root / "extra-materialization.json"),
        "extra_tokens": extra,
        "gates": {
            "base_prefix_exact": True,
            "same_pinned_tokenizer": True,
            "unique_document_ids_across_segments": True,
            "sufficient_unique_tokens": train_tokens >= REQUIRED_TRAIN_TOKENS,
            "immutable_source_revisions": True,
        },
    }
    atomic_write_json(args.output_root / "manifest.json", provenance)
    _emit({"phase": "complete", **provenance})


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "command", choices=("initialize", "materialize-extra", "tokenize", "assemble")
    )
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--base-stream", type=Path, required=True)
    result.add_argument("--base-tokenizer", type=Path, required=True)
    result.add_argument("--base-train", type=Path, required=True)
    result.add_argument("--base-validation", type=Path, required=True)
    result.add_argument("--secondary", type=Path, required=True)
    result.add_argument("--secondary-checkpoint", type=Path, required=True)
    result.add_argument("--segment", choices=("secondary", "extra"))
    result.add_argument("--extra-text-bytes", type=int, default=DEFAULT_EXTRA_TEXT_BYTES)
    result.add_argument("--source-batch-rows", type=int, default=8192)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "initialize":
        initialize(args)
    elif args.command == "materialize-extra":
        materialize_extra(args)
    elif args.command == "tokenize":
        if args.segment is None:
            raise ValueError("tokenize requires --segment")
        tokenize_segment(args)
    else:
        assemble(args)


if __name__ == "__main__":
    main()
