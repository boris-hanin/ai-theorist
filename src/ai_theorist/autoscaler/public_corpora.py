from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from time import sleep
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .pretraining import TokenizedTextCorpus, TokenizedTextSpec
from .study import atomic_write_json
from .tokenization import (
    PINNED_TOKENIZER_REGISTRY,
    TOKEN_STREAM_FORMAT,
    TOKEN_STREAM_PACKING_CONTRACT,
    builtin_byte_tokenizer_manifest,
    load_token_stream_manifest,
    load_tokenizer_manifest,
    materialize_pinned_token_streams,
    resolve_pinned_tokenizer,
    tokenizer_definition_fingerprint,
)


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


PUBLIC_CORPUS_CATALOG: Dict[str, Dict[str, Any]] = {
    "fineweb_edu": {
        "name": "FineWeb-Edu sample-10BT",
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "text_field": "text",
        "id_field": "id",
        "license": "ODC-By 1.0",
        "data_card_url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
        "train_offset": 0,
        "validation_offset": 5_000_000,
        "row_count": 9_672_101,
    },
    "openwebtext": {
        "name": "OpenWebText",
        "dataset": "Skylion007/openwebtext",
        "config": "plain_text",
        "split": "train",
        "text_field": "text",
        "id_field": None,
        "license": "CC0 packaging; source text rights remain with their owners",
        "data_card_url": "https://huggingface.co/datasets/Skylion007/openwebtext",
        "train_offset": 0,
        "validation_offset": 4_000_000,
        "row_count": 8_013_769,
    },
    "slimpajama": {
        "name": "SlimPajama-627B",
        "dataset": "cerebras/SlimPajama-627B",
        "config": "default",
        "split": "train",
        "text_field": "text",
        "id_field": None,
        "license": "Apache-2.0 dataset release; constituent source licenses apply",
        "data_card_url": "https://huggingface.co/datasets/cerebras/SlimPajama-627B",
        "train_offset": 0,
        "validation_offset": 0,
        "row_count": 627_000_000,
        "direct_shards": {
            "train": {
                "chunk": 1,
                "count": 6_000,
                "path_template": "train/chunk1/example_train_{index}.jsonl.zst",
            },
            "validation": {
                "chunk": 1,
                "count": 6_000,
                "path_template": (
                    "validation/chunk1/example_holdout_{index}.jsonl.zst"
                ),
            },
        },
    },
}


@dataclass(frozen=True)
class PublicCorpusSpec:
    source: str = "fineweb_edu"
    tokenizer: str = "byte_v1"
    train_bytes: int = 33_554_432
    validation_bytes: int = 4_194_304
    maximum_documents_per_split: int = 50_000
    token_shard_tokens: int = 16_777_216
    acquisition_backend: str = "viewer_rows"
    source_batch_rows: int = 8_192
    train_primary_bytes: Optional[int] = None
    train_secondary_offset: Optional[int] = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PublicCorpusSpec":
        allowed = {
            "source",
            "tokenizer",
            "train_bytes",
            "validation_bytes",
            "maximum_documents_per_split",
            "token_shard_tokens",
            "acquisition_backend",
            "source_batch_rows",
            "train_primary_bytes",
            "train_secondary_offset",
        }
        extras = sorted(set(payload) - allowed)
        if extras:
            raise ValueError(f"Unknown public corpus field(s): {', '.join(extras)}")
        result = cls(**dict(payload))
        if result.source not in PUBLIC_CORPUS_CATALOG:
            raise ValueError(
                "source must be one of " + ", ".join(sorted(PUBLIC_CORPUS_CATALOG))
            )
        if result.tokenizer not in {"byte_v1", *PINNED_TOKENIZER_REGISTRY}:
            raise ValueError(
                "tokenizer must be byte_v1 or one of "
                + ", ".join(sorted(PINNED_TOKENIZER_REGISTRY))
            )
        if result.acquisition_backend not in {
            "viewer_rows",
            "parquet",
            "zstd_shards",
        }:
            raise ValueError(
                "acquisition_backend must be viewer_rows, parquet, or zstd_shards"
            )
        if result.acquisition_backend == "zstd_shards" and not PUBLIC_CORPUS_CATALOG[
            result.source
        ].get("direct_shards"):
            raise ValueError(
                "zstd_shards acquisition is only available for a direct-shard corpus"
            )
        for name, value, minimum, maximum in (
            ("train_bytes", result.train_bytes, 65_536, 2_199_023_255_552),
            ("validation_bytes", result.validation_bytes, 16_384, 274_877_906_944),
            (
                "maximum_documents_per_split",
                result.maximum_documents_per_split,
                100,
                100_000_000,
            ),
            ("token_shard_tokens", result.token_shard_tokens, 1_024, 268_435_456),
            ("source_batch_rows", result.source_batch_rows, 128, 131_072),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < minimum or value > maximum:
                raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
        segmented = (
            result.train_primary_bytes is not None,
            result.train_secondary_offset is not None,
        )
        if segmented[0] != segmented[1]:
            raise ValueError(
                "train_primary_bytes and train_secondary_offset must be set together"
            )
        if segmented[0]:
            primary_bytes = result.train_primary_bytes
            secondary_offset = result.train_secondary_offset
            if (
                isinstance(primary_bytes, bool)
                or not isinstance(primary_bytes, int)
                or not 16_384 <= primary_bytes < result.train_bytes
            ):
                raise ValueError(
                    "train_primary_bytes must be an integer in "
                    "[16384, train_bytes)"
                )
            catalog = PUBLIC_CORPUS_CATALOG[result.source]
            if (
                isinstance(secondary_offset, bool)
                or not isinstance(secondary_offset, int)
                or not int(catalog["validation_offset"]) < secondary_offset
                < int(catalog["row_count"])
            ):
                raise ValueError(
                    "train_secondary_offset must be an integer after the validation "
                    "offset and before the source row count"
                )
            if result.acquisition_backend != "parquet":
                raise ValueError(
                    "segmented training currently requires the resumable parquet backend"
                )
        return result

    @property
    def fingerprint(self) -> str:
        spec_payload = {
            key: value for key, value in asdict(self).items() if value is not None
        }
        payload = json.dumps(
            {
                **spec_payload,
                "tokenizer_definition_fingerprint": tokenizer_definition_fingerprint(
                    self.tokenizer
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8")).hexdigest()[:16]


def _source_intervals(metadata: Mapping[str, Any]) -> List[Tuple[int, int]]:
    segments = metadata.get("source_segments")
    rows = segments if isinstance(segments, list) and segments else [metadata]
    intervals: List[Tuple[int, int]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("source segment metadata must contain objects")
        first = int(row["first_source_row"])
        last = int(row["last_source_row"])
        if first < 0 or last < first:
            raise ValueError("source segment row interval is invalid")
        intervals.append((first, last))
    for index, current in enumerate(intervals):
        for other in intervals[index + 1 :]:
            if not (current[1] < other[0] or other[1] < current[0]):
                raise ValueError("source segments overlap")
    return intervals


def _source_ranges_are_disjoint(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> bool:
    first_split = first.get("source_split")
    second_split = second.get("source_split")
    if (
        isinstance(first_split, str)
        and isinstance(second_split, str)
        and first_split != second_split
    ):
        return True
    return all(
        left_last < right_first or right_last < left_first
        for left_first, left_last in _source_intervals(first)
        for right_first, right_last in _source_intervals(second)
    )


def public_corpus_catalog() -> List[Dict[str, Any]]:
    return [
        {"id": source, **dict(metadata)}
        for source, metadata in PUBLIC_CORPUS_CATALOG.items()
    ]


def _hugging_face_headers(accept: str) -> Dict[str, str]:
    headers = {
        "Accept": accept,
        "User-Agent": "ai-theorist-autoscaler/0.1 public-corpus-materializer",
    }
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token is None:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.is_file():
            candidate = token_path.read_text(encoding="utf-8").strip()
            token = candidate or None
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _json_request(
    url: str, timeout: float = 60.0, maximum_attempts: int = 10
) -> Dict[str, Any]:
    """Read one public JSON endpoint with bounded transient-failure retries."""
    retryable_statuses = {429, 500, 502, 503, 504}
    for attempt in range(maximum_attempts):
        request = Request(
            url,
            headers=_hugging_face_headers("application/json"),
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise RuntimeError(
                    "public dataset endpoint returned a non-object response"
                )
            return payload
        except HTTPError as error:
            if error.code in {401, 403} and "huggingface.co" in url:
                raise PermissionError(
                    "Hugging Face denied access to the selected corpus. Accept its "
                    "dataset terms and provide HF_TOKEN (or log in with "
                    "huggingface-cli) without placing the token in a config file."
                ) from error
            if error.code not in retryable_statuses or attempt + 1 >= maximum_attempts:
                raise
            retry_after = error.headers.get("Retry-After") if error.headers else None
            try:
                requested_delay = float(retry_after) if retry_after is not None else 0.0
            except ValueError:
                requested_delay = 0.0
            sleep(max(requested_delay, min(30.0, float(2**attempt))))
        except URLError:
            if attempt + 1 >= maximum_attempts:
                raise
            sleep(min(30.0, float(2**attempt)))
    raise RuntimeError("unreachable public dataset retry state")


def _source_revision(dataset: str) -> str:
    payload = _json_request(f"https://huggingface.co/api/datasets/{dataset}/revision/main")
    revision = payload.get("sha")
    if not isinstance(revision, str) or len(revision) < 12:
        raise RuntimeError("Hugging Face did not return a source revision")
    return revision


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _parquet_inventory(catalog: Mapping[str, Any]) -> Dict[str, Any]:
    query = urlencode({"dataset": catalog["dataset"]})
    payload = _json_request(
        "https://datasets-server.huggingface.co/parquet?" + query,
        timeout=120.0,
    )
    rows = payload.get("parquet_files")
    if not isinstance(rows, list):
        raise RuntimeError("dataset server did not return a parquet inventory")
    files = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("config") != catalog["config"]
            or row.get("split") != catalog["split"]
        ):
            continue
        url = row.get("url")
        filename = row.get("filename")
        size = row.get("size")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise RuntimeError("parquet inventory contains an invalid URL")
        if not isinstance(filename, str) or not filename:
            filename = Path(url.split("?", 1)[0]).name
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            size = None
        files.append({"url": url, "filename": filename, "bytes": size})
    if not files:
        raise RuntimeError("no parquet files matched the selected dataset config/split")
    fingerprint = sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"files": files, "fingerprint": fingerprint}


def _download_parquet_file(
    entry: Mapping[str, Any], directory: Path
) -> Dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    url = str(entry["url"])
    expected_bytes = entry.get("bytes")
    key = sha256(url.encode("utf-8")).hexdigest()[:20]
    output_path = directory / f"{key}.parquet"
    partial_path = output_path.with_suffix(".parquet.partial")
    if output_path.is_file():
        if expected_bytes is None or output_path.stat().st_size == int(expected_bytes):
            return {
                "path": str(output_path.resolve()),
                "url": url,
                "bytes": output_path.stat().st_size,
                "sha256": _hash_file(output_path),
            }
        output_path.unlink()
    start = partial_path.stat().st_size if partial_path.is_file() else 0
    headers = _hugging_face_headers("application/octet-stream")
    if start:
        headers["Range"] = f"bytes={start}-"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=300.0) as response:
        response_status = getattr(response, "status", None)
        status = int(response_status if response_status is not None else response.getcode())
        if start and status != 206:
            start = 0
        mode = "ab" if start else "wb"
        with partial_path.open(mode) as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    if expected_bytes is not None and partial_path.stat().st_size != int(expected_bytes):
        raise RuntimeError(
            "downloaded parquet size disagrees with the frozen inventory"
        )
    os.replace(partial_path, output_path)
    return {
        "path": str(output_path.resolve()),
        "url": url,
        "bytes": output_path.stat().st_size,
        "sha256": _hash_file(output_path),
    }


def _path_inside(directory: Path, value: Any, label: str) -> Path:
    path = Path(str(value)).resolve()
    try:
        path.relative_to(directory.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the corpus directory") from exc
    return path


def _cached_manifest(directory: Path) -> Optional[Dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("status") != "complete" or manifest.get("schema_version") != 3:
            return None
        observed_fingerprint = manifest.get("manifest_fingerprint")
        unsigned = dict(manifest)
        unsigned.pop("manifest_fingerprint", None)
        expected_fingerprint = sha256(
            json.dumps(
                unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        if observed_fingerprint != expected_fingerprint:
            return None
        for split in ("train", "validation"):
            metadata = manifest["splits"][split]
            path = _path_inside(directory, metadata["path"], f"{split} path")
            if not path.is_file() or _hash_file(path) != metadata["file_sha256"]:
                return None
        tokenizer_manifest_path = _path_inside(
            directory, manifest["tokenizer_manifest_path"], "tokenizer manifest path"
        )
        load_tokenizer_manifest(tokenizer_manifest_path, verify_assets=True)
        stream_manifest_path = manifest.get("token_stream_manifest_path")
        if stream_manifest_path:
            load_token_stream_manifest(
                _path_inside(
                    directory, stream_manifest_path, "token stream manifest path"
                ),
                verify_files=True,
            )
        return manifest
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _materialize_split(
    *,
    catalog: Mapping[str, Any],
    start_offset: int,
    target_bytes: int,
    maximum_documents: int,
    output_path: Path,
    split_name: str,
    progress: ProgressCallback,
    completed_bytes: int,
    total_bytes: int,
    source_revision: str,
) -> Tuple[Dict[str, Any], int]:
    partial_path = output_path.with_name(f".{output_path.name}.partial")
    checkpoint_path = output_path.with_name(f".{output_path.name}.partial.json")
    contract = {
        "schema_version": 1,
        "source_revision": source_revision,
        "start_offset": start_offset,
        "target_bytes": target_bytes,
        "maximum_documents": maximum_documents,
    }
    checkpoint: Optional[Dict[str, Any]] = None
    if checkpoint_path.is_file() and partial_path.is_file():
        try:
            with checkpoint_path.open("r", encoding="utf-8") as checkpoint_handle:
                candidate = json.load(checkpoint_handle)
            if all(candidate.get(key) == value for key, value in contract.items()):
                file_position = int(candidate["file_position"])
                if 0 <= file_position <= partial_path.stat().st_size:
                    checkpoint = candidate
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            checkpoint = None

    if checkpoint is None:
        document_count = 0
        text_bytes = 0
        first_row: Optional[int] = None
        last_row: Optional[int] = None
        offset = start_offset
        handle = partial_path.open("wb")
    else:
        document_count = int(checkpoint["document_count"])
        text_bytes = int(checkpoint["text_bytes"])
        first_row = checkpoint.get("first_row")
        last_row = checkpoint.get("last_row")
        offset = int(checkpoint["offset"])
        handle = partial_path.open("r+b")
        handle.truncate(int(checkpoint["file_position"]))
        handle.seek(0, os.SEEK_END)
        if progress is not None:
            progress(
                {
                    "phase": "materializing",
                    "completed": min(
                        total_bytes, completed_bytes + min(text_bytes, target_bytes)
                    ),
                    "total": total_bytes,
                    "message": (
                        f"Resuming {split_name}: {document_count:,} documents, "
                        f"{text_bytes / (1024 * 1024):.1f} MiB"
                    ),
                }
            )

    with handle:
        try:
            while text_bytes < target_bytes and document_count < maximum_documents:
                remaining_documents = maximum_documents - document_count
                length = min(100, remaining_documents)
                query = urlencode(
                    {
                        "dataset": catalog["dataset"],
                        "config": catalog["config"],
                        "split": catalog["split"],
                        "offset": offset,
                        "length": length,
                    }
                )
                payload = _json_request(
                    "https://datasets-server.huggingface.co/rows?" + query
                )
                rows = payload.get("rows")
                if not isinstance(rows, list) or not rows:
                    raise RuntimeError(
                        f"dataset server returned no rows at offset {offset}"
                    )
                for wrapper in rows:
                    if not isinstance(wrapper, dict) or not isinstance(
                        wrapper.get("row"), dict
                    ):
                        raise RuntimeError("dataset server returned a malformed row")
                    row = wrapper["row"]
                    text = row.get(catalog["text_field"])
                    if not isinstance(text, str) or not text:
                        continue
                    truncated = wrapper.get("truncated_cells", [])
                    if catalog["text_field"] in truncated:
                        raise RuntimeError("dataset server truncated a text document")
                    row_index = wrapper.get("row_idx")
                    if isinstance(row_index, bool) or not isinstance(row_index, int):
                        raise RuntimeError("dataset server omitted a numeric row index")
                    source_id = (
                        row.get(catalog["id_field"])
                        if catalog.get("id_field")
                        else None
                    )
                    record = {
                        "text": text,
                        "source_row": row_index,
                        "source_id": source_id,
                    }
                    encoded_record = (
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    ).encode("utf-8")
                    handle.write(encoded_record)
                    encoded_size = len(text.encode("utf-8"))
                    text_bytes += encoded_size
                    document_count += 1
                    first_row = row_index if first_row is None else first_row
                    last_row = row_index
                    if text_bytes >= target_bytes or document_count >= maximum_documents:
                        break
                offset += len(rows)
                handle.flush()
                os.fsync(handle.fileno())
                atomic_write_json(
                    checkpoint_path,
                    {
                        **contract,
                        "offset": offset,
                        "document_count": document_count,
                        "text_bytes": text_bytes,
                        "first_row": first_row,
                        "last_row": last_row,
                        "file_position": handle.tell(),
                    },
                )
                if progress is not None:
                    progress(
                        {
                            "phase": "materializing",
                            "completed": min(
                                total_bytes,
                                completed_bytes + min(text_bytes, target_bytes),
                            ),
                            "total": total_bytes,
                            "message": (
                                f"Preparing {split_name}: {document_count:,} documents, "
                                f"{text_bytes / (1024 * 1024):.1f} MiB"
                            ),
                        }
                    )
            if text_bytes < target_bytes:
                raise RuntimeError(
                    f"{split_name} reached its document cap before the byte target"
                )
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            # The checkpoint points only to fsynced complete records, so a later
            # attempt can safely truncate and resume this exact source revision.
            raise
    os.replace(partial_path, output_path)
    checkpoint_path.unlink(missing_ok=True)
    return (
        {
            "path": str(output_path.resolve()),
            "documents": document_count,
            "text_bytes": text_bytes,
            "file_bytes": output_path.stat().st_size,
            "first_source_row": first_row,
            "last_source_row": last_row,
            "file_sha256": _hash_file(output_path),
        },
        text_bytes,
    )


def _materialize_split_parquet(
    *,
    catalog: Mapping[str, Any],
    inventory: Mapping[str, Any],
    parquet_cache: Path,
    start_offset: int,
    target_bytes: int,
    maximum_documents: int,
    output_path: Path,
    split_name: str,
    progress: ProgressCallback,
    completed_bytes: int,
    total_bytes: int,
    source_revision: str,
    source_batch_rows: int,
) -> Tuple[Dict[str, Any], int]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "parquet acquisition requires pyarrow; install the project runtime dependencies"
        ) from exc
    partial_path = output_path.with_name(f".{output_path.name}.partial")
    checkpoint_path = output_path.with_name(f".{output_path.name}.partial.json")
    contract = {
        "schema_version": 1,
        "source_revision": source_revision,
        "inventory_fingerprint": inventory["fingerprint"],
        "start_offset": start_offset,
        "target_bytes": target_bytes,
        "maximum_documents": maximum_documents,
        "source_batch_rows": source_batch_rows,
    }
    checkpoint: Optional[Dict[str, Any]] = None
    if checkpoint_path.is_file() and partial_path.is_file():
        try:
            with checkpoint_path.open("r", encoding="utf-8") as handle:
                candidate = json.load(handle)
            if all(candidate.get(key) == value for key, value in contract.items()):
                position = int(candidate["file_position"])
                if 0 <= position <= partial_path.stat().st_size:
                    checkpoint = candidate
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            checkpoint = None
    if checkpoint is None:
        document_count = 0
        text_bytes = 0
        first_row: Optional[int] = None
        last_row: Optional[int] = None
        next_source_row = start_offset
        source_files: List[Dict[str, Any]] = []
        handle = partial_path.open("wb")
    else:
        document_count = int(checkpoint["document_count"])
        text_bytes = int(checkpoint["text_bytes"])
        first_row = checkpoint.get("first_row")
        last_row = checkpoint.get("last_row")
        next_source_row = int(checkpoint["next_source_row"])
        source_files = list(checkpoint.get("source_files", []))
        handle = partial_path.open("r+b")
        handle.truncate(int(checkpoint["file_position"]))
        handle.seek(0, os.SEEK_END)
        if progress is not None:
            progress(
                {
                    "phase": "materializing",
                    "completed": min(
                        total_bytes, completed_bytes + min(text_bytes, target_bytes)
                    ),
                    "total": total_bytes,
                    "message": (
                        f"Resuming {split_name}: {document_count:,} documents, "
                        f"{text_bytes / (1024 * 1024):.1f} MiB"
                    ),
                }
            )
    known_file_urls = {str(row.get("url")) for row in source_files}
    global_row = 0
    with handle:
        for entry in inventory["files"]:
            local_file = _download_parquet_file(entry, parquet_cache)
            parquet_file = parquet.ParquetFile(local_file["path"])
            file_rows = int(parquet_file.metadata.num_rows)
            file_end = global_row + file_rows
            if file_end <= next_source_row:
                global_row = file_end
                continue
            if local_file["url"] not in known_file_urls:
                source_files.append(local_file)
                known_file_urls.add(local_file["url"])
            columns = [str(catalog["text_field"])]
            if catalog.get("id_field"):
                columns.append(str(catalog["id_field"]))
            batch_start = global_row
            for batch in parquet_file.iter_batches(
                batch_size=source_batch_rows, columns=columns
            ):
                batch_rows = len(batch)
                batch_end = batch_start + batch_rows
                if batch_end <= next_source_row:
                    batch_start = batch_end
                    continue
                values = batch.to_pydict()
                texts = values[str(catalog["text_field"])]
                source_ids = (
                    values[str(catalog["id_field"])]
                    if catalog.get("id_field")
                    else [None] * batch_rows
                )
                for local_index, text in enumerate(texts):
                    row_index = batch_start + local_index
                    if row_index < next_source_row or row_index < start_offset:
                        continue
                    if not isinstance(text, str) or not text:
                        continue
                    record = {
                        "text": text,
                        "source_row": row_index,
                        "source_id": source_ids[local_index],
                    }
                    handle.write(
                        (
                            json.dumps(
                                record,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        ).encode("utf-8")
                    )
                    text_bytes += len(text.encode("utf-8"))
                    document_count += 1
                    first_row = row_index if first_row is None else first_row
                    last_row = row_index
                    if (
                        text_bytes >= target_bytes
                        or document_count >= maximum_documents
                    ):
                        break
                next_source_row = batch_end
                handle.flush()
                os.fsync(handle.fileno())
                atomic_write_json(
                    checkpoint_path,
                    {
                        **contract,
                        "next_source_row": next_source_row,
                        "document_count": document_count,
                        "text_bytes": text_bytes,
                        "first_row": first_row,
                        "last_row": last_row,
                        "file_position": handle.tell(),
                        "source_files": source_files,
                    },
                )
                if progress is not None:
                    progress(
                        {
                            "phase": "materializing",
                            "completed": min(
                                total_bytes,
                                completed_bytes + min(text_bytes, target_bytes),
                            ),
                            "total": total_bytes,
                            "message": (
                                f"Preparing {split_name}: {document_count:,} documents, "
                                f"{text_bytes / (1024 * 1024):.1f} MiB"
                            ),
                        }
                    )
                if text_bytes >= target_bytes or document_count >= maximum_documents:
                    break
                batch_start = batch_end
            global_row = file_end
            if text_bytes >= target_bytes or document_count >= maximum_documents:
                break
        if text_bytes < target_bytes:
            raise RuntimeError(
                f"{split_name} reached the parquet inventory/document cap before "
                "the byte target"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial_path, output_path)
    checkpoint_path.unlink(missing_ok=True)
    return (
        {
            "path": str(output_path.resolve()),
            "documents": document_count,
            "text_bytes": text_bytes,
            "file_bytes": output_path.stat().st_size,
            "first_source_row": first_row,
            "last_source_row": last_row,
            "file_sha256": _hash_file(output_path),
            "source_parquet_files": source_files,
            "source_inventory_fingerprint": inventory["fingerprint"],
        },
        text_bytes,
    )


def _download_hugging_face_source_file(
    *,
    dataset: str,
    revision: str,
    repository_path: str,
    output_path: Path,
    maximum_attempts: int = 10,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f".{output_path.name}.partial")
    if output_path.is_file():
        return {
            "path": str(output_path.resolve()),
            "repository_path": repository_path,
            "source_revision": revision,
            "bytes": output_path.stat().st_size,
            "sha256": _hash_file(output_path),
        }
    url = (
        f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/"
        f"{repository_path}"
    )
    retryable_statuses = {429, 500, 502, 503, 504}
    for attempt in range(maximum_attempts):
        start = partial_path.stat().st_size if partial_path.is_file() else 0
        headers = _hugging_face_headers("application/octet-stream")
        if start:
            headers["Range"] = f"bytes={start}-"
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=300.0) as response:
                response_status = getattr(response, "status", None)
                status = int(
                    response_status
                    if response_status is not None
                    else response.getcode()
                )
                if start and status != 206:
                    start = 0
                mode = "ab" if start else "wb"
                with partial_path.open(mode) as handle:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            break
        except HTTPError as error:
            if error.code in {401, 403}:
                raise PermissionError(
                    "Hugging Face denied a SlimPajama shard. Accept the dataset "
                    "terms and provide HF_TOKEN (or log in with huggingface-cli)."
                ) from error
            if (
                error.code not in retryable_statuses
                or attempt + 1 >= maximum_attempts
            ):
                raise
            retry_after = error.headers.get("Retry-After") if error.headers else None
            try:
                requested_delay = (
                    float(retry_after) if retry_after is not None else 0.0
                )
            except ValueError:
                requested_delay = 0.0
            sleep(max(requested_delay, min(30.0, float(2**attempt))))
        except URLError:
            if attempt + 1 >= maximum_attempts:
                raise
            sleep(min(30.0, float(2**attempt)))
    else:
        raise RuntimeError("unreachable SlimPajama shard retry state")
    if not partial_path.is_file() or partial_path.stat().st_size == 0:
        raise RuntimeError("Hugging Face returned an empty source shard")
    os.replace(partial_path, output_path)
    return {
        "path": str(output_path.resolve()),
        "repository_path": repository_path,
        "source_revision": revision,
        "bytes": output_path.stat().st_size,
        "sha256": _hash_file(output_path),
    }


def _iter_binary_lines(stream: Any) -> Iterator[bytes]:
    pending = b""
    while True:
        chunk = stream.read(8 * 1024 * 1024)
        if not chunk:
            break
        parts = (pending + chunk).split(b"\n")
        pending = parts.pop()
        for line in parts:
            yield line
    if pending:
        yield pending


def _materialize_split_zstd_shards(
    *,
    catalog: Mapping[str, Any],
    source_split: str,
    target_bytes: int,
    maximum_documents: int,
    output_path: Path,
    shard_cache: Path,
    split_name: str,
    progress: ProgressCallback,
    completed_bytes: int,
    total_bytes: int,
    source_revision: str,
) -> Tuple[Dict[str, Any], int]:
    try:
        import pyarrow as arrow
    except ImportError as exc:
        raise RuntimeError(
            "SlimPajama zstd-shard acquisition requires pyarrow"
        ) from exc
    direct = catalog.get("direct_shards")
    if not isinstance(direct, Mapping) or not isinstance(
        direct.get(source_split), Mapping
    ):
        raise ValueError("selected corpus has no direct shard contract for this split")
    split_contract = dict(direct[source_split])
    shard_count = int(split_contract["count"])
    path_template = str(split_contract["path_template"])
    partial_path = output_path.with_name(f".{output_path.name}.partial")
    checkpoint_path = output_path.with_name(f".{output_path.name}.partial.json")
    contract = {
        "schema_version": 1,
        "dataset": str(catalog["dataset"]),
        "source_revision": source_revision,
        "source_split": source_split,
        "target_bytes": target_bytes,
        "maximum_documents": maximum_documents,
        "shard_count": shard_count,
        "path_template": path_template,
    }
    checkpoint: Optional[Dict[str, Any]] = None
    if checkpoint_path.is_file() and partial_path.is_file():
        try:
            candidate = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if all(candidate.get(key) == value for key, value in contract.items()):
                position = int(candidate["file_position"])
                if 0 <= position <= partial_path.stat().st_size:
                    checkpoint = candidate
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            checkpoint = None
    if checkpoint is None:
        document_count = 0
        text_bytes = 0
        next_shard_index = 0
        source_files: List[Dict[str, Any]] = []
        handle = partial_path.open("wb")
    else:
        document_count = int(checkpoint["document_count"])
        text_bytes = int(checkpoint["text_bytes"])
        next_shard_index = int(checkpoint["next_shard_index"])
        source_files = list(checkpoint.get("source_files", []))
        handle = partial_path.open("r+b")
        handle.truncate(int(checkpoint["file_position"]))
        handle.seek(0, os.SEEK_END)
        if progress is not None:
            progress(
                {
                    "phase": "materializing",
                    "completed": min(
                        total_bytes, completed_bytes + min(text_bytes, target_bytes)
                    ),
                    "total": total_bytes,
                    "message": (
                        f"Resuming {split_name}: {document_count:,} documents, "
                        f"{text_bytes / (1024 * 1024):.1f} MiB"
                    ),
                }
            )

    with handle:
        for shard_index in range(next_shard_index, shard_count):
            repository_path = path_template.format(index=shard_index)
            local_path = (
                shard_cache
                / source_revision
                / source_split
                / f"{shard_index:06d}.jsonl.zst"
            )
            source_file = _download_hugging_face_source_file(
                dataset=str(catalog["dataset"]),
                revision=source_revision,
                repository_path=repository_path,
                output_path=local_path,
            )
            source_file["shard_index"] = shard_index
            source_files.append(source_file)
            with arrow.input_stream(str(local_path), compression="detect") as stream:
                record_index = 0
                for raw in _iter_binary_lines(stream):
                    if text_bytes >= target_bytes or document_count >= maximum_documents:
                        break
                    if not raw.strip():
                        continue
                    try:
                        row = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError(
                            f"malformed SlimPajama JSONL in {repository_path}"
                        ) from exc
                    text = row.get(str(catalog["text_field"])) if isinstance(row, dict) else None
                    if not isinstance(text, str) or not text:
                        record_index += 1
                        continue
                    record = {
                        "text": text,
                        "source_row": document_count,
                        "source_id": (
                            f"{source_split}/chunk{split_contract['chunk']}/"
                            f"{shard_index}:{record_index}"
                        ),
                    }
                    handle.write(
                        (
                            json.dumps(
                                record, ensure_ascii=False, separators=(",", ":")
                            )
                            + "\n"
                        ).encode("utf-8")
                    )
                    text_bytes += len(text.encode("utf-8"))
                    document_count += 1
                    record_index += 1
            next_shard_index = shard_index + 1
            handle.flush()
            os.fsync(handle.fileno())
            atomic_write_json(
                checkpoint_path,
                {
                    **contract,
                    "next_shard_index": next_shard_index,
                    "document_count": document_count,
                    "text_bytes": text_bytes,
                    "file_position": handle.tell(),
                    "source_files": source_files,
                },
            )
            if progress is not None:
                progress(
                    {
                        "phase": "materializing",
                        "completed": min(
                            total_bytes,
                            completed_bytes + min(text_bytes, target_bytes),
                        ),
                        "total": total_bytes,
                        "message": (
                            f"Preparing {split_name}: {document_count:,} documents, "
                            f"{text_bytes / (1024 * 1024):.1f} MiB, "
                            f"{next_shard_index:,} source shards"
                        ),
                    }
                )
            if text_bytes >= target_bytes or document_count >= maximum_documents:
                break
        if text_bytes < target_bytes:
            raise RuntimeError(
                f"{split_name} reached its shard inventory/document cap before "
                "the byte target"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial_path, output_path)
    checkpoint_path.unlink(missing_ok=True)
    return (
        {
            "path": str(output_path.resolve()),
            "documents": document_count,
            "text_bytes": text_bytes,
            "file_bytes": output_path.stat().st_size,
            "first_source_row": 0,
            "last_source_row": document_count - 1,
            "source_split": source_split,
            "file_sha256": _hash_file(output_path),
            "source_direct_files": source_files,
            "source_inventory_fingerprint": sha256(
                json.dumps(
                    [
                        {
                            "repository_path": row["repository_path"],
                            "bytes": row["bytes"],
                            "sha256": row["sha256"],
                        }
                        for row in source_files
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
        text_bytes,
    )


def _concatenate_training_segments(
    segments: List[Dict[str, Any]], output_path: Path
) -> Dict[str, Any]:
    if len(segments) < 2:
        raise ValueError("segmented training requires at least two source segments")
    intervals = []
    for row in segments:
        intervals.extend(_source_intervals(row))
    for index, current in enumerate(intervals):
        for other in intervals[index + 1 :]:
            if not (current[1] < other[0] or other[1] < current[0]):
                raise ValueError("segmented training source rows overlap")

    partial_path = output_path.with_name(f".{output_path.name}.partial")
    with partial_path.open("wb") as destination:
        for row in segments:
            source_path = Path(str(row["path"])).resolve()
            with source_path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(partial_path, output_path)

    source_files: List[Dict[str, Any]] = []
    known_urls = set()
    for row in segments:
        for source_file in row.get("source_parquet_files", []):
            url = str(source_file.get("url"))
            if url not in known_urls:
                source_files.append(dict(source_file))
                known_urls.add(url)
    segment_metadata = [
        {
            key: value
            for key, value in row.items()
            if key
            in {
                "documents",
                "text_bytes",
                "file_bytes",
                "file_sha256",
                "first_source_row",
                "last_source_row",
                "source_inventory_fingerprint",
                "source_parquet_files",
            }
        }
        for row in segments
    ]
    return {
        "path": str(output_path.resolve()),
        "documents": sum(int(row["documents"]) for row in segments),
        "text_bytes": sum(int(row["text_bytes"]) for row in segments),
        "file_bytes": output_path.stat().st_size,
        "first_source_row": min(int(row["first_source_row"]) for row in segments),
        "last_source_row": max(int(row["last_source_row"]) for row in segments),
        "file_sha256": _hash_file(output_path),
        "source_segments": segment_metadata,
        "source_parquet_files": source_files,
        "source_inventory_fingerprint": segments[0].get(
            "source_inventory_fingerprint"
        ),
    }


def materialize_public_corpus(
    spec: PublicCorpusSpec,
    output_root: Path,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    catalog = PUBLIC_CORPUS_CATALOG[spec.source]
    directory = output_root / spec.fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    cached = _cached_manifest(directory)
    if cached is not None:
        if progress is not None:
            progress(
                {
                    "phase": "complete",
                    "completed": spec.train_bytes + spec.validation_bytes,
                    "total": spec.train_bytes + spec.validation_bytes,
                    "message": "Frozen corpus already materialized",
                }
            )
        return cached

    total_bytes = spec.train_bytes + spec.validation_bytes
    if progress is not None:
        progress(
            {
                "phase": "resolving_source",
                "completed": 0,
                "total": total_bytes,
                "message": f"Resolving {catalog['name']} source revision",
            }
        )
    revision_before = _source_revision(str(catalog["dataset"]))
    inventory_before = (
        _parquet_inventory(catalog)
        if spec.acquisition_backend == "parquet"
        else None
    )
    train_path = directory / "train.jsonl"
    validation_path = directory / "validation.jsonl"
    if spec.acquisition_backend == "zstd_shards":
        shard_cache = directory / "source-shards"
        train, _ = _materialize_split_zstd_shards(
            catalog=catalog,
            source_split="train",
            target_bytes=spec.train_bytes,
            maximum_documents=spec.maximum_documents_per_split,
            output_path=train_path,
            shard_cache=shard_cache,
            split_name="training corpus",
            progress=progress,
            completed_bytes=0,
            total_bytes=total_bytes,
            source_revision=revision_before,
        )
        validation, _ = _materialize_split_zstd_shards(
            catalog=catalog,
            source_split="validation",
            target_bytes=spec.validation_bytes,
            maximum_documents=spec.maximum_documents_per_split,
            output_path=validation_path,
            shard_cache=shard_cache,
            split_name="held-out corpus",
            progress=progress,
            completed_bytes=spec.train_bytes,
            total_bytes=total_bytes,
            source_revision=revision_before,
        )
    elif inventory_before is None:
        train, _ = _materialize_split(
            catalog=catalog,
            start_offset=int(catalog["train_offset"]),
            target_bytes=spec.train_bytes,
            maximum_documents=spec.maximum_documents_per_split,
            output_path=train_path,
            split_name="training corpus",
            progress=progress,
            completed_bytes=0,
            total_bytes=total_bytes,
            source_revision=revision_before,
        )
        validation, _ = _materialize_split(
            catalog=catalog,
            start_offset=int(catalog["validation_offset"]),
            target_bytes=spec.validation_bytes,
            maximum_documents=spec.maximum_documents_per_split,
            output_path=validation_path,
            split_name="held-out corpus",
            progress=progress,
            completed_bytes=spec.train_bytes,
            total_bytes=total_bytes,
            source_revision=revision_before,
        )
    else:
        parquet_cache = directory / "source-parquet"
        if spec.train_primary_bytes is None:
            train, _ = _materialize_split_parquet(
                catalog=catalog,
                inventory=inventory_before,
                parquet_cache=parquet_cache,
                start_offset=int(catalog["train_offset"]),
                target_bytes=spec.train_bytes,
                maximum_documents=spec.maximum_documents_per_split,
                output_path=train_path,
                split_name="training corpus",
                progress=progress,
                completed_bytes=0,
                total_bytes=total_bytes,
                source_revision=revision_before,
                source_batch_rows=spec.source_batch_rows,
            )
        else:
            primary_path = directory / "train-primary.jsonl"
            secondary_path = directory / "train-secondary.jsonl"
            primary, _ = _materialize_split_parquet(
                catalog=catalog,
                inventory=inventory_before,
                parquet_cache=parquet_cache,
                start_offset=int(catalog["train_offset"]),
                target_bytes=int(spec.train_primary_bytes),
                maximum_documents=spec.maximum_documents_per_split,
                output_path=primary_path,
                split_name="primary training corpus",
                progress=progress,
                completed_bytes=0,
                total_bytes=total_bytes,
                source_revision=revision_before,
                source_batch_rows=spec.source_batch_rows,
            )
            secondary_bytes = spec.train_bytes - int(spec.train_primary_bytes)
            secondary, _ = _materialize_split_parquet(
                catalog=catalog,
                inventory=inventory_before,
                parquet_cache=parquet_cache,
                start_offset=int(spec.train_secondary_offset),
                target_bytes=secondary_bytes,
                maximum_documents=spec.maximum_documents_per_split,
                output_path=secondary_path,
                split_name="secondary training corpus",
                progress=progress,
                completed_bytes=int(spec.train_primary_bytes),
                total_bytes=total_bytes,
                source_revision=revision_before,
                source_batch_rows=spec.source_batch_rows,
            )
            train = _concatenate_training_segments(
                [primary, secondary], train_path
            )
        validation, _ = _materialize_split_parquet(
            catalog=catalog,
            inventory=inventory_before,
            parquet_cache=parquet_cache,
            start_offset=int(catalog["validation_offset"]),
            target_bytes=spec.validation_bytes,
            maximum_documents=spec.maximum_documents_per_split,
            output_path=validation_path,
            split_name="held-out corpus",
            progress=progress,
            completed_bytes=spec.train_bytes,
            total_bytes=total_bytes,
            source_revision=revision_before,
            source_batch_rows=spec.source_batch_rows,
        )
    revision_after = _source_revision(str(catalog["dataset"]))
    if revision_before != revision_after:
        raise RuntimeError(
            "public corpus source revision changed during materialization; rerun "
            "to freeze one revision"
        )
    if inventory_before is not None:
        inventory_after = _parquet_inventory(catalog)
        if inventory_before["fingerprint"] != inventory_after["fingerprint"]:
            raise RuntimeError(
                "parquet source inventory changed during materialization; rerun "
                "to freeze one conversion"
            )
    if not _source_ranges_are_disjoint(train, validation):
        raise RuntimeError("training and validation source-row ranges overlap")
    tokenizer_directory = directory / "tokenizer"
    token_stream_manifest_path: Optional[Path] = None

    def postprocess_progress(event: Dict[str, Any]) -> None:
        if progress is not None:
            progress(
                {
                    **event,
                    "completed": total_bytes,
                    "total": total_bytes,
                }
            )

    if spec.tokenizer == "byte_v1":
        tokenizer_directory.mkdir(parents=True, exist_ok=True)
        tokenizer_manifest = builtin_byte_tokenizer_manifest()
        tokenizer_manifest_path = tokenizer_directory / "manifest.json"
        atomic_write_json(tokenizer_manifest_path, tokenizer_manifest)
        corpus = TokenizedTextCorpus(
            TokenizedTextSpec(
                str(train_path.resolve()),
                str(validation_path.resolve()),
                tokenizer="byte_v1",
                text_field="text",
                maximum_bytes=max(
                    train_path.stat().st_size, validation_path.stat().st_size
                )
                + 1024,
            ),
            context_length=128,
            vocab_size=260,
        )
        corpus_fingerprint = corpus.fingerprint
        dataset_identity_fingerprint = corpus.identity_fingerprint
        training_tokens = int(corpus.train_tokens.numel())
        validation_tokens = int(corpus.validation_tokens.numel())
    else:
        resolved_tokenizer = resolve_pinned_tokenizer(
            spec.tokenizer, tokenizer_directory, postprocess_progress
        )
        tokenizer_manifest = resolved_tokenizer.manifest
        tokenizer_manifest_path = resolved_tokenizer.manifest_path
        token_stream_directory = directory / "token-streams"
        token_stream_manifest = materialize_pinned_token_streams(
            tokenizer=resolved_tokenizer,
            train_path=train_path,
            validation_path=validation_path,
            output_directory=token_stream_directory,
            text_field="text",
            shard_token_limit=spec.token_shard_tokens,
            progress=postprocess_progress,
            source_fingerprints={
                "train": str(train["file_sha256"]),
                "validation": str(validation["file_sha256"]),
            },
        )
        token_stream_manifest_path = token_stream_directory / "manifest.json"
        load_token_stream_manifest(token_stream_manifest_path, verify_files=True)
        corpus_fingerprint = str(token_stream_manifest["content_fingerprint"])
        dataset_identity_fingerprint = str(token_stream_manifest["fingerprint"])
        training_tokens = int(token_stream_manifest["splits"]["train"]["tokens"])
        validation_tokens = int(
            token_stream_manifest["splits"]["validation"]["tokens"]
        )
    manifest: Dict[str, Any] = {
        "schema_version": 3,
        "status": "complete",
        "id": spec.fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spec": {
            key: value for key, value in asdict(spec).items() if value is not None
        },
        "source": {
            "provider": (
                "Hugging Face Parquet export"
                if spec.acquisition_backend == "parquet"
                else "Hugging Face immutable compressed shards"
                if spec.acquisition_backend == "zstd_shards"
                else "Hugging Face Dataset Viewer API"
            ),
            "dataset": catalog["dataset"],
            "config": catalog["config"],
            "split": catalog["split"],
            "revision": revision_before,
            "license": catalog["license"],
            "data_card_url": catalog["data_card_url"],
            "acquisition_backend": spec.acquisition_backend,
            "parquet_inventory_fingerprint": (
                inventory_before["fingerprint"]
                if inventory_before is not None
                else None
            ),
        },
        "tokenizer": spec.tokenizer,
        "tokenizer_definition_fingerprint": tokenizer_definition_fingerprint(
            spec.tokenizer
        ),
        "tokenizer_fingerprint": tokenizer_manifest["fingerprint"],
        "tokenizer_manifest_path": str(tokenizer_manifest_path.resolve()),
        "tokenizer_vocab_size": tokenizer_manifest["vocab_size"],
        "token_stream_manifest_path": (
            str(token_stream_manifest_path.resolve())
            if token_stream_manifest_path is not None
            else None
        ),
        "corpus_fingerprint": corpus_fingerprint,
        "dataset_identity_fingerprint": dataset_identity_fingerprint,
        "training_tokens": training_tokens,
        "validation_tokens": validation_tokens,
        "splits": {"train": train, "validation": validation},
    }
    manifest["manifest_fingerprint"] = sha256(
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    atomic_write_json(directory / "manifest.json", manifest)
    if progress is not None:
        progress(
            {
                "phase": "complete",
                "completed": total_bytes,
                "total": total_bytes,
                "message": (
                    f"Frozen corpus ready · {manifest['training_tokens']:,} train tokens · "
                    f"fingerprint {corpus_fingerprint[:12]}"
                ),
            }
        )
    return manifest


def _load_verified_raw_snapshot(manifest_path: Path) -> Dict[str, Any]:
    """Load the immutable raw-text portion of a completed public corpus."""

    resolved_manifest = manifest_path.expanduser().resolve()
    directory = resolved_manifest.parent
    with resolved_manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("source corpus manifest must contain an object")
    if manifest.get("schema_version") != 3 or manifest.get("status") != "complete":
        raise ValueError("source corpus manifest must be a completed schema-3 snapshot")
    observed_fingerprint = manifest.get("manifest_fingerprint")
    unsigned = dict(manifest)
    unsigned.pop("manifest_fingerprint", None)
    expected_fingerprint = sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    if observed_fingerprint != expected_fingerprint:
        raise ValueError("source corpus manifest fingerprint mismatch")
    verified_splits: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "validation"):
        metadata = manifest.get("splits", {}).get(split)
        if not isinstance(metadata, dict):
            raise ValueError(f"source corpus is missing {split} metadata")
        path = _path_inside(directory, metadata.get("path"), f"{split} path")
        if not path.is_file() or _hash_file(path) != metadata.get("file_sha256"):
            raise ValueError(f"source corpus {split} bytes failed verification")
        verified_splits[split] = {**metadata, "path": str(path)}
    if not _source_ranges_are_disjoint(
        verified_splits["train"], verified_splits["validation"]
    ):
        raise ValueError("source corpus train and validation rows overlap")
    return {
        **manifest,
        "manifest_path": str(resolved_manifest),
        "splits": verified_splits,
    }


def retokenize_public_corpus(
    source_manifest_path: Path,
    *,
    tokenizer_id: str,
    output_root: Path,
    token_shard_tokens: int = 16_777_216,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    """Retokenize verified raw text without reacquiring the public dataset."""

    if tokenizer_id not in PINNED_TOKENIZER_REGISTRY:
        raise ValueError("retokenization requires an allow-listed pinned tokenizer")
    if (
        isinstance(token_shard_tokens, bool)
        or not isinstance(token_shard_tokens, int)
        or not 1_024 <= token_shard_tokens <= 268_435_456
    ):
        raise ValueError("token_shard_tokens must be in [1024, 268435456]")
    source = _load_verified_raw_snapshot(source_manifest_path)
    identity_payload = {
        "source_manifest_fingerprint": source["manifest_fingerprint"],
        "tokenizer_definition_fingerprint": tokenizer_definition_fingerprint(
            tokenizer_id
        ),
        "token_stream_format": TOKEN_STREAM_FORMAT,
        "packing_contract": TOKEN_STREAM_PACKING_CONTRACT,
        "token_shard_tokens": token_shard_tokens,
    }
    identity = sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    directory = output_root / identity
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
            fingerprint = cached.get("manifest_fingerprint")
            unsigned = dict(cached)
            unsigned.pop("manifest_fingerprint", None)
            expected = sha256(
                json.dumps(
                    unsigned,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            tokenizer_manifest = directory / "tokenizer" / "manifest.json"
            stream_manifest = directory / "token-streams" / "manifest.json"
            if (
                cached.get("status") == "complete"
                and cached.get("schema_version") == 1
                and fingerprint == expected
                and cached.get("source_manifest_fingerprint")
                == source["manifest_fingerprint"]
                and cached.get("tokenizer") == tokenizer_id
                and cached.get("tokenizer_definition_fingerprint")
                == identity_payload["tokenizer_definition_fingerprint"]
                and cached.get("token_stream_format") == TOKEN_STREAM_FORMAT
                and cached.get("packing_contract")
                == TOKEN_STREAM_PACKING_CONTRACT
                and cached.get("token_shard_tokens") == token_shard_tokens
            ):
                verified_tokenizer = load_tokenizer_manifest(
                    tokenizer_manifest, verify_assets=True
                )
                verified_stream = load_token_stream_manifest(
                    stream_manifest, verify_files=True
                )
                if (
                    verified_tokenizer.get("id") != tokenizer_id
                    or verified_tokenizer.get("fingerprint")
                    != cached.get("tokenizer_fingerprint")
                    or verified_stream.get("fingerprint")
                    != cached.get("dataset_identity_fingerprint")
                    or verified_stream.get("content_fingerprint")
                    != cached.get("corpus_fingerprint")
                ):
                    raise ValueError("retokenized corpus cache identity mismatch")
                if progress is not None:
                    progress(
                        {
                            "phase": "complete",
                            "completed": 1,
                            "total": 1,
                            "message": "Retokenized corpus already materialized",
                        }
                    )
                return cached
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    tokenizer = resolve_pinned_tokenizer(
        tokenizer_id, directory / "tokenizer", progress
    )
    stream = materialize_pinned_token_streams(
        tokenizer=tokenizer,
        train_path=Path(source["splits"]["train"]["path"]),
        validation_path=Path(source["splits"]["validation"]["path"]),
        output_directory=directory / "token-streams",
        text_field="text",
        shard_token_limit=token_shard_tokens,
        progress=progress,
        source_fingerprints={
            "train": str(source["splits"]["train"]["file_sha256"]),
            "validation": str(source["splits"]["validation"]["file_sha256"]),
        },
    )
    stream_manifest_path = directory / "token-streams" / "manifest.json"
    load_token_stream_manifest(stream_manifest_path, verify_files=True)
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "kind": "retokenized_public_corpus",
        "id": identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest_path": source["manifest_path"],
        "source_manifest_fingerprint": source["manifest_fingerprint"],
        "source": source["source"],
        "source_splits": source["splits"],
        "tokenizer": tokenizer_id,
        "tokenizer_definition_fingerprint": tokenizer_definition_fingerprint(
            tokenizer_id
        ),
        "tokenizer_fingerprint": tokenizer.manifest["fingerprint"],
        "tokenizer_manifest_path": str(tokenizer.manifest_path.resolve()),
        "tokenizer_vocab_size": tokenizer.manifest["vocab_size"],
        "token_stream_manifest_path": str(stream_manifest_path.resolve()),
        "corpus_fingerprint": stream["content_fingerprint"],
        "dataset_identity_fingerprint": stream["fingerprint"],
        "training_tokens": stream["splits"]["train"]["tokens"],
        "validation_tokens": stream["splits"]["validation"]["tokens"],
        "token_stream_format": TOKEN_STREAM_FORMAT,
        "packing_contract": TOKEN_STREAM_PACKING_CONTRACT,
        "token_shard_tokens": token_shard_tokens,
    }
    manifest["manifest_fingerprint"] = sha256(
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    atomic_write_json(manifest_path, manifest)
    if progress is not None:
        progress(
            {
                "phase": "complete",
                "completed": 1,
                "total": 1,
                "message": (
                    f"Retokenized corpus ready · {manifest['training_tokens']:,} "
                    f"train tokens · fingerprint {manifest['corpus_fingerprint'][:12]}"
                ),
            }
        )
    return manifest
