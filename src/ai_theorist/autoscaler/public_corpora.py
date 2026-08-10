from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from time import sleep
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .pretraining import TokenizedTextCorpus, TokenizedTextSpec
from .study import atomic_write_json
from .tokenization import (
    PINNED_TOKENIZER_REGISTRY,
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
}


@dataclass(frozen=True)
class PublicCorpusSpec:
    source: str = "fineweb_edu"
    tokenizer: str = "byte_v1"
    train_bytes: int = 33_554_432
    validation_bytes: int = 4_194_304
    maximum_documents_per_split: int = 50_000
    token_shard_tokens: int = 16_777_216

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PublicCorpusSpec":
        allowed = {
            "source",
            "tokenizer",
            "train_bytes",
            "validation_bytes",
            "maximum_documents_per_split",
            "token_shard_tokens",
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
        for name, value, minimum, maximum in (
            ("train_bytes", result.train_bytes, 65_536, 536_870_912),
            ("validation_bytes", result.validation_bytes, 16_384, 67_108_864),
            (
                "maximum_documents_per_split",
                result.maximum_documents_per_split,
                100,
                100_000,
            ),
            ("token_shard_tokens", result.token_shard_tokens, 1_024, 268_435_456),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < minimum or value > maximum:
                raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
        return result

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                **asdict(self),
                "tokenizer_definition_fingerprint": tokenizer_definition_fingerprint(
                    self.tokenizer
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8")).hexdigest()[:16]


def public_corpus_catalog() -> List[Dict[str, Any]]:
    return [
        {"id": source, **dict(metadata)}
        for source, metadata in PUBLIC_CORPUS_CATALOG.items()
    ]


def _json_request(
    url: str, timeout: float = 60.0, maximum_attempts: int = 10
) -> Dict[str, Any]:
    """Read one public JSON endpoint with bounded transient-failure retries."""
    retryable_statuses = {429, 500, 502, 503, 504}
    for attempt in range(maximum_attempts):
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "ai-theorist-autoscaler/0.1 public-corpus-materializer",
            },
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
        if manifest.get("status") != "complete" or manifest.get("schema_version") != 2:
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
    train_path = directory / "train.jsonl"
    validation_path = directory / "validation.jsonl"
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
    revision_after = _source_revision(str(catalog["dataset"]))
    if revision_before != revision_after:
        raise RuntimeError(
            "FineWeb source revision changed during materialization; rerun to freeze one revision"
        )
    if not (
        int(train["last_source_row"]) < int(validation["first_source_row"])
        or int(validation["last_source_row"]) < int(train["first_source_row"])
    ):
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
        "schema_version": 2,
        "status": "complete",
        "id": spec.fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spec": asdict(spec),
        "source": {
            "provider": "Hugging Face Dataset Viewer API",
            "dataset": catalog["dataset"],
            "config": catalog["config"],
            "split": catalog["split"],
            "revision": revision_before,
            "license": catalog["license"],
            "data_card_url": catalog["data_card_url"],
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
