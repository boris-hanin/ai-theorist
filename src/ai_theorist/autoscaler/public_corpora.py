from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .pretraining import TokenizedTextCorpus, TokenizedTextSpec
from .study import atomic_write_json


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
    train_bytes: int = 33_554_432
    validation_bytes: int = 4_194_304
    maximum_documents_per_split: int = 50_000

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PublicCorpusSpec":
        allowed = {
            "source",
            "train_bytes",
            "validation_bytes",
            "maximum_documents_per_split",
        }
        extras = sorted(set(payload) - allowed)
        if extras:
            raise ValueError(f"Unknown public corpus field(s): {', '.join(extras)}")
        result = cls(**dict(payload))
        if result.source not in PUBLIC_CORPUS_CATALOG:
            raise ValueError(
                "source must be one of " + ", ".join(sorted(PUBLIC_CORPUS_CATALOG))
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
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < minimum or value > maximum:
                raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
        return result

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()[:16]


def public_corpus_catalog() -> List[Dict[str, Any]]:
    return [
        {"id": source, **dict(metadata)}
        for source, metadata in PUBLIC_CORPUS_CATALOG.items()
    ]


def _json_request(url: str, timeout: float = 60.0) -> Dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "ai-theorist-autoscaler/0.1 public-corpus-materializer",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError("public dataset endpoint returned a non-object response")
    return payload


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


def _cached_manifest(directory: Path) -> Optional[Dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("status") != "complete":
            return None
        for split in ("train", "validation"):
            metadata = manifest["splits"][split]
            path = Path(metadata["path"])
            if not path.is_file() or _hash_file(path) != metadata["file_sha256"]:
                return None
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
) -> Tuple[Dict[str, Any], int]:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent, text=True
    )
    document_count = 0
    text_bytes = 0
    first_row: Optional[int] = None
    last_row: Optional[int] = None
    offset = start_offset
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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
                    handle.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    )
                    handle.write("\n")
                    encoded_size = len(text.encode("utf-8"))
                    text_bytes += encoded_size
                    document_count += 1
                    first_row = row_index if first_row is None else first_row
                    last_row = row_index
                    if text_bytes >= target_bytes or document_count >= maximum_documents:
                        break
                offset += len(rows)
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
            handle.flush()
            os.fsync(handle.fileno())
        if text_bytes < target_bytes:
            raise RuntimeError(
                f"{split_name} reached its document cap before the byte target"
            )
        os.replace(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
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

    corpus = TokenizedTextCorpus(
        TokenizedTextSpec(
            str(train_path.resolve()),
            str(validation_path.resolve()),
            tokenizer="byte_v1",
            text_field="text",
            maximum_bytes=max(train_path.stat().st_size, validation_path.stat().st_size)
            + 1024,
        ),
        context_length=128,
        vocab_size=260,
    )
    manifest: Dict[str, Any] = {
        "schema_version": 1,
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
        "tokenizer": "byte_v1",
        "corpus_fingerprint": corpus.fingerprint,
        "training_tokens": int(corpus.train_tokens.numel()),
        "validation_tokens": int(corpus.validation_tokens.numel()),
        "splits": {"train": train, "validation": validation},
    }
    atomic_write_json(directory / "manifest.json", manifest)
    if progress is not None:
        progress(
            {
                "phase": "complete",
                "completed": total_bytes,
                "total": total_bytes,
                "message": (
                    f"Frozen corpus ready · {manifest['training_tokens']:,} train tokens · "
                    f"fingerprint {corpus.fingerprint[:12]}"
                ),
            }
        )
    return manifest
