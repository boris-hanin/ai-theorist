from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from time import sleep
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .study import atomic_write_json


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]
TOKENIZER_MANIFEST_SCHEMA_VERSION = 1
TOKEN_STREAM_MANIFEST_SCHEMA_VERSION = 1
PINNED_TOKENIZERS_PACKAGE_VERSION = "0.21.4"


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_uint32_shard(path: Path) -> Tuple[str, int, int, int]:
    digest = sha256()
    token_count = 0
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            if len(chunk) % 4:
                raise ValueError("uint32 token shard has an unaligned byte length")
            digest.update(chunk)
            values = np.frombuffer(chunk, dtype="<u4")
            if values.size:
                chunk_minimum = int(values.min())
                chunk_maximum = int(values.max())
                minimum = (
                    chunk_minimum if minimum is None else min(minimum, chunk_minimum)
                )
                maximum = (
                    chunk_maximum if maximum is None else max(maximum, chunk_maximum)
                )
                token_count += int(values.size)
    if minimum is None or maximum is None:
        raise ValueError("uint32 token shard is empty")
    return digest.hexdigest(), token_count, minimum, maximum


def _safe_manifest_path(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{label} must be a non-empty relative path")
    root = base.resolve()
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its manifest directory") from exc
    return candidate


def _token_id_hash(token_ids: Sequence[int]) -> str:
    digest = sha256()
    for token_id in token_ids:
        if token_id < 0 or token_id >= 2**32:
            raise ValueError("token IDs must fit in unsigned 32-bit storage")
        digest.update(int(token_id).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


@dataclass(frozen=True)
class TokenizerAssetDefinition:
    name: str
    sha256: str


@dataclass(frozen=True)
class TokenizerCanaryDefinition:
    text: str
    token_count: int
    token_ids_sha256: str


@dataclass(frozen=True)
class PinnedTokenizerDefinition:
    id: str
    name: str
    implementation: str
    repository: str
    revision: str
    tokenizer_file: str
    package: str
    package_version: str
    vocab_size: int
    special_tokens: Mapping[str, Optional[str]]
    special_token_ids: Mapping[str, Optional[int]]
    document_separator_token_id: int
    assets: Tuple[TokenizerAssetDefinition, ...]
    canaries: Tuple[TokenizerCanaryDefinition, ...]

    @property
    def definition_fingerprint(self) -> str:
        return _fingerprint(asdict(self))


OLMO2_1124 = PinnedTokenizerDefinition(
    id="olmo2_1124",
    name="OLMo 2 (November 2024)",
    implementation="huggingface_tokenizers_json_v1",
    repository="allenai/OLMo-2-1124-7B",
    revision="35a7ed2e8347efe11760bcaa1f758a3b2d978a90",
    tokenizer_file="tokenizer.json",
    package="tokenizers",
    package_version=PINNED_TOKENIZERS_PACKAGE_VERSION,
    vocab_size=100_278,
    special_tokens={
        "bos": "<|endoftext|>",
        "eos": "<|endoftext|>",
        "pad": "<|pad|>",
        "unknown": "<|endoftext|>",
        "extra_id_0": "<|extra_id_0|>",
    },
    special_token_ids={
        "bos": 100_257,
        "eos": 100_257,
        "pad": 100_277,
        "unknown": 100_257,
        "extra_id_0": 100_256,
    },
    document_separator_token_id=100_257,
    assets=(
        TokenizerAssetDefinition(
            "tokenizer.json",
            "73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca",
        ),
        TokenizerAssetDefinition(
            "tokenizer_config.json",
            "91c69c665697785ace4ec7dbd159e1839bc9bb5033ab05a56bb0547521dc9ab0",
        ),
        TokenizerAssetDefinition(
            "special_tokens_map.json",
            "3c6bf7c09d5473c303cee8575a22bb51e5153c17d177a721b43cd4785c6d09ae",
        ),
        TokenizerAssetDefinition(
            "vocab.json",
            "9e14712c91b37c7aab74b1306baa46ac342d620637a4b44523cdc3aec7d24195",
        ),
        TokenizerAssetDefinition(
            "merges.txt",
            "b6fe424e334903f7fb84d3a106d9730455f4744b9fe3c21ee136d97a00e72502",
        ),
    ),
    canaries=(
        TokenizerCanaryDefinition(
            "Autoscaler",
            3,
            "24beedbd394216a75d3d732a8bc6372b900968218da8f34c476a96b7c26d348a",
        ),
        TokenizerCanaryDefinition(
            "νGPT scales across width.\n",
            7,
            "d0d2ba5e723d134b132575dcd73d7cc60b80810d1a6af26560a67714dd45ae3a",
        ),
        TokenizerCanaryDefinition(
            "<|endoftext|>",
            1,
            "f830a739ce8c937be2e89730ef37e7d7b740540ad33030bd41bb281ea8939af5",
        ),
    ),
)


PINNED_TOKENIZER_REGISTRY: Dict[str, PinnedTokenizerDefinition] = {
    OLMO2_1124.id: OLMO2_1124,
}


def builtin_byte_tokenizer_manifest() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": TOKENIZER_MANIFEST_SCHEMA_VERSION,
        "id": "byte_v1",
        "name": "Deterministic UTF-8 bytes",
        "implementation": "ai_theorist_builtin_byte_v1",
        "implementation_version": "1",
        "repository": None,
        "revision": None,
        "definition_fingerprint": None,
        "vocab_size": 260,
        "special_tokens": {
            "bos": "<BOS>",
            "eos": "<EOS>",
            "pad": "<PAD>",
            "document_separator": "<DOCUMENT_SEPARATOR>",
        },
        "special_token_ids": {
            "bos": 256,
            "eos": 257,
            "pad": 258,
            "document_separator": 259,
        },
        "pipeline": {
            "normalization": "none",
            "encoding": "UTF-8 bytes",
            "document_encoding": "BOS + bytes + EOS",
            "between_document_separator": 259,
        },
        "assets": [],
        "canaries": [
            {
                "text": "Autoscaler",
                "token_count": 12,
                "token_ids_sha256": _token_id_hash(
                    [256, *"Autoscaler".encode("utf-8"), 257]
                ),
            },
            {
                "text": "νGPT scales across width.\n",
                "token_count": 29,
                "token_ids_sha256": _token_id_hash(
                    [256, *"νGPT scales across width.\n".encode("utf-8"), 257]
                ),
            },
        ],
    }
    payload["definition_fingerprint"] = _fingerprint(
        {
            "id": payload["id"],
            "implementation": payload["implementation"],
            "implementation_version": payload["implementation_version"],
            "vocab_size": payload["vocab_size"],
            "special_tokens": payload["special_tokens"],
            "special_token_ids": payload["special_token_ids"],
            "pipeline": payload["pipeline"],
        }
    )
    payload["fingerprint"] = _fingerprint(payload)
    return payload


def tokenizer_catalog() -> List[Dict[str, Any]]:
    byte = builtin_byte_tokenizer_manifest()
    result = [
        {
            "id": "byte_v1",
            "name": byte["name"],
            "kind": "builtin",
            "vocab_size": byte["vocab_size"],
            "revision": None,
            "definition_fingerprint": byte["definition_fingerprint"],
            "tokenizer_fingerprint": byte["fingerprint"],
            "document_separator_token_id": byte["special_token_ids"][
                "document_separator"
            ],
        }
    ]
    for definition in PINNED_TOKENIZER_REGISTRY.values():
        result.append(
            {
                "id": definition.id,
                "name": definition.name,
                "kind": "pinned_remote",
                "vocab_size": definition.vocab_size,
                "repository": definition.repository,
                "revision": definition.revision,
                "definition_fingerprint": definition.definition_fingerprint,
                "tokenizer_fingerprint": None,
                "document_separator_token_id": definition.document_separator_token_id,
            }
        )
    return result


def tokenizer_definition_fingerprint(tokenizer_id: str) -> str:
    if tokenizer_id == "byte_v1":
        return str(builtin_byte_tokenizer_manifest()["definition_fingerprint"])
    try:
        return PINNED_TOKENIZER_REGISTRY[tokenizer_id].definition_fingerprint
    except KeyError as exc:
        raise ValueError(f"unknown pinned tokenizer: {tokenizer_id}") from exc


def tokenizer_vocab_size(tokenizer_id: str) -> int:
    if tokenizer_id == "byte_v1":
        return 260
    try:
        return PINNED_TOKENIZER_REGISTRY[tokenizer_id].vocab_size
    except KeyError as exc:
        raise ValueError(f"unknown pinned tokenizer: {tokenizer_id}") from exc


def _download_asset(url: str, output_path: Path, maximum_attempts: int = 5) -> None:
    retryable_statuses = {429, 500, 502, 503, 504}
    partial = output_path.with_name(f".{output_path.name}.partial")
    for attempt in range(maximum_attempts):
        request = Request(
            url,
            headers={
                "Accept": "application/octet-stream",
                "User-Agent": "ai-theorist-autoscaler/0.1 pinned-tokenizer-resolver",
            },
        )
        try:
            with urlopen(request, timeout=120.0) as response, partial.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, output_path)
            return
        except HTTPError as error:
            partial.unlink(missing_ok=True)
            if error.code not in retryable_statuses or attempt + 1 >= maximum_attempts:
                raise
        except URLError:
            partial.unlink(missing_ok=True)
            if attempt + 1 >= maximum_attempts:
                raise
        sleep(min(16.0, float(2**attempt)))
    raise RuntimeError("unreachable tokenizer asset retry state")


@dataclass(frozen=True)
class ResolvedTokenizer:
    definition: PinnedTokenizerDefinition
    tokenizer: Any
    manifest: Dict[str, Any]
    manifest_path: Path

    def encode_document(self, text: str) -> List[int]:
        token_ids = list(self.tokenizer.encode(text, add_special_tokens=False).ids)
        token_ids.append(self.definition.document_separator_token_id)
        return token_ids


def _tokenizer_pipeline(tokenizer_path: Path) -> Dict[str, Any]:
    with tokenizer_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("tokenizer.json must contain a JSON object")
    return {
        "normalizer": payload.get("normalizer"),
        "pre_tokenizer": payload.get("pre_tokenizer"),
        "post_processor": payload.get("post_processor"),
        "decoder": payload.get("decoder"),
        "model_type": (
            payload.get("model", {}).get("type")
            if isinstance(payload.get("model"), dict)
            else None
        ),
    }


def resolve_pinned_tokenizer(
    tokenizer_id: str,
    output_directory: Path,
    progress: ProgressCallback = None,
) -> ResolvedTokenizer:
    try:
        definition = PINNED_TOKENIZER_REGISTRY[tokenizer_id]
    except KeyError as exc:
        raise ValueError(f"unknown pinned tokenizer: {tokenizer_id}") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", definition.revision):
        raise ValueError("pinned tokenizer revision must be a full immutable Git commit")

    try:
        import tokenizers
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            f"{tokenizer_id} requires tokenizers=={definition.package_version}"
        ) from exc
    if tokenizers.__version__ != definition.package_version:
        raise RuntimeError(
            f"{tokenizer_id} requires tokenizers=={definition.package_version}; "
            f"found {tokenizers.__version__}"
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    assets_directory = output_directory / "assets"
    assets_directory.mkdir(parents=True, exist_ok=True)
    asset_rows = []
    for index, asset in enumerate(definition.assets):
        path = assets_directory / asset.name
        if path.is_file():
            observed_hash = _hash_file(path)
            if observed_hash != asset.sha256:
                raise ValueError(
                    f"pinned tokenizer asset failed SHA-256 verification: {asset.name}"
                )
        else:
            if progress is not None:
                progress(
                    {
                        "phase": "resolving_tokenizer",
                        "completed": index,
                        "total": len(definition.assets),
                        "message": f"Downloading pinned tokenizer asset {asset.name}",
                    }
                )
            _download_asset(
                "https://huggingface.co/"
                f"{definition.repository}/resolve/{definition.revision}/{asset.name}",
                path,
            )
            observed_hash = _hash_file(path)
            if observed_hash != asset.sha256:
                path.unlink(missing_ok=True)
                raise ValueError(
                    f"downloaded tokenizer asset failed SHA-256 verification: {asset.name}"
                )
        asset_rows.append(
            {
                "name": asset.name,
                "path": f"assets/{asset.name}",
                "sha256": observed_hash,
                "bytes": path.stat().st_size,
            }
        )

    tokenizer_path = assets_directory / definition.tokenizer_file
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    observed_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if observed_vocab_size != definition.vocab_size:
        raise ValueError(
            f"pinned tokenizer vocabulary mismatch: {observed_vocab_size} != "
            f"{definition.vocab_size}"
        )
    for name, token in definition.special_tokens.items():
        expected = definition.special_token_ids[name]
        observed = tokenizer.token_to_id(token) if token is not None else None
        if observed != expected:
            raise ValueError(
                f"pinned tokenizer special token mismatch for {name}: {observed} != {expected}"
            )

    canary_rows = []
    for canary in definition.canaries:
        ids = list(tokenizer.encode(canary.text, add_special_tokens=False).ids)
        observed_hash = _token_id_hash(ids)
        if len(ids) != canary.token_count or observed_hash != canary.token_ids_sha256:
            raise ValueError(
                "pinned tokenizer canary mismatch; implementation or assets changed for "
                f"{canary.text!r}"
            )
        canary_rows.append(
            {
                "text": canary.text,
                "token_count": len(ids),
                "token_ids_sha256": observed_hash,
            }
        )

    manifest: Dict[str, Any] = {
        "schema_version": TOKENIZER_MANIFEST_SCHEMA_VERSION,
        "id": definition.id,
        "name": definition.name,
        "implementation": definition.implementation,
        "implementation_package": definition.package,
        "implementation_version": tokenizers.__version__,
        "repository": definition.repository,
        "revision": definition.revision,
        "definition_fingerprint": definition.definition_fingerprint,
        "vocab_size": definition.vocab_size,
        "special_tokens": dict(definition.special_tokens),
        "special_token_ids": {
            **dict(definition.special_token_ids),
            "document_separator": definition.document_separator_token_id,
        },
        "pipeline": _tokenizer_pipeline(tokenizer_path),
        "assets": asset_rows,
        "canaries": canary_rows,
    }
    manifest["fingerprint"] = _fingerprint(manifest)
    manifest_path = output_directory / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    return ResolvedTokenizer(definition, tokenizer, manifest, manifest_path)


def load_tokenizer_manifest(
    manifest_path: Path, *, verify_assets: bool = True
) -> Dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("tokenizer manifest must contain a JSON object")
    if manifest.get("schema_version") != TOKENIZER_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported tokenizer manifest schema version")
    fingerprint = manifest.get("fingerprint")
    unsigned = dict(manifest)
    unsigned.pop("fingerprint", None)
    if not isinstance(fingerprint, str) or fingerprint != _fingerprint(unsigned):
        raise ValueError("tokenizer manifest fingerprint mismatch")
    tokenizer_id = manifest.get("id")
    if tokenizer_id == "byte_v1":
        expected = builtin_byte_tokenizer_manifest()
        if manifest != expected:
            raise ValueError("built-in byte tokenizer manifest contract mismatch")
    elif tokenizer_id in PINNED_TOKENIZER_REGISTRY:
        definition = PINNED_TOKENIZER_REGISTRY[str(tokenizer_id)]
        expected_assets = {asset.name: asset.sha256 for asset in definition.assets}
        observed_assets = {
            str(asset.get("name")): str(asset.get("sha256"))
            for asset in manifest.get("assets", [])
            if isinstance(asset, dict)
        }
        expected_canaries = {
            (canary.text, canary.token_count, canary.token_ids_sha256)
            for canary in definition.canaries
        }
        observed_canaries = {
            (
                str(canary.get("text")),
                int(canary.get("token_count", -1)),
                str(canary.get("token_ids_sha256")),
            )
            for canary in manifest.get("canaries", [])
            if isinstance(canary, dict)
        }
        if (
            manifest.get("name") != definition.name
            or manifest.get("definition_fingerprint")
            != definition.definition_fingerprint
            or manifest.get("repository") != definition.repository
            or manifest.get("revision") != definition.revision
            or manifest.get("implementation") != definition.implementation
            or manifest.get("implementation_package") != definition.package
            or manifest.get("implementation_version") != definition.package_version
            or manifest.get("vocab_size") != definition.vocab_size
            or manifest.get("special_tokens") != dict(definition.special_tokens)
            or {
                key: manifest.get("special_token_ids", {}).get(key)
                for key in definition.special_token_ids
            }
            != dict(definition.special_token_ids)
            or manifest.get("special_token_ids", {}).get("document_separator")
            != definition.document_separator_token_id
            or observed_assets != expected_assets
            or observed_canaries != expected_canaries
        ):
            raise ValueError("tokenizer manifest does not match its allow-listed definition")
    else:
        raise ValueError(f"tokenizer manifest ID is not allow-listed: {tokenizer_id}")
    if verify_assets:
        verified_asset_paths: Dict[str, Path] = {}
        for asset in manifest.get("assets", []):
            if not isinstance(asset, dict):
                raise ValueError("tokenizer manifest asset must be an object")
            path = _safe_manifest_path(
                manifest_path.parent, asset.get("path"), "tokenizer asset path"
            )
            if not path.is_file() or _hash_file(path) != asset.get("sha256"):
                raise ValueError(
                    f"tokenizer manifest asset verification failed: {asset.get('name')}"
                )
            verified_asset_paths[str(asset.get("name"))] = path
        if tokenizer_id in PINNED_TOKENIZER_REGISTRY:
            definition = PINNED_TOKENIZER_REGISTRY[str(tokenizer_id)]
            tokenizer_path = verified_asset_paths.get(definition.tokenizer_file)
            if tokenizer_path is None or manifest.get("pipeline") != _tokenizer_pipeline(
                tokenizer_path
            ):
                raise ValueError("tokenizer manifest pipeline does not match tokenizer.json")
    return manifest


def _iter_documents(path: Path, text_field: str) -> Iterator[str]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get(text_field), str):
                    raise ValueError(
                        f"{path}:{line_number} must contain string field {text_field!r}"
                    )
                yield row[text_field]
        return
    yield path.read_text(encoding="utf-8")


def _iter_documents_with_offsets(
    path: Path, text_field: str, start_offset: int
) -> Iterator[Tuple[str, int]]:
    if path.suffix.lower() != ".jsonl":
        if start_offset:
            raise ValueError("plain-text tokenization cannot resume at a nonzero offset")
        yield path.read_text(encoding="utf-8"), path.stat().st_size
        return
    with path.open("rb") as handle:
        if start_offset < 0 or start_offset > path.stat().st_size:
            raise ValueError("tokenization checkpoint source offset is invalid")
        handle.seek(start_offset)
        line_number = 0
        while True:
            raw = handle.readline()
            if not raw:
                break
            line_number += 1
            next_offset = handle.tell()
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{path}: malformed UTF-8 JSONL after byte offset {start_offset}"
                ) from exc
            if not isinstance(row, dict) or not isinstance(row.get(text_field), str):
                raise ValueError(
                    f"{path}: record after byte offset {start_offset} must contain "
                    f"string field {text_field!r}"
                )
            yield row[text_field], next_offset


def _write_token_shard(path: Path, token_ids: Sequence[int]) -> Dict[str, Any]:
    if not token_ids:
        raise ValueError("cannot write an empty token shard")
    array = np.asarray(token_ids, dtype="<u4")
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("wb") as handle:
        array.tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    return {
        "path": path.name,
        "tokens": int(array.size),
        "bytes": path.stat().st_size,
        "sha256": _hash_file(path),
        "minimum_token_id": int(array.min()),
        "maximum_token_id": int(array.max()),
    }


def _materialize_token_split(
    *,
    split: str,
    source_path: Path,
    output_directory: Path,
    text_field: str,
    tokenizer: ResolvedTokenizer,
    shard_token_limit: int,
    progress: ProgressCallback,
    source_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    source_fingerprint = source_fingerprint or _hash_file(source_path)
    checkpoint_path = output_directory / f".{split}.tokenization-checkpoint.json"
    contract = {
        "schema_version": 1,
        "split": split,
        "source_path": str(source_path.resolve()),
        "source_bytes": source_path.stat().st_size,
        "source_sha256": source_fingerprint,
        "text_field": text_field,
        "tokenizer_fingerprint": tokenizer.manifest["fingerprint"],
        "shard_token_limit": shard_token_limit,
    }
    checkpoint: Optional[Dict[str, Any]] = None
    if checkpoint_path.is_file():
        try:
            with checkpoint_path.open("r", encoding="utf-8") as handle:
                candidate = json.load(handle)
            if all(candidate.get(key) == value for key, value in contract.items()):
                checkpoint = candidate
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            checkpoint = None
    if checkpoint is None:
        shards: List[Dict[str, Any]] = []
        documents = 0
        total_tokens = 0
        source_offset = 0
    else:
        shards = list(checkpoint.get("shards", []))
        documents = int(checkpoint.get("documents", 0))
        total_tokens = int(checkpoint.get("tokens", 0))
        source_offset = int(checkpoint.get("source_offset", 0))
        for shard in shards:
            path = _safe_manifest_path(
                output_directory, shard.get("path"), "tokenization checkpoint shard"
            )
            if (
                not path.is_file()
                or path.stat().st_size != int(shard.get("bytes", -1))
                or _hash_file(path) != shard.get("sha256")
            ):
                raise ValueError(
                    "tokenization checkpoint references a missing or altered shard"
                )
    retained = {str(row["path"]) for row in shards}
    for orphan in output_directory.glob(f"{split}-*.bin"):
        if orphan.name not in retained:
            orphan.unlink()
    buffer = array("I")
    if buffer.itemsize != 4:
        raise RuntimeError("token sharding requires a 32-bit unsigned array implementation")
    buffer_end_offset = source_offset

    def flush() -> None:
        nonlocal buffer, source_offset
        if not buffer:
            return
        shard_path = output_directory / f"{split}-{len(shards):05d}.bin"
        shards.append(_write_token_shard(shard_path, buffer))
        source_offset = buffer_end_offset
        atomic_write_json(
            checkpoint_path,
            {
                **contract,
                "source_offset": source_offset,
                "documents": documents,
                "tokens": total_tokens,
                "shards": shards,
            },
        )
        buffer = array("I")

    for document, next_offset in _iter_documents_with_offsets(
        source_path, text_field, source_offset
    ):
        encoded = tokenizer.encode_document(document)
        if not encoded:
            raise ValueError("the pinned tokenizer produced an empty document")
        if max(encoded) >= tokenizer.definition.vocab_size:
            raise ValueError("the pinned tokenizer emitted an out-of-vocabulary token")
        if buffer and len(buffer) + len(encoded) > shard_token_limit:
            flush()
        buffer.extend(encoded)
        buffer_end_offset = next_offset
        documents += 1
        total_tokens += len(encoded)
        if progress is not None and documents % 1000 == 0:
            progress(
                {
                    "phase": "tokenizing",
                    "completed": documents,
                    "total": 0,
                    "message": f"Tokenized {documents:,} {split} documents",
                }
            )
    flush()
    if not shards:
        raise ValueError(f"{split} corpus contains no tokenizable documents")
    checkpoint_path.unlink(missing_ok=True)
    return {
        "documents": documents,
        "tokens": total_tokens,
        "source_sha256": source_fingerprint,
        "shards": shards,
    }


def materialize_pinned_token_streams(
    *,
    tokenizer: ResolvedTokenizer,
    train_path: Path,
    validation_path: Path,
    output_directory: Path,
    text_field: str = "text",
    shard_token_limit: int = 16_777_216,
    progress: ProgressCallback = None,
    source_fingerprints: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    if shard_token_limit < 1024:
        raise ValueError("shard_token_limit must be at least 1024")
    output_directory.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = output_directory / "manifest.json"
    if existing_manifest_path.is_file():
        try:
            return load_token_stream_manifest(
                existing_manifest_path, verify_files=True
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    source_fingerprints = dict(source_fingerprints or {})
    train = _materialize_token_split(
        split="train",
        source_path=train_path,
        output_directory=output_directory,
        text_field=text_field,
        tokenizer=tokenizer,
        shard_token_limit=shard_token_limit,
        progress=progress,
        source_fingerprint=source_fingerprints.get("train"),
    )
    validation = _materialize_token_split(
        split="validation",
        source_path=validation_path,
        output_directory=output_directory,
        text_field=text_field,
        tokenizer=tokenizer,
        shard_token_limit=shard_token_limit,
        progress=progress,
        source_fingerprint=source_fingerprints.get("validation"),
    )
    content_fingerprint = _fingerprint(
        {
            split: [
                {"sha256": row["sha256"], "tokens": row["tokens"]}
                for row in metadata["shards"]
            ]
            for split, metadata in (("train", train), ("validation", validation))
        }
    )
    packing = {
        "contract": "document_eos_concatenation_v1",
        "add_special_tokens": False,
        "append_document_separator": True,
        "document_separator_token_id": tokenizer.definition.document_separator_token_id,
        "cross_document_attention": True,
        "cross_shard_windows": False,
        "shard_token_limit": shard_token_limit,
    }
    manifest: Dict[str, Any] = {
        "schema_version": TOKEN_STREAM_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "format": "sharded_uint32_le_v1",
        "dtype": "uint32_le",
        "tokenizer_id": tokenizer.definition.id,
        "tokenizer_fingerprint": tokenizer.manifest["fingerprint"],
        "tokenizer_manifest_path": "../tokenizer/manifest.json",
        "vocab_size": tokenizer.definition.vocab_size,
        "packing": packing,
        "content_fingerprint": content_fingerprint,
        "splits": {"train": train, "validation": validation},
    }
    manifest["fingerprint"] = _fingerprint(manifest)
    atomic_write_json(output_directory / "manifest.json", manifest)
    return manifest


def load_token_stream_manifest(
    manifest_path: Path, *, verify_files: bool = True
) -> Dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("token stream manifest must contain a JSON object")
    if manifest.get("schema_version") != TOKEN_STREAM_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported token stream manifest schema version")
    if manifest.get("status") != "complete" or manifest.get("format") != "sharded_uint32_le_v1":
        raise ValueError("unsupported or incomplete token stream manifest")
    fingerprint = manifest.get("fingerprint")
    unsigned = dict(manifest)
    unsigned.pop("fingerprint", None)
    if not isinstance(fingerprint, str) or fingerprint != _fingerprint(unsigned):
        raise ValueError("token stream manifest fingerprint mismatch")

    corpus_root = manifest_path.parent.parent
    tokenizer_manifest_path = _safe_manifest_path(
        corpus_root,
        str(Path(manifest_path.parent.name) / str(manifest.get("tokenizer_manifest_path", ""))),
        "tokenizer manifest path",
    )
    tokenizer_manifest = load_tokenizer_manifest(
        tokenizer_manifest_path, verify_assets=verify_files
    )
    if tokenizer_manifest.get("fingerprint") != manifest.get("tokenizer_fingerprint"):
        raise ValueError("token stream and tokenizer manifest fingerprints disagree")
    if tokenizer_manifest.get("id") != manifest.get("tokenizer_id"):
        raise ValueError("token stream and tokenizer manifest IDs disagree")
    if tokenizer_manifest.get("vocab_size") != manifest.get("vocab_size"):
        raise ValueError("token stream and tokenizer vocabulary sizes disagree")
    packing = manifest.get("packing")
    if (
        not isinstance(packing, dict)
        or packing.get("contract") != "document_eos_concatenation_v1"
        or packing.get("document_separator_token_id")
        != tokenizer_manifest.get("special_token_ids", {}).get("document_separator")
    ):
        raise ValueError("token stream packing contract is invalid")

    if verify_files:
        for split in ("train", "validation"):
            metadata = manifest.get("splits", {}).get(split)
            if not isinstance(metadata, dict) or not isinstance(metadata.get("shards"), list):
                raise ValueError(f"token stream manifest is missing {split} shards")
            token_total = 0
            for shard in metadata["shards"]:
                if not isinstance(shard, dict):
                    raise ValueError("token stream shard must be an object")
                path = _safe_manifest_path(
                    manifest_path.parent, shard.get("path"), "token shard path"
                )
                expected_bytes = int(shard.get("tokens", -1)) * 4
                if (
                    not path.is_file()
                    or path.stat().st_size != expected_bytes
                ):
                    raise ValueError(
                        f"token stream shard verification failed: {shard.get('path')}"
                    )
                observed_hash, observed_tokens, observed_minimum, observed_maximum = (
                    _inspect_uint32_shard(path)
                )
                if (
                    observed_hash != shard.get("sha256")
                    or observed_tokens != int(shard["tokens"])
                    or observed_minimum != int(shard.get("minimum_token_id", -1))
                    or observed_maximum != int(shard.get("maximum_token_id", -1))
                    or observed_minimum < 0
                    or observed_maximum >= int(manifest["vocab_size"])
                ):
                    raise ValueError(
                        f"token stream shard content mismatch: {shard.get('path')}"
                    )
                token_total += observed_tokens
            if token_total != metadata.get("tokens"):
                raise ValueError(f"token count mismatch for {split} token stream")
    expected_content_fingerprint = _fingerprint(
        {
            split: [
                {"sha256": row["sha256"], "tokens": row["tokens"]}
                for row in manifest["splits"][split]["shards"]
            ]
            for split in ("train", "validation")
        }
    )
    if manifest.get("content_fingerprint") != expected_content_fingerprint:
        raise ValueError("token stream content fingerprint mismatch")
    return manifest


def token_stream_identity(manifest_path: Path) -> Dict[str, Any]:
    manifest = load_token_stream_manifest(manifest_path, verify_files=True)
    return {
        "format": manifest["format"],
        "fingerprint": manifest["fingerprint"],
        "content_fingerprint": manifest["content_fingerprint"],
        "tokenizer_id": manifest["tokenizer_id"],
        "tokenizer_fingerprint": manifest["tokenizer_fingerprint"],
        "vocab_size": manifest["vocab_size"],
        "packing": manifest["packing"],
        "training_tokens": manifest["splits"]["train"]["tokens"],
        "validation_tokens": manifest["splits"]["validation"]["tokens"],
    }
