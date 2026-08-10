from hashlib import sha256
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from ai_theorist.autoscaler import public_corpora
from ai_theorist.autoscaler.public_corpora import (
    PublicCorpusSpec,
    materialize_public_corpus,
    public_corpus_catalog,
)
from ai_theorist.autoscaler.tokenization import (
    PINNED_TOKENIZERS_PACKAGE_VERSION,
    PinnedTokenizerDefinition,
    TokenizerAssetDefinition,
    TokenizerCanaryDefinition,
)


def test_public_corpus_contract_is_allow_listed_and_bounded() -> None:
    sources = {row["id"] for row in public_corpus_catalog()}
    assert sources == {"fineweb_edu", "openwebtext"}
    assert PublicCorpusSpec.from_dict({"source": "fineweb_edu"}).source == "fineweb_edu"
    with pytest.raises(ValueError, match="source must be"):
        PublicCorpusSpec.from_dict({"source": "arbitrary/url"})
    with pytest.raises(ValueError, match="train_bytes"):
        PublicCorpusSpec.from_dict({"train_bytes": 1})
    with pytest.raises(ValueError, match="Unknown public corpus"):
        PublicCorpusSpec.from_dict({"url": "https://example.com"})
    large = PublicCorpusSpec.from_dict(
        {
            "train_bytes": 2_000_000_000,
            "validation_bytes": 100_000_000,
            "maximum_documents_per_split": 2_000_000,
            "acquisition_backend": "parquet",
        }
    )
    assert large.train_bytes == 2_000_000_000
    assert large.acquisition_backend == "parquet"


def test_json_request_retries_rate_limits(monkeypatch) -> None:
    attempts = []
    sleeps = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(request, timeout):
        attempts.append((request.full_url, timeout))
        if len(attempts) == 1:
            raise HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                {"Retry-After": "0"},
                None,
            )
        return FakeResponse()

    monkeypatch.setattr(public_corpora, "urlopen", fake_urlopen)
    monkeypatch.setattr(public_corpora, "sleep", sleeps.append)
    assert public_corpora._json_request("https://example.test/data") == {"ok": True}
    assert len(attempts) == 2
    assert sleeps == [1.0]


def test_split_materialization_resumes_from_fsynced_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    requested_offsets = []
    fail_second_request = True

    def fake_request(url: str, timeout: float = 60.0):
        nonlocal fail_second_request
        query = parse_qs(urlparse(url).query)
        offset = int(query["offset"][0])
        length = int(query["length"][0])
        requested_offsets.append(offset)
        if offset == 100 and fail_second_request:
            fail_second_request = False
            raise RuntimeError("simulated interrupted download")
        return {
            "rows": [
                {
                    "row_idx": row_index,
                    "row": {"text": "x" * 10, "id": f"doc-{row_index}"},
                    "truncated_cells": [],
                }
                for row_index in range(offset, offset + length)
            ]
        }

    monkeypatch.setattr(public_corpora, "_json_request", fake_request)
    output_path = tmp_path / "train.jsonl"
    arguments = dict(
        catalog=public_corpora.PUBLIC_CORPUS_CATALOG["fineweb_edu"],
        start_offset=0,
        target_bytes=1_500,
        maximum_documents=200,
        output_path=output_path,
        split_name="training corpus",
        progress=None,
        completed_bytes=0,
        total_bytes=1_500,
        source_revision="a" * 40,
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        public_corpora._materialize_split(**arguments)
    assert requested_offsets == [0, 100]

    metadata, _ = public_corpora._materialize_split(**arguments)
    assert requested_offsets == [0, 100, 100]
    assert metadata["documents"] == 150
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 150


def test_parquet_materializer_streams_batches_with_global_row_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    parquet_path = tmp_path / "fixture.parquet"
    pq.write_table(
        pa.table(
            {
                "text": [f"document-{index}-" + "x" * 20 for index in range(50)],
                "id": [f"id-{index}" for index in range(50)],
            }
        ),
        parquet_path,
    )
    inventory = {
        "files": [
            {
                "url": "https://example.test/fixture.parquet",
                "filename": "fixture.parquet",
                "bytes": parquet_path.stat().st_size,
            }
        ],
        "fingerprint": "f" * 64,
    }

    def local_file(_entry, _directory):
        return {
            "path": str(parquet_path),
            "url": "https://example.test/fixture.parquet",
            "bytes": parquet_path.stat().st_size,
            "sha256": sha256(parquet_path.read_bytes()).hexdigest(),
        }

    monkeypatch.setattr(public_corpora, "_download_parquet_file", local_file)
    output = tmp_path / "slice.jsonl"
    metadata, _ = public_corpora._materialize_split_parquet(
        catalog=public_corpora.PUBLIC_CORPUS_CATALOG["fineweb_edu"],
        inventory=inventory,
        parquet_cache=tmp_path / "cache",
        start_offset=10,
        target_bytes=100,
        maximum_documents=20,
        output_path=output,
        split_name="training corpus",
        progress=None,
        completed_bytes=0,
        total_bytes=100,
        source_revision="a" * 40,
        source_batch_rows=128,
    )
    rows = [__import__("json").loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["source_row"] == 10
    assert rows[0]["source_id"] == "id-10"
    assert metadata["first_source_row"] == 10
    assert metadata["source_inventory_fingerprint"] == "f" * 64
    assert metadata["source_parquet_files"][0]["sha256"]


def test_materializer_freezes_disjoint_rows_and_reuses_verified_cache(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    def fake_request(url: str, timeout: float = 60.0):
        calls.append(url)
        if "api/datasets" in url:
            return {"sha": "a" * 40}
        query = parse_qs(urlparse(url).query)
        offset = int(query["offset"][0])
        length = int(query["length"][0])
        rows = []
        for row_index in range(offset, offset + length):
            rows.append(
                {
                    "row_idx": row_index,
                    "row": {
                        "text": f"Document {row_index} " + ("scaling data " * 100),
                        "id": f"doc-{row_index}",
                    },
                    "truncated_cells": [],
                }
            )
        return {"rows": rows, "num_rows_total": 9_672_101, "partial": False}

    monkeypatch.setattr(public_corpora, "_json_request", fake_request)
    spec = PublicCorpusSpec(
        source="fineweb_edu",
        train_bytes=65_536,
        validation_bytes=16_384,
        maximum_documents_per_split=100,
    )
    events = []
    result = materialize_public_corpus(spec, tmp_path, events.append)
    assert result["status"] == "complete"
    assert result["source"]["revision"] == "a" * 40
    assert result["source"]["license"] == "ODC-By 1.0"
    assert result["training_tokens"] > 65_536
    assert result["validation_tokens"] > 16_384
    assert result["splits"]["train"]["last_source_row"] < result["splits"]["validation"]["first_source_row"]
    assert len(result["corpus_fingerprint"]) == 64
    assert events[-1]["phase"] == "complete"

    call_count = len(calls)
    cached = materialize_public_corpus(spec, tmp_path)
    assert cached == result
    assert len(calls) == call_count


def test_public_materializer_builds_verified_pinned_token_stream(
    tmp_path: Path, monkeypatch
) -> None:
    def token_hash(token_ids) -> str:
        digest = sha256()
        for token_id in token_ids:
            digest.update(int(token_id).to_bytes(4, "little"))
        return digest.hexdigest()

    seed_tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "Document": 1,
                "<|endoftext|>": 2,
                "<|extra_id_0|>": 3,
            },
            unk_token="[UNK]",
        )
    )
    seed_tokenizer.pre_tokenizer = Whitespace()
    seed_path = tmp_path / "seed-tokenizer.json"
    seed_tokenizer.save(str(seed_path))
    asset_hash = sha256(seed_path.read_bytes()).hexdigest()
    definition = PinnedTokenizerDefinition(
        id="test_public_pinned",
        name="Test public tokenizer",
        implementation="huggingface_tokenizers_json_v1",
        repository="example/test-tokenizer",
        revision="b" * 40,
        tokenizer_file="tokenizer.json",
        package="tokenizers",
        package_version=PINNED_TOKENIZERS_PACKAGE_VERSION,
        vocab_size=4,
        special_tokens={
            "bos": None,
            "eos": "<|endoftext|>",
            "pad": None,
            "unknown": "[UNK]",
            "extra_id_0": "<|extra_id_0|>",
        },
        special_token_ids={
            "bos": None,
            "eos": 2,
            "pad": None,
            "unknown": 0,
            "extra_id_0": 3,
        },
        document_separator_token_id=2,
        assets=(TokenizerAssetDefinition("tokenizer.json", asset_hash),),
        canaries=(TokenizerCanaryDefinition("Document", 1, token_hash([1])),),
    )
    monkeypatch.setitem(
        public_corpora.PINNED_TOKENIZER_REGISTRY, definition.id, definition
    )
    original_resolver = public_corpora.resolve_pinned_tokenizer

    def local_resolver(tokenizer_id, output_directory, progress=None):
        assets = output_directory / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        (assets / "tokenizer.json").write_bytes(seed_path.read_bytes())
        return original_resolver(tokenizer_id, output_directory, progress)

    def fake_request(url: str, timeout: float = 60.0):
        if "api/datasets" in url:
            return {"sha": "c" * 40}
        query = parse_qs(urlparse(url).query)
        offset = int(query["offset"][0])
        length = int(query["length"][0])
        return {
            "rows": [
                {
                    "row_idx": row_index,
                    "row": {
                        "text": "Document " + ("scaling data " * 100),
                        "id": f"doc-{row_index}",
                    },
                    "truncated_cells": [],
                }
                for row_index in range(offset, offset + length)
            ]
        }

    monkeypatch.setattr(public_corpora, "resolve_pinned_tokenizer", local_resolver)
    monkeypatch.setattr(public_corpora, "_json_request", fake_request)
    spec = PublicCorpusSpec(
        source="fineweb_edu",
        tokenizer=definition.id,
        train_bytes=65_536,
        validation_bytes=16_384,
        maximum_documents_per_split=100,
        token_shard_tokens=1024,
    )
    result = materialize_public_corpus(spec, tmp_path / "corpora")
    assert result["tokenizer"] == definition.id
    assert result["tokenizer_vocab_size"] == 4
    assert result["token_stream_manifest_path"]
    assert len(result["tokenizer_fingerprint"]) == 64
    assert len(result["dataset_identity_fingerprint"]) == 64
    assert result["training_tokens"] > 0

    cached = materialize_public_corpus(spec, tmp_path / "corpora")
    assert cached == result
