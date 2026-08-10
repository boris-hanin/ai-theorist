from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

from ai_theorist.autoscaler import public_corpora
from ai_theorist.autoscaler.public_corpora import (
    PublicCorpusSpec,
    materialize_public_corpus,
    public_corpus_catalog,
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
