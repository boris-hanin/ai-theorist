from pathlib import Path
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
