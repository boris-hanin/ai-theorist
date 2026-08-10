from hashlib import sha256
import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from ai_theorist.autoscaler import tokenization
from ai_theorist.autoscaler.api import _campaign_data_identity
from ai_theorist.autoscaler.pretraining import (
    TokenizedTextCorpus,
    TokenizedTextSpec,
    compile_standard_pretraining_plan,
)
from ai_theorist.autoscaler.tokenization import (
    PINNED_TOKENIZERS_PACKAGE_VERSION,
    PinnedTokenizerDefinition,
    TokenizerAssetDefinition,
    TokenizerCanaryDefinition,
    load_tokenizer_manifest,
    load_token_stream_manifest,
    materialize_pinned_token_streams,
    resolve_pinned_tokenizer,
    token_stream_identity,
    tokenizer_catalog,
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _token_hash(token_ids) -> str:
    digest = sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, "little"))
    return digest.hexdigest()


def _test_definition(tmp_path: Path, monkeypatch) -> PinnedTokenizerDefinition:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "Autoscaler": 1,
                "<|endoftext|>": 2,
                "<|extra_id_0|>": 3,
                "unused-4": 4,
                "unused-5": 5,
                "unused-6": 6,
                "unused-7": 7,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    assets = tmp_path / "tokenizer" / "assets"
    assets.mkdir(parents=True)
    tokenizer_path = assets / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    definition = PinnedTokenizerDefinition(
        id="test_pinned",
        name="Test pinned tokenizer",
        implementation="huggingface_tokenizers_json_v1",
        repository="example/test-tokenizer",
        revision="a" * 40,
        tokenizer_file="tokenizer.json",
        package="tokenizers",
        package_version=PINNED_TOKENIZERS_PACKAGE_VERSION,
        vocab_size=8,
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
        assets=(TokenizerAssetDefinition("tokenizer.json", _hash(tokenizer_path)),),
        canaries=(
            TokenizerCanaryDefinition("Autoscaler", 1, _token_hash([1])),
        ),
    )
    monkeypatch.setitem(tokenization.PINNED_TOKENIZER_REGISTRY, definition.id, definition)
    return definition


def _write_documents(path: Path, count: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(json.dumps({"text": "Autoscaler", "row": index}) + "\n")


def _materialized_fixture(tmp_path: Path, monkeypatch):
    definition = _test_definition(tmp_path, monkeypatch)
    resolved = resolve_pinned_tokenizer(definition.id, tmp_path / "tokenizer")
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_documents(train, 600)
    _write_documents(validation, 20)
    manifest = materialize_pinned_token_streams(
        tokenizer=resolved,
        train_path=train,
        validation_path=validation,
        output_directory=tmp_path / "token-streams",
        shard_token_limit=1024,
    )
    return definition, resolved, manifest, tmp_path / "token-streams" / "manifest.json"


def test_catalog_exposes_immutable_remote_definitions() -> None:
    catalog = {row["id"]: row for row in tokenizer_catalog()}
    assert catalog["byte_v1"]["tokenizer_fingerprint"]
    olmo = catalog["olmo2_1124"]
    assert len(olmo["revision"]) == 40
    assert olmo["vocab_size"] == 100_278
    assert len(olmo["definition_fingerprint"]) == 64
    mistral = catalog["mistral_7b_v03"]
    assert mistral["repository"] == "mistralai/Mistral-7B-v0.3"
    assert len(mistral["revision"]) == 40
    assert mistral["vocab_size"] == 32_768
    assert mistral["document_separator_token_id"] == 2
    assert len(mistral["definition_fingerprint"]) == 64


def test_pinned_tokenizer_resolution_and_sharded_stream_are_reproducible(
    tmp_path: Path, monkeypatch
) -> None:
    definition, resolved, manifest, manifest_path = _materialized_fixture(
        tmp_path, monkeypatch
    )
    assert resolved.manifest["definition_fingerprint"] == definition.definition_fingerprint
    assert resolved.manifest["implementation_version"] == PINNED_TOKENIZERS_PACKAGE_VERSION
    assert len(manifest["splits"]["train"]["shards"]) == 2
    assert manifest["packing"]["document_separator_token_id"] == 2

    identity = token_stream_identity(manifest_path)
    assert identity["tokenizer_id"] == definition.id
    assert identity["vocab_size"] == 8
    corpus = TokenizedTextCorpus(
        TokenizedTextSpec(
            tokenizer=definition.id,
            token_stream_manifest_path=str(manifest_path),
            maximum_bytes=16_384,
        ),
        context_length=2,
        vocab_size=8,
    )
    first = corpus.sample_batch(
        "train", 8, torch.Generator().manual_seed(7), "cpu"
    )
    second = corpus.sample_batch(
        "train", 8, torch.Generator().manual_seed(7), "cpu"
    )
    assert all(left.equal(right) for left, right in zip(first, second))
    assert corpus.identity_fingerprint == manifest["fingerprint"]
    assert corpus.tokenizer_fingerprint == resolved.manifest["fingerprint"]
    assert corpus.tokenizer_is_pinned is True


def test_tampered_shard_and_tokenizer_asset_are_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    _, _, manifest, manifest_path = _materialized_fixture(tmp_path, monkeypatch)
    shard = manifest_path.parent / manifest["splits"]["train"]["shards"][0]["path"]
    shard.write_bytes(shard.read_bytes()[:-4] + b"BAD!")
    with pytest.raises(ValueError, match="token stream shard"):
        load_token_stream_manifest(manifest_path)

    _, _, _, second_manifest_path = _materialized_fixture(
        tmp_path / "second", monkeypatch
    )
    tokenizer_asset = second_manifest_path.parent.parent / "tokenizer" / "assets" / "tokenizer.json"
    tokenizer_asset.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="asset verification failed"):
        load_token_stream_manifest(second_manifest_path)


def test_self_consistent_manifest_cannot_override_allowlisted_separator(
    tmp_path: Path, monkeypatch
) -> None:
    _, resolved, _, _ = _materialized_fixture(tmp_path, monkeypatch)
    manifest = json.loads(resolved.manifest_path.read_text(encoding="utf-8"))
    manifest["special_token_ids"]["document_separator"] = 7
    manifest.pop("fingerprint")
    manifest["fingerprint"] = tokenization._fingerprint(manifest)
    resolved.manifest_path.write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="allow-listed definition"):
        load_tokenizer_manifest(resolved.manifest_path)


def test_token_sharding_resumes_from_last_atomic_shard(
    tmp_path: Path, monkeypatch
) -> None:
    definition = _test_definition(tmp_path, monkeypatch)
    resolved = resolve_pinned_tokenizer(definition.id, tmp_path / "tokenizer")
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_documents(train, 700)
    _write_documents(validation, 20)
    original_atomic_write = tokenization.atomic_write_json
    interrupted = False

    def interrupt_after_checkpoint(path, payload):
        nonlocal interrupted
        original_atomic_write(path, payload)
        if path.name == ".train.tokenization-checkpoint.json" and not interrupted:
            interrupted = True
            raise RuntimeError("simulated tokenization interruption")

    monkeypatch.setattr(tokenization, "atomic_write_json", interrupt_after_checkpoint)
    with pytest.raises(RuntimeError, match="simulated tokenization interruption"):
        materialize_pinned_token_streams(
            tokenizer=resolved,
            train_path=train,
            validation_path=validation,
            output_directory=tmp_path / "token-streams",
            shard_token_limit=1024,
        )
    checkpoint = tmp_path / "token-streams" / ".train.tokenization-checkpoint.json"
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    first_shard = tmp_path / "token-streams" / payload["shards"][0]["path"]
    first_hash = _hash(first_shard)

    monkeypatch.setattr(tokenization, "atomic_write_json", original_atomic_write)
    manifest = materialize_pinned_token_streams(
        tokenizer=resolved,
        train_path=train,
        validation_path=validation,
        output_directory=tmp_path / "token-streams",
        shard_token_limit=1024,
    )
    assert manifest["splits"]["train"]["documents"] == 700
    assert manifest["splits"]["train"]["tokens"] == 1400
    assert _hash(first_shard) == first_hash
    assert not checkpoint.exists()


def test_manifest_vocab_and_ambiguous_dataset_inputs_are_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    definition, _, _, manifest_path = _materialized_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="requires vocab_size 8"):
        TokenizedTextCorpus(
            TokenizedTextSpec(
                tokenizer=definition.id,
                token_stream_manifest_path=str(manifest_path),
            ),
            context_length=2,
            vocab_size=9,
        )
    with pytest.raises(ValueError, match="either token_stream_manifest_path"):
        TokenizedTextSpec.from_dict(
            {
                "train_path": "train.bin",
                "validation_path": "validation.bin",
                "tokenizer": definition.id,
                "token_stream_manifest_path": str(manifest_path),
            }
        )


def test_campaign_identity_binds_verified_tokenizer_stream_and_model(
    tmp_path: Path, monkeypatch
) -> None:
    definition, resolved, manifest, manifest_path = _materialized_fixture(
        tmp_path, monkeypatch
    )
    config = {
        "architecture": {"vocab_size": 8},
        "dataset": {
            "tokenizer": definition.id,
            "token_stream_manifest_path": str(manifest_path),
        },
    }
    identity = _campaign_data_identity(config)
    assert identity["fingerprint"] == manifest["fingerprint"]
    assert identity["tokenizer_fingerprint"] == resolved.manifest["fingerprint"]

    plan = compile_standard_pretraining_plan(
        {
            "model": {
                "vocab_size": 8,
                "context_length": 2,
                "width": 8,
                "depth": 1,
                "num_heads": 2,
                "mlp_multiplier": 2,
            },
            "dataset": config["dataset"],
            "runtime": {
                "precision": "fp32",
                "attention_backend": "math",
                "distributed": "none",
                "num_processes": 1,
            },
            "scales": [{"name": "S1", "width": 8, "depth": 1, "num_heads": 2}],
            "batch_examples": [1, 2, 3, 4],
            "total_tokens": 24,
            "optimizers": [{"name": "adam", "learning_rates": [0.001]}],
        }
    )
    assert plan["dataset_identity"]["fingerprint"] == manifest["fingerprint"]

    wrong_tokenizer = {**config, "dataset": {**config["dataset"], "tokenizer": "wrong"}}
    with pytest.raises(ValueError, match="tokenizer does not match"):
        _campaign_data_identity(wrong_tokenizer)
    wrong_vocab = {**config, "architecture": {"vocab_size": 9}}
    with pytest.raises(ValueError, match="requires vocab_size 8"):
        _campaign_data_identity(wrong_vocab)
