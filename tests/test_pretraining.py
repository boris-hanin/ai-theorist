from dataclasses import replace
import time

import numpy as np
import pytest
import torch

from ai_theorist.autoscaler.batch_scaling import OptimizerHyperparameters
from ai_theorist.autoscaler.api import CampaignStore
from ai_theorist.autoscaler.campaign_jobs import compile_fsdp_launch
from ai_theorist.autoscaler.pretraining import (
    ByteTokenizer,
    DistributedContext,
    PretrainingRuntimeSpec,
    StandardTransformer,
    StandardTransformerSpec,
    TokenizedTextCorpus,
    TokenizedTextSpec,
    compile_standard_pretraining_plan,
    preflight_runtime,
    run_standard_pretraining_batch_census,
    run_standard_pretraining_trial,
)


def _corpus_spec(tmp_path):
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    train.write_text(
        "Scaling experiments need exact token counts. " * 20,
        encoding="utf-8",
    )
    validation.write_text(
        "Held-out text measures next-token prediction. " * 10,
        encoding="utf-8",
    )
    return TokenizedTextSpec(str(train), str(validation))


def _model_spec():
    return StandardTransformerSpec(
        vocab_size=260,
        context_length=4,
        width=8,
        depth=1,
        num_heads=2,
        mlp_multiplier=2,
    )


def test_byte_tokenizer_and_real_text_corpus_are_deterministic(tmp_path) -> None:
    tokenizer = ByteTokenizer()
    text = "νGPT and ordinary UTF-8"
    tokens = tokenizer.encode(text)
    assert tokenizer.decode(tokens) == text

    spec = _corpus_spec(tmp_path)
    first = TokenizedTextCorpus(spec, context_length=4)
    second = TokenizedTextCorpus(spec, context_length=4)
    assert first.fingerprint == second.fingerprint
    generator_a = torch.Generator().manual_seed(5)
    generator_b = torch.Generator().manual_seed(5)
    batch_a = first.sample_batch("train", 3, generator_a, "cpu")
    batch_b = second.sample_batch("train", 3, generator_b, "cpu")
    assert all(torch.equal(left, right) for left, right in zip(batch_a, batch_b))


def test_memory_mapped_uint16_token_stream(tmp_path) -> None:
    train = tmp_path / "train.bin"
    validation = tmp_path / "validation.bin"
    np.tile(np.arange(32, dtype="<u2"), 4).tofile(train)
    np.tile(np.arange(16, dtype="<u2"), 4).tofile(validation)
    corpus = TokenizedTextCorpus(
        TokenizedTextSpec(
            str(train),
            str(validation),
            tokenizer="uint16_bin_v1",
        ),
        context_length=4,
        vocab_size=64,
    )
    inputs, targets = corpus.sample_batch(
        "train", 2, torch.Generator().manual_seed(3), "cpu"
    )
    assert inputs.shape == targets.shape == (2, 4)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_standard_transformer_is_causal_and_ties_embeddings() -> None:
    spec = _model_spec()
    model = StandardTransformer(spec, attention_backend="math").eval()
    assert model.language_model_head.weight is model.token_embedding.weight
    tokens = torch.tensor([[1, 2, 3, 4]])
    changed = tokens.clone()
    changed[0, -1] = 5
    with torch.no_grad():
        original = model(tokens)
        modified = model(changed)
    assert torch.equal(original[:, :-1], modified[:, :-1])
    assert not torch.equal(original[:, -1], modified[:, -1])


def test_standard_trial_supports_fp32_and_cpu_bf16(tmp_path) -> None:
    corpus = TokenizedTextCorpus(_corpus_spec(tmp_path), context_length=4)
    runtime = PretrainingRuntimeSpec(precision="fp32", attention_backend="math")
    context = DistributedContext(0, 1, 0, "cpu")
    optimizer = OptimizerHyperparameters("adam", 0.001, beta2=0.99)
    first, _ = run_standard_pretraining_trial(
        model_spec=_model_spec(),
        corpus=corpus,
        runtime=runtime,
        distributed_context=context,
        optimizer_spec=optimizer,
        total_tokens=16,
        batch_examples=1,
        seed=7,
        validation_interval=1,
        validation_examples=4,
    )
    second, _ = run_standard_pretraining_trial(
        model_spec=_model_spec(),
        corpus=corpus,
        runtime=runtime,
        distributed_context=context,
        optimizer_spec=optimizer,
        total_tokens=16,
        batch_examples=1,
        seed=7,
        validation_interval=1,
        validation_examples=4,
    )
    assert first.final_validation_loss == second.final_validation_loss
    assert first.model_family == "standard_pre_norm_transformer_tokenized_text"
    assert first.metadata["uses_torch_sdpa"] is True

    bf16, _ = run_standard_pretraining_trial(
        model_spec=_model_spec(),
        corpus=corpus,
        runtime=replace(runtime, precision="bf16"),
        distributed_context=context,
        optimizer_spec=optimizer,
        total_tokens=8,
        batch_examples=1,
        seed=9,
        validation_interval=1,
        validation_examples=2,
    )
    assert bf16.metadata["precision"] == "bf16"
    assert bf16.final_validation_loss > 0


def test_flash_requires_cuda_and_fsdp_plan_is_explicit(tmp_path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="FlashAttention requires a CUDA"):
        preflight_runtime(
            _model_spec(),
            PretrainingRuntimeSpec(precision="bf16", attention_backend="flash"),
            "cpu",
        )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (8, 0))
    diagnostics = preflight_runtime(
        _model_spec(),
        PretrainingRuntimeSpec(precision="bf16", attention_backend="auto"),
        "cuda",
    )
    assert diagnostics["device"] == "cuda"
    dataset = _corpus_spec(tmp_path)
    config = {
        "model": {
            "vocab_size": 260,
            "context_length": 4,
            "width": 8,
            "depth": 1,
            "num_heads": 2,
            "mlp_multiplier": 2,
        },
        "dataset": {
            "train_path": dataset.train_path,
            "validation_path": dataset.validation_path,
        },
        "runtime": {
            "precision": "bf16",
            "attention_backend": "flash",
            "distributed": "fsdp",
            "num_processes": 2,
        },
        "scales": [{"name": "S1", "width": 8, "depth": 1, "num_heads": 2}],
        "batch_examples": [2, 4, 6, 8],
        "total_tokens": 96,
        "optimizers": [{"name": "adamw", "learning_rates": [0.001]}],
    }
    plan = compile_standard_pretraining_plan(config)
    assert plan["runtime"]["distributed"] == "fsdp"
    assert plan["capabilities"]["single_node_fsdp"] is True


def test_real_text_batch_census_runs_all_three_estimators(tmp_path) -> None:
    dataset = _corpus_spec(tmp_path)
    config = {
        "model": {
            "vocab_size": 260,
            "context_length": 4,
            "width": 8,
            "depth": 1,
            "num_heads": 2,
            "mlp_multiplier": 2,
        },
        "dataset": {
            "train_path": dataset.train_path,
            "validation_path": dataset.validation_path,
        },
        "runtime": {
            "precision": "fp32",
            "attention_backend": "math",
            "distributed": "none",
            "num_processes": 1,
        },
        "scales": [{"name": "S1", "width": 8, "depth": 1, "num_heads": 2}],
        "batch_examples": [1, 2, 4, 8],
        "total_tokens": 32,
        "checkpoint_tokens": 8,
        "continuation_tokens": 32,
        "target_validation_loss": 6.0,
        "validation_interval": 1,
        "validation_examples": 4,
        "gradient_noise_samples": 8,
        "seeds": [3],
        "optimizers": [
            {"name": "adam", "learning_rates": [0.001], "beta2": 0.99}
        ],
    }
    events = []
    result = run_standard_pretraining_batch_census(config, progress=events.append)
    assert result["status"] == "completed"
    assert result["dataset"]["training_tokens"] > 32
    assert len(result["records"]) == 4
    analysis = result["scale_optimizer_analyses"][0]
    assert {"steps_to_target", "direct_checkpoint", "gradient_noise", "consensus"} <= set(
        analysis
    )
    assert events[-1]["phase"] == "complete"


def test_campaign_store_runs_and_resumes_same_pretraining_job(tmp_path) -> None:
    dataset = _corpus_spec(tmp_path)
    config = {
        "model": {
            "vocab_size": 260,
            "context_length": 4,
            "width": 8,
            "depth": 1,
            "num_heads": 2,
            "mlp_multiplier": 2,
        },
        "dataset": {
            "train_path": dataset.train_path,
            "validation_path": dataset.validation_path,
        },
        "runtime": {
            "precision": "fp32",
            "attention_backend": "math",
            "distributed": "none",
            "num_processes": 1,
        },
        "scales": [{"name": "S1", "width": 8, "depth": 1, "num_heads": 2}],
        "batch_examples": [1, 2, 4, 8],
        "total_tokens": 32,
        "checkpoint_tokens": 8,
        "continuation_tokens": 32,
        "target_validation_loss": 6.0,
        "validation_interval": 1,
        "validation_examples": 4,
        "gradient_noise_samples": 8,
        "seeds": [3],
        "optimizers": [{"name": "adam", "learning_rates": [0.001]}],
    }
    store = CampaignStore(tmp_path / "runs")
    job = store.create("standard_pretraining_census", config, "cpu")
    deadline = time.time() + 10
    while job["status"] in {"queued", "running"} and time.time() < deadline:
        time.sleep(0.02)
        job = store.get(job["id"])
    assert job["status"] == "completed", job.get("error")
    assert job["result"]["status"] == "completed"
    resumed = store.create("standard_pretraining_census", config, "cpu")
    assert resumed["id"] == job["id"]
    assert resumed["status"] == "completed"


def test_fsdp_launch_uses_torchrun_module(tmp_path) -> None:
    command = compile_fsdp_launch(tmp_path / "config.json", tmp_path / "result.json", 4)
    assert command[1:5] == ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4"]
    assert "ai_theorist.autoscaler.pretraining_worker" in command
