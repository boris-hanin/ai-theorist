from dataclasses import replace
import time

import numpy as np
import pytest
import torch
import ai_theorist.autoscaler.pretraining as pretraining

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
    runtime_checkpoint_due,
    synchronized_runtime_checkpoint_due,
    run_standard_pretraining_batch_census,
    run_standard_pretraining_trial,
)


def test_runtime_checkpoint_supports_wall_clock_or_step_cadence() -> None:
    runtime = PretrainingRuntimeSpec.from_dict(
        {"checkpoint_interval_steps": 10, "checkpoint_interval_seconds": 60}
    )
    assert not runtime_checkpoint_due(
        runtime, step=3, total_steps=100, last_checkpoint_at=100, now=159
    )
    assert runtime_checkpoint_due(
        runtime, step=3, total_steps=100, last_checkpoint_at=100, now=160
    )
    assert runtime_checkpoint_due(
        runtime, step=10, total_steps=100, last_checkpoint_at=100, now=101
    )
    assert not runtime_checkpoint_due(
        runtime, step=100, total_steps=100, last_checkpoint_at=100, now=101
    )
    assert not runtime_checkpoint_due(
        PretrainingRuntimeSpec(),
        step=100,
        total_steps=100,
        last_checkpoint_at=100,
        now=1000,
    )


def test_runtime_retained_checkpoint_tpp_contract_is_strict() -> None:
    runtime = PretrainingRuntimeSpec.from_dict(
        {"retained_checkpoint_tokens_per_parameter": [10, 20.0, 40]}
    )
    assert runtime.retained_checkpoint_tokens_per_parameter == (10.0, 20.0, 40.0)
    with pytest.raises(ValueError, match="unique, increasing"):
        PretrainingRuntimeSpec.from_dict(
            {"retained_checkpoint_tokens_per_parameter": [20, 10]}
        )
    with pytest.raises(ValueError, match="must be a sequence"):
        PretrainingRuntimeSpec.from_dict(
            {"retained_checkpoint_tokens_per_parameter": "10,20"}
        )


def test_distributed_wall_clock_checkpoint_uses_primary_decision(monkeypatch) -> None:
    runtime = PretrainingRuntimeSpec.from_dict(
        {"checkpoint_interval_seconds": 60}
    )
    context = pretraining.DistributedContext(1, 8, 1, "cpu")

    def broadcast(marker, src):
        assert src == 0
        marker.fill_(1)

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    assert synchronized_runtime_checkpoint_due(
        runtime,
        context,
        step=3,
        total_steps=100,
        last_checkpoint_at=100,
        now=101,
    )


def test_ddp_checkpoint_is_one_shared_rank_zero_state(tmp_path, monkeypatch) -> None:
    runtime = PretrainingRuntimeSpec.from_dict(
        {"distributed": "ddp", "num_processes": 8, "resume": True}
    )
    primary = DistributedContext(0, 8, 0, "cpu")
    replica = DistributedContext(1, 8, 1, "cpu")
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(17)
    base = tmp_path / "trial.resume"
    barriers = []

    monkeypatch.setattr(
        torch.distributed, "barrier", lambda: barriers.append("barrier")
    )
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda tensor, op: None
    )
    pretraining.save_runtime_checkpoint(
        base_path=base,
        model=model,
        plain_model=model,
        optimizer=optimizer,
        context=primary,
        runtime=runtime,
        identity_fingerprint="a" * 64,
        step=7,
        generator=generator,
        extra={"tokens_seen": 123},
    )
    shared = base.with_suffix(".pt")
    assert shared.is_file()
    assert not pretraining.runtime_checkpoint_path(base, primary).exists()
    checkpoint = torch.load(shared, map_location="cpu", weights_only=False)
    assert checkpoint["schema_version"] == 2
    assert checkpoint["distributed"] == "ddp"
    assert checkpoint["rank"] == 0

    replica_model = torch.nn.Linear(3, 2)
    replica_optimizer = torch.optim.Adam(replica_model.parameters(), lr=0.01)
    resumed = pretraining.load_runtime_checkpoint(
        base_path=base,
        model=replica_model,
        plain_model=replica_model,
        optimizer=replica_optimizer,
        context=replica,
        runtime=runtime,
        identity_fingerprint="a" * 64,
        generator=torch.Generator(),
    )
    assert resumed == {"step": 7, "extra": {"tokens_seen": 123}}
    assert all(
        torch.equal(left, right)
        for left, right in zip(model.parameters(), replica_model.parameters())
    )
    pretraining.clear_runtime_checkpoint(base, primary, runtime)
    assert not shared.exists()
    assert barriers == ["barrier", "barrier"]


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


def test_memory_mapped_uint32_stream_preserves_large_token_ids(tmp_path) -> None:
    train = tmp_path / "train-u32.bin"
    validation = tmp_path / "validation-u32.bin"
    values = np.array([0, 65_535, 100_257, 100_277] * 20, dtype="<u4")
    values.tofile(train)
    values.tofile(validation)
    corpus = TokenizedTextCorpus(
        TokenizedTextSpec(
            str(train),
            str(validation),
            tokenizer="uint32_bin_v1",
        ),
        context_length=4,
        vocab_size=100_278,
    )
    inputs, targets = corpus.sample_batch(
        "train", 16, torch.Generator().manual_seed(4), "cpu"
    )
    assert int(torch.maximum(inputs.max(), targets.max())) > 65_535
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
    assert corpus.tokenizer_is_pinned is False


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


def test_gradient_accumulation_matches_full_batch(tmp_path) -> None:
    corpus = TokenizedTextCorpus(_corpus_spec(tmp_path), context_length=4)
    context = DistributedContext(0, 1, 0, "cpu")
    optimizer = OptimizerHyperparameters("adam", 0.001, beta2=0.99)
    full, _ = run_standard_pretraining_trial(
        model_spec=_model_spec(),
        corpus=corpus,
        runtime=PretrainingRuntimeSpec(attention_backend="math"),
        distributed_context=context,
        optimizer_spec=optimizer,
        total_tokens=32,
        batch_examples=4,
        seed=13,
        validation_interval=1,
        validation_examples=4,
    )
    accumulated, _ = run_standard_pretraining_trial(
        model_spec=_model_spec(),
        corpus=corpus,
        runtime=PretrainingRuntimeSpec(
            attention_backend="math",
            gradient_accumulation_steps=2,
            activation_checkpointing=True,
        ),
        distributed_context=context,
        optimizer_spec=optimizer,
        total_tokens=32,
        batch_examples=4,
        seed=13,
        validation_interval=1,
        validation_examples=4,
    )
    assert accumulated.final_validation_loss == pytest.approx(
        full.final_validation_loss, rel=1e-6, abs=1e-7
    )
    assert accumulated.accumulation_steps == 2
    assert accumulated.metadata["activation_checkpointing"] is True


def test_midtrial_checkpoint_resumes_exact_optimizer_and_sampling_state(
    tmp_path, monkeypatch
) -> None:
    corpus = TokenizedTextCorpus(_corpus_spec(tmp_path), context_length=4)
    context = DistributedContext(0, 1, 0, "cpu")
    optimizer = OptimizerHyperparameters("adam", 0.001, beta2=0.99)
    runtime = PretrainingRuntimeSpec(
        attention_backend="math",
        checkpoint_interval_steps=1,
        resume=True,
    )
    interrupted_cache = tmp_path / "interrupted"
    original_save = pretraining.save_runtime_checkpoint
    calls = 0

    def interrupt_after_first_checkpoint(**kwargs):
        nonlocal calls
        original_save(**kwargs)
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated worker interruption")

    monkeypatch.setattr(
        pretraining, "save_runtime_checkpoint", interrupt_after_first_checkpoint
    )
    with pytest.raises(RuntimeError, match="simulated worker interruption"):
        run_standard_pretraining_trial(
            model_spec=_model_spec(),
            corpus=corpus,
            runtime=runtime,
            distributed_context=context,
            optimizer_spec=optimizer,
            total_tokens=32,
            batch_examples=2,
            seed=17,
            validation_interval=1,
            validation_examples=4,
            cache_directory=interrupted_cache,
        )
    monkeypatch.setattr(pretraining, "save_runtime_checkpoint", original_save)
    resumed, _ = run_standard_pretraining_trial(
        model_spec=_model_spec(),
        corpus=corpus,
        runtime=runtime,
        distributed_context=context,
        optimizer_spec=optimizer,
        total_tokens=32,
        batch_examples=2,
        seed=17,
        validation_interval=1,
        validation_examples=4,
        cache_directory=interrupted_cache,
    )
    clean, _ = run_standard_pretraining_trial(
        model_spec=_model_spec(),
        corpus=corpus,
        runtime=runtime,
        distributed_context=context,
        optimizer_spec=optimizer,
        total_tokens=32,
        batch_examples=2,
        seed=17,
        validation_interval=1,
        validation_examples=4,
        cache_directory=tmp_path / "clean",
    )
    assert resumed.metadata["resumed_from_step"] == 1
    assert resumed.final_validation_loss == clean.final_validation_loss
    assert resumed.validation_checkpoints == clean.validation_checkpoints
    assert not list(interrupted_cache.glob("*.resume.pt"))


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
    assert plan["capabilities"]["single_node_ddp"] is True
    assert plan["capabilities"]["gradient_accumulation"] is True
    assert plan["capabilities"]["activation_checkpointing"] is True
    assert plan["capabilities"]["mid_trial_resume"] is True


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
