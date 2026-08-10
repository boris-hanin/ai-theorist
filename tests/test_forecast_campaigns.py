from hashlib import sha256
import json
from pathlib import Path
import shutil

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from ai_theorist.autoscaler import forecast_campaigns, tokenization
from ai_theorist.autoscaler.forecast_campaigns import (
    _sample_rank_partitioned_batch,
    bind_real_text_scaling_config,
    compile_real_text_scaling_plan,
    jiang_parameter_count,
    nugpt_parameter_count,
    run_real_text_scaling_campaign,
)
from ai_theorist.autoscaler.pretraining import DistributedContext
from ai_theorist.autoscaler.forecast_fleet import (
    aggregate_forecast_fleet_cache,
    assign_forecast_fleet_tasks,
    build_forecast_fleet_tasks,
    run_forecast_fleet_shard,
    select_forecast_fleet_learning_rate,
)
from ai_theorist.autoscaler.jiang_chizat import (
    JiangChizatReference,
    JiangChizatShape,
    JiangChizatTransformer,
)
from ai_theorist.autoscaler.normalized_transformer import NormalizedTransformer
from ai_theorist.autoscaler.schema import ArchitectureTemplate, ScaleLevel
from ai_theorist.autoscaler.tokenization import (
    PINNED_TOKENIZERS_PACKAGE_VERSION,
    PinnedTokenizerDefinition,
    TokenizerAssetDefinition,
    TokenizerCanaryDefinition,
    materialize_pinned_token_streams,
    resolve_pinned_tokenizer,
)


def _token_hash(token_ids) -> str:
    digest = sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, "little"))
    return digest.hexdigest()


def _stream(tmp_path: Path, monkeypatch) -> Path:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "Autoscaler": 1,
                "<|endoftext|>": 2,
                "<|extra_id_0|>": 3,
                **{f"unused-{index}": index for index in range(4, 16)},
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
        id="forecast_test",
        name="Forecast test tokenizer",
        implementation="huggingface_tokenizers_json_v1",
        repository="example/forecast-test",
        revision="b" * 40,
        tokenizer_file="tokenizer.json",
        package="tokenizers",
        package_version=PINNED_TOKENIZERS_PACKAGE_VERSION,
        vocab_size=16,
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
        assets=(
            TokenizerAssetDefinition(
                "tokenizer.json", sha256(tokenizer_path.read_bytes()).hexdigest()
            ),
        ),
        canaries=(TokenizerCanaryDefinition("Autoscaler", 1, _token_hash([1])),),
    )
    monkeypatch.setitem(
        tokenization.PINNED_TOKENIZER_REGISTRY, definition.id, definition
    )
    resolved = resolve_pinned_tokenizer(definition.id, tmp_path / "tokenizer")
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text(
        "".join(json.dumps({"text": "Autoscaler"}) + "\n" for _ in range(400)),
        encoding="utf-8",
    )
    validation.write_text(
        "".join(json.dumps({"text": "Autoscaler"}) + "\n" for _ in range(100)),
        encoding="utf-8",
    )
    materialize_pinned_token_streams(
        tokenizer=resolved,
        train_path=train,
        validation_path=validation,
        output_directory=tmp_path / "token-streams",
        shard_token_limit=1024,
    )
    return tmp_path / "token-streams" / "manifest.json"


def _jiang_config(tmp_path: Path, manifest_path: Path):
    shapes = [(1, 4), (1, 6), (1, 8), (2, 8), (2, 10), (2, 12)]
    targets = [
        jiang_parameter_count(
            vocab_size=16,
            context_length=2,
            depth=depth,
            residual_width=width,
            hidden_width=max(2, int(round((width / depth) / 2)) * 2),
        )
        for depth, width in shapes
    ]
    return {
        "run_profile": "smoke",
        "architecture": {
            "block_type": "jiang_chizat_transformer",
            "vocab_size": 16,
            "context_length": 2,
            "head_dimension": 2,
            "reference_depth": 1,
            "reference_hidden_width": 4,
            "reference_residual_width": 4,
        },
        "dataset": {
            "task_type": "tokenized_text",
            "tokenizer": "forecast_test",
            "token_stream_manifest_path": str(manifest_path),
            "maximum_bytes": 1_000_000,
        },
        "ladder": {
            "target_parameters": targets,
            "depths": [depth for depth, _ in shapes],
            "rho_lm_over_d": 1.0,
            "hidden_width_multiple": 2,
            "tokens_per_parameter": 0.02,
            "heldout_scale_count": 1,
            "minimum_parameter_span": 2.0,
            "maximum_repetition_ratio": 1.0,
            "maximum_parameter_error_fraction": 0.001,
            "target_forecasts": [targets[-1] * 2],
            "maximum_extrapolation_factor": 3.0,
        },
        "optimizer": {
            "name": "adam",
            "learning_rates": [0.0001, 0.001, 0.01],
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-12,
        },
        "schedule": "cosine_to_10_percent",
        "batch_examples": 1,
        "validation_examples": 4,
        "validation_interval_steps": 2,
        "seeds": [3],
        "bootstrap_samples": 0,
        "run_negative_control": False,
        "runtime": {
            "precision": "fp32",
            "attention_backend": "math",
            "distributed": "none",
            "num_processes": 1,
            "gradient_accumulation_steps": 1,
            "activation_checkpointing": True,
            "checkpoint_interval_steps": 1,
            "resume": True,
        },
        "cache_directory": str(tmp_path / "trials"),
    }


def test_analytic_counts_match_both_theory_models() -> None:
    jiang = JiangChizatTransformer(
        JiangChizatShape(2, 8, 8, 2),
        vocab_size=16,
        context_length=4,
        reference=JiangChizatReference(1, 4, 4),
    )
    assert sum(parameter.numel() for parameter in jiang.parameters()) == (
        jiang_parameter_count(
            vocab_size=16,
            context_length=4,
            depth=2,
            residual_width=8,
            hidden_width=8,
        )
    )
    architecture = ArchitectureTemplate.from_dict(
        {
            "block_type": "normalized_transformer",
            "activation": "silu",
            "vocab_size": 16,
            "context_length": 4,
            "head_dimension": 2,
            "mlp_multiplier": 2,
            "reference_width": 8,
            "reference_depth": 2,
        }
    )
    nugpt = NormalizedTransformer(architecture, ScaleLevel("S", 8, 2))
    assert sum(parameter.numel() for parameter in nugpt.parameters()) == (
        nugpt_parameter_count(
            vocab_size=16, depth=2, width=8, mlp_multiplier=2
        )
    )


def test_ddp_sampling_partitions_the_same_global_draw_as_one_gpu() -> None:
    class DeterministicCorpus:
        def sample_batch(self, _split, count, generator, _device):
            inputs = torch.randint(0, 1000, (count, 4), generator=generator)
            return inputs, inputs + 1

    corpus = DeterministicCorpus()
    single_generator = torch.Generator().manual_seed(123)
    expected_inputs, expected_targets = _sample_rank_partitioned_batch(
        corpus,
        "train",
        8,
        single_generator,
        DistributedContext(0, 1, 0, "cpu"),
    )
    partitions = []
    target_partitions = []
    for rank in range(2):
        inputs, targets = _sample_rank_partitioned_batch(
            corpus,
            "train",
            4,
            torch.Generator().manual_seed(123),
            DistributedContext(rank, 2, rank, "cpu"),
        )
        partitions.append(inputs)
        target_partitions.append(targets)
    assert torch.equal(torch.cat(partitions), expected_inputs)
    assert torch.equal(torch.cat(target_partitions), expected_targets)


def test_plan_compiles_exact_vocab_aware_constant_tpp_ladder(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch)
    plan = compile_real_text_scaling_plan(_jiang_config(tmp_path, manifest_path))
    assert plan["architecture_contract"]["rho_lm_over_d"] == pytest.approx(1.0)
    assert plan["architecture_contract"]["tied_embeddings"] is True
    assert len(plan["scales"]) == 6
    assert plan["scales"][-1]["heldout"] is True
    assert all(row["relative_parameter_error"] == 0.0 for row in plan["scales"])
    assert all(row["tokens_per_parameter"] > 0.0 for row in plan["scales"])
    assert plan["measurement_contract"]["validation_microbatch_examples"] == 1
    changed = _jiang_config(tmp_path, manifest_path)
    changed["validation_microbatch_examples"] = 2
    assert compile_real_text_scaling_plan(changed)["fingerprint"] != plan["fingerprint"]


def test_forecast_binding_replaces_placeholder_and_compiles_verified_plan(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch)
    config = _jiang_config(tmp_path, manifest_path)
    config["dataset"]["token_stream_manifest_path"] = "__TOKEN_STREAM_MANIFEST__"

    bound, summary = bind_real_text_scaling_config(config, manifest_path)

    assert bound["dataset"]["token_stream_manifest_path"] == str(
        manifest_path.resolve()
    )
    assert config["dataset"]["token_stream_manifest_path"] == (
        "__TOKEN_STREAM_MANIFEST__"
    )
    assert summary["status"] == "bound"
    assert summary["dataset_identity"]["tokenizer_id"] == "forecast_test"
    assert summary["plan_fingerprint"] == compile_real_text_scaling_plan(bound)[
        "fingerprint"
    ]


def test_forecast_binding_refuses_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest does not exist"):
        bind_real_text_scaling_config({}, tmp_path / "missing.json")


def test_mistral_jiang_preset_is_gate_eligible_before_gpu_allocation(
    monkeypatch,
) -> None:
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "autoscaler"
        / "jiang_mistral_100m_forecast.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        forecast_campaigns,
        "token_stream_identity",
        lambda _path: {
            "format": "sharded_uint32_le_v1",
            "fingerprint": "a" * 64,
            "content_fingerprint": "b" * 64,
            "tokenizer_id": "mistral_7b_v03",
            "tokenizer_fingerprint": "c" * 64,
            "vocab_size": 32_768,
            "packing": {"contract": "document_eos_concatenation_v1"},
            "training_tokens": 3_000_000_000,
            "validation_tokens": 100_000_000,
        },
    )

    plan = compile_real_text_scaling_plan(config)

    assert plan["require_gate_eligible_plan"] is True
    assert plan["fit_parameter_span"] >= plan["minimum_parameter_span"]
    assert max(row["repetition_ratio"] for row in plan["scales"]) <= 1.0
    assert plan["scales"][-1]["heldout"] is True
    assert plan["scales"][-1]["parameters"] == 99_709_568
    assert all(row["rho_lm_over_d"] == pytest.approx(4.0) for row in plan["scales"])
    assert all(row["rho_relative_error"] == pytest.approx(0.0) for row in plan["scales"])
    assert plan["target_forecasts"][-1] == 1_000_000_000
    assert (
        plan["target_forecasts"][-1] / plan["scales"][-1]["parameters"]
        <= plan["maximum_extrapolation_factor"]
    )


def test_gate_eligible_plan_refuses_an_undersized_corpus(monkeypatch) -> None:
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "autoscaler"
        / "jiang_mistral_100m_forecast.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        forecast_campaigns,
        "token_stream_identity",
        lambda _path: {
            "format": "sharded_uint32_le_v1",
            "fingerprint": "a" * 64,
            "content_fingerprint": "b" * 64,
            "tokenizer_id": "mistral_7b_v03",
            "tokenizer_fingerprint": "c" * 64,
            "vocab_size": 32_768,
            "packing": {},
            "training_tokens": 1_000_000,
            "validation_tokens": 100_000,
        },
    )

    with pytest.raises(ValueError, match="corpus repetition gate"):
        compile_real_text_scaling_plan(config)


def test_smoke_campaign_runs_accelerated_checkpointed_theory_path(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch)
    config = _jiang_config(tmp_path, manifest_path)
    result = run_real_text_scaling_campaign(config)
    assert result["status"] == "completed"
    assert len(result["scales"]) == 6
    assert len(result["hidden_scale_backtests"]) == 1
    assert result["dataset"]["tokenizer_is_pinned"] is True
    first = result["records"][0]
    assert first["metadata"]["activation_checkpointing"] is True
    assert first["metadata"]["optimizer_group_audit"]["complete"] is True
    assert {
        group["name"]
        for group in first["metadata"]["peak_parameter_group_contract"]
    } == {
        "jiang_embeddings",
        "jiang_norms",
        "jiang_attention_qkv",
        "jiang_attention_output",
        "jiang_ffn_up",
        "jiang_ffn_down",
        "jiang_other_biases",
    }
    assert not list((tmp_path / "trials").glob("*.resume.pt"))


def test_nugpt_refuses_definition_breaking_fsdp(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _stream(tmp_path, monkeypatch)
    config = _jiang_config(tmp_path, manifest_path)
    config["architecture"] = {
        "block_type": "normalized_transformer",
        "activation": "silu",
        "vocab_size": 16,
        "context_length": 2,
        "head_dimension": 2,
        "mlp_multiplier": 1,
        "reference_width": 4,
        "reference_depth": 1,
    }
    widths = [4, 6, 8, 10, 12, 14]
    depths = [1, 1, 1, 2, 2, 2]
    config["ladder"].update(
        target_parameters=[
            nugpt_parameter_count(
                vocab_size=16, depth=depth, width=width, mlp_multiplier=1
            )
            for width, depth in zip(widths, depths)
        ],
        depths=depths,
        target_forecasts=[5000],
    )
    config["runtime"].update(distributed="fsdp", num_processes=2)
    config["batch_examples"] = 2
    config["validation_examples"] = 4
    with pytest.raises(ValueError, match="refuses FSDP"):
        compile_real_text_scaling_plan(config)


def test_fleet_tasks_remove_duplicate_reference_runs_and_balance_flops(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch)
    config = _jiang_config(tmp_path, manifest_path)
    plan = compile_real_text_scaling_plan(config)
    tuning = build_forecast_fleet_tasks(plan, phase="tune")
    ladder = build_forecast_fleet_tasks(
        plan,
        phase="ladder",
        selected_learning_rate=0.001,
        run_negative_control=False,
    )
    assert len(tuning) == 3
    assert len(ladder) == 5
    assert all(task.scale_name != "S1" for task in ladder)
    assignments = assign_forecast_fleet_tasks(ladder, 2)
    assert {task.task_id for rows in assignments for task in rows} == {
        task.task_id for task in ladder
    }
    loads = [sum(task.estimated_flops for task in rows) for rows in assignments]
    assert max(loads) / min(loads) < 1.5


def test_two_shard_fleet_reuses_exact_caches_for_canonical_analysis(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch)
    config = _jiang_config(tmp_path, manifest_path)
    tune_roots = [tmp_path / "tune-0", tmp_path / "tune-1"]
    for shard_index, output in enumerate(tune_roots):
        run_forecast_fleet_shard(
            config,
            phase="tune",
            shard_index=shard_index,
            shard_count=2,
            output_directory=output,
            device="cpu",
        )
    tune_caches = [path / "trials" for path in tune_roots]
    selection = select_forecast_fleet_learning_rate(config, tune_caches)
    assert selection["selected_learning_rate"] in config["optimizer"][
        "learning_rates"
    ]

    ladder_roots = [tmp_path / "ladder-0", tmp_path / "ladder-1"]
    for shard_index, output in enumerate(ladder_roots):
        run_forecast_fleet_shard(
            config,
            phase="ladder",
            shard_index=shard_index,
            shard_count=2,
            selected_learning_rate=selection["selected_learning_rate"],
            output_directory=output,
            device="cpu",
        )
    result = aggregate_forecast_fleet_cache(
        config,
        cache_directories=[
            *tune_caches,
            *(path / "trials" for path in ladder_roots),
        ],
        output_directory=tmp_path / "aggregate",
    )
    assert result["status"] == "completed"
    assert len(result["records"]) == 9
    aggregate = json.loads(
        (tmp_path / "aggregate" / "fleet-aggregate.json").read_text()
    )
    assert aggregate["physical_trial_count"] == 8
    assert aggregate["logical_trial_count"] == 9

    conflict = tmp_path / "conflicting-cache"
    conflict.mkdir()
    original = next(tune_caches[0].glob("*.json"))
    duplicate = conflict / original.name
    shutil.copy2(original, duplicate)
    payload = json.loads(duplicate.read_text())
    payload["final_validation_loss"] += 1
    duplicate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting duplicate"):
        select_forecast_fleet_learning_rate(
            config, [*tune_caches, conflict]
        )
