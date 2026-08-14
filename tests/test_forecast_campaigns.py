from hashlib import sha256
from copy import deepcopy
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from ai_theorist.autoscaler import forecast_campaigns, tokenization
from ai_theorist.autoscaler.forecast_campaigns import (
    _sample_rank_partitioned_batch,
    bind_real_text_scaling_config,
    completep_parameter_count,
    compile_real_text_scaling_plan,
    forecast_trial_cache_identity,
    jiang_moe_parameter_counts,
    jiang_parameter_count,
    nugpt_parameter_count,
    run_real_text_scaling_campaign,
)
from ai_theorist.autoscaler.pretraining import (
    DistributedContext,
    PretrainingRuntimeSpec,
    TokenizedTextCorpus,
)
from ai_theorist.autoscaler.forecast_fleet import (
    aggregate_forecast_fleet_cache,
    assign_forecast_fleet_tasks,
    build_forecast_fleet_tasks,
    run_forecast_fleet_shard,
    select_forecast_fleet_learning_rate,
)
from ai_theorist.autoscaler.forecast_critical_batch import (
    ForecastCriticalBatchTask,
    build_forecast_critical_batch_tasks,
    compile_conservative_batch_warmup,
    compile_forecast_critical_batch_plan,
    run_forecast_critical_batch_task,
)
from ai_theorist.autoscaler.critical_batch import CriticalBatchEstimate
from ai_theorist.autoscaler.jiang_chizat import (
    JiangChizatReference,
    JiangChizatShape,
    JiangChizatTransformer,
)
from ai_theorist.autoscaler.jiang_moe import (
    JIANG_MOE_REPORTED_LR_MULTIPLIERS,
    JiangMoEReference,
    JiangMoEShape,
    JiangMoETransformer,
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


def _stream(tmp_path: Path, monkeypatch, *, vocab_size: int = 16) -> Path:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "Autoscaler": 1,
                "<|endoftext|>": 2,
                "<|extra_id_0|>": 3,
                **{f"unused-{index}": index for index in range(4, vocab_size)},
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
        vocab_size=vocab_size,
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


def _jiang_moe_config(tmp_path: Path, manifest_path: Path):
    shapes = [(1, 4), (1, 6), (1, 8), (2, 8), (2, 12), (2, 16)]
    counts = [
        jiang_moe_parameter_counts(
            vocab_size=16,
            context_length=2,
            depth=depth,
            residual_width=width,
            expert_width=max(2, int(round((width / depth) / 2)) * 2),
            num_experts=4,
            active_experts=1,
        )
        for depth, width in shapes
    ]
    config = _jiang_config(tmp_path, manifest_path)
    config["architecture"] = {
        "block_type": "jiang_moe_transformer",
        "vocab_size": 16,
        "context_length": 2,
        "head_dimension": 2,
        "reference_depth": 1,
        "reference_hidden_width": 4,
        "reference_residual_width": 4,
        "reference_num_experts": 4,
        "reference_active_experts": 1,
        "num_experts": 4,
        "active_experts": 1,
        "router_gamma": 1.0,
        "initialization_std": 2.0 ** -6,
    }
    config["ladder"] = {
        **config["ladder"],
        "target_parameters": [row["active_parameters"] for row in counts],
        "depths": [depth for depth, _ in shapes],
        "residual_widths": [width for _, width in shapes],
        "expert_widths": [
            max(2, int(round((width / depth) / 2)) * 2)
            for depth, width in shapes
        ],
        "target_parameter_axis": "active_parameters",
        "token_budget_parameter_axis": "active_parameters",
        "fit_parameter_axis": "active_non_embedding_parameters",
        "num_experts": [4] * len(shapes),
        "active_experts": [1] * len(shapes),
        "target_forecasts": [counts[-1]["active_non_embedding_parameters"] * 2],
    }
    config["ladder"].pop("rho_lm_over_d")
    config["ladder"].pop("maximum_rho_relative_error", None)
    config["optimizer"] = {
        **config["optimizer"],
        "learning_rate_multipliers": dict(JIANG_MOE_REPORTED_LR_MULTIPLIERS),
        "expert_bias_learning_rate": 0.01,
    }
    config["runtime"]["activation_checkpointing"] = False
    return config


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


def test_jiang_moe_plan_preserves_source_parameterization_and_active_axes(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch)
    config = _jiang_moe_config(tmp_path, manifest_path)
    plan = compile_real_text_scaling_plan(config)
    contract = plan["architecture_contract"]
    assert contract["router_gamma"] == 1.0
    assert contract["rho_lm_over_d_is_not_a_source_transfer_invariant"] is True
    assert contract["optional_declared_rho_lm_over_d"] is None
    assert contract["fixed_active_expert_fraction"] == pytest.approx(0.25)
    assert contract["attention_scale"] == "QK^T/d_head"
    assert contract["residual_branch_scale"] == "1/L"
    assert contract["moe_mixing"] == (
        "sigmoid-weighted hard top-A divided by A"
    )
    assert all(row["parameters"] > row["active_parameters"] for row in plan["scales"])
    assert all(
        row["target_parameter_axis"] == "active_parameters"
        and row["token_budget_parameter_axis"] == "active_parameters"
        and row["active_experts"] / row["num_experts"] == pytest.approx(0.25)
        for row in plan["scales"]
    )

    invalid = deepcopy(config)
    invalid["architecture"]["router_gamma"] = 0.5
    with pytest.raises(ValueError, match="router_gamma=1"):
        compile_real_text_scaling_plan(invalid)
    invalid = deepcopy(config)
    invalid["optimizer"]["learning_rate_multipliers"][
        "jiang_moe_router"
    ] = 1.0
    with pytest.raises(ValueError, match="exact Appendix-D.1"):
        compile_real_text_scaling_plan(invalid)
    invalid = deepcopy(config)
    invalid["ladder"]["active_experts"][-1] = 2
    with pytest.raises(ValueError, match="fixed A/E sparsity"):
        compile_real_text_scaling_plan(invalid)
    invalid = deepcopy(config)
    invalid["runtime"]["distributed"] = "fsdp"
    invalid["runtime"]["num_processes"] = 2
    with pytest.raises(ValueError, match="requires DDP or one GPU"):
        compile_real_text_scaling_plan(invalid)


def test_jiang_moe_rho32_transfer_pilot_is_exact_and_fully_factorial(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch, vocab_size=50_257)
    config = json.loads(
        Path(
            "configs/autoscaler/jiang_moe_slimpajama_rho32_transfer_pilot.json"
        ).read_text(encoding="utf-8")
    )
    config["dataset"]["token_stream_manifest_path"] = str(manifest_path)
    config["dataset"]["tokenizer"] = "forecast_test"
    plan = compile_real_text_scaling_plan(config)
    assert [
        (row["depth"], row["width"], row["hidden_width"])
        for row in plan["scales"]
    ] == [
        (2, 128, 2048),
        (4, 256, 2048),
        (6, 384, 2048),
        (8, 512, 2048),
        (12, 768, 2048),
        (16, 1024, 2048),
    ]
    assert all(row["rho_lm_over_d"] == 32.0 for row in plan["scales"])
    assert all(
        row["width"]
        / (row["hidden_width"] * row["num_experts"] * row["depth"])
        == pytest.approx(1 / 128)
        for row in plan["scales"]
    )
    assert plan["architecture_contract"]["optional_declared_rho_lm_over_d"] == 32.0
    assert len(build_forecast_fleet_tasks(plan, phase="tune")) == 24
    ladder = build_forecast_fleet_tasks(
        plan,
        phase="ladder",
        selected_learning_rate=plan["learning_rates"][4],
        run_negative_control=True,
    )
    assert len(ladder) == 18
    assert sum(row.optimizer_mode == "theory" for row in ladder) == 15
    assert sum(row.optimizer_mode == "wrong_global" for row in ladder) == 3
    assert all(row["presented_tokens"] == 6_553_600 for row in plan["scales"])


def test_jiang_moe_rho32_adaptive_lower_bracket_remains_fully_factorial(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch, vocab_size=50_257)
    config = json.loads(
        Path(
            "configs/autoscaler/jiang_moe_slimpajama_rho32_transfer_pilot_lrbracket_v2.json"
        ).read_text(encoding="utf-8")
    )
    config["dataset"]["token_stream_manifest_path"] = str(manifest_path)
    config["dataset"]["tokenizer"] = "forecast_test"
    plan = compile_real_text_scaling_plan(config)
    assert plan["learning_rates"] == [
        2.0**-10,
        2.0**-9,
        2.0**-8,
        2.0**-7,
        2.0**-6,
        2.0**-5,
        2.0**-4,
    ]
    assert len(build_forecast_fleet_tasks(plan, phase="tune")) == 21
    assert [
        (row["depth"], row["width"], row["hidden_width"])
        for row in plan["scales"]
    ] == [
        (2, 128, 2048),
        (4, 256, 2048),
        (6, 384, 2048),
        (8, 512, 2048),
        (12, 768, 2048),
        (16, 1024, 2048),
    ]


def test_jiang_moe_rho32_active_1b_ladder_uses_active_tpp_and_exact_geometry(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch, vocab_size=50_257)
    config = json.loads(
        Path(
            "configs/autoscaler/jiang_moe_slimpajama_rho32_active_1b.json"
        ).read_text(encoding="utf-8")
    )
    config["dataset"]["token_stream_manifest_path"] = str(manifest_path)
    config["dataset"]["tokenizer"] = "forecast_test"
    # The synthetic test stream is intentionally tiny. Production
    # preregistration separately enforces no repetition against SlimPajama.
    config["ladder"]["require_gate_eligible_plan"] = False
    config["ladder"]["maximum_repetition_ratio"] = 1_000_000.0
    plan = compile_real_text_scaling_plan(config)
    assert [
        (row["depth"], row["width"], row["hidden_width"])
        for row in plan["scales"]
    ] == [
        (16, 512, 1024),
        (16, 768, 1536),
        (16, 1024, 2048),
        (16, 1280, 2560),
        (16, 1536, 3072),
        (16, 1792, 3584),
        (16, 2048, 4096),
        (16, 2304, 4608),
        (16, 2624, 5248),
    ]
    assert all(row["rho_lm_over_d"] == 32.0 for row in plan["scales"])
    assert all(
        row["token_budget_parameter_axis"] == "active_parameters"
        and row["target_parameter_axis"] == "active_non_embedding_parameters"
        for row in plan["scales"]
    )
    endpoint = plan["scales"][-1]
    assert endpoint["active_parameters"] == 1_014_509_312
    assert endpoint["active_non_embedding_parameters"] == 881_963_200
    assert endpoint["parameters"] == 2_336_879_360
    assert endpoint["heldout"] is True
    assert len(build_forecast_fleet_tasks(plan, phase="tune")) == 8
    ladder = build_forecast_fleet_tasks(
        plan,
        phase="ladder",
        selected_learning_rate=plan["learning_rates"][4],
        run_negative_control=False,
    )
    assert len(ladder) == 8
    assert {task.scale_name for task in ladder} == {
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
        "S8",
        "S9",
    }

    source = tmp_path / "active-tpp-config.json"
    canary = tmp_path / "active-tpp-canary.json"
    source.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_forecast_runtime_canary.py",
            str(source),
            str(manifest_path),
            str(canary),
            "--steps",
            "3",
            "--learning-rate",
            str(plan["learning_rates"][4]),
            "--batch-examples",
            "128",
            "--gradient-accumulation-steps",
            "16",
        ],
    )
    runpy.run_path("scripts/prepare_forecast_runtime_canary.py", run_name="__main__")
    canary_plan = compile_real_text_scaling_plan(
        json.loads(canary.read_text(encoding="utf-8"))
    )
    assert canary_plan["scales"][-1]["optimizer_steps"] == 3


def test_jiang_moe_rho32_active_1b_10tpp_ladder_is_exact(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch, vocab_size=32_768)
    config = json.loads(
        Path(
            "configs/autoscaler/"
            "jiang_moe_fineweb_mistral_rho32_active_1b_10tpp.json"
        ).read_text(encoding="utf-8")
    )
    config["dataset"]["token_stream_manifest_path"] = str(manifest_path)
    config["dataset"].pop("token_stream_verification_receipt_path")
    config["dataset"]["tokenizer"] = "forecast_test"
    config["ladder"]["require_gate_eligible_plan"] = False
    config["ladder"]["maximum_repetition_ratio"] = 1_000_000_000.0
    plan = compile_real_text_scaling_plan(config)

    assert [
        (row["depth"], row["width"], row["hidden_width"])
        for row in plan["scales"]
    ] == [
        (16, 512, 1024),
        (16, 768, 1536),
        (16, 1024, 2048),
        (16, 1280, 2560),
        (16, 1536, 3072),
        (16, 1792, 3584),
        (16, 2048, 4096),
        (16, 2304, 4608),
        (16, 2688, 5376),
    ]
    assert [
        row["active_non_embedding_parameters"] for row in plan["scales"]
    ] == [
        33_678_400,
        75_683_392,
        134_465_600,
        210_025_024,
        302_361_664,
        411_475_520,
        537_366_592,
        680_034_880,
        925_494_592,
    ]
    assert all(row["depth"] * row["hidden_width"] / row["width"] == 32.0
               for row in plan["scales"])
    assert all(
        abs(row["presented_tokens"] / row["active_parameters"] - 10.0)
        <= 0.002
        for row in plan["scales"]
    )
    endpoint = plan["scales"][-1]
    assert endpoint["parameters"] == 2_401_916_224
    assert endpoint["active_parameters"] == 1_014_263_104
    assert endpoint["non_embedding_parameters"] == 2_313_147_712
    assert endpoint["active_non_embedding_parameters"] == 925_494_592
    assert endpoint["presented_tokens"] == 10_142_613_504
    assert endpoint["optimizer_steps"] == 77_382
    assert endpoint["heldout"] is True


def test_jiang_moe_real_text_trial_audits_every_lr_epsilon_and_init_group(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch)
    config = _jiang_moe_config(tmp_path, manifest_path)
    plan = compile_real_text_scaling_plan(config)
    scale = plan["scales"][3]
    corpus = TokenizedTextCorpus(
        forecast_campaigns.forecast_tokenized_text_spec(config),
        context_length=2,
        vocab_size=16,
    )
    cache = tmp_path / "moe-trials"
    cache.mkdir()
    record = forecast_campaigns._run_trial(
        config=config,
        plan=plan,
        scale=scale,
        corpus=corpus,
        runtime=PretrainingRuntimeSpec.from_dict(config["runtime"]),
        context=DistributedContext(0, 1, 0, "cpu"),
        eta=0.001,
        seed=3,
        optimizer_mode="theory",
        cache_directory=cache,
    )
    audit = record.metadata["optimizer_group_audit"]
    assert audit["complete"] is True
    assert audit["disjoint"] is True
    assert len(audit["groups"]) == 8
    groups = {row["name"]: row for row in audit["groups"]}
    assert groups["jiang_moe_router"]["learning_rate"] == pytest.approx(
        0.001 * (scale["width"] / 4) ** -1 / 16
    )
    assert groups["jiang_moe_expert_down"]["learning_rate"] == pytest.approx(
        0.001 * (scale["hidden_width"] / 4) ** -1 / 16
    )
    assert groups["jiang_moe_expert_up"]["adam_epsilon"] == pytest.approx(
        1e-12
        * (scale["hidden_width"] / 4) ** -1
        * (scale["depth"] / 1) ** -1
    )
    assert groups["jiang_moe_expert_down"]["adam_epsilon"] == pytest.approx(
        1e-12
        * (scale["width"] / 4)
        * (scale["hidden_width"] / 4) ** -2
        * (scale["depth"] / 1) ** -1
    )
    initialization = audit["initialization_contract"]
    assert initialization["router_gamma"] == 1.0
    assert initialization["attention_value_std"] == pytest.approx(
        initialization["attention_qko_std"] / 16
    )
    assert initialization["expert_down_std"] == pytest.approx(
        initialization["attention_qko_std"]
        * initialization["ffn_ratio_ratio"] ** -1
        / 4
    )
    assert audit["manual_expert_bias"]["learning_rate"] == 0.01
    assert record.metadata["parameter_accounting"]["total_parameters"] > (
        record.metadata["parameter_accounting"]["active_parameters_per_token"]
    )
    assert record.metadata["diagnostics"]["maximum_absolute_expert_bias"] > 0.0


def test_forecast_single_seed_requires_explicit_exploratory_disclosure(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch)
    config = _jiang_config(tmp_path, manifest_path)
    config["run_profile"] = "forecast"
    with pytest.raises(ValueError, match="requires at least three seeds"):
        compile_real_text_scaling_plan(config)

    config["exploratory_single_seed"] = True
    plan = compile_real_text_scaling_plan(config)
    assert plan["seeds"] == [3]
    assert plan["exploratory_single_seed"] is True


def test_forecast_critical_batch_plan_and_conservative_schedule(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch)
    config = _jiang_config(tmp_path, manifest_path)
    source = compile_real_text_scaling_plan(config)
    config["critical_batch"] = {
        "source_plan_fingerprint": source["fingerprint"],
        "source_selection_sha256": "a" * 64,
        "selected_learning_rate": 0.001,
        "selected_weight_decay_tau_ema": None,
        "reference_batch_examples": 1,
        "initial_batch_examples": 1,
        "microbatch_examples": 1,
        "batch_examples": [1, 2, 4, 8],
        "checkpoint_tokens": [32, 64, 96],
        "continuation_tokens": 1024,
        "pilot_tokens": 32,
        "eta_multipliers": [0.0625, 0.125, 0.25, 0.5, 1.0],
        "seeds": [3, 5, 7],
        "pilot_seed_count": 2,
        "loss_tolerance": 0.01,
        "safety_fraction": 0.8,
    }
    plan = compile_forecast_critical_batch_plan(config)
    assert plan["source_forecast_plan_fingerprint"] == source["fingerprint"]
    assert len(build_forecast_critical_batch_tasks(plan, phase="pilot")) == 10
    assert len(build_forecast_critical_batch_tasks(plan, phase="baseline")) == 3
    assert len(build_forecast_critical_batch_tasks(plan, phase="branch")) == 36

    estimates = [
        (
            tokens,
            CriticalBatchEstimate(
                "local_branched",
                critical,
                lower,
                upper,
                True,
                {},
            ),
        )
        for tokens, critical, lower, upper in (
            (32, 6.0, 4.0, 8.0),
            (64, 12.0, 8.0, 16.0),
            (96, 24.0, 16.0, 32.0),
        )
    ]
    schedule = compile_conservative_batch_warmup(
        checkpoints=estimates,
        candidate_batch_examples=[1, 2, 4, 8],
        context_length=2,
        initial_batch_examples=1,
        reference_batch_examples=1,
        safety_fraction=0.8,
    )
    assert schedule["qualified"]
    assert [stage["batch_examples"] for stage in schedule["stages"]] == [1, 2, 4]
    assert not schedule["uses_extrapolated_batch"]


def test_forecast_critical_batch_pilot_runs_the_faithful_model(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch)
    config = _jiang_config(tmp_path, manifest_path)
    source = compile_real_text_scaling_plan(config)
    config["critical_batch"] = {
        "source_plan_fingerprint": source["fingerprint"],
        "source_selection_sha256": "b" * 64,
        "selected_learning_rate": 0.001,
        "selected_weight_decay_tau_ema": None,
        "reference_batch_examples": 1,
        "initial_batch_examples": 1,
        "microbatch_examples": 1,
        "batch_examples": [1, 2, 4, 8],
        "checkpoint_tokens": [32, 64, 96],
        "continuation_tokens": 1024,
        "pilot_tokens": 32,
        "eta_multipliers": [0.0625, 0.125, 0.25, 0.5, 1.0],
        "seeds": [3, 5, 7],
        "pilot_seed_count": 2,
    }
    result = run_forecast_critical_batch_task(
        config,
        task=ForecastCriticalBatchTask("pilot", 3, eta_multiplier=0.25),
        root=tmp_path / "cbs",
        device="cpu",
    )
    assert result["stop_tokens"] == 32
    assert result["eta_actual"] == pytest.approx(0.00025)
    assert result["peak_parameter_group_contract"]
    assert result["final_validation_loss"] > 0
    baseline = run_forecast_critical_batch_task(
        config,
        task=ForecastCriticalBatchTask("baseline", 3),
        root=tmp_path / "cbs",
        device="cpu",
        selected_eta_multiplier=0.25,
    )
    assert len(baseline["checkpoints"]) == 3
    branch = run_forecast_critical_batch_task(
        config,
        task=ForecastCriticalBatchTask(
            "branch", 3, checkpoint_tokens=32, batch_examples=8
        ),
        root=tmp_path / "cbs",
        device="cpu",
        selected_eta_multiplier=0.25,
    )
    assert branch["start_tokens"] == 32
    assert branch["stop_tokens"] == 1056
    assert branch["gradient_accumulation_steps"] == 8


def test_forecast_plan_compiles_measured_variable_batch_in_token_time(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch)
    fixed = _jiang_config(tmp_path, manifest_path)
    fixed["ladder"]["tokens_per_parameter"] = 1.0
    fixed_plan = compile_real_text_scaling_plan(fixed)
    dynamic = deepcopy(fixed)
    dynamic["batch_schedule"] = {
        "reference_batch_examples": 1,
        "microbatch_examples": 1,
        "learning_rate_rule": "adam_sqrt",
        "uses_extrapolated_batch": False,
        "source_critical_batch_result_sha256": "c" * 64,
        "stages": [
            {"start_tokens": 0, "batch_examples": 1},
            {"start_tokens": 32, "batch_examples": 2},
        ],
    }
    dynamic_plan = compile_real_text_scaling_plan(dynamic)
    assert dynamic_plan["fingerprint"] != fixed_plan["fingerprint"]
    assert dynamic_plan["batch_schedule"]["learning_rate_rule"] == "adam_sqrt"
    assert dynamic_plan["scales"][-1]["optimizer_steps"] < (
        dynamic_plan["scales"][-1]["presented_tokens"] // 2
    )
    assert dynamic_plan["execution_order"][0] == (
        "verify_critical_batch_result_and_measured_lower_bounds"
    )
    corpus = TokenizedTextCorpus(
        forecast_campaigns.forecast_tokenized_text_spec(dynamic),
        context_length=2,
        vocab_size=16,
    )
    runtime = PretrainingRuntimeSpec.from_dict(dynamic["runtime"])
    (tmp_path / "dynamic-trials").mkdir()
    record = forecast_campaigns._run_trial(
        config=dynamic,
        plan=dynamic_plan,
        scale=dynamic_plan["scales"][-1],
        corpus=corpus,
        runtime=runtime,
        context=DistributedContext(0, 1, 0, "cpu"),
        eta=0.001,
        seed=3,
        optimizer_mode="theory",
        cache_directory=tmp_path / "dynamic-trials",
    )
    assert len(record.metadata["batch_schedule_trace"]) == 2
    assert record.validation_checkpoints[-1]["tokens"] == record.total_tokens
    assert record.accumulation_steps == 2


def test_completep_comparison_plan_tunes_reference_then_only_target(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch)
    identity = tokenization.token_stream_identity(manifest_path)
    targets = [
        completep_parameter_count(
            vocab_size=16,
            context_length=2,
            depth=depth,
            width=width,
            mlp_multiplier=2,
        )
        for depth, width in ((1, 4), (2, 8))
    ]
    config = {
        "run_profile": "comparison",
        "architecture": {
            "block_type": "completep_transformer",
            "vocab_size": 16,
            "context_length": 2,
            "head_dimension": 2,
            "mlp_multiplier": 2,
            "reference_depth": 1,
            "reference_width": 4,
            "initialization_std": 0.02,
            "activation": "relu_squared",
            "position_encoding": "learned_absolute",
        },
        "dataset": {
            "task_type": "tokenized_text",
            "tokenizer": "forecast_test",
            "token_stream_manifest_path": str(manifest_path),
            "maximum_bytes": 1_000_000,
        },
        "ladder": {
            "target_parameters": targets,
            "depths": [1, 2],
            "tokens_per_parameter": 0.02,
            "heldout_scale_count": 1,
            "reference_scale_index": 0,
            "maximum_parameter_error_fraction": 0.001,
            "minimum_parameter_span": 1.0,
            "maximum_repetition_ratio": 1.0,
            "target_forecasts": [],
        },
        "optimizer": {
            "name": "adamw",
            "learning_rates": [0.0001, 0.001, 0.01],
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-16,
            "weight_decay_tau_ema": 0.1407,
            "fused": False,
        },
        "schedule": "linear_warmup_decay_to_zero",
        "batch_examples": 1,
        "validation_examples": 3,
        "validation_microbatch_examples": 1,
        "validation_interval_steps": 1,
        "seeds": [3, 5, 7],
        "runtime": {
            "precision": "fp32",
            "attention_backend": "math",
            "distributed": "none",
            "num_processes": 1,
            "gradient_accumulation_steps": 1,
            "activation_checkpointing": False,
            "resume": True,
        },
        "comparison_contract": {
            "baseline_plan_fingerprint": "a" * 64,
            "baseline_aggregate_sha256": "b" * 64,
            "baseline_dataset_fingerprint": identity["fingerprint"],
            "baseline_tokenizer_fingerprint": identity["tokenizer_fingerprint"],
            "baseline_architecture": "jiang_chizat_transformer",
            "baseline_parameters": targets[-1],
            "baseline_mean_validation_loss": 4.0,
            "baseline_seed_losses": [4.1, 4.0, 3.9],
        },
        "run_negative_control": False,
    }
    plan = compile_real_text_scaling_plan(config)
    assert plan["campaign"] == "real_text_100m_completep_comparison"
    assert plan["planned_grid_trials"] == 12
    assert plan["architecture_contract"]["parameterization"] == (
        "completep_alpha_1_adamw"
    )
    tune = build_forecast_fleet_tasks(plan, phase="tune")
    target = build_forecast_fleet_tasks(
        plan,
        phase="ladder",
        selected_learning_rate=0.001,
        run_negative_control=False,
    )
    assert len(tune) == 9
    assert len(target) == 3
    assert {task.scale_name for task in target} == {"S2"}
    assert {task.optimizer_mode for task in target} == {"theory"}

    jointly_tuned = json.loads(json.dumps(config))
    jointly_tuned["optimizer"].pop("weight_decay_tau_ema")
    jointly_tuned["optimizer"]["weight_decay_tau_ema_grid"] = [0.05, 0.1, 0.2]
    joint_plan = compile_real_text_scaling_plan(jointly_tuned)
    assert joint_plan["planned_grid_trials"] == 30
    assert joint_plan["weight_decay_tau_ema_grid"] == [0.05, 0.1, 0.2]
    joint_tune = build_forecast_fleet_tasks(joint_plan, phase="tune")
    assert len(joint_tune) == 27
    assert {task.weight_decay_tau_ema for task in joint_tune} == {
        0.05,
        0.1,
        0.2,
    }
    with pytest.raises(ValueError, match="selected_weight_decay_tau_ema"):
        build_forecast_fleet_tasks(
            joint_plan,
            phase="ladder",
            selected_learning_rate=0.001,
            run_negative_control=False,
        )
    joint_target = build_forecast_fleet_tasks(
        joint_plan,
        phase="ladder",
        selected_learning_rate=0.001,
        selected_weight_decay_tau_ema=0.1,
        run_negative_control=False,
    )
    assert {task.weight_decay_tau_ema for task in joint_target} == {0.1}


def test_jiang_adamw_plan_jointly_tunes_eta_and_tau_ema(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch)
    config = _jiang_config(tmp_path, manifest_path)
    config["optimizer"].update(
        name="adamw",
        weight_decay_tau_ema_grid=[0.05, 0.1, 0.2],
    )
    plan = compile_real_text_scaling_plan(config)
    assert plan["architecture_contract"]["parameterization"] == (
        "jiang_completep_adamw"
    )
    assert plan["weight_decay_tau_ema_grid"] == [0.05, 0.1, 0.2]
    assert plan["architecture_contract"]["theory"]["optimizer"] == "adamw"
    assert plan["tuning_trials"] == 9
    tasks = build_forecast_fleet_tasks(plan, phase="tune")
    assert len(tasks) == 9
    assert len({task.task_id for task in tasks}) == 9


def test_joint_eta_tau_fleet_records_are_distinct_and_select_both_coordinates(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch)
    config = _jiang_config(tmp_path, manifest_path)
    config["optimizer"].update(
        name="adamw",
        weight_decay_tau_ema_grid=[0.05, 0.1, 0.2],
    )
    config["run_negative_control"] = False
    tune_roots = [tmp_path / "joint-tune-0", tmp_path / "joint-tune-1"]
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
    assert len([path for cache in tune_caches for path in cache.glob("*.json")]) == 9
    selection = select_forecast_fleet_learning_rate(config, tune_caches)
    assert selection["selected_learning_rate"] in config["optimizer"][
        "learning_rates"
    ]
    assert selection["selected_weight_decay_tau_ema"] in config["optimizer"][
        "weight_decay_tau_ema_grid"
    ]
    assert "learning_rate_optimum_is_interior" in selection
    assert "weight_decay_optimum_is_interior" in selection


def test_extension_profile_binds_one_seed_and_frozen_parent_contract(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch)
    config = _jiang_config(tmp_path, manifest_path)
    provisional = compile_real_text_scaling_plan(config)
    target = provisional["scales"][-1]
    config["run_profile"] = "extension"
    config["extension_contract"] = {
        "parent_plan_fingerprint": "a" * 64,
        "parent_dataset_fingerprint": "b" * 64,
        "parent_aggregate_sha256": "c" * 64,
        "selected_learning_rate": 0.001,
        "target_scale": target["name"],
        "target_seed": 3,
        "expected_target_parameters": target["parameters"],
    }
    plan = compile_real_text_scaling_plan(config)
    assert plan["run_profile"] == "extension"
    assert plan["extension_contract"]["selected_learning_rate"] == 0.001
    assert plan["scales"][-1]["heldout"] is True
    assert plan["planned_grid_trials"] == 1
    tasks = build_forecast_fleet_tasks(
        plan,
        phase="ladder",
        selected_learning_rate=0.001,
        run_negative_control=False,
    )
    assert len(tasks) == 1
    assert tasks[0].scale_name == target["name"]
    assert tasks[0].seed == 3
    with pytest.raises(ValueError, match="skip tuning"):
        build_forecast_fleet_tasks(plan, phase="tune")

    endpoint_only = json.loads(json.dumps(config))
    endpoint_only["ladder"]["target_forecasts"] = []
    endpoint_plan = compile_real_text_scaling_plan(endpoint_only)
    assert endpoint_plan["target_forecasts"] == []
    assert endpoint_plan["planned_grid_trials"] == 1

    smoke = json.loads(json.dumps(endpoint_only))
    smoke["run_profile"] = "smoke"
    smoke.pop("extension_contract")
    smoke_plan = compile_real_text_scaling_plan(smoke)
    assert smoke_plan["target_forecasts"] == []

    invalid = json.loads(json.dumps(config))
    invalid["seeds"] = [3, 5]
    with pytest.raises(ValueError, match="exactly one seed"):
        compile_real_text_scaling_plan(invalid)

    invalid = json.loads(json.dumps(config))
    invalid["run_negative_control"] = True
    with pytest.raises(ValueError, match="refuses a wrong-LR control"):
        compile_real_text_scaling_plan(invalid)


def test_jiang_500m_rho32_geometry_is_exact(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch, vocab_size=32_768)
    config = json.loads(
        Path("configs/autoscaler/jiang_mistral_10tpp_calibration_200m.json")
        .read_text(encoding="utf-8")
    )
    config["dataset"]["token_stream_manifest_path"] = str(manifest_path)
    config["dataset"]["tokenizer"] = "forecast_test"
    config["run_profile"] = "smoke"
    config["ladder"]["target_parameters"][-1] = 498_723_456
    config["ladder"]["depths"][-1] = 8
    config["ladder"]["target_forecasts"] = []
    config["ladder"]["heldout_scale_count"] = 1
    config["ladder"]["require_gate_eligible_plan"] = False
    config["runtime"].update(
        distributed="ddp",
        num_processes=8,
        gradient_accumulation_steps=1,
    )
    config["validation_interval_steps"] = 2_378
    plan = compile_real_text_scaling_plan(config)
    target = plan["scales"][-1]
    assert target["parameters"] == 498_723_456
    assert target["non_embedding_parameters"] == 428_436_096
    assert (target["depth"], target["width"], target["hidden_width"]) == (
        8,
        2_112,
        8_448,
    )
    assert target["num_heads"] == 33
    assert target["rho_lm_over_d"] == 32.0
    assert target["tokens_per_parameter"] == pytest.approx(10.0, abs=0.001)


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


def test_forecast_retains_full_update_aligned_horizon_checkpoints(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch)
    config = _jiang_config(tmp_path, manifest_path)
    config["runtime"]["retained_checkpoint_tokens_per_parameter"] = [0.01, 0.02]
    plan = compile_real_text_scaling_plan(config)
    contract = plan["retained_checkpoint_contract"]
    assert contract["state"] == "model_optimizer_and_sampling_generator"
    scale = plan["scales"][0]
    rows = contract["scales"][scale["name"]]
    assert [row["requested_tokens_per_parameter"] for row in rows] == [
        0.01,
        0.02,
    ]
    assert rows[-1]["optimizer_step"] == scale["optimizer_steps"]

    corpus = TokenizedTextCorpus(
        forecast_campaigns.forecast_tokenized_text_spec(config),
        context_length=2,
        vocab_size=16,
    )
    cache = tmp_path / "retained-trials"
    cache.mkdir()
    record = forecast_campaigns._run_trial(
        config=config,
        plan=plan,
        scale=scale,
        corpus=corpus,
        runtime=PretrainingRuntimeSpec.from_dict(config["runtime"]),
        context=DistributedContext(0, 1, 0, "cpu"),
        eta=0.001,
        seed=3,
        optimizer_mode="theory",
        cache_directory=cache,
    )
    retained = record.metadata["retained_checkpoints"]
    assert len(retained) == 2
    assert {
        int(row["step"]) for row in record.validation_checkpoints
    }.issuperset({row["optimizer_step"] for row in retained})
    assert all(
        Path(row["base_path"]).with_suffix(".pt").is_file()
        for row in retained
    )
    assert not any(cache.glob("*.resume.pt"))

    payload = torch.load(
        Path(retained[-1]["base_path"]).with_suffix(".pt"),
        map_location="cpu",
        weights_only=False,
    )
    assert payload["step"] == scale["optimizer_steps"]
    assert payload["extra"]["tokens_seen"] == scale["presented_tokens"]
    assert "model_state_dict" in payload
    assert "optimizer_state_dict" in payload


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


def test_adamw_tau_ema_100m_presets_compile_exact_joint_grids(monkeypatch) -> None:
    config_root = Path(__file__).parents[1] / "configs" / "autoscaler"
    identity = {
        "format": "sharded_uint32_le_v1",
        "fingerprint": "1b854ee220230e0421acd8312d313a72d396de2234474ec20f63ba1ce4f1d703",
        "content_fingerprint": "b" * 64,
        "tokenizer_id": "mistral_7b_v03",
        "tokenizer_fingerprint": "d52f662783555cbf11f6a0cd8af35016652cda033389db471813c7d30f6958c5",
        "vocab_size": 32_768,
        "packing": {"contract": "document_eos_concatenation_v1"},
        "training_tokens": 3_080_501_458,
        "validation_tokens": 129_177_154,
    }
    monkeypatch.setattr(forecast_campaigns, "token_stream_identity", lambda _path: identity)

    jiang = json.loads(
        (config_root / "jiang_mistral_100m_adamw_tau_ema.json").read_text()
    )
    jiang_plan = compile_real_text_scaling_plan(jiang)
    assert jiang_plan["architecture_contract"]["parameterization"] == (
        "jiang_completep_adamw"
    )
    assert jiang_plan["optimizer_contract"]["name"] == "adamw"
    assert jiang_plan["tuning_trials"] == 120
    assert jiang_plan["scale_trials"] == 21
    assert jiang_plan["negative_control_trials"] == 0
    assert jiang_plan["planned_grid_trials"] == 141

    completep = json.loads(
        (config_root / "completep_mistral_100m_adamw_tau_ema.json").read_text()
    )
    completep_plan = compile_real_text_scaling_plan(completep)
    assert completep_plan["tuning_trials"] == 120
    assert completep_plan["scale_trials"] == 3
    assert completep_plan["planned_grid_trials"] == 123
    assert completep_plan["weight_decay_tau_ema_grid"] == [
        0.035175,
        0.07035,
        0.1407,
        0.2814,
        0.5628,
    ]


def test_jiang_rho32_preset_has_exact_reference_and_l8_endpoint(monkeypatch) -> None:
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "autoscaler"
        / "jiang_mistral_100m_rho32_adamw_tau_ema.json"
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
            "training_tokens": 3_080_501_458,
            "validation_tokens": 129_177_154,
        },
    )

    plan = compile_real_text_scaling_plan(config)

    assert plan["architecture_contract"]["rho_lm_over_d"] == pytest.approx(32.0)
    assert plan["architecture_contract"]["parameterization"] == (
        "jiang_completep_adamw"
    )
    assert plan["weight_decay_tau_ema_grid"] == [
        0.035175,
        0.07035,
        0.1407,
        0.2814,
        0.5628,
    ]
    assert [row["parameters"] for row in plan["scales"]] == [
        2_428_160,
        5_446_144,
        9_352_320,
        18_864_000,
        25_789_440,
        82_263_552,
        106_984_192,
    ]
    assert all(row["rho_lm_over_d"] == pytest.approx(32.0) for row in plan["scales"])
    assert all(row["rho_relative_error"] == pytest.approx(0.0) for row in plan["scales"])
    assert plan["scales"][0]["depth"] == 2
    assert plan["scales"][0]["width"] == 64
    assert plan["scales"][0]["hidden_width"] == 1_024
    assert plan["scales"][-1]["depth"] == 8
    assert plan["scales"][-1]["width"] == 896
    assert plan["scales"][-1]["hidden_width"] == 3_584
    assert plan["scales"][-1]["hidden_width"] / plan["scales"][-1]["width"] == 4
    assert plan["scales"][-1]["heldout"] is True
    assert plan["fit_parameter_span"] >= plan["minimum_parameter_span"]
    assert plan["planned_grid_trials"] == 141

    expanded = json.loads(
        config_path.with_name(
            "jiang_mistral_100m_rho32_adamw_tau_ema_expanded.json"
        ).read_text(encoding="utf-8")
    )
    expanded_plan = compile_real_text_scaling_plan(expanded)
    assert expanded_plan["scales"] == plan["scales"]
    assert expanded_plan["weight_decay_tau_ema_grid"] == [
        1.1256,
        2.2512,
        4.5024,
        9.0048,
    ]
    assert expanded_plan["tuning_trials"] == 96


def test_fixed_budget_presets_freeze_budget_and_parameter_axes(monkeypatch) -> None:
    config_root = Path(__file__).parents[1] / "configs" / "autoscaler"
    identity = {
        "format": "sharded_uint32_le_v1",
        "fingerprint": "1" * 64,
        "content_fingerprint": "2" * 64,
        "tokenizer_id": "mistral_7b_v03",
        "tokenizer_fingerprint": "3" * 64,
        "vocab_size": 32_768,
        "packing": {"contract": "document_eos_concatenation_v1"},
        "training_tokens": 6_200_000_000,
        "validation_tokens": 100_000_000,
    }
    monkeypatch.setattr(forecast_campaigns, "token_stream_identity", lambda _path: identity)

    jiang = json.loads(
        (config_root / "jiang_mistral_fixed_budget_100m_rho32_adamw.json").read_text()
    )
    completep = json.loads(
        (config_root / "completep_mistral_fixed_budget_100m_adamw.json").read_text()
    )
    jiang_plan = compile_real_text_scaling_plan(jiang)
    completep_plan = compile_real_text_scaling_plan(completep)
    jiang_without_refinement = deepcopy(jiang)
    jiang_without_refinement["optimizer"].pop("learning_rate_refinement")
    inherited_jiang_plan = compile_real_text_scaling_plan(
        jiang_without_refinement
    )
    assert jiang_plan["learning_rate_refinement"][
        "inherited_reference_plan_fingerprint"
    ] == inherited_jiang_plan["fingerprint"]
    reference = jiang_plan["scales"][
        jiang_plan["architecture_contract"]["reference_scale_index"]
    ]
    runtime = PretrainingRuntimeSpec.from_dict(jiang["runtime"])
    inherited_identity = forecast_trial_cache_identity(
        config=jiang_without_refinement,
        plan=inherited_jiang_plan,
        scale=reference,
        dataset_fingerprint=identity["fingerprint"],
        runtime=runtime,
        eta=0.03,
        seed=11,
        optimizer_mode="theory",
    )
    refined_identity = forecast_trial_cache_identity(
        config=jiang,
        plan=jiang_plan,
        scale=reference,
        dataset_fingerprint=identity["fingerprint"],
        runtime=runtime,
        eta=0.03,
        seed=11,
        optimizer_mode="theory",
    )
    assert refined_identity == inherited_identity

    for plan in (jiang_plan, completep_plan):
        assert plan["run_profile"] == "fixed_budget_scan"
        assert plan["fit_parameter_axis"] == "non_embedding_parameters"
        assert plan["measurement_contract"]["validation_seed"] == 424242
        assert plan["measurement_contract"][
            "validation_windows_are_identical_across_trials"
        ] is True
        assert plan["fixed_budget_contract"] == {
            "batch_examples": 512,
            "batch_tokens": 262144,
            "optimizer_steps": 1144,
            "presented_tokens": 299892736,
            "identical_at_every_scale": True,
            "batch_learning_rate_scaling": "none; eta tuned at this exact batch",
        }
        assert {row["optimizer_steps"] for row in plan["scales"]} == {1144}
        assert {row["presented_tokens"] for row in plan["scales"]} == {299892736}
        assert all(
            row["parameters"] > row["non_embedding_parameters"] > 0
            for row in plan["scales"]
        )
        assert plan["architecture_contract"][
            "layer_norm_numerical_epsilon"
        ] == pytest.approx(1e-5)

    assert jiang_plan["architecture_contract"]["reference_scale_index"] == 3
    assert jiang_plan["architecture_contract"]["unembedding_forward_scale"] == (
        "(D/D0)^(-1)"
    )
    assert all(row["rho_lm_over_d"] == 32.0 for row in jiang_plan["scales"])
    assert jiang_plan["learning_rates"] == [
        0.003,
        0.01,
        0.015,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.1,
        0.18,
        0.3,
    ]
    assert jiang_plan["tuning_task_learning_rates"] == [
        0.003,
        0.01,
        0.03,
        0.06,
        0.1,
        0.18,
        0.3,
        0.015,
        0.02,
        0.04,
        0.05,
    ]
    assert jiang_plan["tuning_trials"] == 25
    assert jiang_plan["scale_trials"] == 21
    assert jiang_plan["planned_grid_trials"] == 46
    jiang_tuning = build_forecast_fleet_tasks(jiang_plan, phase="tune")
    assert len(jiang_tuning) == 25
    assert [task.ordinal for task in jiang_tuning] == list(range(25))
    assert all(task.seed in {11, 29, 47} for task in jiang_tuning[:21])
    assert [task.eta for task in jiang_tuning[21:]] == [0.015, 0.02, 0.04, 0.05]
    assert {task.seed for task in jiang_tuning[21:]} == {11}
    inherited_ladder = build_forecast_fleet_tasks(
        jiang_plan,
        phase="ladder",
        selected_learning_rate=0.03,
        run_negative_control=False,
    )
    assert len(inherited_ladder) == 21
    refined_ladder = build_forecast_fleet_tasks(
        jiang_plan,
        phase="ladder",
        selected_learning_rate=0.02,
        run_negative_control=False,
    )
    assert len(refined_ladder) == 23
    assert {
        (task.scale_name, task.seed)
        for task in refined_ladder
        if task.scale_name == reference["name"]
    } == {(reference["name"], 29), (reference["name"], 47)}

    assert completep_plan["optimizer_contract"][
        "include_zero_weight_decay_control"
    ] is True
    assert completep_plan["architecture_contract"]["attention_scale"] == "QK^T/N"
    assert completep_plan["tuning_trials"] == 126
    assert completep_plan["scale_trials"] == 21
    assert completep_plan["planned_grid_trials"] == 147
    tuning = build_forecast_fleet_tasks(completep_plan, phase="tune")
    assert sum(task.weight_decay_tau_ema is None for task in tuning) == 21
    zero_decay_ladder = build_forecast_fleet_tasks(
        completep_plan,
        phase="ladder",
        selected_learning_rate=0.01,
        selected_weight_decay_tau_ema=None,
        run_negative_control=False,
    )
    assert len(zero_decay_ladder) == 21
    assert all(task.weight_decay_tau_ema is None for task in zero_decay_ladder)


def test_completep_baseline_preparation_requires_verified_jiang_adamw(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    template = root / "configs" / "autoscaler" / (
        "completep_mistral_100m_adamw_tau_ema.json"
    )
    selected_tau = 0.1407
    target = {
        "name": "S7",
        "parameters": 99_709_568,
        "seed_losses": [3.7, 3.6, 3.65],
        "mean_validation_loss": 3.65,
    }
    result = {
        "status": "completed",
        "plan_fingerprint": "a" * 64,
        "architecture_contract": {"parameterization": "jiang_completep_adamw"},
        "reference_tuning": {
            "learning_rate_optimum_is_interior": True,
            "weight_decay_optimum_is_interior": True,
            "selected_weight_decay_tau_ema": selected_tau,
        },
        "dataset": {
            "fingerprint": "b" * 64,
            "tokenizer_fingerprint": "c" * 64,
        },
        "scales": [target],
        "records": [
            {
                "optimizer": {"name": "adamw"},
                "metadata": {
                    "scale": {"name": "S7"},
                    "optimizer_mode": "theory",
                    "weight_decay_tau_ema": selected_tau,
                    "optimizer_group_audit": {
                        "theory": {"optimizer": "adamw"}
                    },
                },
            }
            for _ in range(3)
        ],
    }
    result_path = tmp_path / "jiang-result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    output = tmp_path / "prepared.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "prepare_completep_comparison_from_jiang.py"),
            str(template),
            str(result_path),
            "--output",
            str(output),
        ],
        check=True,
    )
    prepared = json.loads(output.read_text(encoding="utf-8"))
    baseline = prepared["comparison_contract"]
    assert baseline["baseline_plan_fingerprint"] == "a" * 64
    assert baseline["baseline_parameters"] == 99_709_568
    assert baseline["baseline_mean_validation_loss"] == pytest.approx(3.65)


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
        "jiang_final_norm",
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


def test_lr_refinement_selects_every_rate_on_the_shared_seed(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _stream(tmp_path, monkeypatch)
    config = _jiang_config(tmp_path, manifest_path)
    config["run_profile"] = "fixed_budget_scan"
    config["seeds"] = [3, 5, 7]
    config["ladder"].pop("tokens_per_parameter")
    config["ladder"].update(
        optimizer_steps=2,
        fit_parameter_axis="non_embedding_parameters",
        target_forecasts=[],
    )
    config["optimizer"]["learning_rate_refinement"] = {
        "learning_rates": [0.002],
        "seeds": [3],
        "exploratory_single_seed": True,
    }
    tune_root = tmp_path / "matched-seed-tune"
    run_forecast_fleet_shard(
        config,
        phase="tune",
        shard_index=0,
        shard_count=1,
        output_directory=tune_root,
        device="cpu",
    )
    for path in (tune_root / "trials").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        eta = float(payload["optimizer"]["learning_rate"])
        seed = int(payload["seed"])
        if eta == 0.001:
            loss = 1.0 if seed == 3 else 10.0
        elif eta == 0.002:
            loss = 2.0
        else:
            loss = 5.0
        payload["final_validation_loss"] = loss
        path.write_text(json.dumps(payload), encoding="utf-8")

    selection = select_forecast_fleet_learning_rate(
        config, [tune_root / "trials"]
    )
    assert selection["selected_learning_rate"] == 0.001
    assert selection["selection_mode"] == (
        "matched_single_seed_across_all_learning_rates"
    )
    assert selection["selected_seed_count"] == 1
    selected = next(
        row
        for row in selection["grid"]
        if row["learning_rate"] == 0.001
    )
    assert selected["mean_validation_loss"] == pytest.approx(7.0)
    assert selected["selection_mean_validation_loss"] == pytest.approx(1.0)


def test_fleet_refuses_boundary_reference_optimum_before_ladder(
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
    for cache in tune_caches:
        for path in cache.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["final_validation_loss"] = -float(
                payload["optimizer"]["learning_rate"]
            )
            path.write_text(json.dumps(payload), encoding="utf-8")

    selection = select_forecast_fleet_learning_rate(config, tune_caches)
    assert selection["selected_learning_rate"] == max(
        config["optimizer"]["learning_rates"]
    )
    assert selection["optimum_is_interior"] is False
    with pytest.raises(ValueError, match="grid boundary"):
        select_forecast_fleet_learning_rate(
            config, tune_caches, require_interior=True
        )
