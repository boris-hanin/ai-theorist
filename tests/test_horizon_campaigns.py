import pytest

from ai_theorist.autoscaler.campaign_jobs import compile_campaign_plan, run_campaign_job
from ai_theorist.autoscaler.horizon_campaigns import run_horizon_transfer_campaign
from ai_theorist.autoscaler.lr_schedules import LearningRateSchedule


def _config():
    return {
        "architecture": {
            "block_type": "normalized_transformer",
            "activation": "silu",
            "vocab_size": 8,
            "context_length": 2,
            "head_dimension": 2,
            "mlp_multiplier": 1,
            "reference_width": 4,
            "reference_depth": 1,
        },
        "dataset": {
            "task_type": "synthetic_markov",
            "n_train": 8,
            "n_validation": 8,
            "noise_std": 0.03,
            "seed": 7,
            "markov_order": 1,
            "markov_states": 2,
        },
        "scale": {"name": "anchor", "width": 4, "repeats": 1},
        "optimizer": {
            "name": "adam",
            "learning_rates": [0.001, 0.01, 0.1],
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-16,
        },
        "presented_tokens": [8, 16, 32, 64],
        "batch_examples": 1,
        "schedules": ["cosine_to_10_percent"],
        "horizon_rules": ["none", "nugpt_one_third", "bjorck_032", "fitted_power"],
        "validation_interval": 4,
        "seeds": [5],
        "minimum_seeds": 1,
        "minimum_fit_horizon_span": 4,
        "bootstrap_samples": 0,
        "maximum_grid_expansion_rounds": 0,
    }


def test_normalized_schedule_shapes_have_exact_endpoints() -> None:
    cosine = LearningRateSchedule.from_payload("cosine_to_10_percent")
    assert cosine.multiplier(1, 11) == pytest.approx(1.0)
    assert cosine.multiplier(11, 11) == pytest.approx(0.1)

    linear = LearningRateSchedule.from_payload("linear_warmup_decay_to_zero")
    assert 0.0 < linear.multiplier(1, 20) < 1.0
    assert linear.multiplier(2, 20) == pytest.approx(1.0)
    assert linear.multiplier(20, 20) == pytest.approx(0.0)

    wsd = LearningRateSchedule.from_payload("wsd")
    assert wsd.multiplier(10, 100) == pytest.approx(1.0)
    assert wsd.multiplier(100, 100) == pytest.approx(0.0)


def test_horizon_plan_counts_fit_transfer_and_oracle_trials() -> None:
    plan = compile_campaign_plan("horizon_transfer", _config())
    assert plan["fit_trials"] == 9
    assert plan["frozen_transfer_trials"] == 4
    assert plan["heldout_oracle_trials"] == 3
    assert plan["planned_grid_trials"] == 16
    assert plan["execution_order"][-1] == "reveal_heldout_oracle_for_regret_only"


def test_horizon_campaign_freezes_rules_before_revealing_oracle() -> None:
    result = run_horizon_transfer_campaign(_config())
    assert result["status"] == "completed"
    assert result["heldout_horizon"] == 64
    assert result["fixed_coordinates"]["unique_tokens"] == 16
    assert result["geometry"][-1]["presented_to_unique_token_ratio"] == 4
    assert result["execution_order"][-1] == "reveal_heldout_oracle_for_regret_only"

    analysis = result["schedule_analyses"][0]
    assert [row["presented_tokens"] for row in analysis["fit_optima"]] == [8, 16, 32]
    assert {row["rule"] for row in analysis["frozen_rule_results"]} == {
        "none",
        "nugpt_one_third",
        "bjorck_032",
        "fitted_power",
    }
    assert all("relative_oracle_regret" in row for row in analysis["frozen_rule_results"])
    assert all(
        record["learning_rate_schedule"] == "cosine_to_0.1"
        for record in result["records"]
    )
    assert all(record["metadata"]["gradient_clipping"] == "none" for record in result["records"])


def test_horizon_campaign_refuses_non_adam_optimizer() -> None:
    config = _config()
    config["optimizer"] = {"name": "sgd", "learning_rates": [0.001, 0.01, 0.1]}
    with pytest.raises(ValueError, match="requires Adam"):
        run_horizon_transfer_campaign(config)


def test_horizon_campaign_runs_through_persistent_job_engine(tmp_path) -> None:
    result = run_campaign_job(
        "horizon_transfer",
        _config(),
        device="cpu",
        output_dir=tmp_path,
    )
    assert result["campaign"] == "horizon_transfer"
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "result.json").is_file()
    assert (tmp_path / "trials").is_dir()


def test_horizon_campaign_freezes_one_fingerprinted_real_text_sample(tmp_path) -> None:
    train_path = tmp_path / "train.txt"
    validation_path = tmp_path / "validation.txt"
    train_path.write_text("alpha beta gamma delta\n" * 32, encoding="utf-8")
    validation_path.write_text("held out epsilon zeta\n" * 16, encoding="utf-8")
    config = _config()
    config["architecture"] = {
        **config["architecture"],
        "vocab_size": 260,
        "context_length": 4,
    }
    config["dataset"] = {
        "task_type": "tokenized_text",
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "tokenizer": "byte_v1",
        "n_train": 8,
        "n_validation": 8,
        "seed": 7,
        "maximum_bytes": 1_000_000,
    }
    config["presented_tokens"] = [16, 32, 64, 128]
    config["cache_directory"] = str(tmp_path / "cache")

    plan = compile_campaign_plan("horizon_transfer", config)
    assert plan["data_mode"] == "frozen_real_text"
    assert plan["execution_order"][0] == "freeze_real_text_corpus_and_sampled_windows"

    result = run_horizon_transfer_campaign(config)
    fingerprint = result["dataset"]["fingerprint"]
    assert len(fingerprint) == 64
    assert result["dataset"]["sampled_unique_training_tokens"] == 32
    assert result["fixed_coordinates"]["unique_tokens"] == 32
    assert all(
        record["metadata"]["dataset"]["corpus_fingerprint"] == fingerprint
        for record in result["records"]
    )
    assert all(
        record["metadata"]["unique_training_tokens"] == 32
        for record in result["records"]
    )


def test_real_text_horizon_requires_byte_tokenizer_vocabulary(tmp_path) -> None:
    config = _config()
    config["dataset"] = {
        "task_type": "tokenized_text",
        "train_path": str(tmp_path / "train.txt"),
        "validation_path": str(tmp_path / "validation.txt"),
        "tokenizer": "byte_v1",
        "n_train": 8,
        "n_validation": 8,
        "seed": 7,
    }
    with pytest.raises(ValueError, match="vocab_size 260"):
        run_horizon_transfer_campaign(config)
