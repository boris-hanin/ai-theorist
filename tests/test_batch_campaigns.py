import pytest

from ai_theorist.autoscaler.batch_campaigns import (
    run_constant_tpp_campaign,
    run_quadratic_calibration,
    run_transformer_batch_census,
    run_transformer_batch_trial,
)
from ai_theorist.autoscaler.batch_scaling import OptimizerHyperparameters
from ai_theorist.autoscaler.campaign_jobs import compile_campaign_plan
from ai_theorist.autoscaler.schema import ArchitectureTemplate, DatasetSpec, ScaleLevel


def _architecture():
    return {
        "block_type": "normalized_transformer",
        "activation": "silu",
        "vocab_size": 8,
        "context_length": 4,
        "head_dimension": 2,
        "mlp_multiplier": 1,
        "reference_width": 4,
        "reference_depth": 1,
    }


def _dataset():
    return {
        "task_type": "synthetic_markov",
        "n_train": 16,
        "n_validation": 8,
        "noise_std": 0.03,
        "seed": 7,
    }


def test_noisy_quadratic_calibration_emits_all_three_estimators() -> None:
    result = run_quadratic_calibration(
        {
            "dimension": 8,
            "condition_number": 5,
            "noise_scale": 0.2,
            "maximum_steps": 30,
            "target_loss": 0.2,
            "batch_tokens": [2, 4, 8, 16],
            "seeds": [1],
            "continuation_tokens": 32,
            "optimizers": [
                {"name": "sgd", "learning_rates": [0.1], "momentum": 0.0}
            ],
        }
    )
    assert result["campaign"] == "paquette_noisy_quadratic_calibration"
    analysis = result["optimizer_analyses"]["sgd"]
    assert {"steps_to_target", "direct_checkpoint", "gradient_noise", "consensus"} <= set(
        analysis
    )
    assert len(result["records"]) >= 4
    assert len(analysis["transfer_rule_calibration"]["rules"]) == 2


def test_transformer_census_runs_sgd_and_adam_on_same_contract() -> None:
    result = run_transformer_batch_census(
        {
            "architecture": _architecture(),
            "dataset": _dataset(),
            "scales": [{"name": "S1", "width": 4, "repeats": 1}],
            "batch_examples": [1, 2, 4, 8],
            "total_tokens": 32,
            "continuation_tokens": 32,
            "checkpoint_tokens": 16,
            "target_validation_loss": 2.2,
            "validation_interval": 1,
            "gradient_noise_samples": 8,
            "seeds": [3],
            "optimizers": [
                {"name": "sgd", "learning_rates": [0.01]},
                {
                    "name": "adam",
                    "learning_rates": [0.001],
                    "beta1": 0.9,
                    "beta2": 0.99,
                },
            ],
        }
    )
    assert {row["optimizer"] for row in result["scale_optimizer_analyses"]} == {
        "sgd",
        "adam",
    }
    assert len(result["records"]) == 8
    assert all("optimizer_timescales" in record for record in result["records"])


def test_web_campaign_plans_report_initial_progress_totals() -> None:
    transformer = {
        "architecture": _architecture(),
        "dataset": _dataset(),
        "scales": [{"name": "S1", "width": 4, "repeats": 1}],
        "batch_examples": [1, 2, 4, 8],
        "total_tokens": 32,
        "seeds": [3, 5],
        "optimizers": [
            {"name": "sgd", "learning_rates": [0.01, 0.03]},
            {"name": "adam", "learning_rates": [0.001]},
        ],
    }
    assert compile_campaign_plan("transformer_census", transformer)[
        "planned_grid_trials"
    ] == 24
    constant_tpp = {
        "architecture": _architecture(),
        "dataset": _dataset(),
        "scales": [
            {"name": "S1", "width": 4, "repeats": 1},
            {"name": "S2", "width": 6, "repeats": 1},
            {"name": "S3", "width": 8, "repeats": 1},
        ],
        "optimizer": {"name": "adam", "learning_rates": [0.001]},
        "tokens_per_parameter": 0.1,
    }
    assert compile_campaign_plan("constant_tpp", constant_tpp)[
        "planned_grid_trials"
    ] == 5


def test_constant_tpp_campaign_holds_out_largest_scale() -> None:
    result = run_constant_tpp_campaign(
        {
            "architecture": _architecture(),
            "dataset": _dataset(),
            "scales": [
                {"name": "S1", "width": 4, "repeats": 1},
                {"name": "S2", "width": 6, "repeats": 1},
                {"name": "S3", "width": 8, "repeats": 1},
            ],
            "optimizer": {
                "name": "adam",
                "learning_rates": [0.001],
                "beta1": 0.9,
                "beta2": 0.99,
            },
            "tokens_per_parameter": 0.1,
            "base_batch_examples": 1,
            "batch_growth_exponent": 0.0,
            "validation_interval": 1,
            "seeds": [5],
            "transfer_rules": ["none", "complete_dp_joint"],
        }
    )
    assert result["heldout_scale"] == "S3"
    assert result["execution_order"][-1] == "heldout_oracle_for_regret_only"
    assert [row["scale"] for row in result["fit_scale_optima"]] == ["S1", "S2"]
    assert len(result["transfer_results"]) == 2
    assert all(row["evaluated"] for row in result["transfer_results"])
    assert result["geometry"][0]["realized_tpp"] == pytest.approx(
        result["geometry"][1]["realized_tpp"], rel=0.1
    )


def test_transformer_trial_cache_round_trips_checkpoint_state(tmp_path) -> None:
    kwargs = {
        "architecture": ArchitectureTemplate.from_dict(_architecture()),
        "dataset": DatasetSpec.from_dict(_dataset()),
        "scale": ScaleLevel("S1", 4, 1),
        "optimizer": OptimizerHyperparameters("adam", 0.001, beta2=0.99),
        "total_tokens": 16,
        "batch_examples": 1,
        "seed": 13,
        "validation_interval": 1,
        "cache_directory": tmp_path,
        "cache_state": True,
    }
    first, first_extra = run_transformer_batch_trial(**kwargs)
    second, second_extra = run_transformer_batch_trial(**kwargs)
    assert second == first
    assert second_extra["initial_validation_loss"] == first_extra["initial_validation_loss"]
    assert second_extra["state_dict"].keys() == first_extra["state_dict"].keys()
