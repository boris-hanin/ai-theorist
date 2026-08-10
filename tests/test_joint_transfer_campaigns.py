import pytest

from ai_theorist.autoscaler.batch_scaling import OptimizerHyperparameters
from ai_theorist.autoscaler.campaign_jobs import compile_campaign_plan, run_campaign_job
from ai_theorist.autoscaler.joint_transfer_campaigns import (
    build_joint_optimizer,
    run_joint_transfer_campaign,
)


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
        "fit_presented_tokens": [16, 32, 64],
        "heldout_presented_tokens": 128,
        "fit_batch_examples": [1, 2, 4],
        "heldout_batch_examples": 8,
        "schedule": "linear_warmup_decay_to_zero",
        "validation_interval": 4,
        "seeds": [5],
        "minimum_seeds": 1,
        "minimum_fit_horizon_span": 4,
        "minimum_fit_batch_span": 4,
        "minimum_axis_fit_r_squared": 0,
        "bootstrap_samples": 0,
        "maximum_grid_expansion_rounds": 0,
    }


def test_joint_optimizer_rules_transform_every_declared_adam_coordinate() -> None:
    source = OptimizerHyperparameters(
        name="adam",
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.99,
        epsilon=1e-8,
    )
    fitted_sde = build_joint_optimizer(
        "horizon_fitted_x_adam_sde_batch",
        source,
        source_tokens=100,
        target_tokens=400,
        source_batch_tokens=8,
        target_batch_tokens=32,
        parameter_count=1000,
        horizon_exponent=0.25,
        batch_exponent=0.4,
    )
    target = fitted_sde["optimizer"]
    assert fitted_sde["valid"]
    assert target.learning_rate == pytest.approx(1e-3 * 2**0.5)
    assert target.epsilon == pytest.approx(5e-9)
    assert target.beta1 == pytest.approx(0.6)
    assert target.beta2 == pytest.approx(0.96)

    complete_dp = build_joint_optimizer(
        "complete_dp_joint",
        source,
        source_tokens=100,
        target_tokens=400,
        source_batch_tokens=8,
        target_batch_tokens=32,
        parameter_count=1000,
        horizon_exponent=0.25,
        batch_exponent=0.4,
    )["optimizer"]
    assert complete_dp.learning_rate == pytest.approx(source.learning_rate)
    assert complete_dp.beta1 == pytest.approx(source.beta1)
    assert complete_dp.beta2 == pytest.approx(source.beta2)


def test_joint_plan_has_axis_crosscheck_and_doubly_heldout_stages() -> None:
    plan = compile_campaign_plan("joint_horizon_batch", _config())
    assert plan["calibration_trials"] == 15
    assert plan["candidate_trials"] == 16
    assert plan["oracle_trials"] == 6
    assert plan["planned_grid_trials"] == 37
    assert plan["composition_crosscheck"] == {
        "presented_tokens": 64,
        "batch_examples": 4,
    }
    assert plan["execution_order"][-1] == "reveal_heldout_oracle_for_regret_only"


def test_joint_campaign_never_uses_heldout_corner_in_axis_fits() -> None:
    result = run_joint_transfer_campaign(_config())
    assert result["status"] == "completed"
    qualification = result["axis_fit_qualification"]
    assert [row["presented_tokens"] for row in qualification["horizon_optima"]] == [
        16,
        32,
        64,
    ]
    assert [row["batch_examples"] for row in qualification["batch_optima"]] == [
        1,
        2,
        4,
    ]
    assert result["composition_crosscheck"]["presented_tokens"] == 64
    assert result["composition_crosscheck"]["batch_examples"] == 4
    assert result["heldout_corner"]["presented_tokens"] == 128
    assert result["heldout_corner"]["batch_examples"] == 8
    assert {row["rule"] for row in result["heldout_corner"]["candidate_results"]} >= {
        "none",
        "horizon_fitted_only",
        "batch_fitted_only",
        "separable_fitted_peak",
        "complete_dp_joint",
    }
    evaluated = [
        row for row in result["heldout_corner"]["candidate_results"] if row["evaluated"]
    ]
    assert all(len(row["peak_parameter_group_contract"]) == 4 for row in evaluated)
    assert all(
        {group["name"] for group in row["peak_parameter_group_contract"]}
        == {"nugpt_input", "nugpt_hidden", "nugpt_output", "nugpt_rescalers"}
        for row in evaluated
    )
    one_third = next(
        row
        for row in result["composition_crosscheck"]["candidate_results"]
        if row["rule"] == "one_third_x_adam_sde_batch"
    )
    separable = next(
        row
        for row in result["composition_crosscheck"]["candidate_results"]
        if row["rule"] == "separable_fitted_peak"
    )
    assert one_third["prerequisite_refusal_reasons"] == []
    assert one_third["theory_assumption_status"] == "unverified_or_outside_subcritical_gate"
    assert separable["prerequisite_refusal_reasons"]
    assert all(
        record["metadata"]["gradient_clipping"] == "none"
        for record in result["records"]
    )


def test_joint_campaign_requires_mechanism_controls() -> None:
    config = _config()
    config["joint_rules"] = ["none", "separable_fitted_peak"]
    with pytest.raises(ValueError, match="mechanism controls"):
        run_joint_transfer_campaign(config)


def test_joint_campaign_runs_through_persistent_job_engine(tmp_path) -> None:
    result = run_campaign_job(
        "joint_horizon_batch",
        _config(),
        device="cpu",
        output_dir=tmp_path,
    )
    assert result["campaign"] == "joint_horizon_batch_transfer"
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "result.json").is_file()
    assert (tmp_path / "trials").is_dir()
