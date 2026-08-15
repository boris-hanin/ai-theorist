import math

import pytest

from ai_theorist.autoscaler.training import TrialResult
from ai_theorist.autoscaler.tuning import (
    CHIZAT_MEAN_FIELD,
    MOE_TABLE1_ADAM,
    NUGPT_MID_ALIGNMENT,
    STANDARD_RESIDUAL_MLP,
    adaptive_tune,
    fixed_eta_noninferiority,
    fixed_eta_transfer_diagnostics,
    normalized_eta_from_raw_learning_rate,
    optimizer_group_learning_rates_from_normalized_eta,
    paired_mean_and_sem,
    raw_learning_rate_from_normalized_eta,
    transfer_rule_name,
)


def fake_trial(rate, seed, loss):
    return TrialResult(
        scale="S1",
        width=8,
        repeats=1,
        seed=seed,
        normalized_learning_rate=rate,
        optimizer="sgd",
        parameter_count=100,
        steps_completed=10,
        final_validation_loss=loss,
        train_loss_trace=[],
        diverged=not math.isfinite(loss),
        device="cpu",
    )


def test_adaptive_tuning_expands_an_edge_until_optimum_is_interior():
    def run(rate, seed):
        loss = 1.0 + (math.log10(rate) + 2.0) ** 2 + seed * 1e-4
        return fake_trial(rate, seed, loss)

    result, trials = adaptive_tune(
        [0.1, 1.0, 10.0],
        [1, 2, 3],
        run,
        max_expansion_rounds=2,
        expansion_factor=10.0,
    )
    assert result.selected_normalized_learning_rate == pytest.approx(0.01)
    assert result.optimum_is_interior
    assert result.expansion_rounds == 2
    assert len(trials) == 15


def test_divergent_candidate_is_not_selected():
    def run(rate, seed):
        return fake_trial(rate, seed, float("inf") if rate == 1.0 else 1.0 + rate)

    result, _ = adaptive_tune(
        [0.01, 0.1, 1.0], [1, 2], run, max_expansion_rounds=0, expansion_factor=10.0
    )
    assert result.selected_normalized_learning_rate == 0.01


def test_one_sem_rule_prefers_conservative_tied_learning_rate():
    losses = {
        0.003: [1.3, 1.3, 1.3, 1.3],
        0.01: [1.0, 1.0, 1.0, 1.0],
        0.03: [0.94, 1.02, 0.94, 1.02],
        0.1: [1.4, 1.4, 1.4, 1.4],
    }

    def run(rate, seed):
        return fake_trial(rate, seed, losses[rate][seed - 1])

    result, _ = adaptive_tune(
        list(losses), [1, 2, 3, 4], run, max_expansion_rounds=0, expansion_factor=3.0
    )
    assert result.numerical_best_normalized_learning_rate == 0.03
    assert result.selected_normalized_learning_rate == 0.01
    assert result.flat_minimum


def test_one_sem_selection_does_not_hide_boundary_numerical_optimum():
    losses = {
        0.001: [1.5, 1.5, 1.5, 1.5],
        0.01: [1.0, 1.0, 1.0, 1.0],
        0.1: [0.7, 1.1, 0.7, 1.1],
    }

    def run(rate, seed):
        return fake_trial(rate, seed, losses[rate][seed - 1])

    result, _ = adaptive_tune(
        list(losses), [1, 2, 3, 4], run, max_expansion_rounds=0, expansion_factor=10.0
    )
    assert result.numerical_best_normalized_learning_rate == 0.1
    assert result.selected_normalized_learning_rate == 0.01
    assert not result.optimum_is_interior


def test_paired_sem_uses_only_common_seed_differences():
    average, sem = paired_mean_and_sem({1: 1.2, 2: 1.1, 3: 5.0}, {1: 1.0, 2: 1.0, 4: 9.0})
    assert average == pytest.approx(0.15)
    assert sem == pytest.approx(0.05)


def test_paired_comparison_requires_replication():
    with pytest.raises(ValueError, match="two shared seeds"):
        paired_mean_and_sem({1: 1.0}, {1: 1.0})


def test_optimizer_specific_learning_rate_transfer():
    assert transfer_rule_name("adam") == "raw_lr_equals_normalized_eta"
    assert transfer_rule_name("sgd") == "raw_lr_equals_normalized_eta_over_sqrt_width"


def test_normalized_eta_conversions_are_parameterization_specific_and_invertible():
    standard_raw = raw_learning_rate_from_normalized_eta(
        STANDARD_RESIDUAL_MLP, "sgd", 0.4, width=400, depth=8
    )
    assert standard_raw == pytest.approx(0.02)
    assert normalized_eta_from_raw_learning_rate(
        STANDARD_RESIDUAL_MLP, "sgd", standard_raw, width=400, depth=8
    ) == pytest.approx(0.4)

    chizat_raw = raw_learning_rate_from_normalized_eta(
        CHIZAT_MEAN_FIELD, "sgd", 3.0, width=256, depth=8, alpha=2.0
    )
    assert chizat_raw == pytest.approx(1536.0)
    assert normalized_eta_from_raw_learning_rate(
        CHIZAT_MEAN_FIELD, "sgd", chizat_raw, width=256, depth=8, alpha=2.0
    ) == pytest.approx(3.0)
    assert raw_learning_rate_from_normalized_eta(
        CHIZAT_MEAN_FIELD, "adam", 1e-3, width=1024, depth=16
    ) == pytest.approx(1e-3)

    moe_rates = optimizer_group_learning_rates_from_normalized_eta(
        MOE_TABLE1_ADAM,
        "adam",
        0.32,
        width=64,
        depth=8,
        expert_width=256,
    )
    assert moe_rates["moe_router"] == pytest.approx(0.005)
    assert moe_rates["moe_up"] == pytest.approx(0.005)
    assert moe_rates["moe_down"] == pytest.approx(0.00125)
    assert moe_rates["readout_weight"] == pytest.approx(0.005)

    nugpt_rates = optimizer_group_learning_rates_from_normalized_eta(
        NUGPT_MID_ALIGNMENT,
        "adam",
        0.003,
        width=512,
        depth=16,
        reference_width=256,
    )
    assert nugpt_rates == {
        "nugpt_input": pytest.approx(0.003 * 2 ** -0.5),
        "nugpt_hidden": pytest.approx(0.003 * 2 ** -0.75),
        "nugpt_output": pytest.approx(0.5 * 0.003 * 2 ** -0.75),
        "nugpt_rescalers": pytest.approx(0.003),
    }
    assert transfer_rule_name("adam", NUGPT_MID_ALIGNMENT) == (
        "nugpt_mid_alignment_group_rates_with_post_step_sphere_projection"
    )


def test_fixed_eta_diagnostics_accept_inverse_sqrt_finite_width_settling():
    widths = [64, 128, 256, 512]
    rows = []
    for width in widths:
        center = 0.01 + 0.08 / math.sqrt(width)
        rows.append({1: center - 1e-4, 2: center + 1e-4, 3: center})
    result = fixed_eta_transfer_diagnostics(widths, rows)
    assert result.accepted
    assert result.settling
    assert result.finite_size_exponent == pytest.approx(-0.5)
    assert result.finite_size_intercept == pytest.approx(0.01)
    assert result.finite_size_r_squared == pytest.approx(1.0)


def test_fixed_eta_diagnostics_reject_runaway_large_width_gap():
    result = fixed_eta_transfer_diagnostics(
        [64, 128, 256, 512],
        [
            {1: 1.00, 2: 1.00},
            {1: 1.01, 2: 1.01},
            {1: 1.03, 2: 1.03},
            {1: 1.10, 2: 1.10},
        ],
    )
    assert not result.accepted
    assert not result.settling


def test_fixed_eta_gate_ignores_aggressive_probe_headroom():
    # The fixed eta is better than the conservative eta, so it transfers. A
    # still more aggressive rate could be better yet without changing this
    # verdict; that belongs to the separate stability-edge report.
    result = fixed_eta_noninferiority(
        {1: 0.80, 2: 0.82, 3: 0.81},
        {1: 1.00, 2: 1.02, 3: 1.01},
    )
    assert result["accepted"]
    assert result["paired_loss_penalty"] < 0.0
