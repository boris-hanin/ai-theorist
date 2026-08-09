import math

import pytest

from ai_theorist.autoscaler.training import TrialResult
from ai_theorist.autoscaler.tuning import (
    adaptive_tune,
    paired_mean_and_sem,
    transfer_learning_rate,
    transfer_rule_name,
)


def fake_trial(rate, seed, loss):
    return TrialResult(
        scale="S1",
        width=8,
        repeats=1,
        seed=seed,
        learning_rate=rate,
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
    assert result.selected_learning_rate == pytest.approx(0.01)
    assert result.optimum_is_interior
    assert result.expansion_rounds == 2
    assert len(trials) == 15


def test_divergent_candidate_is_not_selected():
    def run(rate, seed):
        return fake_trial(rate, seed, float("inf") if rate == 1.0 else 1.0 + rate)

    result, _ = adaptive_tune(
        [0.01, 0.1, 1.0], [1, 2], run, max_expansion_rounds=0, expansion_factor=10.0
    )
    assert result.selected_learning_rate == 0.01


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
    assert result.numerical_best_learning_rate == 0.03
    assert result.selected_learning_rate == 0.01
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
    assert result.numerical_best_learning_rate == 0.1
    assert result.selected_learning_rate == 0.01
    assert not result.optimum_is_interior


def test_paired_sem_uses_only_common_seed_differences():
    average, sem = paired_mean_and_sem({1: 1.2, 2: 1.1, 3: 5.0}, {1: 1.0, 2: 1.0, 4: 9.0})
    assert average == pytest.approx(0.15)
    assert sem == pytest.approx(0.05)


def test_paired_comparison_requires_replication():
    with pytest.raises(ValueError, match="two shared seeds"):
        paired_mean_and_sem({1: 1.0}, {1: 1.0})


def test_optimizer_specific_learning_rate_transfer():
    assert transfer_learning_rate("adam", 1e-3, 256, 1024) == pytest.approx(1e-3)
    assert transfer_learning_rate("sgd", 0.02, 256, 1024) == pytest.approx(0.01)
    assert transfer_rule_name("adam") == "constant_global_learning_rate"
    assert transfer_rule_name("sgd") == "inverse_sqrt_width_learning_rate"
