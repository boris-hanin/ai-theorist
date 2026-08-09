import copy
from pathlib import Path

import pytest
import torch

from ai_theorist.autoscaler.schema import StudySpec, default_study_spec
from ai_theorist.autoscaler.training import make_optimizer, train_trial


def tiny_spec(optimizer="adam", *, steps=8, microbatch_size=None):
    data = copy.deepcopy(default_study_spec(optimizer, quick=True).to_dict())
    data["dataset"] = {"n_train": 32, "n_validation": 24, "noise_std": 0.0, "seed": 7}
    data["horizon"] = {"steps": steps, "batch_size": 8, "microbatch_size": microbatch_size}
    data["seeds"] = [3, 5]
    data["validation"]["bootstrap_samples"] = 0
    return StudySpec.from_dict(data)


def tiny_moe_spec(*, steps=4, microbatch_size=None):
    data = copy.deepcopy(
        default_study_spec("adam", quick=True, block_type="pre_norm_moe").to_dict()
    )
    data["dataset"] = {"n_train": 32, "n_validation": 24, "noise_std": 0.0, "seed": 7}
    data["horizon"] = {"steps": steps, "batch_size": 8, "microbatch_size": microbatch_size}
    data["scales"] = [
        {
            "name": f"S{index + 1}",
            "width": width,
            "repeats": index + 1,
            "expert_width": 2 * width,
        }
        for index, width in enumerate((4, 6, 8, 10, 12))
    ]
    data["seeds"] = [3, 5]
    data["validation"]["bootstrap_samples"] = 0
    return StudySpec.from_dict(data)


def test_sgd_step_matches_closed_form():
    spec = tiny_spec("sgd")
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    parameter.grad = torch.tensor([0.5, -0.25])
    optimizer = make_optimizer(torch.nn.ParameterList([parameter]), spec, 0.1)
    optimizer.step()
    assert torch.allclose(parameter, torch.tensor([0.95, -1.975]))


def test_adam_first_step_matches_bias_corrected_formula():
    spec = tiny_spec("adam")
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    gradient = torch.tensor([0.5, -0.25])
    parameter.grad = gradient.clone()
    optimizer = make_optimizer(torch.nn.ParameterList([parameter]), spec, 0.01)
    optimizer.step()
    expected = torch.tensor([1.0, -2.0]) - 0.01 * gradient / (gradient.abs() + spec.optimizer.epsilon)
    assert torch.allclose(parameter, expected, atol=1e-7, rtol=1e-7)


def test_trial_is_deterministic_and_reports_final_loss():
    spec = tiny_spec("adam")
    scale = spec.scales[0]
    first = train_trial(spec, scale, 1e-3, 3)
    second = train_trial(spec, scale, 1e-3, 3)
    assert first.final_validation_loss == second.final_validation_loss
    assert first.train_loss_trace == second.train_loss_trace
    assert first.steps_completed == spec.horizon.steps
    assert not first.diverged


def test_checkpoint_resume_matches_uninterrupted(tmp_path: Path):
    spec = tiny_spec("adam", steps=10)
    scale = spec.scales[0]
    uninterrupted = train_trial(spec, scale, 1e-3, 3)
    checkpoint = tmp_path / "trial.pt"
    partial = train_trial(
        spec,
        scale,
        1e-3,
        3,
        checkpoint_path=checkpoint,
        checkpoint_every=2,
        stop_after_steps=4,
    )
    assert partial.steps_completed == 4
    resumed = train_trial(spec, scale, 1e-3, 3, checkpoint_path=checkpoint, checkpoint_every=2)
    assert resumed.steps_completed == uninterrupted.steps_completed
    assert resumed.final_validation_loss == uninterrupted.final_validation_loss
    assert resumed.train_loss_trace == uninterrupted.train_loss_trace


def test_gradient_accumulation_matches_full_batch():
    full = tiny_spec("sgd", steps=6, microbatch_size=None)
    accumulated = tiny_spec("sgd", steps=6, microbatch_size=2)
    full_result = train_trial(full, full.scales[0], 1e-2, 3)
    accumulated_result = train_trial(accumulated, accumulated.scales[0], 1e-2, 3)
    assert accumulated_result.final_validation_loss == pytest.approx(
        full_result.final_validation_loss, rel=1e-6, abs=1e-7
    )


def test_moe_trial_uses_group_rates_and_reports_loads():
    spec = tiny_moe_spec()
    scale = spec.scales[0]
    result = train_trial(spec, scale, 0.04, 3, raw_learning_rate=0.04)
    assert result.raw_learning_rates == {
        "adapters_and_norms": pytest.approx(0.04),
        "readout_weight": pytest.approx(0.01),
        "readout_bias": pytest.approx(0.04),
        "moe_router": pytest.approx(0.01),
        "moe_up": pytest.approx(0.01),
        "moe_down": pytest.approx(0.005),
    }
    assert result.routing_loads is not None
    assert len(result.routing_loads) == scale.repeats
    assert all(
        sum(load) == pytest.approx(spec.architecture.active_experts)
        for load in result.routing_loads
    )
    assert result.max_routing_load_imbalance is not None


def test_moe_microbatching_preserves_balance_update_semantics():
    full = tiny_moe_spec(steps=3, microbatch_size=None)
    accumulated = tiny_moe_spec(steps=3, microbatch_size=2)
    full_result = train_trial(full, full.scales[0], 0.02, 3, raw_learning_rate=0.02)
    accumulated_result = train_trial(
        accumulated, accumulated.scales[0], 0.02, 3, raw_learning_rate=0.02
    )
    assert accumulated_result.final_validation_loss == pytest.approx(
        full_result.final_validation_loss, rel=1e-6, abs=1e-7
    )
