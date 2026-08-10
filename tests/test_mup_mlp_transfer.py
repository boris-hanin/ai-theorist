import pytest
import torch

from ai_theorist.autoscaler.lr_contract import audit_optimizer_groups
from mup_mlp_transfer import MuPResidualMLP, json_safe, run_trial, theory_for


def model(width=64, reference_width=32):
    return MuPResidualMLP(
        input_dimension=8,
        output_dimension=1,
        width=width,
        reference_width=reference_width,
        depth=2,
    )


def rates(net, optimizer, rule="mup"):
    return {
        group["name"]: group["lr"]
        for group in net.optimizer_parameter_groups(0.01, optimizer=optimizer, rule=rule)
    }


def test_mup_adam_and_sgd_have_different_group_scalings():
    net = model()
    assert rates(net, "adam") == {
        "mup_input_matrix": pytest.approx(0.01),
        "mup_hidden_matrices": pytest.approx(0.005),
        "mup_width_vectors": pytest.approx(0.01),
        "mup_output_weight": pytest.approx(0.01),
        "mup_output_bias": pytest.approx(0.01),
    }
    assert rates(net, "sgd") == {
        "mup_input_matrix": pytest.approx(0.02),
        "mup_hidden_matrices": pytest.approx(0.01),
        "mup_width_vectors": pytest.approx(0.02),
        "mup_output_weight": pytest.approx(0.02),
        "mup_output_bias": pytest.approx(0.01),
    }
    assert set(rates(net, "adam", "global_lr_control").values()) == {0.01}


@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
def test_mup_groups_cover_every_trainable_tensor_once(optimizer):
    net = model()
    groups = net.optimizer_parameter_groups(0.01, optimizer=optimizer)
    report = audit_optimizer_groups(net, groups, theory_for(optimizer))
    assert report["complete"] is True
    assert report["disjoint"] is True


def test_mup_readout_divides_features_by_width_multiplier():
    net = model(width=64, reference_width=32)
    with torch.no_grad():
        net.readout.weight.fill_(1.0)
        net.readout.bias.zero_()
    features = torch.ones(2, 64)
    assert torch.equal(net.readout(features / net.width_multiplier), torch.full((2, 1), 32.0))


def test_tiny_mup_trial_executes_and_records_group_rates():
    trial, audit = run_trial(
        optimizer_name="adam",
        rule="mup",
        width=16,
        reference_width=16,
        depth=1,
        eta=1e-3,
        steps=2,
        batch_size=4,
        seed=11,
        input_dimension=4,
        n_train=16,
        n_validation=8,
        dataset_seed=1729,
        device=torch.device("cpu"),
    )
    assert trial.diverged is False
    assert trial.raw_learning_rates["mup_hidden_matrices"] == pytest.approx(1e-3)
    assert audit["complete"] is True


def test_nonfinite_failed_control_diagnostics_are_strict_json_safe():
    assert json_safe({"slope": float("nan"), "rows": [1.0, float("inf")]}) == {
        "slope": None,
        "rows": [1.0, None],
    }
