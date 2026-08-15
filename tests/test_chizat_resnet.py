import math

import pytest
import torch

from ai_theorist.autoscaler.chizat_resnet import Chizat2LPResNet, Chizat2LPShape


def make_model(*, L=2, M=3, D=5):
    torch.manual_seed(11)
    return Chizat2LPResNet(
        Chizat2LPShape(depth=L, hidden_width=M, embedding_dimension=D),
        input_dimension=2,
        output_dimension=1,
        dtype=torch.float64,
    )


def rates(model, *, rule="lmd", reference_shape=None):
    return {
        group["name"]: group["lr"]
        for group in model.optimizer_parameter_groups(
            eta_u=0.1,
            eta_v=0.3,
            rule=rule,
            reference_shape=reference_shape,
        )
    }


def test_forward_is_literal_equation_22():
    model = make_model(L=2, M=3, D=5)
    x = torch.tensor([[0.2, -0.7]], dtype=torch.float64)
    hidden = x @ model.input_map.T
    for layer in range(2):
        hidden = hidden + torch.tanh(hidden @ model.U[layer].T / 5.0) @ model.V[layer] / 6.0
    expected = hidden @ model.output_map / 5.0
    assert torch.allclose(model(x), expected)


def test_only_particles_are_trainable_and_contract_is_complete():
    model = make_model()
    assert {name for name, _ in model.named_parameters()} == {"U", "V"}
    assert {name for name, _ in model.named_buffers()} == {"input_map", "output_map"}
    audit = model.optimizer_contract_audit(eta_u=0.1, eta_v=0.3)
    assert audit["complete"] is True
    assert audit["disjoint"] is True
    assert [group["name"] for group in audit["groups"]] == ["particle_u", "particle_v"]


def test_critical_mlu_initialization_has_sqrt_d_entrywise_std():
    model = make_model(L=8, M=64, D=32)
    target = math.sqrt(32)
    assert model.U.detach().std().item() == pytest.approx(target, rel=0.03)
    assert model.V.detach().std().item() == pytest.approx(target, rel=0.03)


def test_equation_23_raw_rates_and_one_factor_controls():
    model = make_model(L=2, M=3, D=5)
    assert rates(model) == {
        "particle_u": pytest.approx(0.1 * 2 * 3 * 5),
        "particle_v": pytest.approx(0.3 * 2 * 3 * 5),
    }
    assert rates(model, rule="omit_l")["particle_u"] == pytest.approx(0.1 * 3 * 5)
    assert rates(model, rule="omit_m")["particle_u"] == pytest.approx(0.1 * 2 * 5)
    assert rates(model, rule="omit_d")["particle_u"] == pytest.approx(0.1 * 2 * 3)
    reference = Chizat2LPShape(depth=1, hidden_width=2, embedding_dimension=4)
    assert rates(model, rule="constant_raw", reference_shape=reference) == {
        "particle_u": pytest.approx(0.1 * 1 * 2 * 4),
        "particle_v": pytest.approx(0.3 * 1 * 2 * 4),
    }


def test_joint_limit_coordinate_is_lm_over_d():
    assert Chizat2LPShape(depth=4, hidden_width=16, embedding_dimension=32).rho == 2.0
