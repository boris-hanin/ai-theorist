import copy
import math
from types import SimpleNamespace

import pytest
import torch

from chizat_lmd_transfer import FixedTaskChizatNet, Shape, group_learning_rates
from chizat_muon import (
    AuxAdamConfig,
    ChizatMuonAdam,
    MuonConfig,
    SingleDeviceMuon,
    chizat_muon_learning_rates,
    learning_rate_adjustment,
    muon_direction,
    validate_semantic_partition,
    zeropower_via_newtonschulz,
)
from chizat_muon_transfer import paired_largest_shape_control


def _net(D=8, M=12, L=2, seed=3):
    shape = Shape("S", L=L, M=M, D=D, dial=1.0)
    return FixedTaskChizatNet(
        shape,
        max_M=M,
        max_D=D,
        d0=5,
        seed=seed,
        device=torch.device("cpu"),
    )


def _polar(matrix):
    u, _, vh = torch.linalg.svd(matrix.double(), full_matrices=False)
    return u @ vh


@pytest.mark.parametrize("shape", [(7, 7), (5, 13), (13, 5)])
def test_newton_schulz_tracks_svd_polar_for_square_and_rectangular_matrices(shape):
    generator = torch.Generator().manual_seed(101 + shape[0])
    gradient = torch.randn(*shape, generator=generator, dtype=torch.float64)
    result = zeropower_via_newtonschulz(gradient)
    polar = _polar(gradient)
    cosine = torch.nn.functional.cosine_similarity(
        result.flatten(), polar.flatten(), dim=0
    )
    singular_values = torch.linalg.svdvals(result)
    assert cosine > 0.97
    assert singular_values.min() > 0.5
    assert singular_values.max() < 1.5


def test_newton_schulz_is_transpose_equivariant_and_zero_safe():
    gradient = torch.randn(5, 11, generator=torch.Generator().manual_seed(8))
    direct = zeropower_via_newtonschulz(gradient)
    transposed = zeropower_via_newtonschulz(gradient.T).T
    assert torch.equal(direct, transposed)
    assert torch.equal(zeropower_via_newtonschulz(torch.zeros_like(gradient)), torch.zeros_like(gradient))


def test_newton_schulz_is_finite_for_tiny_and_rank_deficient_gradients():
    left = torch.randn(9, 1, generator=torch.Generator().manual_seed(19))
    right = torch.randn(1, 6, generator=torch.Generator().manual_seed(20))
    for gradient in (left @ right, 1e-30 * (left @ right)):
        result = zeropower_via_newtonschulz(gradient)
        assert torch.isfinite(result).all()
        assert result.shape == gradient.shape


def test_rms_matching_removes_rectangular_shape_dependence():
    observed = []
    original = []
    for rows, columns in ((4, 64), (64, 4), (16, 16)):
        gradient = torch.randn(
            rows, columns, generator=torch.Generator().manual_seed(rows + columns)
        )
        direction = zeropower_via_newtonschulz(gradient)
        observed.append(
            float(direction.square().mean().sqrt())
            * learning_rate_adjustment("match_rms_adamw", gradient.shape)
        )
        original.append(
            float(direction.square().mean().sqrt())
            * learning_rate_adjustment("original", gradient.shape)
        )
    assert max(observed) - min(observed) < 0.05
    assert max(original) / min(original) > 2.0


def test_muon_one_step_matches_frozen_equations():
    parameter = torch.nn.Parameter(
        torch.tensor([[1.0, -2.0, 0.5], [0.3, -0.7, 1.4]], dtype=torch.float64)
    )
    gradient = torch.tensor(
        [[0.2, -0.3, 0.7], [0.8, -0.1, 0.4]], dtype=torch.float64
    )
    config = MuonConfig(momentum=0.8, nesterov=True, weight_decay=0.1)
    optimizer = SingleDeviceMuon([parameter], lr=0.03, config=config)
    parameter.grad = gradient.clone()
    before = parameter.detach().clone()
    buffer = torch.zeros_like(gradient)
    direction = muon_direction(gradient.clone(), buffer, config)
    effective_lr = 0.03 * learning_rate_adjustment(config.adjustment, parameter.shape)
    expected = before * (1.0 - 0.03 * config.weight_decay) - effective_lr * direction
    optimizer.step()
    assert torch.allclose(parameter, expected, atol=1e-12, rtol=1e-12)


def test_nesterov_mutation_changes_second_step_direction():
    first = torch.randn(4, 7, generator=torch.Generator().manual_seed(1))
    second = torch.randn(4, 7, generator=torch.Generator().manual_seed(2))
    buffer_a = torch.zeros_like(first)
    buffer_b = torch.zeros_like(first)
    with_nesterov = MuonConfig(momentum=0.8, nesterov=True)
    without_nesterov = MuonConfig(momentum=0.8, nesterov=False)
    muon_direction(first, buffer_a, with_nesterov)
    muon_direction(first, buffer_b, without_nesterov)
    direction_a = muon_direction(second, buffer_a, with_nesterov)
    direction_b = muon_direction(second, buffer_b, without_nesterov)
    assert not torch.allclose(direction_a, direction_b, atol=1e-4, rtol=1e-4)


def test_muon_rejects_non_matrix_parameters():
    with pytest.raises(ValueError, match="2D matrices"):
        SingleDeviceMuon([torch.nn.Parameter(torch.ones(5))])


def test_trained_boundaries_have_nested_initialization_and_explicit_roles():
    small = FixedTaskChizatNet(
        Shape("small", 2, 8, 8, 1.0),
        max_M=16,
        max_D=16,
        d0=5,
        seed=9,
        device=torch.device("cpu"),
    )
    large = FixedTaskChizatNet(
        Shape("large", 2, 16, 16, 2.0),
        max_M=16,
        max_D=16,
        d0=5,
        seed=9,
        device=torch.device("cpu"),
    )
    assert torch.equal(small.embed, large.embed[:, :8])
    assert torch.equal(small.unembed * 8, large.unembed[:8] * 16)
    assert set(small.parameter_groups()) == {"embed", "U", "W", "unembed"}
    assert len(small.params()) == 2 * small.shape.L + 2


def test_sgd_boundary_learning_rates_are_dimension_normalized():
    shape = Shape("S", L=4, M=32, D=16, dial=1.0)
    rates = group_learning_rates(
        "group_natural_LMD", shape, 0.25, reference_D=16
    )
    assert rates["embed"] == pytest.approx(4.0)
    assert rates["unembed"] == pytest.approx(0.25 / 16)
    frozen = group_learning_rates("freeze_embed", shape, 0.25, reference_D=16)
    assert frozen["embed"] == 0.0
    assert frozen["unembed"] == rates["unembed"]
    assert frozen["U"] == rates["U"]
    assert frozen["W"] == rates["W"]


def test_chizat_muon_transfer_coordinate_has_distinct_semantic_group_rates():
    rates = chizat_muon_learning_rates(
        "group_rms_D", L=8, M=128, D=32, eta=0.01
    )
    assert rates == pytest.approx(
        {
            "embed": 0.01,
            "U": 0.01,
            "W": math.sqrt(32) * 0.01,
            "unembed": 0.01 / 32,
        }
    )
    wrong = chizat_muon_learning_rates(
        "wrong_constant_unembed", L=8, M=128, D=32, eta=0.01
    )
    assert wrong["unembed"] == pytest.approx(0.01)
    assert wrong["U"] == rates["U"]
    assert wrong["W"] == rates["W"]


def test_semantic_routing_keeps_2d_boundaries_out_of_muon():
    net = _net()
    rates = chizat_muon_learning_rates(
        "group_rms_D", L=net.shape.L, M=net.shape.M, D=net.shape.D, eta=1e-3
    )
    optimizer = ChizatMuonAdam(net, rates)
    muon_ids = {
        id(parameter)
        for group in optimizer.muon.param_groups
        for parameter in group["params"]
    }
    auxiliary_ids = {
        id(parameter)
        for group in optimizer.auxiliary.param_groups
        for parameter in group["params"]
    }
    assert id(net.embed) in auxiliary_ids
    assert id(net.unembed) in auxiliary_ids
    assert id(net.embed) not in muon_ids
    assert id(net.unembed) not in muon_ids
    assert muon_ids == {id(parameter) for parameter in (*net.U, *net.W)}


def test_partition_audit_rejects_double_routing_and_missing_parameters():
    parameters = [torch.nn.Parameter(torch.ones(2, 2)) for _ in range(3)]
    with pytest.raises(ValueError, match="more than once"):
        validate_semantic_partition(parameters, parameters[:2], parameters[1:])
    with pytest.raises(ValueError, match="1 missing"):
        validate_semantic_partition(parameters, parameters[:1], parameters[1:2])


def _training_step(net, optimizer, X, y):
    for parameter in net.params():
        parameter.requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    loss = net.loss(X, y)
    loss.backward()
    optimizer.step()


def test_hybrid_optimizer_checkpoint_resume_matches_uninterrupted():
    shape = Shape("S", 2, 10, 8, 1.0)
    rates = chizat_muon_learning_rates("group_rms_D", L=2, M=10, D=8, eta=1e-3)
    uninterrupted = _net(D=8, M=10, L=2, seed=17)
    split = _net(D=8, M=10, L=2, seed=17)
    X = torch.randn(12, 5, generator=torch.Generator().manual_seed(31), dtype=torch.float64)
    y = torch.randn(12, generator=torch.Generator().manual_seed(32), dtype=torch.float64)
    uninterrupted_optimizer = ChizatMuonAdam(uninterrupted, rates)
    split_optimizer = ChizatMuonAdam(split, rates)
    for _ in range(4):
        _training_step(uninterrupted, uninterrupted_optimizer, X, y)
    for _ in range(2):
        _training_step(split, split_optimizer, X, y)
    checkpoint_parameters = [parameter.detach().clone() for parameter in split.params()]
    checkpoint_optimizer = copy.deepcopy(split_optimizer.state_dict())
    resumed = _net(D=8, M=10, L=2, seed=17)
    with torch.no_grad():
        for parameter, saved in zip(resumed.params(), checkpoint_parameters):
            parameter.copy_(saved)
    resumed_optimizer = ChizatMuonAdam(resumed, rates)
    resumed_optimizer.load_state_dict(checkpoint_optimizer)
    for _ in range(2):
        _training_step(resumed, resumed_optimizer, X, y)
    for expected, actual in zip(uninterrupted.params(), resumed.params()):
        assert torch.equal(expected, actual)


def test_hybrid_optimizer_gradient_accumulation_matches_full_batch():
    rates = chizat_muon_learning_rates("group_rms_D", L=2, M=10, D=8, eta=1e-3)
    full = _net(D=8, M=10, L=2, seed=23)
    accumulated = _net(D=8, M=10, L=2, seed=23)
    X = torch.randn(12, 5, generator=torch.Generator().manual_seed(41), dtype=torch.float64)
    y = torch.randn(12, generator=torch.Generator().manual_seed(42), dtype=torch.float64)
    full_optimizer = ChizatMuonAdam(full, rates)
    accumulated_optimizer = ChizatMuonAdam(accumulated, rates)
    for parameter in (*full.params(), *accumulated.params()):
        parameter.requires_grad_(True)

    full_optimizer.zero_grad(set_to_none=True)
    full.loss(X, y).backward()
    full_optimizer.step()

    accumulated_optimizer.zero_grad(set_to_none=True)
    (0.5 * accumulated.loss(X[:6], y[:6])).backward()
    (0.5 * accumulated.loss(X[6:], y[6:])).backward()
    accumulated_optimizer.step()
    for expected, actual in zip(full.params(), accumulated.params()):
        assert torch.allclose(expected, actual, atol=2e-8, rtol=2e-8)


def test_hybrid_optimizer_is_deterministic_on_repeated_cpu_runs():
    rates = chizat_muon_learning_rates("group_rms_D", L=2, M=10, D=8, eta=1e-3)
    first = _net(D=8, M=10, L=2, seed=29)
    second = _net(D=8, M=10, L=2, seed=29)
    X = torch.randn(12, 5, generator=torch.Generator().manual_seed(51), dtype=torch.float64)
    y = torch.randn(12, generator=torch.Generator().manual_seed(52), dtype=torch.float64)
    first_optimizer = ChizatMuonAdam(first, rates)
    second_optimizer = ChizatMuonAdam(second, rates)
    for _ in range(3):
        _training_step(first, first_optimizer, X, y)
        _training_step(second, second_optimizer, X, y)
    for expected, actual in zip(first.params(), second.params()):
        assert torch.equal(expected, actual)


def test_paired_largest_shape_control_catches_worse_rule_even_if_finite():
    trials = []
    for seed, primary, wrong in ((0, 0.0010, 0.0060), (1, 0.0012, 0.0070), (2, 0.0008, 0.0055)):
        trials.append(
            SimpleNamespace(
                rule="group_rms_D", label="J5", normalized_eta=0.1,
                seed=seed, final_loss=primary,
            )
        )
        trials.append(
            SimpleNamespace(
                rule="wrong", label="J5", normalized_eta=0.1,
                seed=seed, final_loss=wrong,
            )
        )
    result = paired_largest_shape_control(
        trials,
        largest_shape=Shape("J5", 16, 1024, 64, 8.0),
        seeds=[0, 1, 2],
        eta=0.1,
        control_rule="wrong",
    )
    assert result["rejected"]
    assert result["mean_control_minus_primary"] > result["tolerance"]
