import pytest
import torch

from ai_theorist.autoscaler.chizat_resnet import Chizat2LPShape
from chizat_equation22_transfer import (
    NamedShape,
    make_fixed_task,
    parse_shapes,
    run_trial,
)


def test_shape_parser_preserves_the_joint_limit_coordinate():
    shapes = parse_shapes("base:1:8:4,large:2:16:16")
    assert [shape.label for shape in shapes] == ["base", "large"]
    assert [shape.shape.rho for shape in shapes] == [2.0, 2.0]


def test_tiny_equation22_trial_is_full_group_audited():
    dtype = torch.float64
    task = make_fixed_task(
        seed=17,
        n_train=8,
        n_validation=8,
        input_dimension=3,
        output_dimension=1,
        dtype=dtype,
    )
    shape = Chizat2LPShape(depth=1, hidden_width=4, embedding_dimension=4)
    trial, audit = run_trial(
        named_shape=NamedShape("tiny", shape),
        reference_shape=shape,
        eta_u=0.01,
        eta_v=0.03,
        rule="lmd",
        seed=11,
        steps=2,
        input_dimension=3,
        output_dimension=1,
        map_seed=23,
        task=task,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    assert trial.diverged is False
    assert trial.raw_learning_rates == {
        "particle_u": pytest.approx(0.01 * 1 * 4 * 4),
        "particle_v": pytest.approx(0.03 * 1 * 4 * 4),
    }
    assert audit["complete"] is True
    assert audit["trainable_parameter_tensors"] == 2


def test_fixed_task_is_independent_of_network_shape():
    first = make_fixed_task(
        seed=19,
        n_train=4,
        n_validation=3,
        input_dimension=2,
        output_dimension=1,
        dtype=torch.float64,
    )
    second = make_fixed_task(
        seed=19,
        n_train=4,
        n_validation=3,
        input_dimension=2,
        output_dimension=1,
        dtype=torch.float64,
    )
    assert all(torch.equal(left, right) for left, right in zip(first, second))
