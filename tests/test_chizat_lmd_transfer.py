import math

import pytest

from chizat_lmd_transfer import (
    Shape,
    Trial,
    group_learning_rates,
    parse_shape,
    progress_report,
    validate_shapes,
)


def test_shape_parser_and_validation():
    shapes = validate_shapes(
        [
            parse_shape("S1:2:32:8:1"),
            parse_shape("S2:4:64:16:2"),
            parse_shape("S3:8:128:32:4"),
            parse_shape("S4:16:256:64:8"),
        ]
    )
    assert shapes[-1] == Shape("S4", 16, 256, 64, 8.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_shapes([shapes[0], shapes[2], shapes[1], shapes[3]])


def test_group_natural_rates_have_distinct_D_powers():
    shape = Shape("S", L=8, M=256, D=32, dial=1.0)
    rates = group_learning_rates("group_natural_LMD", shape, 0.25, reference_D=32)
    assert rates["U"] == pytest.approx(16.0)
    assert rates["W"] == pytest.approx(16_384.0)
    assert rates["W"] / rates["U"] == pytest.approx(shape.D ** 2)


def test_incoherent_rates_differ_by_one_power_of_D():
    shape = Shape("S", L=8, M=256, D=32, dial=1.0)
    rates = group_learning_rates(
        "group_incoherent_LM_sqrtD", shape, 0.25, reference_D=32
    )
    assert rates["W"] / rates["U"] == pytest.approx(shape.D)


def test_mutations_remove_the_declared_axis():
    shape = Shape("S", L=8, M=256, D=32, dial=1.0)
    correct = group_learning_rates("group_natural_LMD", shape, 0.25, reference_D=32)
    omit_l = group_learning_rates("omit_L", shape, 0.25, reference_D=32)
    omit_m = group_learning_rates("omit_M", shape, 0.25, reference_D=32)
    assert omit_l["U"] == pytest.approx(correct["U"] / shape.L)
    assert omit_m["W"] == pytest.approx(correct["W"] / shape.M)
    single = group_learning_rates("single_LM_reference_D", shape, 0.25, reference_D=32)
    assert math.isclose(single["U"], single["W"])


def test_progress_gate_rejects_vanishing_improvement():
    shapes = [
        Shape("S1", 2, 16, 8, 1.0),
        Shape("S2", 3, 24, 12, 1.5),
        Shape("S3", 4, 32, 16, 2.0),
        Shape("S4", 6, 48, 24, 3.0),
    ]
    trials = []
    for shape in shapes:
        for seed in (0, 1):
            trials.append(
                Trial(
                    shape.label,
                    shape.L,
                    shape.M,
                    shape.D,
                    shape.dial,
                    seed,
                    0.003,
                    "group_incoherent_LM_sqrtD",
                    "both",
                    {"U": 1.0, "W": 1.0},
                    {0: 1.0, 8: 0.999999},
                    0.999999,
                    False,
                )
            )
    report = progress_report(
        trials,
        shapes,
        [0, 1],
        rule="group_incoherent_LM_sqrtD",
        minimum_progress=1e-3,
    )
    assert report["accepted"] is False
    assert report["checkpoints"][-1]["nontrivial"] is False
