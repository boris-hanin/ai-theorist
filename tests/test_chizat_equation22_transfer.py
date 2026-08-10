import math

import pytest
import torch

from ai_theorist.autoscaler.chizat_resnet import Chizat2LPShape
from chizat_equation22_transfer import (
    NamedShape,
    Trial,
    make_fixed_task,
    parse_shapes,
    run_trial,
    select_best_pair,
    summarize,
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


def test_divergent_boundary_probes_bracket_but_do_not_invalidate_transfer():
    shapes = parse_shapes("base:1:8:4,large:2:16:16")
    records = []
    for named_shape in shapes:
        for eta_u in (0.1, 0.3, 1.0):
            for eta_v in (0.1, 0.3, 1.0):
                for seed in (11, 29):
                    diverged = eta_u == 1.0 and eta_v == 1.0
                    # A tempting but incomplete candidate must not beat a
                    # complete paired-seed optimum.
                    partial = eta_u == 0.1 and eta_v == 0.1 and seed == 29
                    loss = 1.0 if (eta_u, eta_v) == (0.3, 0.3) else 2.0
                    if eta_u == 0.1 and eta_v == 0.1 and seed == 11:
                        loss = 0.5
                    if diverged or partial:
                        loss = float("inf")
                    records.append(
                        Trial(
                            label=named_shape.label,
                            L=named_shape.shape.depth,
                            M=named_shape.shape.hidden_width,
                            D=named_shape.shape.embedding_dimension,
                            LM_over_D=named_shape.shape.rho,
                            seed=seed,
                            rule="lmd",
                            eta_u=eta_u,
                            eta_v=eta_v,
                            raw_learning_rates={"particle_u": 1.0, "particle_v": 1.0},
                            initial_validation_loss=3.0,
                            final_training_loss=loss,
                            final_validation_loss=loss,
                            fractional_validation_progress=(
                                -float("inf") if not math.isfinite(loss) else (3.0 - loss) / 3.0
                            ),
                            u_relative_rms_movement=1.0,
                            v_relative_rms_movement=1.0,
                            diverged=not math.isfinite(loss),
                        )
                    )
    assert select_best_pair(records, "base") == (0.3, 0.3)
    report = summarize(
        records,
        [],
        shapes=shapes,
        reference_label="base",
        drift_tolerance_decades=0.55,
        oracle_tolerance=1.25,
        eta_us=(0.1, 0.3, 1.0),
        eta_vs=(0.1, 0.3, 1.0),
    )
    assert report["exploratory_divergent_trial_count"] > 0
    assert report["verdict"] == "PASS"
