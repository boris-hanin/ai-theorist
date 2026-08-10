import math

import pytest
import torch

from ai_theorist.autoscaler.jiang_chizat import JiangChizatReference
from jiang_chizat_transfer import (
    Shape,
    group_feature_velocity_audit,
    parse_shape,
    progress_report,
    run_trial,
    synthetic_markov_data,
    validate_shapes,
)


def tiny_shapes():
    return validate_shapes(
        [
            parse_shape("S1:1:8:8:1"),
            parse_shape("S2:1:12:12:2"),
            parse_shape("S3:2:16:16:3"),
            parse_shape("S4:2:24:24:4"),
        ],
        head_dimension=4,
    )


def test_shape_parser_tracks_rho_and_head_geometry():
    shapes = tiny_shapes()
    assert shapes[2] == Shape("S3", 2, 16, 16, 3.0)
    assert shapes[2].rho == pytest.approx(2.0)
    with pytest.raises(ValueError, match="divisible"):
        validate_shapes(
            [
                Shape("S1", 1, 8, 8, 1),
                Shape("S2", 1, 8, 10, 2),
                Shape("S3", 1, 8, 12, 3),
                Shape("S4", 1, 8, 16, 4),
            ],
            head_dimension=4,
        )


def test_synthetic_task_is_shape_independent_and_deterministic():
    first = synthetic_markov_data(
        vocab_size=16,
        context_length=8,
        n_train=16,
        n_validation=8,
        seed=17,
        noise_probability=0.03,
        device=torch.device("cpu"),
    )
    second = synthetic_markov_data(
        vocab_size=16,
        context_length=8,
        n_train=16,
        n_validation=8,
        seed=17,
        noise_probability=0.03,
        device=torch.device("cpu"),
    )
    assert all(torch.equal(left, right) for left, right in zip(first, second))


def test_tiny_trial_records_attention_movement_and_group_coordinates():
    shape = tiny_shapes()[0]
    result = run_trial(
        shape,
        reference=JiangChizatReference(1, 8, 8),
        head_dimension=4,
        vocab_size=16,
        context_length=8,
        n_train=16,
        n_validation=8,
        dataset_seed=17,
        eta=1e-3,
        epsilon0=1e-12,
        steps=2,
        batch_size=4,
        seed=11,
        rule="primary",
        device=torch.device("cpu"),
    )
    assert result.diverged is False
    assert math.isfinite(result.final_validation_loss)
    assert result.validation_loss_checkpoints.keys() >= {0, 1, 2}
    assert result.raw_learning_rates["jiang_attention_qkv"] == pytest.approx(1e-3 / 16.0)
    assert result.raw_learning_rates["jiang_attention_output"] == pytest.approx(1e-3)
    assert result.attention_movement["per_entry_attention_logit_delta_rms"] > 0.0
    assert result.attention_movement["head_averaged_attention_delta_rms"] > 0.0


def test_group_only_feature_velocity_audit_routes_every_semantic_group():
    result = group_feature_velocity_audit(
        tiny_shapes()[0],
        reference=JiangChizatReference(1, 8, 8),
        head_dimension=4,
        vocab_size=16,
        context_length=4,
        n_train=8,
        n_validation=8,
        dataset_seed=17,
        eta=1e-3,
        epsilon0=1e-12,
        batch_size=4,
        seed=11,
        device=torch.device("cpu"),
    )
    velocities = result["final_hidden_feature_velocity_rms_over_eta"]
    assert set(velocities) == {
        "jiang_embeddings",
        "jiang_norms",
        "jiang_attention_qkv",
        "jiang_attention_output",
        "jiang_ffn_up",
        "jiang_ffn_down",
        "jiang_other_biases",
    }
    assert all(math.isfinite(value) and value > 0.0 for value in velocities.values())


def test_progress_gate_rejects_vanishing_or_negative_progress():
    shapes = tiny_shapes()
    trials = [
        run_trial(
            shape,
            reference=JiangChizatReference(1, 8, 8),
            head_dimension=4,
            vocab_size=16,
            context_length=4,
            n_train=8,
            n_validation=8,
            dataset_seed=17,
            eta=1e-12,
            epsilon0=1e-12,
            steps=1,
            batch_size=4,
            seed=seed,
            rule="primary",
            device=torch.device("cpu"),
        )
        for shape in shapes
        for seed in (11, 29)
    ]
    report = progress_report(trials, shapes, [11, 29], rule="primary")
    assert report["accepted"] is False
    assert report["checkpoints"][-1]["nontrivial"] is False
