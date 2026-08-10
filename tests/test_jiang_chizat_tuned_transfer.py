import pytest
import torch
from types import SimpleNamespace

from ai_theorist.autoscaler.jiang_chizat import (
    JIANG_DENSE_REPORTED_LR_MULTIPLIERS,
    JiangChizatReference,
)
from jiang_chizat_transfer import parse_shape, run_trial, validate_shapes
from jiang_chizat_tuned_transfer import (
    best_eta,
    best_multiplier,
    fixed_eta_analysis,
    primary_analysis,
)


def shapes():
    return validate_shapes(
        [
            parse_shape("S1:1:8:8:1"),
            parse_shape("S2:1:12:12:2"),
            parse_shape("S3:2:12:16:3"),
            parse_shape("S4:2:16:24:4"),
        ],
        4,
    )


def test_tuned_dense_trial_records_schedule_multipliers_and_full_contract():
    shape = shapes()[0]
    trial = run_trial(
        shape,
        reference=JiangChizatReference(1, 8, 8),
        head_dimension=4,
        vocab_size=16,
        context_length=4,
        n_train=8,
        n_validation=8,
        dataset_seed=17,
        eta=1e-3,
        epsilon0=1e-12,
        steps=2,
        batch_size=4,
        seed=11,
        rule="primary",
        device=torch.device("cpu"),
        learning_rate_multipliers={"jiang_attention_qkv": 0.5 / 16.0},
        warmup_steps=1,
    )
    assert trial.diverged is False
    assert trial.warmup_steps == 1
    assert trial.learning_rate_multipliers["jiang_attention_qkv"] == pytest.approx(0.5 / 16.0)
    assert trial.learning_rate_multipliers["jiang_ffn_down"] == pytest.approx(
        JIANG_DENSE_REPORTED_LR_MULTIPLIERS["jiang_ffn_down"]
    )
    assert trial.optimizer_group_contract["complete"] is True
    assert trial.raw_learning_rates["jiang_attention_qkv"] == pytest.approx(0.5e-3 / 16.0)


def test_best_eta_uses_reference_validation_loss():
    shape = shapes()[0]
    trials = []
    for eta in (1e-4, 1e-3):
        trial = run_trial(
            shape,
            reference=JiangChizatReference(1, 8, 8),
            head_dimension=4,
            vocab_size=16,
            context_length=4,
            n_train=8,
            n_validation=8,
            dataset_seed=17,
            eta=eta,
            epsilon0=1e-12,
            steps=1,
            batch_size=4,
            seed=11,
            rule="primary",
            device=torch.device("cpu"),
        )
        trials.append(trial)
    assert best_eta(trials, (1e-4, 1e-3)) in {1e-4, 1e-3}


def test_relative_multiplier_gate_requires_improvement_for_every_seed():
    def rows(losses):
        return [
            SimpleNamespace(seed=seed, final_validation_loss=loss, diverged=False)
            for seed, loss in zip((11, 29, 47), losses)
        ]

    trials = {
        0.5: rows((0.98, 0.98, 1.01)),
        1.0: rows((1.0, 1.0, 1.0)),
        2.0: rows((0.99, 0.99, 0.99)),
    }
    assert best_multiplier(
        trials, minimum_relative_improvement=0.005
    ) == pytest.approx(2.0)


def test_dense_primary_tunes_reference_argmin_and_reports_conservative_point():
    etas = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    rows = []
    for shape in shapes():
        for eta in etas:
            for seed in (11, 29, 47):
                best = 3e-3 if shape.label == "S2" else 1e-2
                loss = 1.0 if eta == best else 1.5
                if eta == 3e-3 and best == 1e-2:
                    loss = 1.05
                rows.append(
                    SimpleNamespace(
                        label=shape.label,
                        rule="primary",
                        normalized_eta=eta,
                        seed=seed,
                        final_validation_loss=loss,
                        validation_loss_checkpoints={0: 2.0},
                        diverged=False,
                    )
                )
    report = primary_analysis(
        rows,
        shapes(),
        etas,
        (11, 29, 47),
        "S1",
        1.10,
    )
    assert report["reference_oracle_eta"] == pytest.approx(1e-2)
    assert report["reference_eta"] == pytest.approx(1e-2)
    assert report["conservative_operating_point"]["eta"] == pytest.approx(3e-3)
    assert report["diagnostics"]["exact_grid_argmin_drift_within_0.35_decades"] is False
    assert report["accepted"] is False
