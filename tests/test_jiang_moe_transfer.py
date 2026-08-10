import pytest
import torch
from types import SimpleNamespace

from ai_theorist.autoscaler.jiang_moe import JiangMoEReference
from jiang_moe_transfer import (
    best_group_multiplier,
    fixed_eta_analysis,
    parse_shape,
    primary_analysis,
    run_trial,
    validate_shapes,
)


def tiny_shapes():
    return validate_shapes(
        [
            parse_shape("S1:1:8:8:2:1:1"),
            parse_shape("S2:1:12:12:4:2:2"),
            parse_shape("S3:2:12:16:4:2:3"),
            parse_shape("S4:2:16:24:8:4:4"),
        ],
        head_dimension=4,
    )


def test_shape_contract_requires_fixed_sparsity():
    shapes = tiny_shapes()
    assert all(shape.kappa == pytest.approx(0.5) for shape in shapes)
    bad = list(shapes)
    bad[-1] = parse_shape("bad:2:16:24:8:2:4")
    with pytest.raises(ValueError, match="fixed A/E"):
        validate_shapes(bad, head_dimension=4)


def test_tiny_full_moe_trial_records_table2_groups_and_manual_bias():
    shape = tiny_shapes()[0]
    trial, audit = run_trial(
        shape,
        reference=JiangMoEReference(1, 8, 8, 2, 1),
        head_dimension=4,
        vocab_size=16,
        context_length=4,
        n_train=8,
        n_validation=8,
        dataset_seed=1729,
        eta=1e-3,
        epsilon0=1e-12,
        expert_bias_learning_rate=0.01,
        steps=1,
        batch_size=4,
        seed=11,
        rule="table2",
        device=torch.device("cpu"),
    )
    assert trial.diverged is False
    assert set(trial.raw_learning_rates) == {
        "jiang_moe_embeddings",
        "jiang_moe_norms",
        "jiang_moe_attention_qkv",
        "jiang_moe_attention_output",
        "jiang_moe_router",
        "jiang_moe_expert_up",
        "jiang_moe_expert_down",
        "jiang_moe_other_biases",
    }
    assert audit["optimizer"]["complete"] is True
    assert audit["manual_expert_bias"]["learning_rate"] == pytest.approx(0.01)


def test_fixed_eta_gate_rejects_negligible_progress():
    shapes = tiny_shapes()
    trials = []
    for shape in shapes:
        for seed in (11, 29, 47):
            trial, _ = run_trial(
                shape,
                reference=JiangMoEReference(1, 8, 8, 2, 1),
                head_dimension=4,
                vocab_size=16,
                context_length=4,
                n_train=8,
                n_validation=8,
                dataset_seed=1729,
                eta=1e-12,
                epsilon0=1e-12,
                expert_bias_learning_rate=1e-12,
                steps=1,
                batch_size=4,
                seed=seed,
                rule="table2",
                device=torch.device("cpu"),
            )
            trials.append(trial)
    result = fixed_eta_analysis(trials, shapes, [11, 29, 47], eta=1e-12, rule="table2")
    assert result["accepted"] is False


def test_moe_relative_multiplier_gate_keeps_source_without_paired_improvement():
    def rows(losses):
        return [
            SimpleNamespace(seed=seed, final_validation_loss=loss, diverged=False)
            for seed, loss in zip((11, 29, 47), losses)
        ]

    trials = {
        0.5: rows((0.98, 0.98, 1.01)),
        1.0: rows((1.0, 1.0, 1.0)),
        2.0: rows((0.999, 0.999, 0.999)),
    }
    assert best_group_multiplier(
        trials, minimum_relative_improvement=0.005
    ) == pytest.approx(1.0)


def test_moe_primary_tunes_reference_argmin_and_reports_conservative_point():
    etas = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    rows = []
    for shape in tiny_shapes():
        for eta in etas:
            for seed in (11, 29, 47):
                best = 3e-3 if shape.label == "S2" else 1e-2
                loss = 1.0 if eta == best else 1.5
                if eta == 3e-3 and best == 1e-2:
                    loss = 1.05
                rows.append(
                    SimpleNamespace(
                        label=shape.label,
                        rule="table2",
                        eta=eta,
                        seed=seed,
                        final_validation_loss=loss,
                        fractional_progress=0.5,
                        maximum_routing_load_deviation=0.1,
                        diverged=False,
                    )
                )
    report = primary_analysis(
        rows,
        tiny_shapes(),
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
