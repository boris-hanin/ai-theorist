import copy
import json
import math
from pathlib import Path

import pytest
import torch

from ai_theorist.autoscaler.model import (
    ChizatResidualMLP,
    build_model,
    dataset_fingerprint,
    make_teacher_dataset,
)
from ai_theorist.autoscaler.matrix import compile_validation_matrix, expand_validation_matrix
from ai_theorist.autoscaler.muon import HybridMuonAdam, zeropower_via_newtonschulz
from ai_theorist.autoscaler.schema import SpecError, StudySpec, default_study_spec
from ai_theorist.autoscaler.study import run_study
from ai_theorist.autoscaler.training import make_optimizer, train_trial
from ai_theorist.autoscaler.tuning import (
    CHIZAT_MEAN_FIELD,
    optimizer_group_learning_rates_from_normalized_eta,
)


def tiny_chizat_spec(optimizer: str = "muon", *, steps: int = 4, kind: str = "tanh_teacher"):
    data = copy.deepcopy(
        default_study_spec(optimizer, quick=True, block_type="chizat_mlp").to_dict()
    )
    data["dataset"] = {
        "kind": kind,
        "n_train": 32,
        "n_validation": 24,
        "noise_std": 0.0,
        "seed": 7,
        "generator_version": 1,
    }
    data["horizon"] = {"steps": steps, "batch_size": 8, "microbatch_size": None}
    data["scales"] = [
        {"name": f"S{index + 1}", "width": width, "repeats": index + 1, "particle_width": width}
        for index, width in enumerate((4, 6, 8, 10, 12))
    ]
    data["seeds"] = [3, 5]
    data["validation"]["bootstrap_samples"] = 0
    return StudySpec.from_dict(data)


def test_muon_schema_is_restricted_to_the_validated_chizat_contract():
    spec = tiny_chizat_spec("muon")
    assert spec.optimizer.momentum == 0.95
    assert spec.optimizer.beta2 == 0.95
    assert spec.optimizer.epsilon == 1e-10
    assert spec.architecture.activation == "tanh"
    assert all(scale.particle_width is not None for scale in spec.scales)

    wrong_architecture = default_study_spec("adam", quick=True).to_dict()
    wrong_architecture["optimizer"] = {"name": "muon"}
    with pytest.raises(SpecError, match="requires the validated chizat_mlp"):
        StudySpec.from_dict(wrong_architecture)

    mutated = spec.to_dict()
    mutated["optimizer"]["nesterov"] = False
    with pytest.raises(SpecError, match="validated Muon slice"):
        StudySpec.from_dict(mutated)


def test_chizat_model_has_trainable_bias_free_semantic_roles():
    spec = tiny_chizat_spec()
    scale = spec.scales[2]
    model = build_model(spec.architecture, scale)
    assert isinstance(model, ChizatResidualMLP)
    roles = model.semantic_parameter_roles()
    assert set(roles) == {"embed", "U", "W", "unembed"}
    assert len(roles["U"]) == len(roles["W"]) == scale.repeats
    assert roles["embed"][0].shape == (spec.architecture.input_dim, scale.width)
    assert roles["unembed"][0].shape == (scale.width, spec.architecture.output_dim)
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert all("bias" not in name for name, _ in model.named_parameters())


def test_chizat_optimizer_coordinates_are_group_specific():
    kwargs = dict(width=16, depth=4, particle_width=32)
    sgd = optimizer_group_learning_rates_from_normalized_eta(
        CHIZAT_MEAN_FIELD, "sgd", 0.1, **kwargs
    )
    assert sgd == pytest.approx(
        {"embed": 1.6, "U": 0.8, "W": 204.8, "unembed": 0.00625}
    )
    for optimizer in ("adam", "muon"):
        rates = optimizer_group_learning_rates_from_normalized_eta(
            CHIZAT_MEAN_FIELD, optimizer, 0.1, **kwargs
        )
        assert rates == pytest.approx(
            {"embed": 0.1, "U": 0.1, "W": 0.4, "unembed": 0.00625}
        )
    wrong = optimizer_group_learning_rates_from_normalized_eta(
        CHIZAT_MEAN_FIELD, "muon", 0.1, rule="wrong_W_D", **kwargs
    )
    assert wrong["W"] == pytest.approx(1.6)


def test_product_muon_routes_only_particle_matrices_through_muon():
    spec = tiny_chizat_spec()
    scale = spec.scales[1]
    model = build_model(spec.architecture, scale)
    optimizer = make_optimizer(
        model, spec, 0.05, normalized_eta=0.05, scale=scale
    )
    assert isinstance(optimizer, HybridMuonAdam)
    roles = model.semantic_parameter_roles()
    muon_ids = {
        id(parameter)
        for group in optimizer.muon.param_groups
        for parameter in group["params"]
    }
    assert muon_ids == {id(parameter) for parameter in (*roles["U"], *roles["W"])}
    assert id(roles["embed"][0]) not in muon_ids
    assert id(roles["unembed"][0]) not in muon_ids


def test_all_chizat_roles_update_in_one_product_step():
    spec = tiny_chizat_spec()
    scale = spec.scales[0]
    torch.manual_seed(9)
    model = build_model(spec.architecture, scale)
    optimizer = make_optimizer(
        model, spec, 0.03, normalized_eta=0.03, scale=scale
    )
    x = torch.randn(12, spec.architecture.input_dim)
    y = torch.randn(12, spec.architecture.output_dim)
    before = {name: [parameter.detach().clone() for parameter in values]
              for name, values in model.semantic_parameter_roles().items()}
    torch.nn.functional.mse_loss(model(x), y).backward()
    optimizer.step()
    after = model.semantic_parameter_roles()
    assert all(
        any(not torch.equal(old, new) for old, new in zip(before[name], after[name]))
        for name in before
    )


def test_product_muon_checkpoint_resume_is_exact(tmp_path: Path):
    spec = tiny_chizat_spec(steps=6)
    scale = spec.scales[0]
    uninterrupted = train_trial(spec, scale, 0.03, 3)
    checkpoint = tmp_path / "muon.pt"
    train_trial(
        spec,
        scale,
        0.03,
        3,
        checkpoint_path=checkpoint,
        checkpoint_every=1,
        stop_after_steps=3,
    )
    resumed = train_trial(
        spec, scale, 0.03, 3, checkpoint_path=checkpoint, checkpoint_every=1
    )
    assert resumed.final_validation_loss == uninterrupted.final_validation_loss
    assert resumed.train_loss_trace == uninterrupted.train_loss_trace
    assert resumed.validation_loss_trace == uninterrupted.validation_loss_trace


def test_dataset_adapters_are_versioned_deterministic_and_distinct():
    outputs = {}
    fingerprints = set()
    for kind in ("linear", "tanh_teacher", "sinusoid_quadratic"):
        spec = tiny_chizat_spec(kind=kind)
        first = make_teacher_dataset(spec.architecture, spec.dataset)
        second = make_teacher_dataset(spec.architecture, spec.dataset)
        assert all(torch.equal(left, right) for left, right in zip(first, second))
        outputs[kind] = first[1]
        fingerprints.add(dataset_fingerprint(spec.architecture, spec.dataset))
    assert len(fingerprints) == 3
    assert not torch.equal(outputs["linear"], outputs["tanh_teacher"])
    assert not torch.equal(outputs["tanh_teacher"], outputs["sinusoid_quadratic"])


def test_product_muon_trial_reports_dataset_and_raw_group_provenance():
    spec = tiny_chizat_spec()
    result = train_trial(spec, spec.scales[0], 0.05, 3)
    assert math.isfinite(result.final_validation_loss)
    assert result.dataset_kind == "tanh_teacher"
    assert len(result.dataset_fingerprint) == 64
    assert result.particle_width == spec.scales[0].particle_width
    assert set(result.raw_learning_rates or {}) == {"embed", "U", "W", "unembed"}


def test_product_newton_schulz_is_transpose_equivariant():
    torch.manual_seed(17)
    matrix = torch.randn(7, 13, dtype=torch.float64)
    direct = zeropower_via_newtonschulz(matrix)
    transposed = zeropower_via_newtonschulz(matrix.mT).mT
    assert torch.allclose(direct, transposed, atol=1e-6, rtol=1e-6)


def test_checked_in_optimizer_dataset_matrix_expands_to_nine_strict_cells():
    path = Path("configs/autoscaler/a100_chizat_optimizer_dataset_matrix.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = expand_validation_matrix(payload)
    compiled = compile_validation_matrix(payload)
    assert len(cells) == compiled["cell_count"] == 9
    assert compiled["execution_policy"] == "sequential_no_gpu_overlap"
    assert {spec.optimizer.name for _, spec in cells} == {"sgd", "adam", "muon"}
    assert {spec.dataset.kind for _, spec in cells} == {
        "linear", "tanh_teacher", "sinusoid_quadratic"
    }
    assert len({spec.fingerprint for _, spec in cells}) == 9


def test_matrix_rejects_unknown_fields():
    path = Path("configs/autoscaler/a100_chizat_optimizer_dataset_matrix.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_everything_anyway"] = True
    with pytest.raises(SpecError, match="Unknown matrix field"):
        expand_validation_matrix(payload)


def test_product_muon_study_reports_full_chizat_transfer_contract(tmp_path: Path):
    data = tiny_chizat_spec(steps=3).to_dict()
    data["tuning"] = {
        "normalized_learning_rates": [0.03, 0.05, 0.08],
        "max_expansion_rounds": 0,
        "expansion_factor": 3.0,
    }
    data["validation"] = {
        "transfer_probe_decades": 0.3,
        "run_negative_control": True,
        "bootstrap_samples": 0,
        "routing_load_tolerance": 0.25,
    }
    spec = StudySpec.from_dict(data)
    result = run_study(spec, output_dir=tmp_path / "chizat-muon-study")

    assert result["status"] == "completed"
    assert result["transfer_rule"] == (
        "chizat_muon_semantic_group_rates_from_normalized_eta"
    )
    assert result["dataset_contract"]["kind"] == "tanh_teacher"
    assert result["fixed_eta_trajectory"]["complete"] is True
    assert result["negative_control"]["rule"] == "wrong_W_D"
    assert all(
        set(row["raw_learning_rates"]) == {"embed", "U", "W", "unembed"}
        and row["particle_width"] is not None
        for row in result["scale_results"]
    )
    assert json.loads(
        (tmp_path / "chizat-muon-study" / "result.json").read_text(encoding="utf-8")
    )["study_fingerprint"] == spec.fingerprint
