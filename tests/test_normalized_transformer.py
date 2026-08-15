import copy
from dataclasses import replace
import math

import pytest
import torch
from torch.nn import functional as F

from ai_theorist.autoscaler.normalized_transformer import (
    NormalizedTransformer,
    make_synthetic_markov_dataset,
)
from ai_theorist.autoscaler.schema import (
    SpecError,
    StudySpec,
    compile_plan,
    default_study_spec,
    parameter_count,
)
from ai_theorist.autoscaler.training import train_trial
from ai_theorist.autoscaler.study import run_study


def tiny_ngpt_spec(*, steps=4, microbatch_size=None):
    data = copy.deepcopy(
        default_study_spec("adam", quick=True, block_type="normalized_transformer").to_dict()
    )
    data["architecture"].update(
        vocab_size=16,
        context_length=8,
        head_dimension=4,
        mlp_multiplier=2,
        reference_width=16,
        reference_depth=2,
    )
    data["dataset"] = {
        "task_type": "synthetic_markov",
        "n_train": 24,
        "n_validation": 16,
        "noise_std": 0.03,
        "seed": 7,
    }
    data["horizon"] = {
        "steps": steps,
        "batch_size": 4,
        "microbatch_size": microbatch_size,
    }
    data["scales"] = [
        {"name": f"S{index + 1}", "width": width, "repeats": repeats}
        for index, (width, repeats) in enumerate(
            ((8, 1), (12, 1), (16, 2), (20, 2), (24, 3))
        )
    ]
    data["seeds"] = [3, 5]
    data["validation"]["bootstrap_samples"] = 0
    data["validation"]["run_negative_control"] = False
    return StudySpec.from_dict(data)


def test_ngpt_plan_and_parameter_count_match_model():
    spec = tiny_ngpt_spec()
    plan = compile_plan(spec)
    assert plan["target_metric"] == "final_validation_cross_entropy"
    assert plan["fixed_token_horizon"] == (
        spec.horizon.steps * spec.horizon.batch_size * spec.architecture.context_length
    )
    assert plan["transfer_rule"] == "nugpt_mid_alignment_group_rates_with_post_step_sphere_projection"
    assert plan["architecture_contract"]["model_family"] == "nuGPT_2026_mid_alignment"
    assert plan["architecture_contract"]["normalization_layers"] == "none"
    assert plan["architecture_contract"]["tied_embeddings"] is False
    assert spec.optimizer.beta2 == pytest.approx(0.95)
    assert spec.optimizer.epsilon == pytest.approx(1e-16)
    for scale in spec.scales:
        model = NormalizedTransformer(spec.architecture, scale)
        assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count(
            spec, scale
        )


def test_ngpt_projection_and_hidden_states_stay_on_unit_sphere():
    spec = tiny_ngpt_spec()
    model = NormalizedTransformer(spec.architecture, spec.scales[2])
    assert model.token_embedding.weight is not model.language_model_head.weight
    with torch.no_grad():
        model.token_embedding.weight[0].mul_(3.0)
        model.blocks[0].attention_output.weight[:, 0].mul_(0.25)
    assert model.sphere_diagnostics()["maximum_matrix_norm_error"] > 0.1
    model.project_normalized_weights()
    inputs, _, _, _ = make_synthetic_markov_dataset(spec.architecture, spec.dataset)
    logits = model(inputs[:3])
    diagnostics = model.sphere_diagnostics()
    assert logits.shape == (3, spec.architecture.context_length, spec.architecture.vocab_size)
    assert diagnostics["maximum_matrix_norm_error"] < 1e-6
    assert diagnostics["maximum_hidden_norm_error"] < 1e-6
    assert diagnostics["mean_attention_alpha"] == pytest.approx(0.05, abs=1e-6)
    assert diagnostics["mean_mlp_alpha"] == pytest.approx(0.05, abs=1e-6)
    assert diagnostics["mean_logit_scale"] == pytest.approx(1.0, abs=1e-6)


def test_uncaptured_ngpt_entropy_is_explicitly_unavailable():
    spec = tiny_ngpt_spec()
    model = NormalizedTransformer(
        spec.architecture,
        spec.scales[1],
        attention_backend="math",
        capture_attention_diagnostics=False,
    )
    inputs, _, _, _ = make_synthetic_markov_dataset(spec.architecture, spec.dataset)
    model(inputs[:2])
    diagnostics = model.sphere_diagnostics()
    assert diagnostics["mean_attention_entropy"] is None
    assert math.isfinite(diagnostics["maximum_matrix_norm_error"])


def test_baseline_ngpt_control_restores_original_rescaler_initialization():
    spec = tiny_ngpt_spec()
    scale = spec.scales[-1]
    model = NormalizedTransformer(
        spec.architecture,
        scale,
        parameterization="baseline_ngpt",
    )
    inputs, _, _, _ = make_synthetic_markov_dataset(spec.architecture, spec.dataset)
    model(inputs[:2])
    diagnostics = model.sphere_diagnostics()
    assert diagnostics["mean_attention_alpha"] == pytest.approx(0.05, abs=1e-6)
    assert diagnostics["mean_mlp_alpha"] == pytest.approx(0.05, abs=1e-6)
    assert diagnostics["mean_logit_scale"] == pytest.approx(1.0, abs=1e-6)
    assert float(model.logit_scale.mean().detach()) == pytest.approx(
        1.0 / math.sqrt(scale.width)
    )


def test_ngpt_attention_is_causal():
    spec = tiny_ngpt_spec()
    model = NormalizedTransformer(spec.architecture, spec.scales[1]).eval()
    tokens = torch.arange(spec.architecture.context_length).remainder(
        spec.architecture.vocab_size
    )[None, :]
    changed = tokens.clone()
    changed[0, -1] = (changed[0, -1] + 3) % spec.architecture.vocab_size
    with torch.no_grad():
        original_logits = model(tokens)
        changed_logits = model(changed)
    assert torch.equal(original_logits[:, :-1], changed_logits[:, :-1])
    assert not torch.equal(original_logits[:, -1], changed_logits[:, -1])


def test_ngpt_accelerated_sdpa_matches_reference_outputs_and_gradients():
    spec = tiny_ngpt_spec()
    scale = spec.scales[1]
    reference = NormalizedTransformer(
        spec.architecture,
        scale,
        attention_backend="math",
        capture_attention_diagnostics=False,
    ).train()
    accelerated = NormalizedTransformer(
        spec.architecture,
        scale,
        attention_backend="auto",
        capture_attention_diagnostics=False,
    ).train()
    accelerated.load_state_dict(reference.state_dict())
    tokens = torch.randint(
        0,
        spec.architecture.vocab_size,
        (2, spec.architecture.context_length),
        generator=torch.Generator().manual_seed(23),
    )
    targets = tokens.roll(-1, dims=1)
    reference_logits = reference(tokens)
    accelerated_logits = accelerated(tokens)
    torch.testing.assert_close(accelerated_logits, reference_logits, rtol=1e-5, atol=1e-6)
    F.cross_entropy(
        reference_logits.reshape(-1, spec.architecture.vocab_size),
        targets.reshape(-1),
    ).backward()
    F.cross_entropy(
        accelerated_logits.reshape(-1, spec.architecture.vocab_size),
        targets.reshape(-1),
    ).backward()
    for (reference_name, reference_parameter), (accelerated_name, accelerated_parameter) in zip(
        reference.named_parameters(), accelerated.named_parameters()
    ):
        assert accelerated_name == reference_name
        torch.testing.assert_close(
            accelerated_parameter.grad,
            reference_parameter.grad,
            rtol=2e-5,
            atol=2e-6,
        )


def test_synthetic_language_data_and_training_are_deterministic():
    spec = tiny_ngpt_spec(steps=3)
    first_data = make_synthetic_markov_dataset(spec.architecture, spec.dataset)
    second_data = make_synthetic_markov_dataset(spec.architecture, spec.dataset)
    assert all(torch.equal(left, right) for left, right in zip(first_data, second_data))
    first = train_trial(spec, spec.scales[0], 0.003, 3)
    second = train_trial(spec, spec.scales[0], 0.003, 3)
    assert first.final_validation_loss == second.final_validation_loss
    assert first.train_loss_trace == second.train_loss_trace
    assert first.normalized_transformer_diagnostics is not None
    assert first.normalized_transformer_diagnostics["maximum_matrix_norm_error"] < 1e-6
    assert math.isfinite(first.normalized_transformer_diagnostics["mean_attention_entropy"])
    assert first.learning_rate_schedule == "cosine_to_10_percent_without_warmup"
    assert first.gradient_clipping == "none_source_faithful"
    assert first.optimizer_group_contract is not None
    assert first.optimizer_group_contract["complete"] is True
    assert first.optimizer_group_contract["disjoint"] is True
    assert first.optimizer_group_contract["optimizer_options"] == {
        "name": "adam",
        "betas": [0.9, 0.95],
        "epsilon": 1e-16,
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "schedule": "cosine_to_10_percent",
        "gradient_clipping": False,
        "matrix_constraint": "unit-sphere projection before every optimization step",
    }
    assert first.train_loss_trace[0]["peak_learning_rate_multiplier"] == pytest.approx(1.0)
    assert first.train_loss_trace[-1]["peak_learning_rate_multiplier"] == pytest.approx(0.1)
    width_multiplier = spec.scales[0].width / spec.architecture.reference_width
    assert first.raw_learning_rates == {
        "nugpt_input": pytest.approx(0.003 * width_multiplier ** -0.5),
        "nugpt_hidden": pytest.approx(0.003 * width_multiplier ** -0.75),
        "nugpt_output": pytest.approx(0.5 * 0.003 * width_multiplier ** -0.75),
        "nugpt_rescalers": pytest.approx(0.003),
    }


def test_joint_language_data_scaling_keeps_nested_train_and_fixed_validation():
    spec = tiny_ngpt_spec()
    small_dataset = replace(spec.dataset, n_train=24, markov_order=3, markov_states=16)
    large_dataset = replace(small_dataset, n_train=48)
    small = make_synthetic_markov_dataset(spec.architecture, small_dataset)
    large = make_synthetic_markov_dataset(spec.architecture, large_dataset)
    assert torch.equal(small[0], large[0][:24])
    assert torch.equal(small[1], large[1][:24])
    assert torch.equal(small[2], large[2])
    assert torch.equal(small[3], large[3])


def test_ngpt_microbatching_matches_full_batch():
    full = tiny_ngpt_spec(steps=3)
    accumulated = tiny_ngpt_spec(steps=3, microbatch_size=2)
    full_result = train_trial(full, full.scales[0], 0.003, 3)
    accumulated_result = train_trial(accumulated, accumulated.scales[0], 0.003, 3)
    assert accumulated_result.final_validation_loss == pytest.approx(
        full_result.final_validation_loss, rel=1e-6, abs=1e-7
    )


def test_ngpt_schema_rejects_sgd_wrong_task_and_invalid_head_geometry():
    data = tiny_ngpt_spec().to_dict()
    data["optimizer"]["name"] = "sgd"
    with pytest.raises(SpecError, match="requires adam"):
        StudySpec.from_dict(data)

    data = tiny_ngpt_spec().to_dict()
    data["dataset"]["task_type"] = "nonlinear_regression"
    with pytest.raises(SpecError, match="requires dataset.task_type synthetic_markov"):
        StudySpec.from_dict(data)

    data = tiny_ngpt_spec().to_dict()
    data["architecture"]["head_dimension"] = 6
    with pytest.raises(SpecError, match="divisible"):
        StudySpec.from_dict(data)


def test_ngpt_end_to_end_study_enforces_normalization_gate(tmp_path):
    data = tiny_ngpt_spec(steps=2).to_dict()
    data["validation"]["run_negative_control"] = True
    data["tuning"] = {
        "normalized_learning_rates": [0.001, 0.003, 0.01],
        "max_expansion_rounds": 0,
        "expansion_factor": 3.0,
    }
    study_spec = StudySpec.from_dict(data)
    output_dir = tmp_path / "ngpt-study"
    result = run_study(study_spec, output_dir=output_dir)
    assert result["status"] == "completed"
    assert result["normalization_quality"]["applicable"] is True
    assert result["normalization_quality"]["accepted"] is True
    assert len(result["normalization_quality"]["scales"]) == 5
    assert all(
        row["maximum_matrix_norm_error"] < 1e-6
        and row["maximum_hidden_norm_error"] < 1e-6
        for row in result["normalization_quality"]["scales"]
    )
    assert result["learning_rate_coordinate"]["parameterization"] == "nugpt_mid_alignment"
    assert result["transfer_rule"] == "nugpt_mid_alignment_group_rates_with_post_step_sphere_projection"
    assert result["negative_control"]["rule"] == (
        "baseline_ngpt_incorrect_single_global_learning_rate"
    )
    assert result["negative_control"]["raw_learning_rates"] == {
        "all": pytest.approx(result["learning_rate_coordinate"]["normalized_eta"])
    }
    control = result["parameterization_control"]
    assert control["name"] == "baseline_ngpt_single_global_learning_rate"
    assert len(control["scale_results"]) == len(study_spec.scales)
    assert len(control["paired_comparisons"]) == len(study_spec.scales)
    assert control["largest_scale_transfer_probe"]["scale"] == study_spec.scales[-1].name

    # Every completed trial is reloaded atomically; no training needs to be repeated.
    resumed = run_study(study_spec, output_dir=output_dir)
    assert resumed["tuning"] == result["tuning"]
    assert resumed["parameterization_control"] == control
    assert resumed["scale_results"] == result["scale_results"]
