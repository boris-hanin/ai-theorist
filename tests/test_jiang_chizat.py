import math

import pytest
import torch
from torch.nn import functional as F

from ai_theorist.autoscaler.jiang_chizat import (
    JIANG_COMPLETEP_ADAMW_THEORY,
    JIANG_DENSE_REPORTED_LR_MULTIPLIERS,
    JiangChizatReference,
    JiangChizatShape,
    JiangChizatTransformer,
)


def make_model(
    shape=JiangChizatShape(depth=2, hidden_width=32, residual_width=16, head_dimension=4),
    **kwargs,
):
    torch.manual_seed(7)
    return JiangChizatTransformer(
        shape,
        vocab_size=16,
        context_length=8,
        reference=JiangChizatReference(depth=2, hidden_width=32, residual_width=16),
        **kwargs,
    )


def test_shape_requires_fixed_valid_head_geometry():
    with pytest.raises(ValueError, match="divisible"):
        JiangChizatShape(depth=2, hidden_width=32, residual_width=18, head_dimension=4)
    shape = JiangChizatShape(depth=4, hidden_width=64, residual_width=32, head_dimension=8)
    assert shape.num_heads == 4
    assert shape.rho == pytest.approx(8.0)


def test_forward_is_causal_and_uses_tied_embeddings():
    model = make_model().eval()
    tokens = torch.arange(8).remainder(16)[None, :]
    changed = tokens.clone()
    changed[0, -1] = 13
    with torch.no_grad():
        logits = model(tokens)
        changed_logits = model(changed)
    assert logits.shape == (1, 8, 16)
    assert torch.equal(logits[:, :-1], changed_logits[:, :-1])
    assert not torch.equal(logits[:, -1], changed_logits[:, -1])
    assert not hasattr(model, "language_model_head")
    diagnostics = model.diagnostics()
    assert math.isfinite(diagnostics["mean_attention_entropy"])
    assert math.isfinite(diagnostics["mean_attention_logit_rms"])


def test_tied_unembedding_retains_completep_width_multiplier() -> None:
    model = make_model(
        JiangChizatShape(
            depth=2,
            hidden_width=64,
            residual_width=32,
            head_dimension=4,
        )
    ).eval()
    tokens = torch.arange(8).remainder(16)[None, :]
    with torch.no_grad():
        hidden = model.forward_features(tokens)
        expected = F.linear(hidden, model.token_embedding.weight) / 2.0
        logits = model(tokens)
    torch.testing.assert_close(logits, expected)
    assert model.diagnostics()["unembedding_forward_scale"] == pytest.approx(0.5)


def test_uncaptured_attention_diagnostics_are_explicitly_unavailable():
    model = make_model(capture_attention_diagnostics=False)
    model(torch.randint(0, 16, (2, 8)))
    diagnostics = model.diagnostics()
    assert diagnostics["mean_attention_entropy"] is None
    assert diagnostics["mean_attention_logit_rms"] is None


def test_mean_field_down_projection_and_fan_in_control_are_distinct():
    shape = JiangChizatShape(
        depth=1, hidden_width=1024, residual_width=256, head_dimension=64
    )
    mean_field = make_model(shape, down_initialization="mean_field")
    fan_in = make_model(shape, down_initialization="fan_in")
    expected_mean_field = math.sqrt(shape.residual_width) / shape.hidden_width / 4.0
    expected_fan_in = 1.0 / math.sqrt(shape.hidden_width) / 4.0
    observed_mean_field = float(mean_field.blocks[0].ffn_down.weight.detach().std())
    observed_fan_in = float(fan_in.blocks[0].ffn_down.weight.detach().std())
    assert observed_mean_field == pytest.approx(expected_mean_field, rel=0.02)
    assert observed_fan_in == pytest.approx(expected_fan_in, rel=0.02)
    assert observed_fan_in / observed_mean_field == pytest.approx(2.0, rel=0.04)


def test_table2_adam_group_rates_and_epsilons():
    shape = JiangChizatShape(depth=4, hidden_width=96, residual_width=32, head_dimension=8)
    model = make_model(shape)
    groups = {
        group["name"]: group
        for group in model.optimizer_parameter_groups(0.01, epsilon0=1e-12)
    }
    assert groups["jiang_embeddings"]["lr"] == pytest.approx(0.01)
    assert groups["jiang_norms"]["lr"] == pytest.approx(0.01)
    assert groups["jiang_final_norm"]["lr"] == pytest.approx(0.01)
    assert groups["jiang_attention_qkv"]["lr"] == pytest.approx(0.005 / 16.0)
    assert groups["jiang_attention_output"]["lr"] == pytest.approx(0.005)
    assert groups["jiang_ffn_up"]["lr"] == pytest.approx(0.005)
    assert groups["jiang_ffn_down"]["lr"] == pytest.approx(0.01 / 3.0 / 16.0)
    assert groups["jiang_other_biases"]["lr"] == pytest.approx(0.01)
    assert groups["jiang_embeddings"]["eps"] == pytest.approx(0.5e-12)
    assert groups["jiang_norms"]["eps"] == pytest.approx(1e-12)
    assert groups["jiang_final_norm"]["eps"] == pytest.approx(0.5e-12)
    assert groups["jiang_attention_qkv"]["eps"] == pytest.approx(0.25e-12)
    assert groups["jiang_attention_output"]["eps"] == pytest.approx(0.25e-12)
    assert groups["jiang_ffn_up"]["eps"] == pytest.approx(1e-12 / 6.0)
    assert groups["jiang_ffn_down"]["eps"] == pytest.approx(1e-12 / 9.0)
    assert groups["jiang_other_biases"]["eps"] == pytest.approx(0.5e-12)

    all_ids = [id(parameter) for group in groups.values() for parameter in group["params"]]
    assert len(all_ids) == len(set(all_ids))
    assert set(all_ids) == {id(parameter) for parameter in model.parameters()}


def test_reference_tuned_group_multiplier_changes_only_named_group():
    model = make_model()
    baseline = {
        group["name"]: group["lr"]
        for group in model.optimizer_parameter_groups(0.01, epsilon0=1e-12)
    }
    tuned = {
        group["name"]: group["lr"]
        for group in model.optimizer_parameter_groups(
            0.01,
            epsilon0=1e-12,
            learning_rate_multipliers={
                "jiang_attention_qkv": (
                    JIANG_DENSE_REPORTED_LR_MULTIPLIERS["jiang_attention_qkv"]
                    / 2.0
                )
            },
        )
    }
    assert tuned["jiang_attention_qkv"] == pytest.approx(
        0.5 * baseline["jiang_attention_qkv"]
    )
    for name, rate in baseline.items():
        if name != "jiang_attention_qkv":
            assert tuned[name] == rate


def test_completep_adamw_decay_scales_rectangular_hidden_groups() -> None:
    shape = JiangChizatShape(
        depth=4,
        hidden_width=128,
        residual_width=32,
        head_dimension=8,
    )
    model = make_model(shape)
    groups = {
        str(group["name"]): group
        for group in model.optimizer_parameter_groups(
            0.01,
            epsilon0=1e-12,
            weight_decay0=0.1,
            optimizer_name="adamw",
        )
    }
    assert groups["jiang_embeddings"]["weight_decay"] == pytest.approx(0.1)
    assert groups["jiang_attention_qkv"]["weight_decay"] == pytest.approx(0.2)
    assert groups["jiang_attention_output"]["weight_decay"] == pytest.approx(0.2)
    assert groups["jiang_ffn_up"]["weight_decay"] == pytest.approx(0.2)
    assert groups["jiang_ffn_down"]["weight_decay"] == pytest.approx(0.4)
    assert groups["jiang_norms"]["weight_decay"] == 0.0
    assert groups["jiang_final_norm"]["weight_decay"] == 0.0
    assert groups["jiang_other_biases"]["weight_decay"] == 0.0
    assert {
        group["theory_contract_id"] for group in groups.values()
    } == {JIANG_COMPLETEP_ADAMW_THEORY.contract_id}

    audit = model.optimizer_contract_audit(
        0.01,
        epsilon0=1e-12,
        weight_decay0=0.1,
        optimizer_name="adamw",
    )
    assert audit["complete"] is True
    assert audit["disjoint"] is True
    assert audit["theory"]["optimizer"] == "adamw"


def test_rho32_endpoint_uses_reference_relative_group_rules() -> None:
    eta = 0.03
    model = JiangChizatTransformer(
        JiangChizatShape(
            depth=8,
            hidden_width=448,
            residual_width=112,
            head_dimension=8,
        ),
        vocab_size=16,
        context_length=8,
        reference=JiangChizatReference(
            depth=2,
            hidden_width=128,
            residual_width=8,
        ),
        capture_attention_diagnostics=False,
    )
    groups = {
        str(group["name"]): group
        for group in model.optimizer_parameter_groups(
            eta,
            epsilon0=1e-12,
            optimizer_name="adamw",
        )
    }
    assert model.shape.rho == pytest.approx(32.0)
    assert groups["jiang_embeddings"]["lr"] == pytest.approx(eta)
    assert groups["jiang_norms"]["lr"] == pytest.approx(eta)
    assert groups["jiang_final_norm"]["lr"] == pytest.approx(eta)
    assert groups["jiang_attention_qkv"]["lr"] == pytest.approx(eta / 14 / 16)
    assert groups["jiang_attention_output"]["lr"] == pytest.approx(eta / 14)
    assert groups["jiang_ffn_up"]["lr"] == pytest.approx(eta / 14)
    assert groups["jiang_ffn_down"]["lr"] == pytest.approx(eta / 3.5 / 16)
    assert groups["jiang_other_biases"]["lr"] == pytest.approx(eta)
    assert groups["jiang_attention_qkv"]["eps"] == pytest.approx(1e-12 / 56)
    assert groups["jiang_final_norm"]["eps"] == pytest.approx(1e-12 / 14)
    assert groups["jiang_ffn_down"]["eps"] == pytest.approx(
        1e-12 * 14 / (3.5**2) / 4
    )
    audit = model.optimizer_contract_audit(
        eta,
        epsilon0=1e-12,
        optimizer_name="adamw",
    )
    assert audit["complete"] is True
    assert audit["disjoint"] is True
    assert audit["trainable_parameter_count"] == sum(
        parameter.numel() for parameter in model.parameters()
    )
    final_norm_group = next(
        row for row in audit["groups"] if row["name"] == "jiang_final_norm"
    )
    assert final_norm_group["parameter_names"] == [
        "final_norm.weight",
        "final_norm.bias",
    ]
    block_norm_group = next(
        row for row in audit["groups"] if row["name"] == "jiang_norms"
    )
    assert all(not name.startswith("final_norm.") for name in block_norm_group["parameter_names"])


def test_omitted_factor_controls_change_only_the_declared_groups():
    shape = JiangChizatShape(depth=4, hidden_width=96, residual_width=32, head_dimension=8)
    model = make_model(shape)
    correct = {
        group["name"]: group["lr"]
        for group in model.optimizer_parameter_groups(0.01, epsilon0=1e-12)
    }
    wrong = {
        group["name"]: group["lr"]
        for group in model.optimizer_parameter_groups(
            0.01,
            epsilon0=1e-12,
            omit_attention_width_factor=True,
            omit_ffn_hidden_width_factor=True,
        )
    }
    assert wrong["jiang_attention_qkv"] == pytest.approx(0.01 / 16.0)
    assert wrong["jiang_attention_output"] == pytest.approx(0.01)
    assert wrong["jiang_ffn_down"] == pytest.approx(0.01 / 16.0)
    assert wrong["jiang_ffn_up"] == correct["jiang_ffn_up"]
    assert wrong["jiang_embeddings"] == correct["jiang_embeddings"]


def test_reported_value_and_down_initialization_constants_are_applied():
    shape = JiangChizatShape(
        depth=1, hidden_width=1024, residual_width=256, head_dimension=64
    )
    model = make_model(shape)
    q_weight, k_weight, v_weight = model.blocks[0].attention.qkv.weight.chunk(3, dim=0)
    base_std = shape.residual_width ** -0.5
    assert float(q_weight.detach().std()) == pytest.approx(base_std, rel=0.03)
    assert float(k_weight.detach().std()) == pytest.approx(base_std, rel=0.03)
    assert float(v_weight.detach().std()) == pytest.approx(base_std / 16.0, rel=0.03)
    assert float(model.blocks[0].ffn_down.weight.detach().std()) == pytest.approx(
        math.sqrt(shape.residual_width) / shape.hidden_width / 4.0,
        rel=0.03,
    )


def test_one_adam_step_is_finite():
    model = make_model().train()
    optimizer = model.make_optimizer(0.001)
    tokens = torch.randint(0, 16, (4, 8), generator=torch.Generator().manual_seed(3))
    targets = tokens.roll(-1, dims=1)
    initial = F.cross_entropy(model(tokens).reshape(-1, 16), targets.reshape(-1))
    optimizer.zero_grad(set_to_none=True)
    initial.backward()
    optimizer.step()
    final = F.cross_entropy(model(tokens).reshape(-1, 16), targets.reshape(-1))
    assert torch.isfinite(initial)
    assert torch.isfinite(final)


def test_accelerated_sdpa_matches_reference_outputs_and_gradients():
    reference = make_model(
        attention_backend="math", capture_attention_diagnostics=False
    ).train()
    accelerated = make_model(
        attention_backend="auto", capture_attention_diagnostics=False
    ).train()
    accelerated.load_state_dict(reference.state_dict())
    tokens = torch.randint(0, 16, (2, 8), generator=torch.Generator().manual_seed(19))
    targets = tokens.roll(-1, dims=1)
    reference_logits = reference(tokens)
    accelerated_logits = accelerated(tokens)
    torch.testing.assert_close(accelerated_logits, reference_logits, rtol=1e-5, atol=1e-6)
    F.cross_entropy(reference_logits.reshape(-1, 16), targets.reshape(-1)).backward()
    F.cross_entropy(accelerated_logits.reshape(-1, 16), targets.reshape(-1)).backward()
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
