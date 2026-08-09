import math

import pytest
import torch
from torch.nn import functional as F

from ai_theorist.autoscaler.jiang_chizat import (
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


def test_mean_field_down_projection_and_fan_in_control_are_distinct():
    shape = JiangChizatShape(
        depth=1, hidden_width=1024, residual_width=256, head_dimension=64
    )
    mean_field = make_model(shape, down_initialization="mean_field")
    fan_in = make_model(shape, down_initialization="fan_in")
    expected_mean_field = math.sqrt(shape.residual_width) / shape.hidden_width
    expected_fan_in = 1.0 / math.sqrt(shape.hidden_width)
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
    assert groups["jiang_attention"]["lr"] == pytest.approx(0.005)
    assert groups["jiang_ffn_up"]["lr"] == pytest.approx(0.005)
    assert groups["jiang_ffn_down"]["lr"] == pytest.approx(0.01 / 3.0)
    assert groups["jiang_embeddings"]["eps"] == pytest.approx(0.5e-12)
    assert groups["jiang_norms"]["eps"] == pytest.approx(1e-12)
    assert groups["jiang_attention"]["eps"] == pytest.approx(0.25e-12)
    assert groups["jiang_ffn_up"]["eps"] == pytest.approx(1e-12 / 6.0)
    assert groups["jiang_ffn_down"]["eps"] == pytest.approx(1e-12 / 9.0)

    all_ids = [id(parameter) for group in groups.values() for parameter in group["params"]]
    assert len(all_ids) == len(set(all_ids))
    assert set(all_ids) == {id(parameter) for parameter in model.parameters()}


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
    assert wrong["jiang_attention"] == pytest.approx(0.01)
    assert wrong["jiang_ffn_down"] == pytest.approx(0.01)
    assert wrong["jiang_ffn_up"] == correct["jiang_ffn_up"]
    assert wrong["jiang_embeddings"] == correct["jiang_embeddings"]


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
