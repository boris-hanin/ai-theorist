import math

import pytest
import torch
from torch.nn import functional as F

from ai_theorist.autoscaler.jiang_moe import (
    JIANG_MOE_REPORTED_LR_MULTIPLIERS,
    JiangMoEReference,
    JiangMoEShape,
    JiangMoETransformer,
)


REFERENCE = JiangMoEReference(
    depth=2,
    residual_width=16,
    expert_width=32,
    num_experts=4,
    active_experts=1,
)


def make_model(shape=None):
    torch.manual_seed(7)
    return JiangMoETransformer(
        shape
        or JiangMoEShape(
            depth=2,
            residual_width=16,
            expert_width=32,
            head_dimension=4,
            num_experts=4,
            active_experts=1,
        ),
        vocab_size=16,
        context_length=8,
        reference=REFERENCE,
    )


def test_moe_requires_fixed_sparsity_and_valid_head_geometry():
    with pytest.raises(ValueError, match="divisible"):
        JiangMoEShape(2, 18, 32, 4, 4, 1)
    with pytest.raises(ValueError, match="fixed active-expert fraction"):
        make_model(JiangMoEShape(2, 16, 32, 4, 8, 1))


def test_table2_groups_cover_all_trainable_parameters_and_scale_lr_and_epsilon():
    shape = JiangMoEShape(
        depth=4,
        residual_width=32,
        expert_width=96,
        head_dimension=8,
        num_experts=8,
        active_experts=2,
    )
    model = make_model(shape)
    groups = {
        group["name"]: group
        for group in model.optimizer_parameter_groups(0.01, epsilon0=1e-12)
    }
    assert groups["jiang_moe_embeddings"]["lr"] == pytest.approx(0.01)
    assert groups["jiang_moe_attention_qkv"]["lr"] == pytest.approx(0.005 / 16.0)
    assert groups["jiang_moe_attention_output"]["lr"] == pytest.approx(0.005)
    assert groups["jiang_moe_router"]["lr"] == pytest.approx(0.005 / 16.0)
    assert groups["jiang_moe_expert_up"]["lr"] == pytest.approx(0.005)
    assert groups["jiang_moe_expert_down"]["lr"] == pytest.approx(0.01 / 3.0 / 16.0)
    assert groups["jiang_moe_other_biases"]["lr"] == pytest.approx(0.01)
    assert groups["jiang_moe_router"]["eps"] == pytest.approx(0.25e-12)
    assert groups["jiang_moe_expert_up"]["eps"] == pytest.approx(1e-12 / 6.0)
    assert groups["jiang_moe_expert_down"]["eps"] == pytest.approx(1e-12 / 9.0)
    assert groups["jiang_moe_other_biases"]["eps"] == pytest.approx(0.5e-12)
    audit = model.optimizer_contract_audit(0.01, epsilon0=1e-12)
    assert audit["complete"] is True
    assert audit["disjoint"] is True


def test_reference_tuned_group_multiplier_changes_only_its_group_rate():
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
                "jiang_moe_router": (
                    JIANG_MOE_REPORTED_LR_MULTIPLIERS["jiang_moe_router"] / 2.0
                )
            },
        )
    }
    assert tuned["jiang_moe_router"] == pytest.approx(
        0.5 * baseline["jiang_moe_router"]
    )
    for name, rate in baseline.items():
        if name != "jiang_moe_router":
            assert tuned[name] == rate


def test_expert_initialization_and_manual_bias_rule_match_table2():
    shape = JiangMoEShape(2, 32, 128, 8, 8, 2)
    model = make_model(shape)
    expert = model.blocks[0].moe.experts[0]
    assert float(expert.up.weight.detach().std()) == pytest.approx(32 ** -0.5, rel=0.12)
    assert float(expert.down.weight.detach().std()) == pytest.approx(
        math.sqrt(32) / 128 / 4.0, rel=0.12
    )
    q_weight, k_weight, v_weight = model.blocks[0].attention.qkv.weight.chunk(3, dim=0)
    assert float(q_weight.detach().std()) == pytest.approx(32 ** -0.5, rel=0.12)
    assert float(k_weight.detach().std()) == pytest.approx(32 ** -0.5, rel=0.12)
    assert float(v_weight.detach().std()) == pytest.approx(32 ** -0.5 / 16.0, rel=0.12)
    moe = model.blocks[0].moe
    with torch.no_grad():
        moe.last_load.copy_(torch.tensor([0.5, 0.25, 0.0, 0.25, 0.5, 0.25, 0.0, 0.25]))
        before = moe.expert_bias.clone()
        moe.update_expert_bias(0.1)
    assert torch.equal(
        moe.expert_bias,
        before - 0.1 * (moe.last_load - shape.sparsity),
    )
    contract = model.manual_parameter_contract(0.1)
    assert contract["learning_rate_formula"].startswith("eta_bias")


def test_moe_forward_is_causal_and_one_full_update_is_finite():
    model = make_model().train()
    tokens = torch.arange(8).remainder(16)[None, :].repeat(2, 1)
    changed = tokens.clone()
    changed[:, -1] = 13
    with torch.no_grad():
        original = model(tokens)
        changed_logits = model(changed)
    assert torch.equal(original[:, :-1], changed_logits[:, :-1])
    groups = model.optimizer_parameter_groups(1e-3, epsilon0=1e-12)
    optimizer = torch.optim.Adam(groups, betas=(0.9, 0.95), weight_decay=0.0)
    targets = tokens.roll(-1, dims=1)
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(tokens).reshape(-1, 16), targets.reshape(-1))
    loss.backward()
    optimizer.step()
    model.update_expert_biases(0.01)
    assert torch.isfinite(loss)
    diagnostics = model.routing_diagnostics()
    assert diagnostics["active_expert_fraction"] == pytest.approx(0.25)
