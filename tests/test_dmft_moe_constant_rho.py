from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from ai_theorist.autoscaler.jiang_moe import (
    JiangMoEReference,
    JiangMoEShape,
    JiangMoETransformer,
)


SCRIPT = Path("skills/dmft-moe/scripts/constant_rho_compatibility.py")
SPEC = importlib.util.spec_from_file_location("constant_rho_compatibility", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
Shape = MODULE.Shape
derive = MODULE.derive


def _shapes():
    return (
        Shape("l2", 2, 256, 16, 4, 1),
        Shape("l4", 4, 256, 32, 4, 1),
        Shape("l8", 8, 256, 64, 4, 1),
        Shape("l16", 16, 256, 128, 4, 1),
    )


def test_constant_rho_makes_residualized_down_init_and_lr_width_only() -> None:
    shapes = _shapes()
    result = derive(shapes[0], shapes)
    assert result["verdict"]["constant_rho_is_parameterisation_compatible"] is True
    assert result["checks"]["effective_down_init_identity_max_relative_error"] < 1e-12
    assert result["checks"]["effective_down_lr_identity_max_relative_error"] < 1e-12
    for row in result["rows"]:
        rD = row["ratios_to_reference"]["D"]
        assert row["initialization_ratios"]["expert_down_after_residual"] == pytest.approx(
            rD**-0.5
        )
        assert row["learning_rate_ratios"]["expert_down_after_residual"] == pytest.approx(
            rD**-1.0
        )


def test_constant_rho_fixed_e_preserves_alpha_star_and_stream_variance() -> None:
    shapes = _shapes()
    result = derive(shapes[0], shapes)
    assert result["checks"]["alpha_star_spread_fraction"] == pytest.approx(0.0)
    assert result["checks"]["stream_init_variance_proxy_spread_fraction"] == pytest.approx(0.0)
    assert [row["alpha_star"] for row in result["rows"]] == pytest.approx([1 / 128] * 4)
    assert [row["stream_init_variance_proxy"] for row in result["rows"]] == pytest.approx(
        [1 / 32] * 4
    )


def test_sqrt_d_over_lm_is_not_a_raw_weight_scale() -> None:
    shapes = _shapes()
    result = derive(shapes[0], shapes)
    assert result["checks"]["constant_rho_stream_variance_slope_in_L"] == pytest.approx(0.0)
    assert result["checks"]["double_depth_stream_variance_slope_in_L"] == pytest.approx(-2.0)
    last = result["rows"][-1]
    rL = last["ratios_to_reference"]["L"]
    correct = last["initialization_ratios"]["expert_down_after_residual"]
    wrong = last["initialization_ratios"]["double_depth_after_residual"]
    assert wrong / correct == pytest.approx(1 / rL)


def test_constant_rho_does_not_keep_expert_ratio_fixed() -> None:
    rows = derive(_shapes()[0], _shapes())["rows"]
    assert [row["alpha_ffn"] for row in rows] == [16.0, 8.0, 4.0, 2.0]
    assert rows[-1]["alpha_ffn_regime"] == "coherent_term_dominant_but_crossover_visible"


def test_audit_matches_the_actual_source_faithful_model_contract() -> None:
    shapes = _shapes()
    rows = derive(shapes[0], shapes)["rows"]
    reference = JiangMoEReference(
        depth=shapes[0].L,
        residual_width=shapes[0].D,
        expert_width=shapes[0].M,
        num_experts=shapes[0].E,
        active_experts=shapes[0].A,
    )

    def model_for(shape):
        return JiangMoETransformer(
            JiangMoEShape(
                depth=shape.L,
                residual_width=shape.D,
                expert_width=shape.M,
                head_dimension=8,
                num_experts=shape.E,
                active_experts=shape.A,
            ),
            vocab_size=8,
            context_length=2,
            reference=reference,
            capture_attention_diagnostics=False,
        )

    base = model_for(shapes[0])
    base_init = base.initialization_contract()
    base_groups = {
        group["name"]: group
        for group in base.optimizer_parameter_groups(0.01, epsilon0=1e-12)
    }
    init_mapping = {
        "embedding_and_unembedding_std": "embedding",
        "attention_qko_std": "attention_qko",
        "attention_value_std": "attention_value",
        "router_std": "router_gamma1",
        "expert_up_std": "expert_up",
        "expert_down_std": "expert_down_raw",
    }
    group_mapping = {
        "jiang_moe_embeddings": "embeddings",
        "jiang_moe_norms": "norms",
        "jiang_moe_attention_qkv": "attention_qkv",
        "jiang_moe_attention_output": "attention_output",
        "jiang_moe_router": "router",
        "jiang_moe_expert_up": "expert_up",
        "jiang_moe_expert_down": "expert_down",
        "jiang_moe_other_biases": "other_biases",
    }
    for shape, row in zip(shapes, rows):
        model = model_for(shape)
        initialization = model.initialization_contract()
        for contract_name, audit_name in init_mapping.items():
            assert initialization[contract_name] / base_init[contract_name] == pytest.approx(
                row["initialization_ratios"][audit_name]
            )
        assert model.residual_width_ratio**-1 == pytest.approx(
            row["initialization_ratios"]["effective_tied_unembedding"]
        )
        groups = {
            group["name"]: group
            for group in model.optimizer_parameter_groups(0.01, epsilon0=1e-12)
        }
        for contract_name, audit_name in group_mapping.items():
            assert groups[contract_name]["lr"] / base_groups[contract_name]["lr"] == pytest.approx(
                row["learning_rate_ratios"][audit_name]
            )
            assert groups[contract_name]["eps"] / base_groups[contract_name]["eps"] == pytest.approx(
                row["adam_epsilon_ratios"][audit_name]
            )
