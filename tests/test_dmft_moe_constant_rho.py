from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


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
