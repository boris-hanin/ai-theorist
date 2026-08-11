from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SELECTOR = _load("select_adaptive_weight_decay")
PREPARE = _load("prepare_weight_decay_extension_configs")


def _row(source: str, eta: float, tau, loss: float):
    return {
        "source": source,
        "learning_rate": eta,
        "weight_decay_tau_ema": tau,
        "mean_validation_loss": loss,
        "sem_validation_loss": 0.01,
        "seed_losses": [loss - 0.01, loss, loss + 0.01],
    }


def test_adaptive_decision_can_select_exact_zero_decay() -> None:
    rows = [
        _row("original_jiang", 0.01, 0.5628, 5.6),
        _row("expanded_jiang", 0.03, 0.5628, 5.46),
        _row("expanded_jiang", 0.03, 1.1256, 5.43),
        _row("expanded_jiang", 0.03, 2.2512, 5.42),
        _row("expanded_jiang", 0.03, 4.5024, 5.41),
        _row("expanded_jiang", 0.03, 9.0048, 5.405),
        _row("zero_jiang", 0.03, None, 5.401),
        _row("zero_jiang", 0.06, None, 5.5),
    ]
    decision = SELECTOR._decision(
        rows,
        finite_sources=("original_jiang", "expanded_jiang"),
        zero_source="zero_jiang",
    )
    assert decision["selected_source"] == "zero_jiang"
    assert decision["selected_weight_decay_mode"] == "zero"
    assert decision["selected_weight_decay_tau_ema"] is None
    assert all(decision["gates"].values())


def test_adaptive_decision_refuses_new_finite_boundary() -> None:
    rows = [
        _row("expanded_jiang", 0.01, 0.5628, 5.6),
        _row("expanded_jiang", 0.03, 1.1256, 5.5),
        _row("expanded_jiang", 0.03, 9.0048, 5.3),
        _row("zero_jiang", 0.03, None, 5.4),
        _row("zero_jiang", 0.06, None, 5.5),
    ]
    decision = SELECTOR._decision(
        rows,
        finite_sources=("expanded_jiang",),
        zero_source="zero_jiang",
    )
    assert decision["selected_weight_decay_tau_ema"] == 9.0048
    assert decision["finite_tau_optimum_is_interior"] is False
    assert not all(decision["gates"].values())


def test_overlap_reproducibility_is_seed_matched() -> None:
    left = _row("original_jiang", 0.03, 0.5628, 5.4)
    right = _row("expanded_jiang", 0.03, 0.5628, 5.4)
    unique, overlaps = SELECTOR._deduplicate(
        [left, right], maximum_overlap_delta=0.005
    )
    assert len(unique) == 1
    assert overlaps == [
        {
            "learning_rate": 0.03,
            "weight_decay_tau_ema": 0.5628,
            "sources": ["original_jiang", "expanded_jiang"],
            "maximum_seed_loss_delta": 0.0,
            "passed": True,
        }
    ]
    right["seed_losses"][0] += 0.01
    with pytest.raises(ValueError, match="reproducibility gate"):
        SELECTOR._deduplicate([left, right], maximum_overlap_delta=0.005)


def test_zero_decay_config_is_explicit_adamw_endpoint() -> None:
    config = {
        "optimizer": {
            "name": "adamw",
            "learning_rates": [0.001, 0.003, 0.01],
            "weight_decay_tau_ema_grid": [0.1, 0.2, 0.4],
        }
    }
    zero = PREPARE._zero_decay(config, "test")
    assert zero["optimizer"]["name"] == "adamw"
    assert zero["optimizer"]["weight_decay"] == 0.0
    assert "weight_decay_tau_ema_grid" not in zero["optimizer"]
    assert config["optimizer"]["weight_decay_tau_ema_grid"] == [0.1, 0.2, 0.4]
