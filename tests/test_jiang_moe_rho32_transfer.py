from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path("scripts/evaluate_jiang_moe_rho32_transfer.py")
SPEC = importlib.util.spec_from_file_location("evaluate_jiang_moe_rho32_transfer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _plan():
    scales = []
    for index, (L, D, M, parameters) in enumerate(
        (
            (2, 128, 2048, 1_187_336),
            (4, 256, 2048, 5_264_912),
            (8, 512, 2048, 25_236_512),
            (16, 1024, 2048, 134_465_600),
        )
    ):
        scales.append(
            {
                "name": f"S{index}",
                "depth": L,
                "width": D,
                "hidden_width": M,
                "num_experts": 4,
                "active_experts": 1,
                "rho_lm_over_d": 32.0,
                "active_non_embedding_parameters": parameters,
            }
        )
    return {
        "seeds": [11, 29, 47],
        "scales": scales,
        "architecture_contract": {"reference_scale_index": 1},
    }


def _record(scale, seed, *, mode="theory", final=9.0):
    return {
        "run_id": f"{scale['name']}-{seed}-{mode}",
        "optimizer": {"learning_rate": 0.08838834764831845},
        "seed": seed,
        "final_validation_loss": final,
        "validation_checkpoints": [
            {"step": 0, "tokens": 0, "validation_loss": 10.0},
            {"step": 100, "tokens": 1000, "validation_loss": 9.5},
            {"step": 200, "tokens": 2000, "validation_loss": final},
        ],
        "metadata": {
            "scale": scale,
            "optimizer_mode": mode,
            "optimizer_group_audit": {
                "complete": True,
                "disjoint": True,
                "groups": [{} for _ in range(8)],
            },
            "diagnostics": {
                "maximum_absolute_load_deviation": 0.05,
                "maximum_absolute_expert_bias": 0.01,
                "routing_token_counts": [100 for _ in range(scale["depth"])],
            },
        },
    }


def test_fixed_eta_transfer_accepts_flat_progress_and_biting_control() -> None:
    plan = _plan()
    records = [
        _record(scale, seed)
        for scale in plan["scales"]
        for seed in plan["seeds"]
    ]
    records.extend(
        _record(plan["scales"][-1], seed, mode="wrong_global", final=9.2)
        for seed in plan["seeds"]
    )
    result = MODULE.evaluate(
        plan,
        {
            "selected_learning_rate": 0.08838834764831845,
            "learning_rate_optimum_is_interior": True,
        },
        records,
    )
    assert result["accepted"] is True
    assert result["log_progress_vs_log_active_nonembedding_parameter_slope"] == 0.0
    assert result["negative_control"]["passed"] is True


def test_fixed_eta_transfer_rejects_a_nonbiting_control() -> None:
    plan = _plan()
    records = [
        _record(scale, seed)
        for scale in plan["scales"]
        for seed in plan["seeds"]
    ]
    records.extend(
        _record(plan["scales"][-1], seed, mode="wrong_global", final=9.0)
        for seed in plan["seeds"]
    )
    result = MODULE.evaluate(
        plan,
        {
            "selected_learning_rate": 0.08838834764831845,
            "learning_rate_optimum_is_interior": True,
        },
        records,
    )
    assert result["accepted"] is False
    assert result["gates"]["wrong_global_lr_control_is_worse"] is False
