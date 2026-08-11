from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]


def _selection(plan: str, tau: float | None, losses: list[float]) -> dict:
    rates = [0.001, 0.003, 0.01]
    grid = []
    for index, rate in enumerate(rates):
        current = [value + abs(index - 1) * 0.2 for value in losses]
        grid.append(
            {
                "learning_rate": rate,
                "weight_decay_tau_ema": tau,
                "seed_losses": current,
                "mean_validation_loss": sum(current) / len(current),
                "sem_validation_loss": 0.01,
            }
        )
    return {
        "plan_fingerprint": plan,
        "selected_learning_rate": 0.003,
        "selected_weight_decay_tau_ema": tau,
        "grid": grid,
    }


def test_prepare_rho32_zero_decay_preserves_every_non_decay_field(tmp_path: Path) -> None:
    source = json.loads(
        (ROOT / "configs/autoscaler/jiang_mistral_100m_rho32_adamw_tau_ema.json")
        .read_text()
    )
    source_path = tmp_path / "finite.json"
    output = tmp_path / "zero.json"
    source_path.write_text(json.dumps(source))
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_rho32_zero_decay_config.py"),
            str(source_path),
            "--output",
            str(output),
        ],
        check=True,
    )
    zero = json.loads(output.read_text())
    zero_optimizer = zero.pop("optimizer")
    source_optimizer = source.pop("optimizer")
    zero.pop("campaign_label")
    assert zero == source
    source_optimizer.pop("weight_decay_tau_ema_grid")
    source_optimizer["weight_decay"] = 0.0
    assert zero_optimizer == source_optimizer


def test_rho32_selector_can_choose_exact_zero_with_interior_eta(tmp_path: Path) -> None:
    finite_path = tmp_path / "finite-selection.json"
    expanded_path = tmp_path / "expanded-selection.json"
    zero_path = tmp_path / "zero-selection.json"
    prereg_path = tmp_path / "preregistration.json"
    output = tmp_path / "decision.json"
    finite_path.write_text(
        json.dumps(_selection("f" * 64, 0.5628, [4.2, 4.21, 4.19]))
    )
    expanded_path.write_text(
        json.dumps(_selection("e" * 64, 2.2512, [4.1, 4.11, 4.09]))
    )
    zero_path.write_text(
        json.dumps(_selection("z" * 64, None, [4.0, 4.01, 3.99]))
    )
    prereg_path.write_text(
        json.dumps(
            {
                "status": "preregistered_rho32_correction",
                "plans": {
                    "jiang_base_finite_tau": {"plan_fingerprint": "f" * 64},
                    "jiang_expanded_finite_tau": {
                        "plan_fingerprint": "e" * 64
                    },
                    "jiang_zero": {"plan_fingerprint": "z" * 64},
                },
                "gates": {"rho32": True},
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/select_rho32_weight_decay.py"),
            str(prereg_path),
            str(finite_path),
            str(expanded_path),
            str(zero_path),
            "--output",
            str(output),
        ],
        check=True,
    )
    decision = json.loads(output.read_text())
    assert decision["status"] == "passed"
    assert decision["selected_source"] == "jiang_zero"
    assert decision["selected_learning_rate"] == 0.003
    assert decision["selected_weight_decay_tau_ema"] is None
    assert decision["optimum_is_interior"] is True
