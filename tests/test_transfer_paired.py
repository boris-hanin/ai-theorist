import numpy as np

from transfer import verdict_from_optima


def test_common_mode_seed_noise_uses_paired_sem():
    common = np.array([-2.0, -1.0, -0.2, 0.4, 1.2, 2.0])
    shifts = np.array([0.0, 0.1, 0.2, 0.4])
    optima = shifts[:, None] + common[None, :]
    result = verdict_from_optima(optima, np.ones(4, dtype=bool), [1, 2, 4, 8])

    assert result["unpaired_sem_log10"] > 0.25
    assert result["paired_sem_log10"] < 1e-12
    assert result["status"] == "FAILS"


def test_grid_edge_remains_underpowered():
    optima = np.tile(np.array([0.0, 0.1, 0.2]), (4, 1))
    result = verdict_from_optima(optima, [True, False, True, True], [1, 2, 4, 8])
    assert result["status"].startswith("UNDER-POWERED (optimum on grid edge)")
