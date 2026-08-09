import numpy as np
import pytest

from ai_theorist.autoscaler.scaling import fit_scaling_law


def test_scaling_law_recovers_synthetic_floor_and_exponent():
    compute = np.geomspace(1e4, 1e9, 9)
    floor, amplitude, exponent = 0.72, 9.0, 0.23
    losses = floor + amplitude * compute ** (-exponent)
    fit = fit_scaling_law(compute, losses, np.full_like(losses, 1e-3), bootstrap_samples=40)
    assert fit.loss_floor == pytest.approx(floor, rel=0.02)
    assert fit.exponent == pytest.approx(exponent, rel=0.08)
    assert fit.r_squared > 0.999
    assert fit.forecastable, fit.refusal_reasons
    assert fit.predict(1e10) == pytest.approx(floor + amplitude * 1e10 ** (-exponent), rel=0.02)


def test_scaling_law_refuses_flat_data():
    compute = np.geomspace(1e4, 1e8, 6)
    losses = np.array([1.0, 1.001, 0.999, 1.0, 1.0005, 0.9995])
    fit = fit_scaling_law(compute, losses, np.full_like(losses, 0.005), bootstrap_samples=0)
    assert not fit.forecastable
    assert any("decreasing" in reason or "too small" in reason for reason in fit.refusal_reasons)


def test_boundary_floor_allows_only_short_range_calibration():
    compute = np.geomspace(1e4, 1e8, 5)
    losses = np.array([0.2, 0.08, 0.055, 0.05, 0.057])
    fit = fit_scaling_law(compute, losses, np.full_like(losses, 0.003), bootstrap_samples=0)
    assert not fit.forecastable
    assert fit.short_range_forecastable
    assert not fit.asymptotic_floor_identifiable
    assert fit.model_kind == "boundary_floor_power_law"
    assert fit.refusal_reasons == ("estimated loss floor is pinned to the smallest observation",)


def test_scaling_law_rejects_invalid_compute_order():
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_scaling_law([1, 3, 2, 4], [4, 3, 2, 1], bootstrap_samples=0)
