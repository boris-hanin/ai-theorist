import numpy as np
import pytest

from ai_theorist.autoscaler.scaling import fit_scaling_ensemble, fit_scaling_law


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


def test_scaling_ensemble_requires_family_agreement_and_rolling_backtests():
    sizes = np.geomspace(1e6, 1e9, 8)
    losses = 5.0 * sizes ** -0.12
    fit = fit_scaling_ensemble(
        sizes,
        losses,
        np.full_like(losses, 1e-4),
        target_size=2e9,
        maximum_family_spread=0.2,
        maximum_backtest_relative_error=0.2,
        bootstrap_samples=20,
    )
    assert fit["certified"], fit["refusal_reasons"]
    assert fit["prediction"] == pytest.approx(5.0 * (2e9) ** -0.12, rel=0.02)
    assert fit["qualified_family_count"] == 3
    assert len(fit["rolling_backtests"]) == 4
    assert fit["prediction_interval_95"][0] <= fit["prediction"]
    assert fit["prediction_interval_95"][1] >= fit["prediction"]


def test_scaling_ensemble_refuses_distant_target_even_with_perfect_curve():
    sizes = np.geomspace(1e6, 1e9, 8)
    losses = 5.0 * sizes ** -0.12
    fit = fit_scaling_ensemble(
        sizes,
        losses,
        np.zeros_like(losses),
        target_size=20e9,
        maximum_extrapolation_factor=10.0,
        bootstrap_samples=0,
    )
    assert not fit["certified"]
    assert fit["prediction"] is None
    assert any("extrapolation factor" in reason for reason in fit["refusal_reasons"])
