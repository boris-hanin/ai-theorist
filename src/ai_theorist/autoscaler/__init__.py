"""Fixed-horizon architecture tuning, transfer, and scaling-law tools.

Public objects are loaded lazily so lightweight parameterization utilities do
not import SciPy-backed scaling code at module-import time.  This matters on
training workers that only execute the PyTorch transfer harness.
"""

from typing import Any


__all__ = [
    "StudySpec",
    "default_study_spec",
    "run_study",
    "run_quadratic_calibration",
    "run_transformer_batch_census",
    "run_constant_tpp_campaign",
    "run_standard_pretraining_batch_census",
]


def __getattr__(name: str) -> Any:
    if name in {"StudySpec", "default_study_spec"}:
        from .schema import StudySpec, default_study_spec

        return {"StudySpec": StudySpec, "default_study_spec": default_study_spec}[name]
    if name == "run_study":
        from .study import run_study

        return run_study
    if name in {
        "run_quadratic_calibration",
        "run_transformer_batch_census",
        "run_constant_tpp_campaign",
        "run_standard_pretraining_batch_census",
    }:
        from .batch_campaigns import (
            run_constant_tpp_campaign,
            run_quadratic_calibration,
            run_transformer_batch_census,
        )
        from .pretraining import run_standard_pretraining_batch_census

        return {
            "run_quadratic_calibration": run_quadratic_calibration,
            "run_transformer_batch_census": run_transformer_batch_census,
            "run_constant_tpp_campaign": run_constant_tpp_campaign,
            "run_standard_pretraining_batch_census": run_standard_pretraining_batch_census,
        }[name]
    raise AttributeError(name)
