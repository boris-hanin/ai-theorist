"""Fixed-horizon architecture tuning, transfer, and scaling-law tools.

Public objects are loaded lazily so lightweight parameterization utilities do
not import SciPy-backed scaling code at module-import time.  This matters on
training workers that only execute the PyTorch transfer harness.
"""

from typing import Any


__all__ = ["StudySpec", "default_study_spec", "run_study"]


def __getattr__(name: str) -> Any:
    if name in {"StudySpec", "default_study_spec"}:
        from .schema import StudySpec, default_study_spec

        return {"StudySpec": StudySpec, "default_study_spec": default_study_spec}[name]
    if name == "run_study":
        from .study import run_study

        return run_study
    raise AttributeError(name)
