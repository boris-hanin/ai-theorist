"""Activation functions and their derivatives.

Every entry is (phi, phidot). Both must be vectorised numpy ufuncs so the
solver and simulator can apply them to whole populations at once.
"""

import numpy as np
from scipy.special import erf as _erf

SQRT2_OVER_PI = np.sqrt(2.0 / np.pi)


def _linear(x):
    return x


def _linear_dot(x):
    return np.ones_like(x)


def _tanh_dot(x):
    t = np.tanh(x)
    return 1.0 - t * t


def _relu(x):
    return np.maximum(x, 0.0)


def _relu_dot(x):
    return (x > 0.0).astype(x.dtype)


def _erf_dot(x):
    return (2.0 / np.sqrt(np.pi)) * np.exp(-x * x)


ACTIVATIONS = {
    "linear": (_linear, _linear_dot),
    "tanh": (np.tanh, _tanh_dot),
    "relu": (_relu, _relu_dot),
    "erf": (_erf, _erf_dot),
}


def get(name):
    """Return (phi, phidot) for a named activation."""
    if name not in ACTIVATIONS:
        raise KeyError(
            "unknown activation %r; available: %s"
            % (name, ", ".join(sorted(ACTIVATIONS)))
        )
    return ACTIVATIONS[name]
