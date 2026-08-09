"""Strict, atomic JSON output for numerical artifacts."""

import json
import math
import os
from numbers import Real


def safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if hasattr(value, "tolist"):
        return safe(value.tolist())
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def dump(value, path, indent=1):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(safe(value), handle, indent=indent, allow_nan=False)
    os.replace(temporary, path)
