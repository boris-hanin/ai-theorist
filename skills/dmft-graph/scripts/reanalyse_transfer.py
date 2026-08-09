"""Re-score a saved transfer artifact with the current paired verdict rule."""

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "skills", "dmft-derivation", "scripts"))

from transfer import verdict_from_optima  # noqa: E402


def safe(value):
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    with open(args.input, encoding="utf-8") as handle:
        source = json.load(handle)
    result = {"source": os.path.basename(args.input),
              "rule": "paired common-random-number SEM (F20)", "legs": {}}
    for name, row in source.items():
        if not isinstance(row, dict) or "per_seed_log10_lr" not in row:
            continue
        scored = verdict_from_optima(row["per_seed_log10_lr"], row["interior"],
                                     row["dial"])
        result["legs"][name] = safe(scored)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1, allow_nan=False)


if __name__ == "__main__":
    main()
