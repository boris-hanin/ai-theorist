"""Reproducible B2 audit of the proposed shared-error susceptibility mechanism.

The historical B2 table was committed only as prose.  This runner restores a
fully specified paired finite-difference experiment; new output must not be
presented as the missing historical raw data.
"""

import argparse
import json
import os

import numpy as np
import torch

from diag_sqrtD import make_data, run, slope


def atomic_dump(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=1, allow_nan=False)
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="susceptibility_out.json")
    parser.add_argument("--seeds", type=int, default=256)
    parser.add_argument("--relative-perturbation", type=float, default=0.05)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    widths = [32, 64] if args.smoke else [32, 64, 128, 256, 512, 1024]
    horizons = [0, 2] if args.smoke else [0, 8, 32, 64, 128]
    seeds = min(args.seeds, 4) if args.smoke else args.seeds
    X, Y = make_data(args.device)
    epsilon = args.relative_perturbation
    Y_perturbed = Y * (1.0 + epsilon)
    perturbation_norm = float((Y_perturbed - Y).pow(2).mean().sqrt())
    out = {
        "schema_version": 1,
        "status": "new reproducible rerun; not historical B2 raw data",
        "definition": "mean(abs(K[Y*(1+eps)]-K[Y])) / rms(eps*Y)",
        "relative_perturbation": epsilon,
        "seeds": seeds,
        "smoke": args.smoke,
        "widths": widths,
        "horizons": {},
    }

    for horizon in horizons:
        means, sems, raw = [], [], []
        for D in widths:
            base, _, _ = run(D, seeds, horizon, device=args.device, X=X, Y=Y)
            pert, _, _ = run(D, seeds, horizon, device=args.device,
                             X=X, Y=Y_perturbed)
            per_seed = np.abs(pert - base) / max(perturbation_norm, 1e-300)
            means.append(float(per_seed.mean()))
            sems.append(float(per_seed.std(ddof=1) / np.sqrt(seeds)))
            raw.append(per_seed.tolist())
        positive = np.asarray(means) > 0
        beta = slope(np.asarray(widths)[positive], np.asarray(means)[positive]) \
            if positive.sum() >= 2 else None
        out["horizons"][str(horizon)] = {
            "chi_mean": means,
            "chi_sem": sems,
            "chi_per_seed": raw,
            "beta": beta,
        }
        atomic_dump(args.output, out)
        print("horizon %3d beta %s" %
              (horizon, "undefined" if beta is None else "%+.4f" % beta), flush=True)


if __name__ == "__main__":
    main()
