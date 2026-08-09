"""HP-transfer sweeps for the transformer parameterisation of 2405.15712.

Three dials -- dimension-per-head N, head count H, depth L -- each swept against
the base learning rate, with the Table 1 scaling

    eta = eta_0 * N * H * L^{2 aL - 1}

applied. Derived independently in `derivations/03-attention.md` D3, where it
lands on Table 1 exactly.

**Each sweep ships a negative control that removes exactly the factor D3
predicts** -- drop `N`, drop `H`, drop `L^{2 aL - 1}`. If a control still
transfers, the sweep is not sensitive to that factor and the result says
nothing (F17's rule). If it fails while the full scaling transfers, the factor
is doing real work.

Verdicts reuse the machinery from the MLP work: the optimum is located per
seed, drift counts only if it exceeds the across-seed scatter AND 0.3 decades,
and UNDER-POWERED is a distinct outcome from a pass.
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "dmft-derivation", "scripts"))
import transfer as T                                            # noqa: E402
import jsonio as J                                              # noqa: E402
import attention as at                                          # noqa: E402


def sweep(dial_name, dials, lr_grid, build, steps, seeds, B=32, S=8, D=16,
          **lr_flags):
    ns = len(seeds)
    per = np.full((len(dials), len(lr_grid), ns), np.inf)
    for i, d in enumerate(dials):
        for j, lr in enumerate(lr_grid):
            for k, sd in enumerate(seeds):
                m = build(d, sd)
                X, y = at.teacher_batch(B, S, D, seed=0)
                per[i, j, k] = at.train(m, X, y, float(lr), steps, **lr_flags)
    return np.median(per, axis=2), per


def run(name, dials, dial_name, build, lr_grid, steps, seeds, controls):
    """Full scaling + one control per predicted factor."""
    out = {}
    losses, per = sweep(dial_name, dials, lr_grid, build, steps, seeds)
    out["full"] = (losses, T.verdict(losses, per, lr_grid, dials))
    for cname, flags in controls.items():
        lo, pe = sweep(dial_name, dials, lr_grid, build, steps, seeds, **flags)
        out[cname] = (lo, T.verdict(lo, pe, lr_grid, dials))
    return out


DUMP = {}


def report(title, dials, dial_name, res):
    print("\n" + title)
    print("  %-28s %-46s %-9s %s" % ("variant", "verdict", "drift", "lr* by " + dial_name))
    for k, (lo, v) in res.items():
        print("  %-28s %-46s %-9.3f %s"
              % (k, v["status"], v["drift_log10"],
                 " ".join("%+.2f" % x for x in v["refined_log10_lr"])))
    DUMP[title] = {"dials": list(map(float, dials)), "dial_name": dial_name,
                   "variants": {k: {"loss": v0.tolist(),
                                    "status": v1["status"],
                                    "drift": float(v1["drift_log10"]),
                                    "sem": float(v1["sem_log10"]),
                                    "lrstar": [float(x) for x in v1["refined_log10_lr"]]}
                                for k, (v0, v1) in res.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default=None, help="dump loss grids for plotting")
    a = ap.parse_args()
    torch.set_num_threads(4)
    np.seterr(all="ignore")
    seeds = (0, 1) if a.quick else (0, 1, 2)
    steps = 30 if a.quick else 40
    # Bracket the measured optimum (eta_0 ~ 0.32) with room on both sides;
    # 30 steps keeps the minimum interior rather than over-converging.
    grid = np.logspace(-1.9, 1.7, 11 if a.quick else 15)

    print("Transformer HP transfer, parameterisation of arXiv:2405.15712 Table 1")
    print("eta = eta_0 * N * H * L^(2 aL - 1);  seeds=%d steps=%d" % (len(seeds), steps))

    # --- N sweep, at both attention exponents ---------------------------
    Ns = [4, 8, 16, 32]
    for aA in (1.0, 0.5):
        res = run("N", Ns, "N",
                  lambda N, sd, _a=aA: at.Transformer(D=16, N=N, H=4, L=2,
                                                      alpha_A=_a, alpha_L=1.0, seed=sd),
                  grid, steps, seeds,
                  {"control: drop N from eta": dict(scale_N=False)})
        report("WIDTH  N = %s   (alpha_A = %.1f, H=4, L=2)" % (Ns, aA), Ns, "N", res)

    # --- H sweep --------------------------------------------------------
    Hs = [2, 4, 8, 16]
    res = run("H", Hs, "H",
              lambda H, sd: at.Transformer(D=16, N=8, H=H, L=2,
                                           alpha_A=1.0, alpha_L=1.0, seed=sd),
              grid, steps, seeds,
              {"control: drop H from eta": dict(scale_H=False)})
    report("HEADS  H = %s   (alpha_A = 1, N=8, L=2)" % Hs, Hs, "H", res)

    # --- L sweep, at both residual exponents ----------------------------
    Ls = [2, 4, 8]
    for aL in (1.0, 0.5):
        res = run("L", Ls, "L",
                  lambda L, sd, _a=aL: at.Transformer(D=16, N=8, H=4, L=L,
                                                      alpha_A=1.0, alpha_L=_a, seed=sd),
                  grid, steps, seeds,
                  {"control: drop L^(2aL-1)": dict(scale_L=False)})
        report("DEPTH  L = %s   (alpha_L = %.1f, N=8, H=4)" % (Ls, aL), Ls, "L", res)

    print("""
Reading these: 'full' should TRANSFER; each control should FAIL. A control that
transfers means the sweep cannot see that factor -- report under-powered, not
success. At alpha_L = 1/2 the depth factor L^(2aL-1) = L^0 is the identity, so
its control is EXPECTED to be inert; that row is a consistency check on the
harness, not evidence.""")
    if a.json:
        DUMP["_lr_grid"] = [float(x) for x in grid]
        J.dump(DUMP, a.json)
        print("wrote " + a.json)


if __name__ == "__main__":
    main()
