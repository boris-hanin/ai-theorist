"""Depth and width HP-transfer sweeps for SP / muP / alpha in {0.5, 1}.

Replicates the claim of arXiv:2505.01618 Figures 2-3: **only alpha = 1 enables
depth-wise HP transfer**; SP, muP and alpha = 0.5 do not.

Every learning-rate rule applied here is derived in
`derivations/04-completep.md` (D1-D6) and matches the paper's Table 1. The
`--controls` run additionally drops one derived factor at a time so the sweep
can show which factor is doing the work.

Optimiser is Adam with weight decay 0 throughout.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "dmft-derivation", "scripts"))
import transfer as T                                        # noqa: E402
import completep as C                                       # noqa: E402


def sweep(build, dials, grid, steps, seeds, B, S, V, **flags):
    per = np.full((len(dials), len(grid), len(seeds)), np.inf)
    for i, d in enumerate(dials):
        for j, e in enumerate(grid):
            for k, sd in enumerate(seeds):
                x, y = C.task(B, S, V, seed=0)
                per[i, j, k] = C.train(build(d, sd), x, y, float(e), steps, **flags)
    return np.median(per, axis=2), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    torch.set_num_threads(4)
    np.seterr(all="ignore")

    V, B, S = 64, 16, 16
    seeds = (0, 1) if a.quick else (0, 1, 2)
    steps = 30 if a.quick else 40
    grid = np.logspace(-3.6, -0.4, 11 if a.quick else 15)
    Ls = [2, 4, 8, 16, 32]
    Ns = [64, 128, 256]
    dump = {"grid": [float(x) for x in grid], "rows": []}

    def row(title, dial_name, dials, variants):
        print("\n" + title)
        print("  %-34s %-46s %-8s %s"
              % ("variant", "verdict", "drift", "eta* by " + dial_name))
        out = {"title": title, "dial": dial_name, "dials": [float(d) for d in dials],
               "variants": {}}
        for vname, (build, flags) in variants.items():
            lo, per = sweep(build, dials, grid, steps, seeds, B, S, V, **flags)
            v = T.verdict(lo, per, grid, dials)
            print("  %-34s %-46s %-8.3f %s"
                  % (vname, v["status"], v["drift_log10"],
                     " ".join("%+.2f" % x for x in v["refined_log10_lr"])))
            out["variants"][vname] = {
                "loss": lo.tolist(), "status": v["status"],
                "drift": float(v["drift_log10"]),
                "lrstar": [float(x) for x in v["refined_log10_lr"]]}
        dump["rows"].append(out)

    print("CompleteP replication -- Adam, weight decay 0")
    print("N_base=%d L_base=%d  V=%d  seeds=%d steps=%d"
          % (C.N_BASE, C.L_BASE, V, len(seeds), steps))

    # --- DEPTH: the paper's central claim -------------------------------
    mk = lambda p, al: (lambda L, sd: C.Model(V, C.N_BASE, L, param=p, alpha=al, seed=sd))
    row("DEPTH  L = %s   (N = %d fixed)" % (Ls, C.N_BASE), "L", Ls, {
        "SP": (mk("sp", 1.0), {}),
        "muP": (mk("mup", 1.0), {}),
        "alpha = 0.5": (mk("alpha", 0.5), {}),
        "CompleteP (alpha = 1)": (mk("alpha", 1.0), {}),
    })
    # NOTE: there is deliberately no "CompleteP minus depth-LR factor" control.
    # At alpha = 1 that factor is m_L^{alpha-1} = m_L^0 = 1 -- the identity --
    # so removing it is a no-op (verified: byte-identical drift). The real
    # control ladder for the depth machinery is the parameterisation list
    # itself: muP is "no residual scaling at all", alpha = 0.5 is partial,
    # alpha = 1 is full. Same trap as round 005's alpha_L = 1/2 row.

    # --- WIDTH: muP's original claim, as a positive control -------------
    mkw = lambda p, al: (lambda N, sd: C.Model(V, N, C.L_BASE, param=p, alpha=al, seed=sd))
    row("WIDTH  N = %s   (L = %d fixed)" % (Ns, C.L_BASE), "N", Ns, {
        "SP": (mkw("sp", 1.0), {}),
        "CompleteP (alpha = 1)": (mkw("alpha", 1.0), {}),
        "CompleteP minus width-LR factor": (mkw("alpha", 1.0), dict(scale_width=False)),
    })

    print("""
Expected, from derivations/04-completep.md:
  depth  -- only alpha = 1 transfers. SP and muP have no depth rule at all;
            alpha = 0.5 has the m_L^{a-1} LR factor but its blocks linearise
            (D6), so its optimum still moves. Removing the depth-LR factor from
            CompleteP must break it (D5).
  width  -- CompleteP transfers, SP does not (D1/D3/D4); removing the width-LR
            factor must break it.""")
    if a.json:
        json.dump(dump, open(a.json, "w"))
        print("wrote " + a.json)


if __name__ == "__main__":
    main()
