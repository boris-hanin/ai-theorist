"""Battery for the deep (general L) nonlinear DMFT solver.

    python3 validate_deep_nonlinear.py [--quick]

The anchor is that with linear `phi` this Monte-Carlo solver must reproduce the
*exact* algebraic deep-linear solver at every depth. That is a strong check: it
exercises the sampling, the forward-mode sensitivities, the per-layer response
kernels, the damped fixed point and the correlator rule, against a reference
with no sampling error at all.
"""

import argparse
import sys
import time

import numpy as np

import dmft_deep_linear as dl
import dmft_deep_nonlinear as dn
import dmft_l2_nonlinear as d2
import sim_deep as sd


class Report(object):
    def __init__(self):
        self.rows = []

    def add(self, name, passed, detail):
        self.rows.append((name, passed, detail))
        print("  [%s] %-40s %s" % ("PASS" if passed else "FAIL", name, detail))

    @property
    def failed(self):
        return [r for r in self.rows if not r[1]]


def check_linear_reduction(rep, cfg):
    """M1: linear phi -> must match the exact algebraic solver at every depth."""
    Kx, y = np.array([[1.0]]), np.array([1.0])
    dt, T = 0.05, cfg["T"]
    for L in cfg["depths"]:
        ref = dl.solve(Kx, y, 1.0, dt, T, L=L, damping=0.4, n_iter=900)
        r = dn.solve(1.0, 1.0, dt, T, L=L, S=cfg["S"], act="linear", seed=0,
                     n_iter=cfg["iters"], damping=0.45)
        gap = float(np.abs(r["f"] - ref["f"][:, 0]).max())
        rep.add("M1 linear reduction at L=%d" % L, gap < cfg["lin_bar"] and r["converged"],
                "max |f_MC - f_exact| = %.2e  (bar %.0e, %d iters)"
                % (gap, cfg["lin_bar"], r["iters"]))


def check_S_convergence(rep, cfg):
    """M2: the residual against the exact solver must be sampling error."""
    Kx, y = np.array([[1.0]]), np.array([1.0])
    dt, T, L = 0.05, cfg["T"], 3
    ref = dl.solve(Kx, y, 1.0, dt, T, L=L, damping=0.4, n_iter=900)
    errs = {}
    for S in (cfg["S"] // 4, cfg["S"]):
        r = dn.solve(1.0, 1.0, dt, T, L=L, S=S, act="linear", seed=0,
                     n_iter=cfg["iters"], damping=0.45)
        errs[S] = float(np.abs(r["f"] - ref["f"][:, 0]).max())
    ratio = errs[cfg["S"] // 4] / max(errs[cfg["S"]], 1e-15)
    rep.add("M2 error shrinks with S (L=3)", ratio > 1.3,
            "err(S/4)/err(S) = %.2f  (%.2e -> %.2e)"
            % (ratio, errs[cfg["S"] // 4], errs[cfg["S"]]))


def check_matches_l2_solver(rep, cfg):
    """M3: at L=2 it must agree with the separately written L=2 solver.

    The two consume random numbers in different orders, so they carry
    INDEPENDENT sample realisations. Their difference must therefore be judged
    against their combined Monte-Carlo floor, each measured by S-halving --
    not against a chosen constant. (A fixed 5e-3 bar failed this at S=4096 on a
    difference of 8.9e-3 that was entirely sampling noise.)
    """
    dt, T = 0.05, cfg["T"]
    kw = dict(act="tanh", seed=0, n_iter=cfg["iters"], damping=0.45)
    a = dn.solve(1.0, 1.0, dt, T, L=2, S=cfg["S"], **kw)
    ah = dn.solve(1.0, 1.0, dt, T, L=2, S=cfg["S"] // 2, **kw)
    b = d2.solve(1.0, 1.0, dt, T, S=cfg["S"], onsager=0.0, **kw)
    bh = d2.solve(1.0, 1.0, dt, T, S=cfg["S"] // 2, onsager=0.0, **kw)
    fa = float(np.abs(a["f"] - ah["f"]).max())
    fb = float(np.abs(b["f"] - bh["f"]).max())
    comb = float(np.hypot(fa, fb))
    gap = float(np.abs(a["f"] - b["f"]).max())
    rep.add("M3 agrees with the L=2 solver", gap < 3.0 * comb,
            "gap %.2e vs combined MC floor %.2e (%.1e, %.1e) = %.1fx"
            % (gap, comb, fa, fb, gap / comb))


def check_antithetic(rep, cfg):
    """M4 (F15): antithetic readout pairs give f(0) = 0 exactly."""
    r = dn.solve(1.0, 1.0, 0.05, 4, L=3, S=cfg["S"], act="tanh", seed=0,
                 n_iter=2, damping=0.5, tol=0)
    rep.add("M4 antithetic f(0) = 0 exactly", abs(r["f"][0]) < 1e-12,
            "|f(0)| = %.2e" % abs(r["f"][0]))


def check_response_ablation(rep, cfg):
    """M5 (F17): the response sector must bite at L >= 2 and be inert at L = 1."""
    X, y = np.array([[1.0]]), np.array([1.0])
    dt, T = 0.05, cfg["T"]
    a1 = dn.solve(1.0, 1.0, dt, T, L=1, S=cfg["S"], act="tanh", seed=0,
                  n_iter=cfg["iters"], damping=0.45)
    b1 = dn.solve(1.0, 1.0, dt, T, L=1, S=cfg["S"], act="tanh", seed=0,
                  n_iter=cfg["iters"], damping=0.45, no_response=True)
    rep.add("M5 ablation inert at L=1", float(np.abs(a1["f"] - b1["f"]).max()) < 1e-12,
            "max |f_full - f_noresp| = %.2e  (A^0 = B^1 = 0 by boundary)"
            % float(np.abs(a1["f"] - b1["f"]).max()))

    for L in [d for d in cfg["depths"] if d >= 2]:
        s = sd.train_seeds(X, y, 1.0, dt, T - 1, cfg["N"], L=L, act="tanh",
                           seeds=range(cfg["seeds"]), record_kernels=False)
        a = dn.solve(1.0, 1.0, dt, T, L=L, S=cfg["S"], act="tanh", seed=0,
                     n_iter=cfg["iters"], damping=0.45)
        b = dn.solve(1.0, 1.0, dt, T, L=L, S=cfg["S"], act="tanh", seed=0,
                     n_iter=cfg["iters"], damping=0.45, no_response=True)
        ga = float(np.abs(a["f"] - s["f"][:, 0]).max())
        gb = float(np.abs(b["f"] - s["f"][:, 0]).max())
        rep.add("M5b responses matter at L=%d" % L, gb > 2.0 * ga,
                "sim gap: full %.2e vs no-response %.2e  (%.1fx worse)"
                % (ga, gb, gb / max(ga, 1e-15)))


def check_sim_match(rep, cfg):
    """M6: nonlinear theory vs finite-width sims at depth, BOTH floors reported."""
    X, y = np.array([[1.0]]), np.array([1.0])
    dt, T = 0.05, cfg["T"]
    for L in [d for d in cfg["depths"] if d >= 2]:
        sA = sd.train_seeds(X, y, 1.0, dt, T - 1, cfg["N"] // 4, L=L, act="tanh",
                            seeds=range(cfg["seeds"]), record_kernels=False)
        sB = sd.train_seeds(X, y, 1.0, dt, T - 1, cfg["N"], L=L, act="tanh",
                            seeds=range(cfg["seeds"]), record_kernels=False)
        sim_floor = float(np.abs(sB["f"][:, 0] - sA["f"][:, 0]).max())
        rA = dn.solve(1.0, 1.0, dt, T, L=L, S=cfg["S"], act="tanh", seed=0,
                      n_iter=cfg["iters"], damping=0.45)
        rB = dn.solve(1.0, 1.0, dt, T, L=L, S=cfg["S"] // 2, act="tanh", seed=0,
                      n_iter=cfg["iters"], damping=0.45)
        mc_floor = float(np.abs(rA["f"] - rB["f"]).max())
        comb = float(np.hypot(sim_floor, mc_floor))
        gap = float(np.abs(rA["f"] - sB["f"][:, 0]).max())
        rep.add("M6 sim match at L=%d" % L, gap < 3.0 * comb,
                "gap %.2e vs combined floor %.2e (sim %.1e, MC %.1e) = %.1fx"
                % (gap, comb, sim_floor, mc_floor, gap / comb))


NOT_COVERED = """
NOT COVERED:
  * P > 1. The solver is single-datapoint by construction; the memory kernels
    do not couple data indices here, so nothing tests that coupling.
  * The equal-time endpoint term is OFF by default (weight 0, as measured in
    round 003 at L=2). It has not been re-measured at L >= 3.
  * Depth-muP / residual architectures: this is a plain MLP.
  * Real datasets: a single synthetic target.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = (dict(T=15, S=4096, iters=45, depths=[1, 2, 3], N=2048, seeds=8,
                lin_bar=3e-2)
           if args.quick else
           dict(T=25, S=16384, iters=80, depths=[1, 2, 3, 4], N=4096, seeds=16,
                lin_bar=2e-2))
    np.seterr(all="ignore")
    print("Deep nonlinear DMFT validation  (%s)" % ("quick" if args.quick else "full"))
    print("depths=%s  S=%d  T=%d\n" % (cfg["depths"], cfg["S"], cfg["T"]))
    rep = Report()
    t0 = time.time()
    for fn in (check_linear_reduction, check_S_convergence, check_matches_l2_solver,
               check_antithetic, check_response_ablation, check_sim_match):
        fn(rep, cfg)
    print("\n%d checks, %d failed, %.1fs"
          % (len(rep.rows), len(rep.failed), time.time() - t0))
    print(NOT_COVERED)
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
