"""Phase 5 battery for the deep linear DMFT solver.

    python3 validate_deep_linear.py [--quick]

The point of this rung: it is algebraically exact, so there is NO Monte-Carlo
floor. Every discrepancy is either discretisation (O(dt), checkable by halving)
or a bug. That makes it the right place to certify the response machinery
before adding sampling noise on top.

Scope, stated up front: this cannot certify the equal-time Onsager diagonal
(F1), which carries phi_ddot and vanishes identically for linear phi. That is
F1b. See the NOT COVERED block.
"""

import argparse
import sys
import time

import numpy as np

import dmft_deep_linear as dl
import dmft_two_layer as d2
import exact
import sim_deep as sd


class Report(object):
    def __init__(self):
        self.rows = []

    def add(self, name, passed, detail):
        self.rows.append((name, passed, detail))
        print("  [%s] %-36s %s" % ("PASS" if passed else "FAIL", name, detail))

    @property
    def failed(self):
        return [r for r in self.rows if not r[1]]


def whitened(P, D=None):
    """X with Kx = I_P exactly."""
    D = D or P
    X = np.zeros((D, P))
    X[:P, :P] = np.sqrt(D) * np.eye(P)
    return X


def check_exact_ode(rep, cfg):
    """D1: L=1 linear whitened single-direction -> the exactly solvable ODE."""
    P = 2
    Kx = np.eye(P)
    y = np.zeros(P)
    y[0] = 1.0
    errs = {}
    for dt, T in ((0.04, cfg["T1"] // 2 + 1), (0.02, cfg["T1"] + 1)):
        r = dl.solve(Kx, y, 1.0, dt, T, L=1, damping=0.5, n_iter=cfg["iters"])
        ref = exact.scalar_ode_linear_whitened(1.0, y[0], r["t"])
        errs[dt] = float(np.abs((y[0] - r["f"][:, 0]) - ref).max())
    ratio = errs[0.04] / max(errs[0.02], 1e-15)
    rep.add("D1 exact scalar ODE (L=1)", errs[0.02] < 2e-2,
            "err = %.2e at dt=0.02  (bar 2e-2)" % errs[0.02])
    rep.add("D1b error is O(dt)", 1.6 < ratio < 2.6,
            "err(dt)/err(dt/2) = %.2f  (expect 2.0)" % ratio)

    # Orthogonal sector must be EXACTLY zero -- no sampling noise here at all.
    r = dl.solve(Kx, y, 1.0, 0.02, cfg["T1"] + 1, L=1, n_iter=cfg["iters"])
    perp = float(np.abs(r["f"][:, 1:]).max())
    rep.add("D1c orthogonal sector exactly 0", perp < 1e-12,
            "max |f_perp| = %.2e  (algebraic solve: no MC floor)" % perp)


def check_reduces_to_two_layer(rep, cfg):
    """D2: at L=1 this solver must reproduce the certified two-layer solver."""
    X = whitened(2, 4)
    Kx = X.T @ X / 4
    y = np.array([1.0, -0.6])
    dt, T = 0.02, cfg["T1"] + 1
    r = dl.solve(Kx, y, 1.0, dt, T, L=1, n_iter=cfg["iters"])
    ref = d2.solve(Kx, y, 1.0, dt, T - 1, S=2 ** 16, act="linear", seed=0,
                   record_kernels=False)
    gap = float(np.abs(r["f"] - ref["f"]).max())
    rep.add("D2 reduces to certified L=1 solver", gap < 2e-2,
            "max |f_alg - f_MC| = %.2e  (MC solver at S=2^16)" % gap)


def check_lazy(rep, cfg):
    """D3: gamma0 -> 0 freezes kernels at H^l = K^x, G^l = 1, K = (L+1)K^x."""
    X = whitened(2, 4)
    Kx = X.T @ X / 4
    y = np.array([1.0, -0.6])
    dt, T, L = 0.02, cfg["T1"] + 1, 3
    r = dl.solve(Kx, y, 0.02, dt, T, L=L, n_iter=cfg["iters"])
    K0 = (L + 1) * Kx
    lazy = exact.lazy_prediction_discrete(K0, y, dt, T - 1)
    gap = float(np.abs(r["f"] - lazy).max())
    rep.add("D3 lazy limit, K0=(L+1)Kx", gap < 5e-3,
            "max |f - lazy| = %.2e at gamma0=0.02, L=%d" % (gap, L))


def check_causality(rep, cfg):
    """D7: responses must be causal and kernels symmetric."""
    X = whitened(2, 4)
    Kx = X.T @ X / 4
    y = np.array([1.0, -0.6])
    P, T, L = 2, cfg["T1"] + 1, 3
    r = dl.solve(Kx, y, 1.0, 0.02, T, L=L, damping=0.4, n_iter=cfg["iters"])
    m = dl.causal_mask(P, T, strict=True)
    acausal = 0.0
    for l in range(1, L):
        acausal = max(acausal, np.abs(r["A"][l][~m]).max(), np.abs(r["B"][l][~m]).max())
    asym = 0.0
    for l in range(1, L + 1):
        asym = max(asym, np.abs(r["H"][l] - r["H"][l].T).max(),
                   np.abs(r["G"][l] - r["G"][l].T).max())
    rep.add("D7 responses causal", acausal < 1e-12,
            "max |A,B| outside the causal mask = %.2e" % acausal)
    rep.add("D7b kernels symmetric", asym < 1e-8,
            "max |H - H^T|, |G - G^T| = %.2e" % asym)


def check_response_ablation(rep, cfg):
    """D5 (F17): the no-response control must bite at L>=2 and be inert at L=1."""
    X = whitened(2, 4)
    Kx = X.T @ X / 4
    y = np.array([1.0, -0.6])
    dt, T = 0.02, cfg["T2"] + 1

    r1 = dl.solve(Kx, y, 1.0, dt, T, L=1, n_iter=cfg["iters"])
    r1n = dl.solve(Kx, y, 1.0, dt, T, L=1, n_iter=cfg["iters"], no_response=True)
    inert = float(np.abs(r1["f"] - r1n["f"]).max())
    rep.add("D5 ablation inert at L=1", inert < 1e-12,
            "max |f_full - f_noresp| = %.2e  (A^0=B^1=0 by boundary)" % inert)

    for L in cfg["depths"]:
        a = dl.solve(Kx, y, 1.0, dt, T, L=L, damping=0.35, n_iter=cfg["iters"])
        b = dl.solve(Kx, y, 1.0, dt, T, L=L, damping=0.35, n_iter=cfg["iters"],
                     no_response=True)
        s = sd.train_seeds(X, y, 1.0, dt, T - 1, cfg["N"], L=L, act="linear",
                           seeds=range(cfg["seeds"]), record_kernels=False)
        ga = float(np.abs(a["f"] - s["f"]).max())
        gb = float(np.abs(b["f"] - s["f"]).max())
        rep.add("D5b responses matter at L=%d" % L, gb > 3.0 * ga,
                "sim gap: full %.2e vs no-response %.2e  (%.1fx worse)"
                % (ga, gb, gb / max(ga, 1e-15)))


def check_resummation(rep, cfg):
    """D5c: the Neumann resummation A = M^{-1}C must be detectable.

    Mutation testing found the battery initially blind to replacing M^{-1}C by
    C: the shift is 0.6% at gamma_0=1, which slipped under the sim-gap bar.
    It is a real O(gamma_0^2 C D) effect and grows with richness, so it gets
    its own check at a gamma_0 where it is unambiguous.
    """
    X = whitened(2, 4)
    Kx = X.T @ X / 4
    y = np.array([1.0, -0.6])
    dt, T, L, g0 = 0.02, cfg["T2"] + 1, 3, 3.0
    a = dl.solve_annealed(Kx, y, g0, dt, T, L=L, damping=0.3, n_iter=cfg["iters"])
    b = dl.solve_annealed(Kx, y, g0, dt, T, L=L, damping=0.3, n_iter=cfg["iters"],
                          no_resum=True)
    shift = float(np.abs(a["f"] - b["f"]).max())
    rep.add("D5c response resummation matters", shift > 5e-3,
            "dropping M^{-1} shifts f by %.2e at gamma0=%.0f  (bar 5e-3)" % (shift, g0))


def check_stiffness(rep, cfg):
    """D8 (F5): a cold start diverges at rich gamma_0; annealing rescues it."""
    X = whitened(2, 4)
    Kx = X.T @ X / 4
    y = np.array([1.0, -0.6])
    dt, T, L, g0 = 0.02, 51, 3, 6.0
    try:
        dl.solve(Kx, y, g0, dt, T, L=L, damping=0.3, n_iter=3000)
        cold = "converged"
    except (dl.Diverged, np.linalg.LinAlgError):
        cold = "diverged"
    try:
        r = dl.solve_annealed(Kx, y, g0, dt, T, L=L, n_iter=3000)
        warm, ok = "converged in %d iters" % r["iters"], np.all(np.isfinite(r["f"]))
    except dl.Diverged:
        warm, ok = "diverged", False
    rep.add("D8 annealing rescues F5 stiffness", ok and cold == "diverged",
            "gamma0=%.0f, horizon %.1f: cold start %s, annealed %s"
            % (g0, dt * (T - 1), cold, warm))


def check_sim_match(rep, cfg):
    """D6: theory vs finite-width nets, judged against the discretisation floor.

    The algebraic solver has no sampling floor, so the analogue of F8's
    "report the floor beside the gap" is the O(dt) discretisation error. It is
    measured here by dt-halving rather than assumed, and the sim gap at the
    widest N must land within a small multiple of it. A fixed absolute bar was
    too loose: mutation testing showed an M/Nt transpose swap producing a
    4x-worse gap that still passed.
    """
    X = whitened(2, 4)
    Kx = X.T @ X / 4
    y = np.array([1.0, -0.6])
    dt, T, L = 0.02, cfg["T2"] + 1, 2
    r = dl.solve(Kx, y, 1.0, dt, T, L=L, damping=0.4, n_iter=cfg["iters"])

    # Discretisation floor, measured.
    r2 = dl.solve(Kx, y, 1.0, dt / 2, 2 * (T - 1) + 1, L=L, damping=0.4,
                  n_iter=cfg["iters"])
    floor = float(np.abs(r["f"] - r2["f"][::2]).max())

    gaps = []
    for N in cfg["widths"]:
        s = sd.train_seeds(X, y, 1.0, dt, T - 1, N, L=L, act="linear",
                           seeds=range(cfg["seeds"]), record_kernels=False)
        gaps.append(float(np.abs(r["f"] - s["f"]).max()))
    detail = "  ".join("N=%d:%.2e" % (N, g) for N, g in zip(cfg["widths"], gaps))
    rep.add("D6 sim gap shrinks with N (L=2)", gaps[0] > gaps[-1] * 1.8, detail)
    rep.add("D6b gap reaches discretisation floor", gaps[-1] < 3.0 * floor,
            "widest-N gap %.2e vs measured dt-floor %.2e" % (gaps[-1], floor))


NOT_COVERED = """
NOT COVERED (do not read a green run as certifying these):
  * The equal-time Onsager diagonal (F1). It carries phi_ddot and so vanishes
    identically for linear phi -- this is exactly F1b, and no linear check of
    any kind can detect it. The minimal detector is nonlinear L=2.
  * F6, F8, F15, F16 -- all Monte-Carlo artifacts. This solver has no sampling,
    so they cannot occur and cannot be tested here.
  * Nonlinear kernel closure of any sort: for linear phi the kernels close
    algebraically, so nothing here exercises single-site sampling, Gaussian
    source generation, or forward-mode sensitivity code.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        cfg = dict(T1=40, T2=30, iters=400, depths=(2,), widths=(512, 4096),
                   N=4096, seeds=3)
    else:
        cfg = dict(T1=60, T2=50, iters=800, depths=(2, 3), widths=(512, 2048, 8192),
                   N=8192, seeds=4)

    print("Deep linear DMFT validation  (%s)" % ("quick" if args.quick else "full"))
    print("depths=%s  widths=%s  seeds=%d\n"
          % (list(cfg["depths"]), list(cfg["widths"]), cfg["seeds"]))
    rep = Report()
    t0 = time.time()
    for fn in (check_exact_ode, check_reduces_to_two_layer, check_lazy,
               check_causality, check_response_ablation, check_resummation,
               check_stiffness, check_sim_match):
        fn(rep, cfg)
    print("\n%d checks, %d failed, %.1fs" % (len(rep.rows), len(rep.failed),
                                             time.time() - t0))
    print(NOT_COVERED)
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
