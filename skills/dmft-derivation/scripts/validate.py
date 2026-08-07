"""Phase 5 mandatory checks, made runnable, for the two-layer (L=1) case.

    python3 validate.py            # full battery
    python3 validate.py --quick    # smaller S / shorter horizon

Every check prints its measured number next to its bar. Bars are set at a few
times the measured Monte-Carlo floor, never tuned to make a run pass; the floor
itself is measured by sample-halving (F8) and reported, because a gap smaller
than its own floor is not evidence either way.

The battery ends with an explicit list of what it does NOT cover. That section
is load-bearing: at L=1 the response functions vanish identically, so nothing
here exercises the response sector, and a green run must not be read as
certifying it.
"""

import argparse
import sys
import time

import numpy as np

import dmft_two_layer as dmft
import exact
import sim_two_layer as sim


class Report(object):
    def __init__(self):
        self.rows = []

    def add(self, name, passed, detail):
        self.rows.append((name, passed, detail))
        flag = "PASS" if passed else "FAIL"
        print("  [%s] %-34s %s" % (flag, name, detail))

    @property
    def failed(self):
        return [r for r in self.rows if not r[1]]


# ---------------------------------------------------------------------------
# C1-C2  Exactly solvable cases (the independent ground truth)
# ---------------------------------------------------------------------------

def check_exact_scalar_ode(rep, cfg):
    """L=1, linear, whitened Kx=I, single output direction."""
    P, y0 = 3, 1.0
    Kx = np.eye(P)
    y = np.zeros(P)
    y[0] = y0
    dt, n = 0.002, cfg["n_ode"]

    errs, perp = [], []
    for seed in range(cfg["seeds"]):
        r = dmft.solve(Kx, y, 1.0, dt, n, S=cfg["S"], act="linear", seed=seed,
                       record_kernels=False)
        ref = exact.scalar_ode_linear_whitened(1.0, y0, r["t"])
        errs.append(np.max(np.abs((y0 - r["f"][:, 0]) - ref)))
        perp.append(np.max(np.abs(r["f"][:, 1:])))
    rms = float(np.sqrt(np.mean(np.square(errs))))
    bar = 1e-2
    rep.add("C1 exact scalar ODE", rms < bar,
            "seed-rms |dDelta| = %.2e  (bar %.0e, %d seeds)" % (rms, bar, cfg["seeds"]))

    # The orthogonal sector does not move in the LIMIT, but the empirical
    # cross-kernel Phi_{mu,1} is an S-sample average, so f_perp is O(1/sqrt(S))
    # rather than exactly zero. Bar tracks that floor.
    perp_max = float(np.max(perp))
    bar_perp = 4.0 / np.sqrt(cfg["S"])
    rep.add("C1b orthogonal sector static", perp_max < bar_perp,
            "max |f_perp| = %.2e  (bar 4/sqrt(S) = %.2e)" % (perp_max, bar_perp))


def check_final_kernel(rep, cfg):
    """H(inf) = I + [(sqrt(1+g0^2 y^2)-1)/y^2] y y^T for the same case.

    This is a t -> infinity statement, so it needs a horizon long enough for
    Delta to actually vanish; Delta(T) is reported so a short-horizon failure
    is not misread as a kernel error.
    """
    P, y0, g0 = 3, 1.0, 1.0
    Kx = np.eye(P)
    y = np.zeros(P)
    y[0] = y0
    dt, n = 0.005, 1600  # T = 8; Delta(T) ~ 1e-10
    r = dmft.solve(Kx, y, g0, dt, n, S=cfg["S"], act="linear", seed=0,
                   record_kernels=True)
    delta_T = float(abs(y0 - r["f"][-1, 0]))
    err = float(np.abs(r["Phi"][-1] - exact.final_kernel_linear_whitened(g0, y)).max())
    # Residual is pure sampling error: measured 5.0e-2 at S=2^12, 1.8e-2 at
    # S=2^14 (the 1/sqrt(S) scaling). Bar is that law with ~3x margin.
    bar = 8.0 / np.sqrt(cfg["S"])
    rep.add("C2 final kernel H(inf)", err < bar and delta_T < 1e-6,
            "max |H - H_exact| = %.2e  (bar %.2e)  Delta(T=%.0f) = %.1e"
            % (err, bar, dt * n, delta_T))


# ---------------------------------------------------------------------------
# C3  t = 0 kernels, three independent routes
# ---------------------------------------------------------------------------

def check_init_kernels(rep, cfg):
    X = sim.random_inputs(8, 3, seed=0)
    Kx = X.T @ X / 8.0

    # Quadrature vs analytic, where a closed form exists. ReLU is excluded:
    # Gauss-Hermite is polynomial-exact and loses accuracy on the kink, so the
    # analytic arccos kernel is authoritative there (measured disagreement
    # ~2e-2 in G is quadrature error, not a bug).
    for act in ("linear", "erf"):
        q = exact.kernels_quadrature(act, Kx)
        a = exact.kernels_analytic(act, Kx)
        err = max(np.abs(q[0] - a[0]).max(), np.abs(q[1] - a[1]).max())
        rep.add("C3 quad==analytic (%s)" % act, err < 1e-10,
                "max |Phi,G diff| = %.2e" % err)

    # MC vs quadrature for a case with no closed form.
    q = exact.kernels_quadrature("tanh", Kx)
    r = dmft.solve(Kx, np.zeros(3), 1.0, 0.001, 0, S=cfg["S_init"], act="tanh",
                   seed=0, record_kernels=True)
    err = max(np.abs(q[0] - r["Phi"][0]).max(), np.abs(q[1] - r["G"][0]).max())
    bar = 2e-2
    rep.add("C3b MC==quadrature (tanh)", err < bar,
            "max |Phi,G diff| = %.2e  (bar %.0e, S=%d)" % (err, bar, cfg["S_init"]))


# ---------------------------------------------------------------------------
# C4  Lazy limit
# ---------------------------------------------------------------------------

def check_lazy_limit(rep, cfg):
    """gamma0 -> 0 must freeze the kernels and give linear NTK dynamics.

    The raw gap against a quadrature NTK mixes two unrelated errors:

      (a) sampling error in the solver's own initial kernel -- common-mode
          across gamma0 at fixed seed, and O(1/sqrt(S));
      (b) the genuine feature-learning deviation -- gamma0-driven and
          S-independent.

    Measured at S=2^12: the raw gap is 1.47e-2 at gamma0=0.05 and 1.46e-2 at
    gamma0=0.4 (ratio 1.0), i.e. entirely (a). Conditioning on the solver's own
    K(0) isolates (b): 4.3e-4 vs 5.8e-3, ratio ~14, and flat in S across 16x.
    So the two are audited separately rather than as one number.
    """
    X = sim.random_inputs(8, 3, seed=0)
    Kx = X.T @ X / 8.0
    y = np.array([1.0, -0.5, 0.7])
    dt, n = 0.002, cfg["n_lazy"]
    _, _, K0q = exact.kernels_quadrature("tanh", Kx)

    phys, raw = {}, {}
    for g0 in (0.4, 0.05):
        r = dmft.solve(Kx, y, g0, dt, n, S=cfg["S"], act="tanh", seed=0)
        phys[g0] = float(np.abs(r["f"] - exact.lazy_prediction(r["K"][0], y, r["t"])).max())
        raw[g0] = float(np.abs(r["f"] - exact.lazy_prediction(K0q, y, r["t"])).max())

    bar = 2e-3
    rep.add("C4 lazy limit g0=0.05", phys[0.05] < bar,
            "|f - (I-e^{-K(0)t})y|max = %.2e  (bar %.0e)" % (phys[0.05], bar))

    ratio = phys[0.4] / max(phys[0.05], 1e-12)
    rep.add("C4b deviation is g0-driven", ratio > 3.0,
            "err(0.4)/err(0.05) = %.1f  (expect >3; S-independent)" % ratio)

    # And the sampling part must actually be sampling: shrink S, gap grows.
    # Seed-averaged, with disjoint seed blocks per S. A single seed will not do:
    # numpy fills row-major, so the S/4 draw is a literal SUBSET of the S draw
    # at the same seed, and the two gaps are positively correlated (measured
    # ratio 1.1, vs 2.0 expected). This is F10 in miniature -- seed-average
    # before comparing.
    def _raw_rms(S_val, seed0):
        vals = []
        for j in range(cfg["seeds"]):
            r = dmft.solve(Kx, y, 0.05, dt, n, S=S_val, act="tanh",
                           seed=seed0 + j, record_kernels=False)
            vals.append(np.abs(r["f"] - exact.lazy_prediction(K0q, y, r["t"])).max())
        return float(np.sqrt(np.mean(np.square(vals))))

    big = _raw_rms(cfg["S"], 0)
    small = _raw_rms(cfg["S"] // 4, 100)
    shrink = small / max(big, 1e-12)
    rep.add("C4c init-kernel gap is sampling", shrink > 1.4,
            "rms gap(S/4)/gap(S) = %.1f  (expect ~2 for 1/sqrt(S), %d seeds each)"
            % (shrink, cfg["seeds"]))


# ---------------------------------------------------------------------------
# C5  Convergence audits: MC floor (F8) and time step
# ---------------------------------------------------------------------------

def check_mc_floor(rep, cfg):
    """Sample-halving (F8). Reports the floor rather than asserting a bar."""
    X = sim.random_inputs(8, 3, seed=0)
    Kx = X.T @ X / 8.0
    y = np.array([1.0, -0.5, 0.7])
    dt, n = 0.002, cfg["n_lazy"]

    full = dmft.solve(Kx, y, 1.0, dt, n, S=cfg["S"], act="tanh", seed=0,
                      record_kernels=False)
    half = dmft.solve(Kx, y, 1.0, dt, n, S=cfg["S"] // 2, act="tanh", seed=0,
                      record_kernels=False)
    floor = float(np.abs(full["f"] - half["f"]).max())
    cfg["mc_floor"] = floor
    # Halving must not be free: a zero shift means the sample count is not
    # actually driving the estimate (wiring bug), not that noise vanished.
    rep.add("C5 MC floor by S-halving", 0.0 < floor < 5e-2,
            "|f(S) - f(S/2)|max = %.2e  (S=%d)" % (floor, cfg["S"]))


def check_dt_convergence(rep, cfg):
    X = sim.random_inputs(8, 3, seed=0)
    Kx = X.T @ X / 8.0
    y = np.array([1.0, -0.5, 0.7])
    T_end = 0.002 * cfg["n_lazy"]

    res = {}
    for dt in (0.004, 0.002):
        n = int(round(T_end / dt))
        r = dmft.solve(Kx, y, 1.0, dt, n, S=cfg["S"], act="tanh", seed=0,
                       record_kernels=False)
        res[dt] = r["f"][-1]
    shift = float(np.abs(res[0.004] - res[0.002]).max())
    bar = 2e-2
    rep.add("C5b dt -> dt/2 stability", shift < bar,
            "|f_T(dt) - f_T(dt/2)|max = %.2e  (bar %.0e)" % (shift, bar))


# ---------------------------------------------------------------------------
# C6  F15: the gamma0^{-1}-amplified readout channel
# ---------------------------------------------------------------------------

def check_antithetic_readout(rep, cfg):
    """Antithetic pairs zero the amplified channel exactly at t=0 (F15)."""
    X = sim.random_inputs(8, 3, seed=0)
    Kx = X.T @ X / 8.0
    y = np.array([1.0, -0.5, 0.7])

    anti = dmft.solve(Kx, y, 0.05, 0.002, 0, S=cfg["S"], act="tanh", seed=0,
                      antithetic=True, record_kernels=False)
    f0_anti = float(np.abs(anti["f"][0]).max())
    rep.add("C6 antithetic f(0) = 0 exactly", f0_anti < 1e-12,
            "|f(0)|max = %.2e" % f0_anti)

    # Without it, |f(0)| ~ 1/(gamma0 sqrt(S)) and GROWS as gamma0 shrinks --
    # the signature that makes F15 masquerade as theory failure.
    raw = {}
    for g0 in (0.5, 0.05):
        r = dmft.solve(Kx, y, g0, 0.002, 0, S=cfg["S"], act="tanh", seed=0,
                       antithetic=False, record_kernels=False)
        raw[g0] = float(np.abs(r["f"][0]).max())
    growth = raw[0.05] / max(raw[0.5], 1e-12)
    rep.add("C6b F15 signature reproduced", growth > 5.0,
            "|f(0)| grows %.1fx as g0: 0.5 -> 0.05 (expect ~10x)" % growth)


# ---------------------------------------------------------------------------
# C7  F16: independently seeded Sobol streams are not independent
# ---------------------------------------------------------------------------

def check_qmc_streams(rep, cfg):
    from scipy.stats import qmc

    d, m = 8, 12
    a = qmc.Sobol(d=d, scramble=True, seed=1).random_base2(m)
    b = qmc.Sobol(d=d, scramble=True, seed=2).random_base2(m)
    same_dim = [abs(np.corrcoef(a[:, k], b[:, k])[0, 1]) for k in range(d)]
    bad = float(np.max(same_dim))

    joint = qmc.Sobol(d=2 * d, scramble=True, seed=1).random_base2(m)
    j1, j2 = joint[:, :d], joint[:, d:]
    good = float(np.max([abs(np.corrcoef(j1[:, k], j2[:, k])[0, 1]) for k in range(d)]))

    rep.add("C7 F16 failure reproduced", bad > 0.5,
            "separately seeded streams: max same-dim |corr| = %.2f" % bad)
    rep.add("C7b F16 fix works", good < bad / 2.0,
            "one joint stream, sliced: max same-dim |corr| = %.2f" % good)


# ---------------------------------------------------------------------------
# C8-C9  Simulation match and the gamma0 trend
# ---------------------------------------------------------------------------

def check_sim_match(rep, cfg):
    X = sim.random_inputs(8, 3, seed=0)
    Kx = X.T @ X / 8.0
    y = np.array([1.0, -0.5, 0.7])
    dt, n, g0 = 0.004, cfg["n_sim"], 1.0

    ref = dmft.solve(Kx, y, g0, dt, n, S=cfg["S"], act="tanh", seed=0,
                     record_kernels=False)
    gaps = []
    for N in cfg["widths"]:
        s = sim.train_seeds(X, y, g0, dt, n, N, act="tanh",
                            seeds=range(cfg["sim_seeds"]), record_kernels=False)
        gaps.append(float(np.abs(s["f"] - ref["f"]).max()))

    floor = cfg.get("mc_floor", 0.0)
    detail = "  ".join("N=%d:%.2e" % (N, g) for N, g in zip(cfg["widths"], gaps))
    # Gap must shrink with width until it reaches the solver's own MC floor.
    shrinks = gaps[0] > gaps[-1] * 1.8
    rep.add("C8 sim gap shrinks with N", shrinks, detail)
    rep.add("C8b gap reaches MC floor", gaps[-1] < max(4.0 * floor, 1e-2),
            "widest-N gap %.2e vs measured MC floor %.2e" % (gaps[-1], floor))


def check_f4_discretisation(rep, cfg):
    """Measure what Euler-marching the prediction ODE actually costs (F4).

    Run both prediction paths on the SAME sources so Monte-Carlo noise cancels
    exactly and only the discretisation difference survives. Two things must
    hold: the difference is real and scales as O(dt), and -- reported, not
    asserted -- its size relative to the MC floor, which is what decides
    whether any theory-vs-sim comparison could have detected it.
    """
    X = sim.random_inputs(8, 3, seed=0)
    Kx = X.T @ X / 8.0
    y = np.array([1.0, -0.5, 0.7])
    g0, T_end = 3.0, 3.0  # rich: F4 grows with gamma0

    diffs = {}
    for dt in (0.02, 0.01):
        n = int(round(T_end / dt))
        kw = dict(act="tanh", seed=0, record_kernels=False)
        a = dmft.solve(Kx, y, g0, dt, n, S=cfg["S"], prediction="correlator", **kw)
        b = dmft.solve(Kx, y, g0, dt, n, S=cfg["S"], prediction="euler", **kw)
        diffs[dt] = float(np.abs(a["f"] - b["f"]).max())

    ratio = diffs[0.02] / max(diffs[0.01], 1e-15)
    rep.add("C10 F4 error is real and O(dt)", 1.5 < ratio < 3.0,
            "|corr - euler| = %.2e (dt=.02) -> %.2e (dt=.01), ratio %.1f"
            % (diffs[0.02], diffs[0.01], ratio))

    floor = cfg.get("mc_floor", 0.0)
    sub = diffs[0.02] < floor
    rep.add("C10b F4 vs MC floor (reported)", True,
            "F4 error %.2e %s MC floor %.2e -- %s"
            % (diffs[0.02], "<" if sub else ">", floor,
               "SUB-FLOOR here: no theory-vs-sim check at this S can detect it"
               if sub else "resolvable by theory-vs-sim at this S"))


def check_gamma0_trend(rep, cfg):
    """Feature learning must appear in BOTH theory and simulation."""
    X = sim.random_inputs(8, 3, seed=0)
    Kx = X.T @ X / 8.0
    y = np.array([1.0, -0.5, 0.7])
    dt, n, N = 0.004, cfg["n_sim"], cfg["widths"][-1]

    thy, sm = [], []
    for g0 in (0.1, 1.0):
        r = dmft.solve(Kx, y, g0, dt, n, S=cfg["S"], act="tanh", seed=0)
        thy.append(dmft.kernel_movement(r))
        s = sim.train_seeds(X, y, g0, dt, n, N, act="tanh",
                            seeds=range(cfg["sim_seeds"]))
        sm.append(float(np.linalg.norm(s["Phi"][-1] - s["Phi"][0])))

    ok = thy[1] > thy[0] and sm[1] > sm[0]
    rep.add("C9 kernel movement grows with g0", ok,
            "theory %.2e -> %.2e ; sim %.2e -> %.2e  (g0: 0.1 -> 1.0)"
            % (thy[0], thy[1], sm[0], sm[1]))


# ---------------------------------------------------------------------------

NOT_COVERED = """
KNOWN BLIND SPOTS (found by mutation-testing this battery -- see scripts/README):
  * Reintroducing F4 (Euler-marching f) at gamma0=1, dt=0.002 changes the C1
    error from 7.4e-4 to 6.7e-4, i.e. NOT detectable there. C10 catches it
    only by comparing the two prediction paths on identical sources so the
    Monte-Carlo noise cancels. Any theory-vs-sim check at S ~ 1e4 is blind to
    F4 in this regime -- the effect is below the floor.
  * A within-step stale-read reordering (reading the updated h when forming
    the w update) is likewise below the bar at small dt.
  Both are O(dt) effects that grow with gamma0 and with the response sector,
  so the deep case needs its own discriminators; do not assume this battery
  transfers.

NOT COVERED by this battery (do not read a green run as certifying these):
  * The response sector. At L=1 the response functions vanish identically
    (A^0 = 0, B^1 = 0), so nothing here exercises A, B, their equal-time
    diagonals (F1), the write-order race (F17), or response-noise
    rectification (F6). An ablation of the response sector at L=1 changes
    nothing -- which is the one case where that is not a red flag, and
    exactly why L=1 cannot certify L>=2 code.
  * The alternating fixed-point solver, damping, and Delta-loop stiffness
    (F5) -- unused at L=1, which is causally integrable.
  * Deep linear algebraic closure, depth-muP residual, attention, MoE.
  * The mean-field closure itself. At L=1 a width-N muP network is EXACTLY
    S = N samples of the single-site process (see scripts/README.md), so
    C8 measures convergence in sample count, not the correctness of a
    closure. The independent tests here are C1-C4.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller S, shorter horizon")
    args = ap.parse_args()

    if args.quick:
        cfg = dict(S=2 ** 12, S_init=2 ** 14, seeds=3, n_ode=600, n_lazy=400,
                   n_sim=300, widths=(256, 2048), sim_seeds=3)
    else:
        cfg = dict(S=2 ** 14, S_init=2 ** 17, seeds=4, n_ode=1500, n_lazy=1000,
                   n_sim=750, widths=(256, 1024, 8192), sim_seeds=4)

    print("DMFT two-layer validation battery  (%s)"
          % ("quick" if args.quick else "full"))
    print("S=%d  seeds=%d  widths=%s" % (cfg["S"], cfg["seeds"], list(cfg["widths"])))
    print()

    rep = Report()
    t0 = time.time()
    # C5 runs early so the measured MC floor is available to later checks.
    for fn in (check_exact_scalar_ode, check_final_kernel, check_init_kernels,
               check_lazy_limit, check_mc_floor, check_dt_convergence,
               check_antithetic_readout, check_qmc_streams,
               check_f4_discretisation, check_sim_match, check_gamma0_trend):
        fn(rep, cfg)

    print()
    print("%d checks, %d failed, %.1fs" % (len(rep.rows), len(rep.failed), time.time() - t0))
    print(NOT_COVERED)
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
