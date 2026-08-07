"""Step-0 scaling audit: pin exponents by measurement, not by citation.

`dmft-master/references/algorithm.md` Step 0: propose exponents, then PIN them
empirically -- "never derive against unpinned exponents." Two instruments, and
the file is explicit that the second is the one that matters for reused
matrices:

  * per-group update RMS vs the dial (fine for per-particle groups);
  * INDUCED FEATURE VELOCITY vs the dial, for reused matrices -- "row-norm
    criteria mislead here (a muP hidden matrix has O(dial^{-1/2}) rows but
    Theta(1) feature velocity at its exponent)."

Feature velocity is measured as RMS(h^l after k steps - h^l at init) divided by
RMS(h^l at init), so it is dimensionless and comparable across widths. A
correct parameterisation gives a slope of 0 in log-dial; a wrong one gives a
nonzero slope, and its sign says whether the network blows up or goes lazy.
"""

import numpy as np

import nets


def feature_velocity(dial_values, build, steps=8, lr=0.05, seeds=(0, 1, 2)):
    """Relative feature movement per layer vs a dial.

    `build(dial, seed)` returns a Net. Returns (velocities, layers) where
    velocities[i, l] is the seed-averaged relative movement of h^{l+1} at
    dial_values[i].
    """
    out = []
    for dial in dial_values:
        per_seed = []
        for s in seeds:
            net = build(dial, s)
            X, y = nets.teacher_data(32, net.D, seed=0)
            h0 = [h.copy() for h in net.features(X)]
            for _ in range(steps):
                net.step(X, y, lr)
            h1 = net.features(X)
            vel = [float(np.sqrt(np.mean((b - a) ** 2)) /
                         max(np.sqrt(np.mean(a ** 2)), 1e-30))
                   for a, b in zip(h0, h1)]
            per_seed.append(vel)
        out.append(np.mean(per_seed, axis=0))
    return np.array(out)


def update_rms(dial_values, build, steps=8, lr=0.05, seeds=(0, 1, 2)):
    """Per-group update RMS vs the dial (the per-particle instrument)."""
    keys = ("W0", "Wh", "w")
    out = {k: [] for k in keys}
    for dial in dial_values:
        acc = {k: [] for k in keys}
        for s in seeds:
            net = build(dial, s)
            X, y = nets.teacher_data(32, net.D, seed=0)
            last = None
            for _ in range(steps):
                _, last = net.step(X, y, lr)
            for k in keys:
                v = last.get(k)
                acc[k].append(np.nan if v is None
                              else float(np.sqrt(np.mean(np.asarray(v) ** 2))))
        for k in keys:
            out[k].append(float(np.mean(acc[k])))
    return {k: np.array(v) for k, v in out.items()}


def log_slope(dial_values, values):
    """Least-squares slope of log(values) vs log(dial). 0 == scale-invariant."""
    d = np.asarray(dial_values, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v) & (v > 0)
    if ok.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(d[ok]), np.log(v[ok]), 1)[0])


def width_audit(param, widths, L=2, D=16, act="tanh", gamma0=1.0, **kw):
    """Feature-velocity slope vs width for a plain MLP."""
    def build(N, seed):
        return nets.Net(D, N, L, param=param, gamma0=gamma0, act=act, seed=seed)
    vel = feature_velocity(widths, build, **kw)
    return vel, [log_slope(widths, vel[:, l]) for l in range(vel.shape[1])]


def depth_audit(alpha, depths, N=256, D=16, act="tanh", block_k=1, gamma0=1.0,
                lr_depth_exp=None, **kw):
    """Feature-velocity slope vs depth for a residual net at a given alpha."""
    def build(L, seed):
        return nets.Net(D, N, L, param="mup", gamma0=gamma0, act=act, seed=seed,
                        residual=True, alpha=alpha, block_k=block_k,
                        lr_depth_exp=lr_depth_exp)
    vel = feature_velocity(depths, build, **kw)
    # Report the LAST residual-stream layer: the accumulated stream movement.
    return vel, log_slope(depths, vel[:, -1])
