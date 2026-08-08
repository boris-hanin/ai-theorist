"""Hyperparameter-transfer harness (validation Leg C).

Sweeps a base learning rate against a scale dial, extracts the optimum at each
dial value, and reports whether it moves. This leg needs no DMFT solver at all
-- it tests the parameterisation table directly -- so it can be run for
architectures whose limiting equations have not been derived yet.

Two rules it enforces, both from the registry:

  * **A negative control is mandatory.** If a mis-scaled parameterisation also
    "transfers", the optimum is simply flat and the test is under-powered. That
    must be reported as under-powered, not as a pass (F17's rule applied here).
  * **Say which mechanism you are claiming.** "The optimum stops moving" can
    mean the optimum genuinely tracks, or that the loss surface has gone flat
    and the argmin is noise. `sharpness` measures the latter, so the two are
    distinguishable (F3's discipline applied to Leg C).
"""

import numpy as np

import nets


def sweep(build, dial_values, lr_grid, steps=200, P=32, D=16, seeds=(0, 1, 2),
          data_seed=0):
    """Returns (losses, per_seed).

    losses[i, j]         seed-median final loss  (for display)
    per_seed[i, j, s]    the individual runs     (for uncertainty on lr*)

    The optimum is located PER SEED and averaged, so its scatter gives an
    uncertainty. Locating it once on seed-averaged losses would give a number
    with no error bar, and "the optimum moved by 0.13 decades" is meaningless
    until you know whether 0.13 is larger than the noise.
    """
    ns = len(seeds)
    per_seed = np.full((len(dial_values), len(lr_grid), ns), np.inf)
    for i, dial in enumerate(dial_values):
        for j, lr in enumerate(lr_grid):
            for k, s in enumerate(seeds):
                net = build(dial, s)
                X, y = nets.teacher_data(P, net.D, seed=data_seed)
                per_seed[i, j, k] = nets.train(net, X, y, lr, steps)
    losses = np.median(per_seed, axis=2)
    return losses, per_seed


def _refined_argmin(row, lg):
    """Parabolic-refined log10(lr*) for one loss curve; nan if unusable."""
    fin = np.isfinite(row)
    if not np.any(fin):
        return np.nan, False
    k = int(np.nanargmin(np.where(fin, row, np.inf)))
    inside = 0 < k < len(row) - 1
    if inside and np.all(fin[k - 1:k + 2]):
        y0, y1, y2 = np.log(np.maximum(row[k - 1:k + 2], 1e-300))
        den = y0 - 2 * y1 + y2
        sh = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        return lg[k] + float(np.clip(sh, -1.0, 1.0)) * (lg[1] - lg[0]), inside
    return lg[k], inside


def verdict(losses, per_seed, lr_grid, dial_values, n_sigma=2.0,
            practical_bar=0.3):
    """Transfer verdict, judged on BOTH effect size and statistical resolution.

    Two bars, because either alone gives the wrong answer:

      * statistical -- the optimum is located per seed, and its across-seed
        scatter gives an uncertainty. Drift within noise is not evidence.
        (Same discipline as the solvers: judge against a MEASURED floor.)
      * practical -- `practical_bar` in decades, default 0.3 (a factor of 2 in
        learning rate). With 5 seeds the statistical test alone resolves a
        0.04-decade drift, i.e. a 10% shift in lr*, and would report that as a
        transfer failure. It is real but it is not what anyone means by
        "the hyperparameter failed to transfer".

    So FAILS requires the drift to be both larger than the practical bar and
    statistically resolved. A drift that is resolved but sub-threshold is
    reported as TRANSFERS with the residual noted, not hidden.

    UNDER-POWERED is a distinct outcome from a pass: if the optimum sits on a
    grid edge, or lr* is too noisy to locate, the sweep has established nothing
    and must not read as a pass.
    """
    lg = np.log10(np.asarray(lr_grid, dtype=float))
    n_dial, _, n_seed = per_seed.shape
    per_dial = np.full((n_dial, n_seed), np.nan)
    inside = np.zeros(n_dial, dtype=bool)
    for i in range(n_dial):
        oks = []
        for s in range(n_seed):
            v, ins = _refined_argmin(per_seed[i, :, s], lg)
            per_dial[i, s] = v
            oks.append(ins)
        inside[i] = all(oks)

    mean = np.nanmean(per_dial, axis=1)
    sem = np.nanstd(per_dial, axis=1, ddof=1) / np.sqrt(n_seed) if n_seed > 1 \
        else np.zeros(n_dial)
    pooled = float(np.sqrt(np.nanmean(sem ** 2))) if n_seed > 1 else 0.0
    drift = float(np.nanmax(mean) - np.nanmin(mean))
    thresh = n_sigma * pooled * np.sqrt(2.0)

    resolved = drift > max(thresh, 1e-9)

    # --- SHAPE of the drift, not just its size (F22) -----------------------
    # max-minus-min is blind to whether a drift is settling or running away.
    # Transfer is an ASYMPTOTIC claim, so a small drift that is monotone AND
    # accelerating is worse evidence than a larger one that is flattening: the
    # first says the exponent is wrong and has not finished showing it.
    m = mean[~np.isnan(mean)]
    if m.size >= 4:
        head = float(np.max(m[:3]) - np.min(m[:3]))
        tail = float(np.max(m[-3:]) - np.min(m[-3:]))
    else:
        head = tail = drift
    # Transfer is an ASYMPTOTIC claim, so the drift must SHRINK toward the large
    # end. `settling` is that test, and it is the one `max - min` cannot make.
    settling = bool(tail <= head * 1.3 + 1e-12)

    if not np.all(inside):
        status = "UNDER-POWERED (optimum on grid edge)"
    elif pooled > 0.25:
        status = "UNDER-POWERED (lr* too noisy to resolve)"
    elif drift > practical_bar and resolved:
        status = "FAILS"
    elif (not settling) and resolved:
        # Sub-threshold but running away: do NOT read as a pass.
        status = ("SUSPECT (drift %.2f dec is NOT settling: %.2f over the "
                  "largest three vs %.2f over the smallest three)"
                  % (drift, tail, head))
    elif resolved:
        status = "TRANSFERS (residual drift resolved but < %.2f dec)" % practical_bar
    else:
        status = "TRANSFERS"
    return {"status": status, "drift_log10": drift, "sem_log10": pooled,
            "tail_drift_log10": tail, "head_drift_log10": head,
            "settling": settling,
            "threshold_log10": float(thresh), "resolved": bool(resolved),
            "refined_log10_lr": mean, "per_seed_log10_lr": per_dial,
            "interior": inside, "dial": np.asarray(dial_values, dtype=float)}


def width_transfer(param, widths, lr_grid, L=1, D=16, act="tanh", gamma0=1.0, **kw):
    def build(N, seed):
        return nets.Net(D, N, L, param=param, gamma0=gamma0, act=act, seed=seed)
    losses, per_seed = sweep(build, widths, lr_grid, **kw)
    return losses, verdict(losses, per_seed, lr_grid, widths)


def depth_transfer(alpha, depths, lr_grid, N=256, D=16, act="tanh", block_k=1,
                   gamma0=1.0, lr_depth_exp=None, **kw):
    def build(L, seed):
        return nets.Net(D, N, L, param="mup", gamma0=gamma0, act=act, seed=seed,
                        residual=True, alpha=alpha, block_k=block_k,
                        lr_depth_exp=lr_depth_exp)
    losses, per_seed = sweep(build, depths, lr_grid, **kw)
    return losses, verdict(losses, per_seed, lr_grid, depths)


def render(name, dial_values, lr_grid, losses, v):
    """Compact text table of the sweep plus the verdict."""
    lines = ["  %s: %s   drift %.3f dec, lr* sem %.3f, threshold %.3f"
             % (name, v["status"], v["drift_log10"], v["sem_log10"],
                v["threshold_log10"]),
             "    dial |" + "".join("%9.1e" % lr for lr in lr_grid) + "   argmin"]
    for i, d in enumerate(dial_values):
        row = "".join("%9.3g" % x if np.isfinite(x) else "      inf"
                      for x in losses[i])
        lines.append("  %6g |%s   10^%.2f" % (d, row, v["refined_log10_lr"][i]))
    return "\n".join(lines)
