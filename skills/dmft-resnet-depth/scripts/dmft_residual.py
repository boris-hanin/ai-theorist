"""Residual DMFT solver with pre-LN, single-site Monte Carlo. P = 1, k = 1 blocks.

Implements the limit derived in `derivations/05-completep-dmft-sgd.md`,
specialised to a one-matrix block so there is one response pair per block rather
than two. Pre-LN throughout.

Architecture (finite N, what the solver is the limit of):

    h^1     = W^0 x / sqrt(D)                        (W^0 frozen)
    hbar^l  = LN(h^l)
    F^l     = (1/sqrt N) W^l phi(hbar^l)
    h^{l+1} = h^l + s F^l ,        s = beta0 L^{-alpha}
    f       = (1/(gamma0 N)) w . hbar^{L+1}

**Layer norm in the limit.** With h having i.i.d.-like entries across the width,
its per-sample mean self-averages to 0 and its variance to `H^l(t,t)`, so

    LN(h^l)  ->  h^l / sqrt(H^l(t,t))

which is a *deterministic* gain — no new stochastic field, just a kernel-set
rescaling. That is the whole reason pre-LN is tractable here.

Single-site system (one scalar trajectory per sample, per layer):

    F^l(t)     = u^l(t) + kappa int_0^t ds [A^l(t,s) + D(s) Phi^l(t,s)] g^{l+1}(s)
    zeta^l(t)  = r^l(t) + kappa int_0^t ds [B^l(t,s) + D(s) G^{l+1}(t,s)] phi(hbar^l(s))
    h^{l+1}(t) = h^l(t) + s F^l(t)
    g^l(t)     = g^{l+1}(t) + (s / sqrt(H^l(t,t))) phi_dot(hbar^l(t)) zeta^l(t)

    u^l ~ GP(0, Phi^l),   Phi^l = <phi(hbar^l) phi(hbar^l)>
    r^l ~ GP(0, G^{l+1}), G^l   = <g^l g^l>
    H^l = <h^l h^l>,      hbar^l(t) = h^l(t)/sqrt(H^l(t,t))
    kappa = d_L gamma0 beta0 L^{-alpha}

Readout (correlator rule, F4):  f(t) = gamma0^{-1} <w(t) hbar^{L+1}(t)>,
with  w(t) = w(0) + d_L gamma0 int_0^t ds D(s) hbar^{L+1}(s),
and   g^{L+1}(t) = w(t)/sqrt(H^{L+1}(t,t)).

STATUS -- NOT YET VALIDATED. First solver-vs-simulation comparison
(pre-LN, k=1, P=1, dt=0.05, T=12, sim seed-averaged over 8):

    L  alpha  plateau   gap vs sim   sim floor   MC floor   gap/combined
    2  1.0    8.0e-07   1.83e-02     3.48e-03    8.84e-03   1.92x
    2  0.5    8.7e-07   4.36e-02     1.90e-03    8.89e-03   4.79x
    4  1.0    9.4e-07   1.15e-02     1.94e-03    2.76e-02   0.42x
    4  0.5    9.4e-07   1.84e-02     6.24e-03    2.56e-02   0.70x

L=4 lands inside the combined floor; L=2 does not (1.9x and 4.8x). Note the
L=4 "agreement" rides on a much looser MC floor (2.6e-2 vs 8.8e-3), so the raw
gaps are comparable across L -- roughly 1-4e-2 throughout. Suggestive, not
validated. Candidates: the k=1 specialisation vs the k=2 derivation; the frozen
W^0 boundary; and the late-time fixed-point instability below.

FIXED-POINT BEHAVIOUR. The residual system is stiffer than the plain-MLP one
(F5): every block adds a feedback path through the stream. The iteration
descends to ~1e-6 and can then destabilise, so `solve` keeps the best iterate
and stops on stall rather than running into a non-finite covariance. `plateau`
reports where it actually got; treat a plateau far above `tol` as a warning.

Discrete-time conventions follow the certified solvers: correlator-rule
predictions (F4), responses from exact forward-mode sensitivities divided by dt
(the density/derivative distinction of round 003), common random numbers, and
antithetic readout pairs (F15).
"""

import numpy as np


class Diverged(RuntimeError):
    """Fixed point diverged (F5). Lower `damping` or anneal."""


def _root(cov):
    if not np.all(np.isfinite(cov)):
        raise Diverged("non-finite covariance -- lower damping (F5)")
    ev, evec = np.linalg.eigh((cov + cov.T) / 2.0)
    return evec @ np.diag(np.sqrt(np.clip(ev, 0.0, None)))


def solve_robust(y, L, alpha, dt, T, dampings=(0.3, 0.1, 0.03, 0.01), **kw):
    """Try progressively heavier damping. The residual fixed point is stiffer
    than the plain-MLP one (F5): every block adds a feedback path through the
    stream, so the effective Delta-loop norm grows with L."""
    kw.pop("damping", None)
    last = None
    for b in dampings:
        try:
            r = solve(y, L, alpha, dt, T, damping=b, **kw)
        except Diverged as e:
            last = e
            continue
        if r["converged"]:
            r["damping_used"] = b
            return r
        last = RuntimeError("did not converge at damping %g" % b)
    raise Diverged("no damping in %s worked: %s" % (list(dampings), last))


def solve(y, L, alpha, dt, T, S=4096, gamma0=1.0, beta0=1.0, Kx=1.0, d_L=None,
          damping=0.3, n_iter=60, tol=1e-6, S_resp=None, no_response=False,
          seed=0, verbose=False, stall=12):
    """Solve the residual DMFT. Returns f(t), kernels per layer, diagnostics."""
    if S % 2:
        raise ValueError("antithetic readout pairs need an even S")
    if d_L is None:
        d_L = L ** (2 * alpha - 1.0)
    s = beta0 * L ** (-alpha)
    kappa = d_L * gamma0 * beta0 * L ** (-alpha)
    Sr = min(S, max(256, 16 * T)) if S_resp is None else min(S_resp, S)
    y = float(y)
    tri = np.tril(np.ones((T, T)), -1)

    phi, phid = np.tanh, lambda x: 1.0 / np.cosh(x) ** 2
    phidd = lambda x: -2.0 * np.tanh(x) / np.cosh(x) ** 2

    # kernels, per layer.  Phi/G/A/B indexed 1..L ; H indexed 1..L+1
    Phi = [None] + [np.full((T, T), float(np.mean(phi(np.random.default_rng(0)
           .standard_normal(20000)) ** 2))) for _ in range(L)]
    G = [None] + [np.ones((T, T)) for _ in range(L + 1)]
    A = [None] + [np.zeros((T, T)) for _ in range(L)]
    B = [None] + [np.zeros((T, T)) for _ in range(L)]
    H = [None] + [np.full((T, T), Kx) for _ in range(L + 1)]
    Dlt = np.full(T, y)
    f = np.zeros(T)

    base = np.random.default_rng(seed)
    zu = [None] + [base.standard_normal((S, T)) for _ in range(L)]
    zr = [None] + [base.standard_normal((S, T)) for _ in range(L)]
    zh1 = base.standard_normal(S) * np.sqrt(Kx)      # h^1, static in t
    half = base.standard_normal(S // 2)
    zw = np.concatenate([half, -half])               # antithetic readout (F15)

    hist, best = [], None
    for it in range(n_iter):
        try:
            u = [None] + [zu[l] @ _root(Phi[l]).T for l in range(1, L + 1)]
            r = [None] + [zr[l] @ _root(G[l + 1]).T for l in range(1, L + 1)]
        except Diverged:
            # Blew up on a later iterate. If we already have a good one, keep it
            # rather than throwing the whole solve away.
            if best is None:
                raise
            break

        h = [None] + [np.zeros((S, T)) for _ in range(L + 1)]
        g = [None] + [np.zeros((S, T)) for _ in range(L + 1)]
        hb = [None] + [np.zeros((S, T)) for _ in range(L + 1)]
        F = [None] + [np.zeros((S, T)) for _ in range(L)]
        zet = [None] + [np.zeros((S, T)) for _ in range(L)]
        Sh = [None] + [np.zeros((Sr, T, T)) for _ in range(L)]   # dF/dr
        Sz = [None] + [np.zeros((Sr, T, T)) for _ in range(L)]   # dzeta/du
        An = [None] + [np.zeros((T, T)) for _ in range(L)]
        Bn = [None] + [np.zeros((T, T)) for _ in range(L)]
        w = np.zeros((S, T))

        for t in range(T):
            past = slice(0, t)
            R = slice(0, Sr)
            h[1][:, t] = zh1
            # -- forward sweep up the stream
            for l in range(1, L + 1):
                Hd = max(float(np.mean(h[l][:, t] ** 2)), 1e-30)
                hb[l][:, t] = h[l][:, t] / np.sqrt(Hd)
                if t:
                    kerA = A[l][t, past] + Dlt[past] * Phi[l][t, past]
                    F[l][:, t] = u[l][:, t] + kappa * dt * (g[l + 1][:, past] @ kerA)
                else:
                    F[l][:, t] = u[l][:, t]
                h[l + 1][:, t] = h[l][:, t] + s * F[l][:, t]
            HdL = max(float(np.mean(h[L + 1][:, t] ** 2)), 1e-30)
            hb[L + 1][:, t] = h[L + 1][:, t] / np.sqrt(HdL)

            # -- readout and prediction (correlator rule, F4)
            w[:, t] = zw + (d_L * gamma0 * dt) * (hb[L + 1][:, past] @ Dlt[past]) \
                if t else zw
            f[t] = float(np.mean(w[:, t] * hb[L + 1][:, t])) / gamma0
            Dlt[t] = y - f[t]
            g[L + 1][:, t] = w[:, t] / np.sqrt(HdL)

            # -- backward sweep down the stream
            for l in range(L, 0, -1):
                Hd = max(float(np.mean(h[l][:, t] ** 2)), 1e-30)
                if t:
                    kerB = B[l][t, past] + Dlt[past] * G[l + 1][t, past]
                    zet[l][:, t] = r[l][:, t] + kappa * dt * (phi(hb[l][:, past]) @ kerB)
                else:
                    zet[l][:, t] = r[l][:, t]
                g[l][:, t] = g[l + 1][:, t] + (s / np.sqrt(Hd)) * \
                    phid(hb[l][:, t]) * zet[l][:, t]

            # -- forward-mode sensitivities -> responses (kernel = S/dt)
            for l in range(1, L + 1):
                Sz[l][:, t, t] = 1.0
                if t:
                    Hd = max(float(np.mean(h[l][:, t] ** 2)), 1e-30)
                    kerA = A[l][t, past] + Dlt[past] * Phi[l][t, past]
                    kerB = B[l][t, past] + Dlt[past] * G[l + 1][t, past]
                    dg = (s / np.sqrt(Hd)) * (
                        phidd(hb[l][R, past]) * zet[l][R, past])[:, :, None] * Sh[l][:, past, :] \
                        + (s / np.sqrt(Hd)) * phid(hb[l][R, past])[:, :, None] * Sz[l][:, past, :]
                    Sh[l][:, t, :] = (kappa * dt) * (kerA @ dg)
                    Sz[l][:, t, :] += (kappa * dt) * (
                        kerB @ (phid(hb[l][R, past])[:, :, None] * Sh[l][:, past, :]))
                An[l][t, :] = np.mean(Sh[l][:, t, :], axis=0) / (kappa * dt)
                Bn[l][t, :] = np.mean(
                    phid(hb[l][:Sr, t])[:, None] * Sz[l][:, t, :], axis=0) / (kappa * dt)

        # -- kernels and damped update
        shift, b = 0.0, damping
        for l in range(1, L + 1):
            Pn = (phi(hb[l]).T @ phi(hb[l])) / S
            Gn = (g[l].T @ g[l]) / S
            Hn = (h[l].T @ h[l]) / S
            shift = max(shift, np.abs(Pn - Phi[l]).max(), np.abs(Gn - G[l]).max())
            Phi[l] = (1 - b) * Phi[l] + b * Pn
            G[l] = (1 - b) * G[l] + b * Gn
            H[l] = (1 - b) * H[l] + b * Hn
            if not no_response:
                A[l] = (1 - b) * A[l] + b * (An[l] * tri)
                B[l] = (1 - b) * B[l] + b * (Bn[l] * tri)
        H[L + 1] = (1 - b) * H[L + 1] + b * ((h[L + 1].T @ h[L + 1]) / S)
        G[L + 1] = (1 - b) * G[L + 1] + b * ((g[L + 1].T @ g[L + 1]) / S)
        hist.append(shift)
        if verbose:
            print("   it %2d shift %.3e f(T)=%.5f" % (it, shift, f[-1]))
        # Keep the best iterate. The residual fixed point descends to ~1e-4 and
        # can then destabilise (every block adds a feedback path through the
        # stream, so the Delta-loop is stiffer than the plain-MLP case, F5).
        # Returning the best-so-far is honest: `plateau` records where it got.
        if not np.isfinite(shift):
            break
        if best is None or shift < best[0]:
            best = (shift, f.copy(), Dlt.copy(), it)
        if shift < tol:
            break
        if it - best[3] >= stall:
            break

    if best is not None:
        f, Dlt = best[1], best[2]
    return {"t": np.arange(T) * dt, "f": f.copy(), "Delta": Dlt.copy(),
            "plateau": (best[0] if best else np.inf),
            "best_iter": (best[3] if best else -1),
            "Phi": Phi, "G": G, "H": H, "A": A, "B": B,
            "kappa": kappa, "s": s, "d_L": d_L, "iters": len(hist),
            "converged": (best is not None and best[0] < tol),
            "final_shift": (best[0] if best else np.inf),
            "L": L, "alpha": alpha, "S": S, "T": T, "dt": dt}
