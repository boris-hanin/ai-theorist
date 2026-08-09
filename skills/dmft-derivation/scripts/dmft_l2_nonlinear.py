"""L=2 nonlinear DMFT with the full response sector, by single-site Monte Carlo.

Scope: L = 2, P = 1. This is the MINIMAL architecture with a nonzero Onsager
term -- A^0 = 0 and B^2 = 0 by the boundaries, but B^1 survives and carries
phi_ddot. P = 1 is sufficient because the Onsager term is diagonal in the data
index, and it keeps the sensitivity arrays two-index (T x T per sample) instead
of four.

Responses come from exact forward-mode sensitivities propagated alongside the
trajectories (never finite differences in production).

The equal-time term
-------------------
`derivations/01-deep-mlp.md` §7 derives, in continuum, that the equal-time
response is a delta function contributing at O(1):

    B^1(t,s) ⊃ gamma_0^{-1} <phi_ddot(h^2(t)) z^2(t)> delta(t-s)

but a delta sitting exactly at the endpoint of int_0^t ds is convention-
dependent (weight 0, 1/2, or 1), and the discrete recursion below gives
S_h2[t,t] = 1 rather than 1/dt, which would make the term O(dt) instead.
Rather than pick, the coefficient is exposed as `onsager` and MEASURED:

    z^1(t) += onsager * <phi_ddot(h^2(t)) z^2(t)> * phi(h^1(t))

Round 003 swept this coefficient against finite-width extrapolations. It found
that `onsager = 0` (strict-past endpoint) fits and `onsager = 1` does not. Zero
is therefore the default; the parameter remains exposed so the endpoint can be
retested in other depths and regimes.

Note the write-order (F17): the Onsager contribution to z^1(t) reads h^1(t),
computed earlier in the same step. Writing it after the read would silently
switch the term off while spot-checks of B^1 still looked right.
"""

import numpy as np

import activations


def _psd_root(cov):
    evals, evecs = np.linalg.eigh((cov + cov.T) / 2.0)
    return evecs @ np.diag(np.sqrt(np.clip(evals, 0.0, None)))


def solve(y, gamma0, dt, T, S=2000, act="tanh", seed=0, Kx=1.0,
          onsager=0.0, damping=0.5, n_iter=60, tol=1e-6,
          no_response=False, verbose=False, S_resp=None):
    """Solve the L=2, P=1 DMFT. Returns f(t), kernels, responses, diagnostics.

    `S_resp` (default min(S, 512)) is the number of samples carried through the
    sensitivity propagation. The kernels cost O(S*T^2) but the sensitivities
    cost O(S*T^3) in time and O(S*T^2) in memory, which dominates everything
    else; at S=32000, T=30 the sensitivity arrays alone are ~1 GB. The
    responses are population averages of smooth quantities and converge on far
    fewer samples than the kernels do, so they are estimated on a subsample.
    `validate_deep_nonlinear.py` checks that doubling S_resp does not move the
    answer -- if it does, this shortcut is not safe at those settings.
    """
    phi, phidot = activations.get(act)
    # phi_ddot by central differences on the activation itself (not on the
    # solve): cheap, and exact for the analytic forms we use.
    def phiddot(x):
        e = 1e-4
        return (phidot(x + e) - phidot(x - e)) / (2 * e)

    rng = np.random.default_rng(seed)
    y = float(y)
    Sr = min(S, 512) if S_resp is None else min(S_resp, S)

    # Kernels on the time grid. Initialise lazily: h^l ~ GP(0, Kx), g ~ O(1).
    Phi1 = np.full((T, T), float(np.mean(phi(rng.standard_normal(20000) * np.sqrt(Kx)) ** 2)))
    Phi2 = Phi1.copy()
    G2 = np.ones((T, T))
    A1 = np.zeros((T, T))
    B1 = np.zeros((T, T))
    Delta = np.full(T, y)
    f = np.zeros(T)

    tri = np.tril(np.ones((T, T)), -1)          # strict past
    hist = []

    # COMMON RANDOM NUMBERS. The base normals are drawn once; each iteration
    # only re-applies the current covariance root. Redrawing them per iteration
    # makes every iterate a different noise realisation, so the fixed-point
    # residual can never fall below the Monte-Carlo floor -- measured: the
    # shift stalled at 2e-2 and the solve never converged.
    if S % 2 != 0:
        raise ValueError("antithetic readout pairs need an even S")
    base = np.random.default_rng(seed)
    zu1 = base.standard_normal(S)
    zr1 = base.standard_normal((S, T))
    zu2 = base.standard_normal((S // 2, T))
    zu2 = np.concatenate([zu2, zu2], axis=0)         # twin shares the carrier
    zr2h = base.standard_normal(S // 2)
    zr2 = np.concatenate([zr2h, -zr2h])              # ...and negates the readout
    # F15: with u^2 shared and r^2 negated, <r^2 phi(u^2)> = 0 exactly, so the
    # gamma_0^{-1}-amplified readout correlator gives f(0) = 0 with no floor.

    for it in range(n_iter):
        u1 = zu1 * np.sqrt(Kx)                           # static in t
        r1 = zr1 @ _psd_root(G2).T                       # GP(0, G^2)
        u2 = zu2 @ _psd_root(Phi1).T                     # GP(0, Phi^1)
        r2 = zr2                                         # G^3 = 1, static

        h1 = np.zeros((S, T)); z1 = np.zeros((S, T)); g1 = np.zeros((S, T))
        h2 = np.zeros((S, T)); z2 = np.zeros((S, T)); g2 = np.zeros((S, T))
        # forward-mode sensitivities  d(field)(t)/d(source)(s), s <= t
        Sh1 = np.zeros((Sr, T, T)); Sz1 = np.zeros((Sr, T, T))
        Sh2 = np.zeros((Sr, T, T)); Sz2 = np.zeros((Sr, T, T))

        ph1 = np.zeros((S, T))
        A1_new = np.zeros((T, T)); B1_new = np.zeros((T, T))

        for t in range(T):
            past = slice(0, t)
            # ---- layer 1 forward
            mem = (gamma0 * dt * Kx) * (g1[:, past] @ Delta[past]) if t else 0.0
            h1[:, t] = u1 + mem
            ph1[:, t] = phi(h1[:, t])
            # ---- layer 2 (independent of z1 at this step)
            if t:
                ker2 = A1[t, past] + Delta[past] * Phi1[t, past]
                h2[:, t] = u2[:, t] + gamma0 * dt * (g2[:, past] @ ker2)
                z2[:, t] = r2 + gamma0 * dt * (phi(h2[:, past]) @ Delta[past])
            else:
                h2[:, t] = u2[:, t]
                z2[:, t] = r2
            g2[:, t] = phidot(h2[:, t]) * z2[:, t]

            # ---- Onsager coefficient at time t (write BEFORE the read, F17)
            ons = float(np.mean(phiddot(h2[:, t]) * z2[:, t]))

            # ---- layer 1 backward
            if t:
                ker1 = B1[t, past] + Delta[past] * G2[t, past]
                z1[:, t] = r1[:, t] + gamma0 * dt * (ph1[:, past] @ ker1)
            else:
                z1[:, t] = r1[:, t]
            if not no_response:
                z1[:, t] += onsager * ons * ph1[:, t]
            g1[:, t] = phidot(h1[:, t]) * z1[:, t]

            # ---- sensitivities (exact forward mode)
            Sz1[:, t, t] = 1.0
            Sh2[:, t, t] = 1.0
            if t:
                R = slice(0, Sr)
                dg1 = (phiddot(h1[R, past]) * z1[R, past])[:, :, None] * Sh1[:, past, :] \
                    + phidot(h1[R, past])[:, :, None] * Sz1[:, past, :]
                # v @ A with A of shape (n, s, k) is a batched matvec over s,
                # BLAS-backed; np.einsum("s,nsk->nk", ...) is the same contraction
                # but ~10x slower here.
                Sh1[:, t, :] = (gamma0 * dt * Kx) * (Delta[past] @ dg1)
                Sz1[:, t, :] += (gamma0 * dt) * (
                    ker1 @ (phidot(h1[R, past])[:, :, None] * Sh1[:, past, :]))

                dg2 = (phiddot(h2[R, past]) * z2[R, past])[:, :, None] * Sh2[:, past, :] \
                    + phidot(h2[R, past])[:, :, None] * Sz2[:, past, :]
                Sh2[:, t, :] += (gamma0 * dt) * (ker2 @ dg2)
                Sz2[:, t, :] = (gamma0 * dt) * (
                    Delta[past] @ (phidot(h2[R, past])[:, :, None] * Sh2[:, past, :]))

            # ---- response rows, written before they are read next step (F17)
            #
            # The DISCRETE sensitivity is S[t,s] = d(field_t)/d(source_s), which
            # equals dt * delta(field(t))/delta(source(s)) -- the continuum
            # object is a density. So the kernel is S/dt. Omitting this divides
            # the response by dt twice (once here, once in the memory sum) and
            # produces a self-consistent but WRONG fixed point: measured a 15-19%
            # error against the exact algebraic solver that did not shrink with S.
            #
            # It also settles the equal-time question. S_h2[t,t] = 1, so
            # B^1[t,t] = gamma_0^{-1} <phi_ddot(h^2) z^2> / dt -- the 1/dt
            # diagonal that `derivations/01-deep-mlp.md` §7 predicts. Weighted by
            # gamma_0*dt in the memory sum it contributes at O(1), confirming the
            # continuum derivation. The apparent conflict noted in this module's
            # docstring was this bookkeeping error, not a real ambiguity.
            A1_new[t, :] = np.mean(phidot(h1[:Sr, t])[:, None] * Sh1[:, t, :],
                                   axis=0) / (gamma0 * dt)
            B1_new[t, :] = np.mean(
                (phiddot(h2[:Sr, t]) * z2[:Sr, t])[:, None] * Sh2[:, t, :]
                + phidot(h2[:Sr, t])[:, None] * Sz2[:, t, :], axis=0) / (gamma0 * dt)

            # ---- prediction by the correlator rule (F4)
            f[t] = float(np.mean(z2[:, t] * phi(h2[:, t]))) / gamma0
            Delta[t] = y - f[t]

        # --- kernel re-estimation ---------------------------------------
        ph2 = phi(h2)
        Phi1_new = (ph1.T @ ph1) / S
        Phi2_new = (ph2.T @ ph2) / S
        G2_new = (g2.T @ g2) / S

        shift = max(np.abs(Phi1_new - Phi1).max(), np.abs(G2_new - G2).max(),
                    np.abs(A1_new * tri - A1).max(), np.abs(B1_new * tri - B1).max())
        b = damping
        Phi1 = (1 - b) * Phi1 + b * Phi1_new
        Phi2 = (1 - b) * Phi2 + b * Phi2_new
        G2 = (1 - b) * G2 + b * G2_new
        A1 = (1 - b) * A1 + b * (A1_new * tri)
        B1 = (1 - b) * B1 + b * (B1_new * tri)
        hist.append(shift)
        if verbose:
            print("   it %2d shift %.3e  f(T)=%.5f" % (it, shift, f[-1]))
        if shift < tol:
            break

    return {"t": np.arange(T) * dt, "f": f.copy(), "Delta": Delta.copy(),
            "Phi1": Phi1, "Phi2": Phi2, "G2": G2, "A1": A1, "B1": B1,
            "B1_diag": np.diag(B1_new), "iters": len(hist),
            "final_shift": hist[-1], "converged": hist[-1] < tol,
            "onsager": onsager, "S": S, "dt": dt, "T": T, "gamma0": gamma0}
