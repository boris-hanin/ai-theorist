"""Deep nonlinear DMFT (general L, P=1) by single-site Monte Carlo.

Generalises `dmft_l2_nonlinear.py` to arbitrary depth. Responses come from exact
forward-mode sensitivities; predictions from the correlator rule (F4); readout
sources are antithetic (F15); the fixed point uses common random numbers.

Structure (from `derivations/01-deep-mlp.md`, verified in round 002):

    h^l(t) = u^l(t) + g0 dt sum_{s<t} [A^{l-1}(t,s) + D(s) Phi^{l-1}(t,s)] g^l(s)
    z^l(t) = r^l(t) + g0 dt sum_{s<t} [B^l(t,s)     + D(s) G^{l+1}(t,s)] phi(h^l(s))
    g^l    = phidot(h^l) z^l
    u^l ~ GP(0, Phi^{l-1}),  r^l ~ GP(0, G^{l+1})
    Phi^0 = Kx,  G^{L+1} = 1,  A^0 = 0,  B^L = 0
    f(t)   = g0^{-1} <z^L(t) phi(h^L(t))>

**Key structural fact that makes this cheap:** within one time step the layers
are INDEPENDENT given the kernels. All inter-layer coupling runs through
Phi, G, A, B, which are held fixed during a sweep and updated between
iterations. So a step is L independent scalar updates, not a chain.

Per layer we need two sensitivity families:

    d(.)/d r^l   ->  A^l     = <phidot(h^l(t)) Sh_r[t,s]> / (g0 dt)
    d(.)/d u^l   ->  B^{l-1} = <phiddot(h^l) z^l Sh_u + phidot(h^l) Sz_u> / (g0 dt)

with A^l needed for l = 1..L-1 and B^{l-1} for l = 2..L, so layer 1 needs only
the r-family and layer L only the u-family.

The `/dt` is not cosmetic: the discrete sensitivity S[t,s] = d(field_t)/d(src_s)
equals dt * delta(field(t))/delta(src(s)), so the kernel is S/dt. Omitting it
divides the response by dt twice and yields a self-consistent but WRONG fixed
point (round 003 measured a 15-19% error that did not shrink with S).

`onsager` defaults to 0: round 003 measured the equal-time endpoint term against
simulations extrapolated to N -> infinity and found weight 0 fits (4.4e-3 vs a
2.6e-3 floor) while weight 1 does not (1.07e-2). The parameter is kept so the
measurement can be repeated at other depths.
"""

import numpy as np

import activations


def _psd_root(cov):
    ev, evec = np.linalg.eigh((cov + cov.T) / 2.0)
    return evec @ np.diag(np.sqrt(np.clip(ev, 0.0, None)))


def solve(y, gamma0, dt, T, L, S=8192, act="tanh", seed=0, Kx=1.0,
          onsager=0.0, damping=0.5, n_iter=60, tol=1e-6, S_resp=None,
          no_response=False, verbose=False):
    """Solve the depth-L, P=1 nonlinear DMFT. Returns f(t), kernels, responses."""
    phi, phidot = activations.get(act)

    def phiddot(x):
        e = 1e-4
        return (phidot(x + e) - phidot(x - e)) / (2 * e)

    if S % 2 != 0:
        raise ValueError("antithetic readout pairs need an even S")
    y = float(y)
    Sr = min(S, 512) if S_resp is None else min(S_resp, S)
    ones = np.ones((T, T))
    tri = np.tril(ones, -1)

    # Kernels. Index 0..L for Phi (Phi[0] = Kx), 1..L+1 for G (G[L+1] = 1).
    rng = np.random.default_rng(seed)
    p0 = float(np.mean(phi(rng.standard_normal(20000) * np.sqrt(Kx)) ** 2))
    Phi = [np.full((T, T), Kx)] + [np.full((T, T), p0) for _ in range(L)]
    G = [None] + [ones.copy() for _ in range(L)] + [ones.copy()]
    A = [np.zeros((T, T)) for _ in range(L + 1)]      # A[0] = 0 always
    B = [np.zeros((T, T)) for _ in range(L + 1)]      # B[L] = 0 always
    Delta = np.full(T, y)
    f = np.zeros(T)

    # Common random numbers; antithetic on the readout source r^L (F15).
    base = np.random.default_rng(seed)
    zu = [None] + [base.standard_normal((S, T)) for _ in range(L)]
    zr = [None] + [base.standard_normal((S, T)) for _ in range(L)]
    half = base.standard_normal((S // 2, T))
    zu[L] = np.concatenate([half, half], axis=0)        # twin shares the carrier
    hr = base.standard_normal(S // 2)
    zr[L] = np.concatenate([hr, -hr])[:, None] * np.ones((1, T))

    hist = []
    for it in range(n_iter):
        u = [None] + [zu[l] @ _psd_root(Phi[l - 1]).T for l in range(1, L + 1)]
        r = [None] + [zr[l] @ _psd_root(G[l + 1]).T if l < L else zr[L]
                      for l in range(1, L + 1)]

        h = [None] + [np.zeros((S, T)) for _ in range(L)]
        z = [None] + [np.zeros((S, T)) for _ in range(L)]
        g = [None] + [np.zeros((S, T)) for _ in range(L)]
        ph = [None] + [np.zeros((S, T)) for _ in range(L)]
        # sensitivity families: r-family for l<L, u-family for l>1
        Shr = [None] + [np.zeros((Sr, T, T)) if l < L else None for l in range(1, L + 1)]
        Szr = [None] + [np.zeros((Sr, T, T)) if l < L else None for l in range(1, L + 1)]
        Shu = [None] + [np.zeros((Sr, T, T)) if l > 1 else None for l in range(1, L + 1)]
        Szu = [None] + [np.zeros((Sr, T, T)) if l > 1 else None for l in range(1, L + 1)]
        A_new = [np.zeros((T, T)) for _ in range(L + 1)]
        B_new = [np.zeros((T, T)) for _ in range(L + 1)]

        for t in range(T):
            past = slice(0, t)
            R = slice(0, Sr)
            kerA, kerB = {}, {}
            # Layers are independent within a step given the kernels.
            for l in range(1, L + 1):
                if t:
                    kerA[l] = A[l - 1][t, past] + Delta[past] * Phi[l - 1][t, past]
                    kerB[l] = B[l][t, past] + Delta[past] * G[l + 1][t, past]
                    h[l][:, t] = u[l][:, t] + (gamma0 * dt) * (g[l][:, past] @ kerA[l])
                else:
                    h[l][:, t] = u[l][:, t]
                ph[l][:, t] = phi(h[l][:, t])

            for l in range(1, L + 1):
                if t:
                    z[l][:, t] = r[l][:, t] + (gamma0 * dt) * (ph[l][:, past] @ kerB[l])
                else:
                    z[l][:, t] = r[l][:, t]
                # Equal-time endpoint term (measured weight 0, round 003).
                if onsager and l < L and not no_response:
                    ons = float(np.mean(phiddot(h[l + 1][:, t]) * z[l + 1][:, t]))
                    z[l][:, t] += onsager * ons * ph[l][:, t]
                g[l][:, t] = phidot(h[l][:, t]) * z[l][:, t]

            # -- forward-mode sensitivities, per layer, per family
            for l in range(1, L + 1):
                if Shr[l] is not None:
                    Szr[l][:, t, t] = 1.0
                    if t:
                        dg = (phiddot(h[l][R, past]) * z[l][R, past])[:, :, None] * Shr[l][:, past, :] \
                            + phidot(h[l][R, past])[:, :, None] * Szr[l][:, past, :]
                        Shr[l][:, t, :] = (gamma0 * dt) * (kerA[l] @ dg)
                        Szr[l][:, t, :] += (gamma0 * dt) * (
                            kerB[l] @ (phidot(h[l][R, past])[:, :, None] * Shr[l][:, past, :]))
                    A_new[l][t, :] = np.mean(
                        phidot(h[l][:Sr, t])[:, None] * Shr[l][:, t, :], axis=0) / (gamma0 * dt)
                if Shu[l] is not None:
                    Shu[l][:, t, t] = 1.0
                    if t:
                        dg = (phiddot(h[l][R, past]) * z[l][R, past])[:, :, None] * Shu[l][:, past, :] \
                            + phidot(h[l][R, past])[:, :, None] * Szu[l][:, past, :]
                        Shu[l][:, t, :] += (gamma0 * dt) * (kerA[l] @ dg)
                        Szu[l][:, t, :] = (gamma0 * dt) * (
                            kerB[l] @ (phidot(h[l][R, past])[:, :, None] * Shu[l][:, past, :]))
                    B_new[l - 1][t, :] = np.mean(
                        (phiddot(h[l][:Sr, t]) * z[l][:Sr, t])[:, None] * Shu[l][:, t, :]
                        + phidot(h[l][:Sr, t])[:, None] * Szu[l][:, t, :],
                        axis=0) / (gamma0 * dt)

            f[t] = float(np.mean(z[L][:, t] * ph[L][:, t])) / gamma0
            Delta[t] = y - f[t]

        # -- kernel re-estimation and damped update
        shift = 0.0
        b = damping
        for l in range(1, L + 1):
            Pn = (ph[l].T @ ph[l]) / S
            Gn = (g[l].T @ g[l]) / S
            shift = max(shift, np.abs(Pn - Phi[l]).max(), np.abs(Gn - G[l]).max())
            Phi[l] = (1 - b) * Phi[l] + b * Pn
            G[l] = (1 - b) * G[l] + b * Gn
            if not no_response:
                if l < L:
                    An = A_new[l] * tri
                    shift = max(shift, np.abs(An - A[l]).max())
                    A[l] = (1 - b) * A[l] + b * An
                if l > 1:
                    Bn = B_new[l - 1] * tri
                    shift = max(shift, np.abs(Bn - B[l - 1]).max())
                    B[l - 1] = (1 - b) * B[l - 1] + b * Bn
        hist.append(shift)
        if verbose:
            print("   it %2d  shift %.3e  f(T)=%.5f" % (it, shift, f[-1]))
        if shift < tol:
            break
        if not np.isfinite(shift):
            raise RuntimeError("nonlinear fixed point diverged at iter %d (F5)" % it)

    ntk = sum(G[l + 1] * Phi[l] for l in range(L + 1))
    return {"t": np.arange(T) * dt, "f": f.copy(), "Delta": Delta.copy(),
            "Phi": Phi, "G": G, "A": A, "B": B, "K": ntk,
            "B_diag": [np.diag(B_new[l]) for l in range(L)],
            "iters": len(hist), "final_shift": hist[-1], "converged": hist[-1] < tol,
            "L": L, "S": S, "dt": dt, "T": T, "gamma0": gamma0, "onsager": onsager}
