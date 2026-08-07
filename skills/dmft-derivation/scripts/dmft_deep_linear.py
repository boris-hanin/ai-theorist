"""Deep linear DMFT: algebraic closure, no Monte Carlo.

Implements `derivations/02-deep-linear.md`. This is the first solver in the
program with a live response sector, and it is exact -- there is no sampling
floor, so a response-sector bug cannot hide under Monte-Carlo noise.

Indexing: (mu, t) -> t*P + mu (time-major), so the causal mask is
block-lower-triangular.

Operator convention: C and D carry the `dt` of the memory integral; the kernels
A, B, H, G do not. So `A_op = M^{-1} C_op` is an operator, and it re-enters the
next layer's operator as `C_op^{l+1} = mask * (A_op^l + dt * H^l diag(Delta))`.
Getting this wrong by one factor of dt is the obvious failure here, and check
C2 below (reduction to the certified two-layer solver) is what catches it.

What this certifies: operator construction, causal masking, the Neumann/M^{-1}
structure, the damped fixed point (F5), and write-before-read ordering (F17),
all with zero sampling floor.

What it CANNOT certify: the equal-time Onsager diagonal (F1). That term carries
phi_ddot and vanishes identically for linear phi -- which is exactly F1b. Only
the nonlinear L>=2 solver can test it.
"""

import numpy as np

import exact


def causal_mask(P, T, strict=True):
    """mask[(mu,t),(al,s)] = 1 iff s < t (strict) or s <= t."""
    tt = np.repeat(np.arange(T), P)
    return (tt[:, None] > tt[None, :]) if strict else (tt[:, None] >= tt[None, :])


def _static(Kmat, P, T):
    """Lift a P x P kernel to PT x PT, constant in both time arguments."""
    return np.tile(Kmat, (T, T))


class Diverged(RuntimeError):
    """The fixed point diverged (F5). Raise damping or anneal gamma_0."""


def solve_annealed(Kx, y, gamma0, dt, T, L, n_rungs=4, dampings=(0.3, 0.1, 0.03),
                   **kw):
    """Anneal gamma_0 upward, warm-starting each rung, dropping damping on failure.

    Both prescribed fixes for F5, applied adaptively. Measured stability map at
    L=3, dt=0.02 (`rounds/002`): the fixed point survives while
    gamma_0 * (T*dt) is below roughly 6, and the damping needed to get there
    falls as that product grows --

        gamma_0=3, T=101 needs beta=0.1;  gamma_0=6, T=51 diverges at every
        fixed beta tried, but converges when annealed.

    consistent with F5's operator norm ~ dt*lambda*T. Raises Diverged if no
    combination works, rather than returning a silently wrong answer.
    """
    kw.pop("damping", None)
    state = None
    res = None
    for k in range(1, n_rungs + 1):
        g = gamma0 * k / n_rungs
        for beta in dampings:
            try:
                res = solve(Kx, y, g, dt, T, L, damping=beta, init=state, **kw)
            except (Diverged, np.linalg.LinAlgError):
                continue
            if res["converged"] and np.all(np.isfinite(res["f"])):
                state = res
                break
        else:
            raise Diverged(
                "annealing failed at gamma_0 = %.3g (rung %d/%d); "
                "gamma_0*horizon = %.2g is past the measured stable edge"
                % (g, k, n_rungs, gamma0 * T * dt))
    return res


def solve(Kx, y, gamma0, dt, T, L, damping=0.5, n_iter=300, tol=1e-10,
          strict=True, verbose=False, no_response=False, no_resum=False,
          init=None):
    """Solve the deep linear DMFT by damped fixed point.

    Returns dict with f (T x P), Delta, kernels H[l], G[l], responses A[l], B[l],
    the NTK, and convergence diagnostics.

    `no_response=True` clamps the interior responses A^l, B^l to zero. This is
    the physical no-response control (F17): it must CHANGE the answer at L >= 2
    and must be inert at L = 1, where the boundaries already force A^0 = B^1 = 0.
    An ablation that changes nothing is a red flag, not a pass.
    """
    Kx = np.asarray(Kx, dtype=float)
    y = np.asarray(y, dtype=float)
    P = len(y)
    PT = P * T
    mask = causal_mask(P, T, strict)
    I = np.eye(PT)

    # Boundaries: H^0 = K^x (static), G^{L+1} = 1 1^T, A^0 = 0, B^L = 0.
    H0 = _static(Kx, P, T)
    Gtop = np.ones((PT, PT))

    # Kernels H[1..L] and G[1..L]; responses A[1..L-1], B[1..L-1] (interior only).
    H = [H0] + [H0.copy() for _ in range(L)]
    G = [None] + [Gtop.copy() for _ in range(L)] + [Gtop]
    A = [np.zeros((PT, PT)) for _ in range(L + 1)]   # A[0] = 0 always
    B = [np.zeros((PT, PT)) for _ in range(L + 1)]   # B[L] = 0 always

    if init is not None:
        # Warm start from a previous solve (gamma_0 annealing, F5).
        H = [h.copy() for h in init["H"]]
        G = [None if g is None else g.copy() for g in init["G"]]
        A = [a.copy() for a in init["A"]]
        B = [b.copy() for b in init["B"]]
        Delta = init["Delta"].reshape(-1).copy()
    else:
        # Lazy initial guess for Delta: K_0 = (L+1) K^x for linear nets.
        K0 = (L + 1) * Kx
        f = exact.lazy_prediction_discrete(K0, y, dt, T - 1)
        Delta = (y[None, :] - f).reshape(-1)         # (PT,) time-major

    hist = []
    for it in range(n_iter):
        dvec = Delta

        # Operators. C^l from below (A^{l-1}, H^{l-1}); D^l from above (B^l, G^{l+1}).
        C = [None] * (L + 1)
        D = [None] * (L + 1)
        for l in range(1, L + 1):
            C[l] = mask * (A[l - 1] + dt * (H[l - 1] * dvec[None, :]))
            D[l] = mask * (B[l] + dt * (G[l + 1] * dvec[None, :]))

        A_new = [np.zeros((PT, PT)) for _ in range(L + 1)]
        B_new = [np.zeros((PT, PT)) for _ in range(L + 1)]
        H_new = [H0] + [None] * L
        G_new = [None] * (L + 2)
        G_new[L + 1] = Gtop

        Minv_store, Ninv_store = {}, {}
        for l in range(1, L + 1):
            M = I - gamma0 ** 2 * (C[l] @ D[l])
            Nt = I - gamma0 ** 2 * (D[l] @ C[l])
            if not (np.all(np.isfinite(M)) and np.all(np.isfinite(Nt))):
                raise Diverged("fixed point diverged at iter %d (F5): raise "
                               "damping or use solve_annealed()" % it)
            Minv_store[l], Ninv_store[l] = M, Nt
            # Responses (deterministic for linear nets).
            # The Neumann resummation M^{-1} is NOT negligible: dropping it
            # shifts f by 0.6% at gamma_0=1 and 1.7% at gamma_0=3 (measured).
            if not no_response:
                try:
                    A_new[l] = C[l].copy() if no_resum else np.linalg.solve(M, C[l])
                    B_new[l - 1] = (D[l].copy() if no_resum
                                    else np.linalg.solve(Nt, D[l]))
                except np.linalg.LinAlgError:
                    raise Diverged("singular M/Nt at iter %d (F5)" % it)

        # Kernels: H sweeps up from H^0, G sweeps down from G^{L+1}.
        for l in range(1, L + 1):
            M = Minv_store[l]
            S = H_new[l - 1] + gamma0 ** 2 * (C[l] @ G[l + 1] @ C[l].T)
            tmp = np.linalg.solve(M, S)
            H_new[l] = np.linalg.solve(M, tmp.T).T
        for l in range(L, 0, -1):
            Nt = Ninv_store[l]
            S = G_new[l + 1] + gamma0 ** 2 * (D[l] @ H[l - 1] @ D[l].T)
            tmp = np.linalg.solve(Nt, S)
            G_new[l] = np.linalg.solve(Nt, tmp.T).T

        # Prediction by the correlator rule (F4), exact:
        #   f = diag[ Nt_L^{-1} ( G^{L+1} C^{LT} + D^L H^{L-1} ) M_L^{-T} ]
        M, Nt = Minv_store[L], Ninv_store[L]
        S = Gtop @ C[L].T + D[L] @ H_new[L - 1]
        Z = np.linalg.solve(Nt, np.linalg.solve(M, S.T).T)
        f_vec = np.diag(Z)
        Delta_new = np.tile(y, T) - f_vec

        # Damped update of everything.
        b = damping
        shift = 0.0
        for l in range(1, L + 1):
            shift = max(shift, np.abs(H_new[l] - H[l]).max(),
                        np.abs(G_new[l] - G[l]).max())
            H[l] = (1 - b) * H[l] + b * H_new[l]
            G[l] = (1 - b) * G[l] + b * G_new[l]
            A[l] = (1 - b) * A[l] + b * A_new[l]
            B[l - 1] = (1 - b) * B[l - 1] + b * B_new[l - 1]
        shift = max(shift, np.abs(Delta_new - Delta).max())
        Delta = (1 - b) * Delta + b * Delta_new
        hist.append(shift)
        if verbose and it % 20 == 0:
            print("  iter %3d  max shift %.3e" % (it, shift))
        if shift < tol:
            break
        # Bail out early on a diverging iterate instead of burning the whole
        # budget. Without this, solve_annealed spends its full n_iter on every
        # rung that was never going to converge.
        if not np.isfinite(shift) or shift > 1e4 * max(hist[0], 1e-12):
            raise Diverged("iterate growing at iter %d (F5): shift %.3e vs "
                           "initial %.3e" % (it, shift, hist[0]))

    f_out = (np.tile(y, T) - Delta).reshape(T, P)
    ntk = np.zeros((PT, PT))
    for l in range(0, L + 1):
        ntk = ntk + G[l + 1] * H[l]

    return {"f": f_out, "Delta": Delta.reshape(T, P), "H": H, "G": G,
            "A": A, "B": B, "K": ntk, "t": np.arange(T) * dt,
            "iters": len(hist), "converged": hist[-1] < tol,
            "final_shift": hist[-1], "history": hist,
            "P": P, "T": T, "L": L, "dt": dt, "gamma0": gamma0}


def equal_time(mat, P, T):
    """Extract the P x P equal-time block at each t from a PT x PT kernel."""
    out = np.zeros((T, P, P))
    for t in range(T):
        out[t] = mat[t * P:(t + 1) * P, t * P:(t + 1) * P]
    return out
