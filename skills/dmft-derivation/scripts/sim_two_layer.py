"""Finite-width two-layer network trained in the matched parameterisation.

The sim side of `references/validation-checks.md`. Must match the DMFT grid in
BOTH parameterisation and discretisation or the comparison is meaningless
(comparing against a standard-PyTorch net is a registered way to get a gap
that grows with N).

    f_mu   = (1/(gamma0 N)) sum_i w_i phi(h_{i mu}),   h = W0 X / sqrt(D)
    W0, w  ~ N(0, 1) entrywise
    theta <- theta + eta * (-grad L),  eta = gamma0^2 N dt,  L = sum_mu (1/2)(y-f)^2

This updates W0 explicitly and recomputes h from it, rather than applying the
analytically reduced h-recursion. That is deliberate: it is an independent
implementation path, so agreement with the solver tests the bookkeeping
instead of assuming it.
"""

import numpy as np

import activations


def train(X, y, gamma0, dt, n_steps, N, act="tanh", seed=0,
          center_init=False, record_kernels=True):
    """Train one finite-width network. X is (D, P)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    D, P = X.shape
    phi, phidot = activations.get(act)
    rng = np.random.default_rng(seed)

    W0 = rng.standard_normal((N, D))
    w = rng.standard_normal(N)
    Xs = X / np.sqrt(D)
    Kx = X.T @ X / D

    h = W0 @ Xs
    f0 = (w @ phi(h)) / (gamma0 * N)

    t = np.arange(n_steps + 1) * dt
    f_hist = np.zeros((n_steps + 1, P))
    loss_hist = np.zeros(n_steps + 1)
    Phi_hist = np.zeros((n_steps + 1, P, P)) if record_kernels else None
    G_hist = np.zeros((n_steps + 1, P, P)) if record_kernels else None

    def _record(k):
        ph = phi(h)
        f = (w @ ph) / (gamma0 * N)
        if center_init:
            f = f - f0  # F10: subtract-init when gamma0*sqrt(N) is not large
        f_hist[k] = f
        loss_hist[k] = 0.5 * float(np.sum((y - f) ** 2))
        if record_kernels:
            pd = phidot(h)
            pw = pd * w[:, None]
            Phi_hist[k] = ph.T @ ph / N
            G_hist[k] = pw.T @ pw / N
        return f

    f = _record(0)
    for k in range(n_steps):
        delta = y - f
        ph = phi(h)
        pd = phidot(h)
        # eta * (-dL/dtheta) with eta = gamma0^2 N dt collapses to gamma0 dt.
        W0 = W0 + (dt * gamma0) * (w[:, None] * ((pd * delta) @ Xs.T))
        w = w + (dt * gamma0) * (ph @ delta)
        h = W0 @ Xs
        f = _record(k + 1)

    return {"t": t, "f": f_hist, "loss": loss_hist, "N": N, "dt": dt,
            "gamma0": gamma0, "act": act, "Kx": Kx,
            "Phi": Phi_hist, "G": G_hist,
            "K": None if not record_kernels else Phi_hist + G_hist * Kx[None, :, :]}


def train_seeds(X, y, gamma0, dt, n_steps, N, act="tanh", seeds=(0, 1, 2, 3),
                center_init=False, record_kernels=True):
    """Seed-average before comparing (F10). Returns mean and per-seed spread."""
    runs = [train(X, y, gamma0, dt, n_steps, N, act=act, seed=s,
                  center_init=center_init, record_kernels=record_kernels)
            for s in seeds]
    out = {"t": runs[0]["t"], "N": N, "dt": dt, "gamma0": gamma0, "act": act,
           "n_seeds": len(seeds)}
    for key in ("f", "loss", "Phi", "G", "K"):
        if runs[0].get(key) is None:
            continue
        stack = np.stack([r[key] for r in runs], axis=0)
        out[key] = stack.mean(axis=0)
        out[key + "_sem"] = stack.std(axis=0, ddof=1) / np.sqrt(len(seeds))
    return out


def whitened_inputs(P):
    """X with Kx = X^T X / D = I_P exactly (D = P)."""
    return np.sqrt(P) * np.eye(P)


def random_inputs(D, P, seed=0):
    """Generic (non-whitened) inputs, columns normalised so diag(Kx) = 1."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((D, P))
    X = X / np.linalg.norm(X, axis=0, keepdims=True) * np.sqrt(D)
    return X
