"""Finite-width deep MLP trained in the matched parameterisation.

The simulation side for L >= 1. Same conventions as `derivations/01-deep-mlp.md`:

    h^1     = W^0 X / sqrt(D)
    h^{l+1} = W^l phi(h^l) / sqrt(N)
    f       = w . phi(h^L) / (gamma_0 N)        [= 1/(gamma sqrt N), gamma = gamma_0 sqrt N]

all weights N(0,1) at init, and the exact discrete form of
`theta_dot = -gamma^2 grad L` with `eta = gamma_0^2 N dt`, which reduces to

    W^0_dot = (gamma_0/sqrt D) sum_mu Delta_mu g^1_mu x_mu^T
    W^l_dot = (gamma_0/sqrt N) sum_mu Delta_mu g^{l+1}_mu phi(h^l_mu)^T
    w_dot   =  gamma_0        sum_mu Delta_mu phi(h^L_mu)

Backprop fields are the O(1) normalisation `g^l = gamma sqrt(N) df/dh^l`:
`g^L = phi_dot(h^L) * w` and `g^l = phi_dot(h^l) * (W^{l T} g^{l+1} / sqrt N)`.
"""

import numpy as np

import activations


def train(X, y, gamma0, dt, n_steps, N, L=1, act="tanh", seed=0,
          center_init=False, record_kernels=True):
    """Train one finite-width depth-L network. X is (D, P)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    D, P = X.shape
    phi, phidot = activations.get(act)
    rng = np.random.default_rng(seed)

    W0 = rng.standard_normal((N, D))
    Wh = [rng.standard_normal((N, N)) for _ in range(L - 1)]
    w = rng.standard_normal(N)
    Xs = X / np.sqrt(D)

    def forward():
        hs = [W0 @ Xs]                                   # h^1, shape (N, P)
        for l in range(L - 1):
            hs.append(Wh[l] @ phi(hs[-1]) / np.sqrt(N))
        return hs

    def backward(hs):
        """Returns g^1..g^L (index 0 is g^1)."""
        gs = [None] * L
        gs[L - 1] = phidot(hs[L - 1]) * w[:, None]       # g^L = phi_dot(h^L) * z^L
        for l in range(L - 2, -1, -1):
            z = Wh[l].T @ gs[l + 1] / np.sqrt(N)
            gs[l] = phidot(hs[l]) * z
        return gs

    hs = forward()
    f0 = (w @ phi(hs[-1])) / (gamma0 * N)

    t = np.arange(n_steps + 1) * dt
    f_hist = np.zeros((n_steps + 1, P))
    loss_hist = np.zeros(n_steps + 1)
    Phi_hist = np.zeros((n_steps + 1, L, P, P)) if record_kernels else None
    G_hist = np.zeros((n_steps + 1, L, P, P)) if record_kernels else None

    def record(k, hs, gs):
        f = (w @ phi(hs[-1])) / (gamma0 * N)
        if center_init:
            f = f - f0
        f_hist[k] = f
        loss_hist[k] = 0.5 * float(np.sum((y - f) ** 2))
        if record_kernels:
            for l in range(L):
                ph = phi(hs[l])
                Phi_hist[k, l] = ph.T @ ph / N
                G_hist[k, l] = gs[l].T @ gs[l] / N
        return f

    gs = backward(hs)
    f = record(0, hs, gs)
    for k in range(n_steps):
        delta = y - f
        # All updates use time-k quantities.
        dW0 = (gamma0 / np.sqrt(D)) * ((gs[0] * delta) @ X.T)
        dWh = [(gamma0 / np.sqrt(N)) * ((gs[l + 1] * delta) @ phi(hs[l]).T)
               for l in range(L - 1)]
        dw = gamma0 * (phi(hs[-1]) @ delta)
        W0 = W0 + dt * dW0
        for l in range(L - 1):
            Wh[l] = Wh[l] + dt * dWh[l]
        w = w + dt * dw
        hs = forward()
        gs = backward(hs)
        f = record(k + 1, hs, gs)

    return {"t": t, "f": f_hist, "loss": loss_hist, "N": N, "L": L, "dt": dt,
            "gamma0": gamma0, "act": act, "Phi": Phi_hist, "G": G_hist}


def train_seeds(X, y, gamma0, dt, n_steps, N, L=1, act="tanh", seeds=(0, 1, 2, 3),
                center_init=False, record_kernels=True):
    """Seed-average before comparing (F10)."""
    runs = [train(X, y, gamma0, dt, n_steps, N, L=L, act=act, seed=s,
                  center_init=center_init, record_kernels=record_kernels)
            for s in seeds]
    out = {"t": runs[0]["t"], "N": N, "L": L, "dt": dt, "gamma0": gamma0,
           "act": act, "n_seeds": len(seeds)}
    for key in ("f", "loss", "Phi", "G"):
        if runs[0].get(key) is None:
            continue
        stack = np.stack([r[key] for r in runs], axis=0)
        out[key] = stack.mean(axis=0)
        out[key + "_sem"] = stack.std(axis=0, ddof=1) / np.sqrt(len(seeds))
    return out
