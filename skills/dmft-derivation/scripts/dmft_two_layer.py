"""Two-layer (L=1) DMFT solver: exact causal co-integration.

Implements `references/numerics.md` §B. At L=1 the response functions vanish
(A^0 = 0, B^1 = 0), so no fixed-point iteration is needed: single-site samples
depend on the population only through the deterministic error signal Delta(t),
and Delta depends on the samples only through equal-time population averages.

Two invariants from `numerics.md` §0 are load-bearing here:

  * predictions come from the correlator rule
        f_mu(t) = gamma0^{-1} <w(t) phi(h_mu(t))>
    read off the population AFTER advancing the states, never from marching
    df/dt = K Delta (F4);
  * the correlator carries an explicit 1/gamma0, so its Monte-Carlo floor is
    O(1/(gamma0 sqrt(S))) and grows as gamma0 shrinks (F15). Antithetic
    readout pairs make it exactly zero at t = 0.

Structural note (see scripts/README.md): at L=1 a width-N network in muP is
*exactly* S = N samples of this process, so this solver and the finite-width
simulator are the same recursion at different sample counts. That makes the
theory-vs-simulation comparison a convergence check in S, not an independent
test of a mean-field closure. The independent tests are the exactly solvable
cases in `exact.py`.
"""

import numpy as np

import activations


def _normal_sources(Kx, n_draw, rng, qmc):
    """Draw (u, w0) with u ~ N(0, Kx) in R^P and w0 ~ N(0, 1).

    F16: when using QMC, draw ONE joint scrambled-Sobol stream of dimension
    P + 1 and slice it, rather than one independently seeded stream per source
    family. Independently seeded scrambles of the same Sobol sequence are
    strongly cross-correlated.
    """
    P = Kx.shape[0]
    if qmc:
        from scipy.stats import qmc as _qmc
        from scipy.stats import norm

        m = int(np.ceil(np.log2(max(n_draw, 2))))
        sob = _qmc.Sobol(d=P + 1, scramble=True, seed=int(rng.integers(2 ** 31)))
        pts = sob.random_base2(m)[:n_draw]
        z = norm.ppf(np.clip(pts, 1e-12, 1.0 - 1e-12))
    else:
        z = rng.standard_normal((n_draw, P + 1))

    # PSD square root of Kx (Kx may be singular when D < P).
    evals, evecs = np.linalg.eigh(Kx)
    root = evecs @ np.diag(np.sqrt(np.clip(evals, 0.0, None)))
    u = z[:, :P] @ root.T
    w0 = z[:, P]
    return u, w0


def sample_sources(Kx, S, seed=0, antithetic=True, qmc=False):
    """Build the single-site source population.

    With `antithetic`, S/2 base draws are each paired with a twin sharing u and
    negating w (F15), so the gamma0^{-1}-amplified readout correlator is
    exactly zero at initialisation.
    """
    rng = np.random.default_rng(seed)
    if antithetic:
        if S % 2 != 0:
            raise ValueError("antithetic sampling needs an even S, got %d" % S)
        u_b, w_b = _normal_sources(Kx, S // 2, rng, qmc)
        u = np.concatenate([u_b, u_b], axis=0)
        w0 = np.concatenate([w_b, -w_b], axis=0)
    else:
        u, w0 = _normal_sources(Kx, S, rng, qmc)
    return u, w0


def solve(Kx, y, gamma0, dt, n_steps, S, act="tanh", seed=0,
          antithetic=True, qmc=False, record_kernels=True,
          prediction="correlator"):
    """Solve the L=1 DMFT system by exact causal co-integration.

    Returns a dict with the time grid, predictions f(t), loss, and (optionally)
    the equal-time kernels Phi(t,t), G(t,t) and NTK K(t,t).

    `prediction` selects how f is obtained:
      "correlator" -- the required path (F4): read f off the population.
      "euler"      -- the registered failure mode, kept ONLY so validate.py can
                      measure the difference it makes. Never use for results.

    The two agree to O(dt^2) per step (expanding the exact update reproduces
    f + dt*K*Delta), so they differ by O(dt*T) overall, growing with gamma0.
    """
    if prediction not in ("correlator", "euler"):
        raise ValueError("prediction must be 'correlator' or 'euler'")
    Kx = np.asarray(Kx, dtype=float)
    y = np.asarray(y, dtype=float)
    P = len(y)
    phi, phidot = activations.get(act)

    u, w0 = sample_sources(Kx, S, seed=seed, antithetic=antithetic, qmc=qmc)
    h = u.copy()
    w = w0.copy()

    t = np.arange(n_steps + 1) * dt
    f_hist = np.zeros((n_steps + 1, P))
    loss_hist = np.zeros(n_steps + 1)
    Phi_hist = np.zeros((n_steps + 1, P, P)) if record_kernels else None
    G_hist = np.zeros((n_steps + 1, P, P)) if record_kernels else None

    def _record(k, f_marched=None):
        ph = phi(h)
        # Correlator rule (F4): read f off the population, do not march it.
        f = (w @ ph) / (gamma0 * S) if f_marched is None else f_marched
        f_hist[k] = f
        loss_hist[k] = 0.5 * float(np.sum((y - f) ** 2))
        if record_kernels:
            pd = phidot(h)
            pw = pd * w[:, None]
            Phi_hist[k] = ph.T @ ph / S
            G_hist[k] = pw.T @ pw / S
        return f

    f = _record(0)
    for k in range(n_steps):
        delta = y - f
        ph = phi(h)
        pd = phidot(h)
        marched = None
        if prediction == "euler":
            pw = pd * w[:, None]
            K_tt = ph.T @ ph / S + (pw.T @ pw / S) * Kx
            marched = f + dt * (K_tt @ delta)
        # Advance every state from the SAME time-k quantities.
        h = h + (dt * gamma0) * w[:, None] * ((pd * delta) @ Kx)
        w = w + (dt * gamma0) * (ph @ delta)
        f = _record(k + 1, marched)

    out = {"t": t, "f": f_hist, "loss": loss_hist, "S": S, "dt": dt,
           "gamma0": gamma0, "act": act, "antithetic": antithetic, "qmc": qmc}
    if record_kernels:
        out["Phi"] = Phi_hist
        out["G"] = G_hist
        out["K"] = Phi_hist + G_hist * Kx[None, :, :]
    return out


def kernel_movement(res):
    """Frobenius norm ||Phi(T,T) - Phi(0,0)||, the feature-learning observable."""
    return float(np.linalg.norm(res["Phi"][-1] - res["Phi"][0]))
