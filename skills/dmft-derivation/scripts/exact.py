"""Closed-form and quadrature references the solver is checked against.

Nothing here uses the solver, so agreement is evidence rather than tautology.
Where two independent routes to the same object exist (analytic formula and
Gauss-Hermite quadrature), both are provided and `validate.py` requires them
to agree before either is used as ground truth (F14).
"""

import numpy as np
from scipy.linalg import expm

import activations


# ---------------------------------------------------------------------------
# Gaussian integrals over the init measure
# ---------------------------------------------------------------------------

def gauss_hermite_pair(f, g, cov, n_nodes=96):
    """<f(u1) g(u2)> for (u1, u2) ~ N(0, cov), cov a 2x2 covariance.

    Exact for polynomials up to degree 2*n_nodes-1 in each variable; for the
    smooth activations used here, converged to machine precision well before
    n_nodes = 96.
    """
    c11, c12, c22 = float(cov[0, 0]), float(cov[0, 1]), float(cov[1, 1])
    s1, s2 = np.sqrt(max(c11, 0.0)), np.sqrt(max(c22, 0.0))

    x, w = np.polynomial.hermite.hermgauss(n_nodes)
    z = np.sqrt(2.0) * x
    wn = w / np.sqrt(np.pi)  # weights for the standard normal

    if s1 == 0.0 or s2 == 0.0:
        # One (or both) marginals are deterministic at zero.
        u1 = np.zeros(n_nodes) if s1 == 0.0 else s1 * z
        u2 = np.zeros(n_nodes) if s2 == 0.0 else s2 * z
        if s1 == 0.0 and s2 == 0.0:
            return float(f(np.zeros(1))[0] * g(np.zeros(1))[0])
        wgt = wn
        return float(np.sum(wgt * f(u1) * g(u2)))

    rho = np.clip(c12 / (s1 * s2), -1.0, 1.0)
    # u1 = s1*z1 ; u2 = s2*(rho*z1 + sqrt(1-rho^2)*z2)
    z1 = z[:, None]
    z2 = z[None, :]
    u1 = s1 * z1 * np.ones_like(z2)
    u2 = s2 * (rho * z1 + np.sqrt(max(1.0 - rho * rho, 0.0)) * z2)
    wgt = wn[:, None] * wn[None, :]
    return float(np.sum(wgt * f(u1) * g(u2)))


def kernels_quadrature(act, Kx, n_nodes=96):
    """Init kernels (Phi0, G0, K0) for a two-layer net by 2D quadrature.

    Phi0_{mu,al} = <phi(u_mu) phi(u_al)>,  u ~ N(0, Kx)
    G0_{mu,al}   = <w^2 phidot(u_mu) phidot(u_al)> = <phidot phidot>  (w indep, var 1)
    K0           = Phi0 + G0 * Kx        (elementwise; L=1 NTK)
    """
    phi, phidot = activations.get(act)
    P = Kx.shape[0]
    Phi0 = np.zeros((P, P))
    G0 = np.zeros((P, P))
    for mu in range(P):
        for al in range(mu, P):
            cov = np.array([[Kx[mu, mu], Kx[mu, al]], [Kx[mu, al], Kx[al, al]]])
            Phi0[mu, al] = Phi0[al, mu] = gauss_hermite_pair(phi, phi, cov, n_nodes)
            G0[mu, al] = G0[al, mu] = gauss_hermite_pair(phidot, phidot, cov, n_nodes)
    return Phi0, G0, Phi0 + G0 * Kx


def kernels_analytic(act, Kx):
    """Init kernels in closed form. Returns None for activations without one."""
    d = np.sqrt(np.diag(Kx))
    outer = np.outer(d, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(outer > 0, Kx / np.where(outer > 0, outer, 1.0), 0.0)
    rho = np.clip(rho, -1.0, 1.0)

    if act == "linear":
        Phi0 = Kx.copy()
        G0 = np.ones_like(Kx)
    elif act == "erf":
        dd = 1.0 + 2.0 * np.diag(Kx)
        denom = np.sqrt(np.outer(dd, dd))
        Phi0 = (2.0 / np.pi) * np.arcsin(np.clip(2.0 * Kx / denom, -1.0, 1.0))
        G0 = (4.0 / np.pi) / np.sqrt(np.outer(dd, dd) - 4.0 * Kx ** 2)
    elif act == "relu":
        theta = np.arccos(rho)
        Phi0 = outer / (2.0 * np.pi) * (np.sin(theta) + (np.pi - theta) * np.cos(theta))
        G0 = (np.pi - theta) / (2.0 * np.pi)
    else:
        return None
    return Phi0, G0, Phi0 + G0 * Kx


# ---------------------------------------------------------------------------
# Exactly solvable trajectories
# ---------------------------------------------------------------------------

def scalar_ode_linear_whitened(gamma0, y, t_grid):
    """L=1, linear, whitened data (Kx = I), single output direction.

        dDelta/dt = -2 sqrt(1 + gamma0^2 (y - Delta)^2) Delta,   Delta(0) = y

    (`equations.md` §3; re-derived independently in `scripts/README.md`.)
    Integrated with RK4 on a refined grid, returned on t_grid.
    """
    def rhs(delta):
        return -2.0 * np.sqrt(1.0 + gamma0 ** 2 * (y - delta) ** 2) * delta

    t_grid = np.asarray(t_grid, dtype=float)
    out = np.empty_like(t_grid)
    delta = float(y)
    out[0] = delta
    for k in range(1, len(t_grid)):
        span = t_grid[k] - t_grid[k - 1]
        n_sub = max(int(np.ceil(span / 1e-4)), 1)
        h = span / n_sub
        for _ in range(n_sub):
            k1 = rhs(delta)
            k2 = rhs(delta + 0.5 * h * k1)
            k3 = rhs(delta + 0.5 * h * k2)
            k4 = rhs(delta + h * k3)
            delta = delta + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[k] = delta
    return out


def final_kernel_linear_whitened(gamma0, y_vec):
    """H(inf) = I + [(sqrt(1+gamma0^2 y^2) - 1)/y^2] y y^T  (`equations.md` §3)."""
    y2 = float(y_vec @ y_vec)
    P = len(y_vec)
    if y2 == 0.0:
        return np.eye(P)
    coef = (np.sqrt(1.0 + gamma0 ** 2 * y2) - 1.0) / y2
    return np.eye(P) + coef * np.outer(y_vec, y_vec)


def lazy_prediction(K0, y, t_grid):
    """Lazy/NTK limit (gamma0 -> 0) under gradient flow: f(t) = (I - e^{-K0 t}) y."""
    return np.array([(np.eye(len(y)) - expm(-K0 * t)) @ y for t in t_grid])


def lazy_prediction_discrete(K0, y, dt, n_steps):
    """Lazy limit under the SAME discrete update the solver uses.

    Separates O(dt) discretisation error from O(gamma0^2) feature-learning
    drift when diagnosing a lazy-check failure.
    """
    P = len(y)
    A = np.eye(P) - dt * K0
    out = np.zeros((n_steps + 1, P))
    delta = y.copy()
    for k in range(n_steps + 1):
        out[k] = y - delta
        delta = A @ delta
    return out
