"""Numerical verification of the sign-sensitive identities in the
Information Mechanics stub (Halmos & Hanin, Aug 2026 draft).

Conventions checked (hbar = m = 1 throughout):
  I_std(mu||rho0) = int |grad log mu - grad log rho0|^2 dmu   (NO 1/8)
  Paper's I       = (1/8) I_std
  Bohm quantum potential  Q_B[rho] = -(1/2) lap(sqrt rho)/sqrt rho
  Paper eq (18)'s         Q_18[rho] = +lap(sqrt rho)/sqrt rho

Claims to test on 1d grids with generic non-Gaussian densities:
  (A) delta I_std / delta mu = -4 lap(sqrt mu)/sqrt mu + 4 lap(sqrt rho0)/sqrt rho0 ??
      -> actually relative version: compute directly by perturbation.
  (B) delta[(1/8) I_std]/delta mu  ==  Q_B[mu] - Q_B[rho0]      (Bohm sign)
      and                          == -(1/2)(Q_18[mu] - Q_18[rho0])
  (C) (1/8) I_std identity vs energy form:
      (1/8) int |glm - glr|^2 mu == (1/2)[ int |grad sqrt mu|^2 + int mu lap(sqrt rho0)/sqrt rho0 ]
      i.e. (1/4) I_std = int |grad sqrt mu|^2 + int mu * lap(sqrt rho0)/sqrt rho0
  (D) lap(sqrt rho)/sqrt rho == (1/4)|grad log rho|^2 + (1/2) lap log rho
  (E) KL: grad dKL/dmu = grad log mu - grad log rho0 ; FP form
"""
import numpy as np

L = 40.0
N = 16384
x = np.linspace(-L / 2, L / 2, N)
dx = x[1] - x[0]


def grad(f):
    return np.gradient(f, dx, edge_order=2)


def lap(f):
    return np.gradient(np.gradient(f, dx, edge_order=2), dx, edge_order=2)


def normalize(f):
    return f / np.trapezoid(f, x)


# generic smooth positive densities (non-Gaussian, asymmetric)
mu = normalize(np.exp(-(x**2) / 2 - 0.3 * np.sin(1.3 * x) - 0.05 * x**3 / (1 + 0.1 * x**2)))
rho0 = normalize(np.exp(-(x**2) / 2.5 + 0.4 * np.cos(0.9 * x)))


def I_std(m):
    glm, glr = grad(np.log(m)), grad(np.log(rho0))
    return np.trapezoid((glm - glr) ** 2 * m, x)


# ---- (A)/(B): first variation of (1/8) I_std by direct perturbation ----
# delta F/delta mu is defined up to a constant; compare after removing the
# mu-weighted mean, and compare gradients too.
rng = np.random.default_rng(0)
# smooth mean-zero perturbation supported inside mu (so mu + eps*h > 0 everywhere)
g = np.sin(2.1 * x + 0.3) * np.exp(-(x**2) / 12)
h = mu * (g - np.trapezoid(mu * g, x))  # int h = 0

eps = 1e-5
F = lambda m: 0.125 * I_std(m)
dF_num = (F(mu + eps * h) - F(mu - eps * h)) / (2 * eps)

QB = lambda r: -0.5 * lap(np.sqrt(r)) / np.sqrt(r)
cand_bohm = QB(mu) - QB(rho0)  # claimed delta F / delta mu (Bohm sign)
cand_18 = (lap(np.sqrt(mu)) / np.sqrt(mu)) - (lap(np.sqrt(rho0)) / np.sqrt(rho0))  # eq(18) sign

pair_bohm = np.trapezoid(cand_bohm * h, x)
pair_18 = np.trapezoid(cand_18 * h, x)
print("(B) <dF, h> numeric        :", dF_num)
print("    <QB(mu)-QB(rho0), h>   :", pair_bohm, "  rel.err:", abs(pair_bohm - dF_num) / abs(dF_num))
print("    <Q18(mu)-Q18(rho0), h> :", pair_18, "  rel.err:", abs(pair_18 - dF_num) / abs(dF_num))
print("    ratio (Q18 pairing)/(numeric):", pair_18 / dF_num)

# ---- (C): energy identity ----
lhs = 0.25 * I_std(mu)
rhs = np.trapezoid(grad(np.sqrt(mu)) ** 2, x) + np.trapezoid(mu * lap(np.sqrt(rho0)) / np.sqrt(rho0), x)
print("(C) (1/4) I_std            :", lhs)
print("    int|g sqrt mu|^2 + int mu lap sqrt rho0/sqrt rho0 :", rhs, "  rel.err:", abs(lhs - rhs) / abs(lhs))

# ---- (D): score expansion ----
lhs_d = lap(np.sqrt(mu)) / np.sqrt(mu)
rhs_d = 0.25 * grad(np.log(mu)) ** 2 + 0.5 * lap(np.log(mu))
i = slice(N // 4, 3 * N // 4)  # interior, away from underflow tails
print("(D) score expansion max rel err (interior):",
      np.max(np.abs(lhs_d[i] - rhs_d[i])) / np.max(np.abs(lhs_d[i])))

# ---- (E): KL first variation ----
KL = lambda m: np.trapezoid(m * np.log(m / rho0), x)
dKL_num = (KL(mu + eps * h) - KL(mu - eps * h)) / (2 * eps)
pair_kl = np.trapezoid((np.log(mu / rho0) + 1.0) * h, x)
print("(E) <dKL,h> numeric:", dKL_num, "  <log(mu/rho0)+1, h>:", pair_kl,
      "  rel.err:", abs(pair_kl - dKL_num) / max(abs(dKL_num), 1e-12))

# FP form: div(mu grad(log mu - log rho0)) == lap mu - div(mu grad log rho0)
fp1 = grad(mu * grad(np.log(mu / rho0)))
fp2 = lap(mu) - grad(mu * grad(np.log(rho0)))
print("    FP-form max rel err (interior):",
      np.max(np.abs(fp1[i] - fp2[i])) / np.max(np.abs(fp1[i])))
