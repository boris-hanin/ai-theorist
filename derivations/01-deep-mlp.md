# Deep MLP DMFT, derived from scratch — and an audit of `equations.md` §1

Derived independently by the cavity route of `00-method.md`, then compared
factor by factor against `skills/dmft-derivation/references/equations.md` §1.

**Result: `equations.md` §1 is confirmed.** Every equation, every factor of
`gamma_0` and `sqrt(N)`, and all four boundary conditions match an independent
derivation. That file previously carried the warning "compiled from a
structured extraction; verify equation-by-equation before high-stakes use" —
for §1 that verification is now done. §3 and §4 are still unaudited.

One thing the derivation adds that is not in `equations.md`: the equal-time
response is a **delta function**, not merely a nonzero value. See §7.

## 1. Setup

Depth-L MLP, width N, P inputs `x_mu` in R^D, all weights i.i.d. `N(0,1)`:

    h^1_mu     = (1/sqrt(D)) W^0 x_mu
    h^{l+1}_mu = (1/sqrt(N)) W^l phi(h^l_mu)          l = 1..L-1
    f_mu       = (1/(gamma sqrt(N))) w^L . phi(h^L_mu),   gamma = gamma_0 sqrt(N)

Loss `L = sum_mu l(f_mu, y_mu)`, error signal `Delta_mu = -dl/df_mu`, dynamics
`theta_dot = -gamma^2 grad_theta L`.

## 2. Backward fields (M1 prerequisite)

Define `g^l_mu = gamma sqrt(N) * df_mu/dh^l_mu`, chosen so `g = O(1)`.

At the readout, `df_mu/dh^L_mu = (1/(gamma sqrt N)) w^L ⊙ phi_dot(h^L_mu)`, so

    g^L = phi_dot(h^L) ⊙ z^L,      z^L = w^L

Below it, `dh^{l+1}_i/dh^l_j = (1/sqrt N) W^l_ij phi_dot(h^l_j)`, giving

    g^l = phi_dot(h^l) ⊙ z^l,      z^l = (1/sqrt N) W^{l T} g^{l+1}

Matches `equations.md` §0. **Note `W^l` appears here transposed and in the
forward pass untransposed — every hidden matrix is a *reused* edge (M3), so
every hidden layer carries a response pair.**

## 3. M1 — integrate the weight dynamics exactly

    df_mu/dW^l_ij = (1/(gamma sqrt N)) g^{l+1}_{i mu} * (1/sqrt N) phi(h^l_{j mu})
    dL/dW^l_ij    = -(1/(gamma N)) sum_mu Delta_mu g^{l+1}_{i mu} phi(h^l_{j mu})

so with `theta_dot = -gamma^2 grad L` and `gamma = gamma_0 sqrt N`:

    W^l_dot = (gamma/N) sum_mu Delta_mu g^{l+1}_mu phi(h^l_mu)^T
            = (gamma_0/sqrt N) sum_mu Delta_mu g^{l+1}_mu phi(h^l_mu)^T

Rank-P, as required. The readout obeys `w_dot = gamma_0 sum_mu Delta_mu phi(h^L_mu)`.

## 4. M2 — split the fields

Forward:

    h^{l+1}_mu(t) = (1/sqrt N) W^l(t) phi(h^l_mu(t))
                  = chi^{l+1}_mu(t)
                    + gamma_0 int_0^t ds sum_alpha Delta_alpha(s) Phi^l_{mu alpha}(t,s) g^{l+1}_alpha(s)

using `Phi^l_{mu alpha}(t,s) = (1/N) phi(h^l_mu(t)) . phi(h^l_alpha(s))`.

Backward, identically:

    z^l_mu(t) = xi^l_mu(t)
                + gamma_0 int_0^t ds sum_alpha Delta_alpha(s) G^{l+1}_{mu alpha}(t,s) phi(h^l_alpha(s))

using `G^{l+1}_{mu alpha}(t,s) = (1/N) g^{l+1}_mu(t) . g^{l+1}_alpha(s)`.

Both memory kernels carry exactly one `gamma_0`. (`gamma^2` from the learning
rate, divided by `gamma` from the readout normalisation, divided by `sqrt(N)`
from the field normalisation: `gamma^2 / (gamma sqrt N) = gamma_0`.)

## 5. M3–M4 — the Onsager terms

`chi^{l+1}_i(t) = (1/sqrt N) sum_j W^l_ij(0) phi(h^l_{j}(t))` is *not* Gaussian:
`h^l_j` depends on `W^l(0)` through `xi^l_j = (1/sqrt N) sum_i W^l_ij g^{l+1}_i`.

Cavity expansion in row `i` (M4):

    phi(h^l_j(t)) = phi(h^{l,\i}_j(t))
                    + sum_alpha int ds [d phi(h^l_j(t)) / d xi^l_j(alpha,s)] (1/sqrt N) W^l_ij g^{l+1}_i(alpha,s)

Substituting, the cavity term is Gaussian with covariance `Phi^l`, and the
correction self-averages (`(1/N) sum_j W_ij^2 -> 1`) to

    chi^{l+1}_i(t) = u^{l+1}_i(t) + gamma_0 int ds sum_alpha A^l_{mu alpha}(t,s) g^{l+1}_i(alpha,s)

with `u^{l+1} ~ GP(0, Phi^l)` and

    A^l_{mu alpha}(t,s) = gamma_0^{-1} <d phi(h^l_mu(t)) / d r^l_alpha(s)>

Transposing the same argument for `xi`:

    xi^l_mu(t) = r^l_mu(t) + gamma_0 int ds sum_alpha B^l_{mu alpha}(t,s) phi(h^l_alpha(s))

with `r^l ~ GP(0, G^{l+1})` and
`B^l_{mu alpha}(t,s) = gamma_0^{-1} <d g^{l+1}_mu(t) / d u^{l+1}_alpha(s)>`.

## 6. The closed system

Combining §4 and §5:

    h^l_mu(t) = u^l_mu(t)
              + gamma_0 int_0^t ds sum_alpha [ A^{l-1}_{mu alpha}(t,s) + Delta_alpha(s) Phi^{l-1}_{mu alpha}(t,s) ]
                                             z^l_alpha(s) phi_dot(h^l_alpha(s))

    z^l_mu(t) = r^l_mu(t)
              + gamma_0 int_0^t ds sum_alpha [ B^l_{mu alpha}(t,s) + Delta_alpha(s) G^{l+1}_{mu alpha}(t,s) ]
                                             phi(h^l_alpha(s))

(using `g^l = phi_dot(h^l) z^l`), self-consistency
`Phi^l = <phi(h^l) phi(h^l)>`, `G^l = <g^l g^l>`, and boundaries

    Phi^0 = K^x,   G^{L+1} = 1 1^T,   A^0 = 0,   B^L = 0

`A^0 = 0` because the inputs do not depend on `W^0`; `B^L = 0` because
`g^{L+1} = 1` identically. **These match `equations.md` line for line.**

Readout identity: with `B^L = 0` and `G^{L+1} = 1`,

    z^L_mu(t) = w(0) + gamma_0 int_0^t ds sum_alpha Delta_alpha(s) phi(h^L_alpha(s)) = w(t)

which is exactly the `w_dot` equation of §3 — an internal consistency check that
the two derivations of the readout agree. Confirms `equations.md` lines 55–56.

### Prediction dynamics and the NTK normalisation

`f_dot = sum_theta (df/dtheta) theta_dot = gamma^2 sum_alpha Delta_alpha sum_theta (df_mu/dtheta)(df_alpha/dtheta)`,
so `K^{NTK} = gamma^2 sum_theta (df_mu/dtheta)(df_alpha/dtheta)`. Layer by layer:

| Parameter | `gamma^2 sum (df df)` | equals |
|---|---|---|
| `W^0` | `gamma^2 (1/(gamma^2 N D)) sum_i g^1_i g^1_i sum_j x_j x_j` | `G^1 * K^x = G^1 Phi^0` |
| `W^l` | `gamma^2 (1/(gamma^2 N^2)) [sum_i g^{l+1} g^{l+1}][sum_j phi phi]` | `G^{l+1} Phi^l` |
| `w^L` | `gamma^2 (1/(gamma^2 N)) sum_i phi(h^L) phi(h^L)` | `Phi^L = G^{L+1} Phi^L` |

Summing: `K^{NTK} = sum_{l=0}^{L} G^{l+1} ⊙ Phi^l`. Confirms `equations.md` line 28,
including the boundary conventions `Phi^0 = K^x` and `G^{L+1} = 1`.

## 7. What the derivation adds: the equal-time response is a delta function

`h^{l+1}(t) = u^{l+1}(t) + gamma_0 (memory over [0,t])`, so `u` enters `h`
directly and `d h(t)/d u(s)` contains `delta(t-s)`. With `g = phi_dot(h) z`,

    B^l_{mu alpha}(t,s) ⊃ gamma_0^{-1} < phi_ddot(h^{l+1}_mu(t)) z^{l+1}_mu(t) > delta_{mu alpha} delta(t-s)

**In discrete time the delta becomes `1/dt`**: the diagonal of the response
kernel is larger than its neighbours by a factor `1/dt` and looks like an
outlier. After the `gamma_0 * dt * B` weighting it contributes at `O(1)`:

    gamma_0 * dt * (gamma_0^{-1} <phi_ddot z> / dt) * phi(h(t)) = <phi_ddot(h(t)) z(t)> phi(h(t))

This sharpens F1 in three ways:

1. It is not "the diagonal happens to be nonzero" — it is a distinct
   instantaneous (Onsager) term with a different `dt`-scaling from the rest of
   the kernel.
2. It explains the 20–50% kernel error: an `O(1)` contribution to the field is
   being dropped, not a small correction.
3. It explains F1b exactly: the term carries `phi_ddot`, so it vanishes
   identically for linear `phi`. **No linear check of any kind can detect it.**
   The minimal architecture that can is nonlinear with `L = 2`, where
   `A^0 = B^2 = 0` but `B^1` retains the Onsager piece.

### MEASURED (round 003): structure confirmed, coefficient FALSIFIED

The `1/dt` prediction is **confirmed**. Measured on the L=2, P=1 solver at fixed
horizon 1.0, erf activation:

| dt | `B^1(t,t)` max | `gamma_0 * dt * B^1(t,t)` |
|---|---|---|
| 0.100 | 5.40 | 0.540 |
| 0.050 | 11.08 | 0.554 |
| 0.025 | 24.21 | 0.605 |

The diagonal doubles as `dt` halves (ratios 2.05, 2.18) while the weighted
contribution stays near 0.55. So the equal-time response really is a delta
representative with `1/dt` magnitude, not an ordinary kernel entry.

**But the O(1) coefficient claim is wrong.** Adding the term to the memory sum
with weight `c`, and comparing against finite-width simulations extrapolated to
`N -> infinity` in `1/N` (four widths, 24 seeds each):

| coefficient `c` | max abs error vs `f_inf` |
|---|---|
| extrapolation residual (the floor) | 2.64e-3 |
| **0** | **4.41e-3** |
| 0.5 | 7.53e-3 |
| 1 | 1.07e-2 |

`c = 0` is favoured; `c = 1` sits at four times the floor. The spread across `c`
(6.3e-3) comfortably exceeds the floor, so unlike the first attempt -- which was
floor-limited and resolved nothing -- this measurement discriminates.

**Conclusion.** The delta-function *structure* is real and the `1/dt` scaling is
confirmed, but the naive step of integrating it against `int_0^t ds` with
weight 1 is not right: a delta sitting exactly at the endpoint of the causal
integral does not contribute. The correct discrete treatment uses the STRICT
past. The claim in the previous version of this section -- that the term
contributes at `O(1)` and that dropping it costs 20-50% -- is not supported by
this measurement and has been withdrawn.

Scope of the refutation: L=2, P=1, erf, `gamma_0 = 1`, horizon 1.0, one task.
It does not follow that F1 is wrong in general -- F1 was recorded from a deep
MLP round with a different solver, where the reported error may have come from
masking something other than the endpoint term. What is established here is
narrower: **in this solver, at these settings, weight 0 fits and weight 1 does
not.**

## 8. Audit summary

| `equations.md` §1 claim | Independent derivation |
|---|---|
| Backward field definitions `g^l`, `z^l` | Confirmed |
| `gamma_0` prefactor on both memory kernels | Confirmed (`gamma^2/(gamma sqrt N)`) |
| Gaussian sources `u^l ~ GP(0,Phi^{l-1})`, `r^l ~ GP(0,G^{l+1})` | Confirmed |
| Response definitions with `gamma_0^{-1}` | Confirmed, and the convention is necessary for `A,B = O(1)` |
| `[A + Delta*Phi]` and `[B + Delta*G]` memory structure | Confirmed |
| Boundaries `Phi^0=K^x`, `G^{L+1}=1`, `A^0=0`, `B^L=0` | Confirmed |
| Readout identity `z^L = w(t)` | Confirmed twice (cavity, and direct `w_dot`) |
| `K^{NTK} = sum_l G^{l+1} Phi^l` | Confirmed layer by layer |
| Rank-P structure of the learned weights | Confirmed |
| Equal-time response | **Extended**: it is a delta function, `1/dt` in discrete time |

No discrepancies found in §1.
