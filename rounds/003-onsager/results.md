# Round 003 — the nonlinear response sector and the Onsager term

**Not pre-registered.** The prediction under test was, however, written down and
committed *before* the solver that tests it existed (`9bba6f8`, Phase 1), which
is most of what pre-registration buys. The outcome below contradicts it.

Date: 2026-08-07
Skill exercised: `dmft-derivation` (nonlinear L=2 branch)
Artifacts: `skills/dmft-derivation/scripts/dmft_l2_nonlinear.py`

## What was predicted

`derivations/01-deep-mlp.md` §7 (committed in Phase 1) derived that the
equal-time response is a delta function:

    B^1(t,s) ⊃ gamma_0^{-1} <phi_ddot(h^2(t)) z^2(t)> delta(t-s)

with two consequences claimed: **(i)** in discrete time the diagonal scales as
`1/dt`, and **(ii)** after the `gamma_0*dt*B` weighting it therefore contributes
at `O(1)`, which would explain F1's recorded 20–50% kernel error.

## Result: (i) confirmed, (ii) falsified

**(i) The 1/dt structure is real.** Fixed horizon 1.0, erf, `gamma_0 = 1`:

| dt | `B^1(t,t)` max | `gamma_0 dt B^1(t,t)` |
|---|---|---|
| 0.100 | 5.40 | 0.540 |
| 0.050 | 11.08 | 0.554 |
| 0.025 | 24.21 | 0.605 |

The diagonal doubles as `dt` halves (ratios 2.05, 2.18) while the weighted
contribution stays near 0.55. This is a delta representative, not an ordinary
kernel entry — exactly as derived.

**(ii) The O(1) coefficient is wrong.** The term was added to the memory sum
with an explicit coefficient `c` and `c` was measured against finite-width
simulations extrapolated to `N -> infinity` in `1/N` (N = 512, 1024, 2048, 4096;
24 seeds each; erf, L=2, P=1):

| `c` | max abs error vs `f_inf` |
|---|---|
| extrapolation residual (**the floor**) | 2.64e-3 |
| **0** | **4.41e-3** |
| 0.5 | 7.53e-3 |
| 1 | 1.07e-2 |

`c = 0` fits; `c = 1` is at 4x the floor. The spread across `c` (6.3e-3) exceeds
the floor, so the measurement discriminates.

**Conclusion:** the delta-function structure is real, but a delta sitting exactly
at the endpoint of `int_0^t ds` does not contribute — the correct discrete
treatment uses the strict past. The `O(1)` claim, and with it the explanation of
F1's 20–50% figure, is withdrawn.

**Scope of the refutation:** one solver, L=2, P=1, erf, `gamma_0 = 1`, horizon
1.0, one task. This does not retract F1 itself; the original deep-MLP round may
have been masking something other than the endpoint term. What is established is
narrower: *in this solver, at these settings, weight 0 fits and weight 1 does
not.*

### The first attempt could not have decided this

Run against raw N=4096 simulations, the four coefficients gave 4.04e-3, 5.55e-3,
7.08e-3, 8.62e-3 — a spread of 4.6e-3 against a finite-width floor of 6.7e-3
(measured as |f(N=4096) − f(N=1024)|). Every coefficient sat inside the floor.
Reporting `c = 0` as the winner from that run would have been reading noise.
The `1/N` extrapolation is what made the question answerable, by cutting the
floor to 2.6e-3.

This is F8's rule doing real work: the first measurement was not weak evidence
for `c = 0`, it was **no evidence at all**, and only the floor said so.

## What the solver is, and what validates it

`dmft_l2_nonlinear.py`: L=2, P=1, single-site Monte Carlo, responses by exact
forward-mode sensitivities propagated alongside the trajectories. P=1 keeps the
sensitivity arrays two-index; L=2 is the minimal depth with a nonzero Onsager
term (`A^0 = B^2 = 0` by boundary, `B^1` survives).

**Primary correctness check:** with linear `phi` the Onsager term vanishes
(`phi_ddot = 0`), so this Monte-Carlo solver must reproduce the *exact*
algebraic deep-linear solver from round 002. It does, converging in S:

| S | max abs f_MC − f_algebraic |
|---|---|
| 2000 | 3.36e-2 |
| 8000 | 8.29e-3 |
| 32000 | 1.72e-3 |

and the response kernel matches to four digits (`A^1 * dt` = 0.0524 vs 0.0524).
That certifies the sampling, the sensitivity propagation, the response kernels,
the fixed point and the correlator rule — everything except the Onsager term.

## Three implementation findings

**1. The response is a density; the discrete sensitivity is not.**
`S[t,s] = d(field_t)/d(source_s)` equals `dt * delta(field(t))/delta(source(s))`,
so the kernel is `S/dt`. Omitting that divides the response by `dt` twice (once
in the estimator, once in the memory sum) and produces a **self-consistent but
wrong fixed point**: a 15–19% error against the exact solver that did *not*
shrink with S. It converged happily and looked fine. The only reason it was
caught is that an exact reference existed to compare against.

This is also what settles the `1/dt` question internally: with the correct
normalisation, `S_h2[t,t] = 1` gives `B^1[t,t] = gamma_0^{-1}<phi_ddot z>/dt`
directly.

**2. Common random numbers are load-bearing for the fixed point.** Redrawing the
Gaussian sources each iteration makes every iterate a different noise
realisation, so the residual cannot fall below the Monte-Carlo floor — measured:
the shift stalled at 2e-2 and the solve never converged. Drawing base normals
once and re-applying the current covariance root each iteration fixes it
(converged in ~20 iterations).

**3. Antithetic readout pairs give `f(0) = 0` exactly** (F15). Sharing `u^2`
across the twin and negating `r^2` makes `<r^2 phi(u^2)> = 0` identically. Before
this, `f(0) = 5.9e-3` at S=8000 — right at `1/(gamma_0 sqrt(S))`.

## Performance note

The sensitivity propagation costs `O(S T^3)` in time and `O(S T^2)` in memory;
at S=32000, T=30 the arrays alone are ~1 GB. Responses are population averages
of smooth quantities and converge on far fewer samples than the kernels, so they
are estimated on a subsample (`S_resp`, default 512). Replacing
`np.einsum("s,nsk->nk", ...)` with a batched `v @ A` matvec cut the per-iteration
cost from ~15s to ~2.5s with bit-identical results.

## Registry changes

F1 updated: the `1/dt` structure moves from "derived, not measured" to
**measured and confirmed**; the `O(1)` consequence is marked **falsified at these
settings**, with the scope stated. The "worth 20–50%" figure is flagged as not
reproduced here and not to be quoted as general.

## Not covered

- L >= 3 nonlinear, and P >= 2 nonlinear (the solver is P=1 by construction).
- Whether F1's original 20–50% figure is reproducible at all — it would need the
  deep nonlinear solver this round did not build.
- `gamma_0` dependence of the conclusion: only `gamma_0 = 1` was tested at the
  discriminating settings.
