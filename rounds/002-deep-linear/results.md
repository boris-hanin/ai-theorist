# Round 002 — derivation audit and the deep linear response sector

**Not pre-registered.** Like round 001, this ran as part of building the
infrastructure the pre-registration template describes. Rounds from 003 onward
should be pre-registered.

Date: 2026-08-07
Skills exercised: `dmft-derivation` (deep MLP, deep linear branches)
Artifacts: `derivations/`, `skills/dmft-derivation/scripts/dmft_deep_linear.py`,
`sim_deep.py`, `validate_deep_linear.py`

## Part A — `equations.md` §1 verified

Re-derived the general deep-MLP DMFT independently by the cavity route
(`derivations/00-method.md`, `01-deep-mlp.md`) and compared factor by factor.

**No discrepancies.** Confirmed: the backward field definitions; the single
`gamma_0` on both memory kernels (it comes out as `gamma^2/(gamma sqrt N)`);
the Gaussian source covariances; the `gamma_0^{-1}` convention in the response
definitions (and that it is *necessary* for `A, B = O(1)`); the
`[A + Delta*Phi]` / `[B + Delta*G]` memory structure; all four boundary
conditions; the readout identity `z^L = w(t)` (obtained twice, by cavity and by
direct `w_dot`); and `K^NTK = sum_l G^{l+1} Phi^l` layer by layer.

That file previously warned it was "compiled from a structured extraction" —
an explicit F14 risk on the program's most load-bearing reference. §1 is off
that list. §3, §4, §5 remain unaudited and are now marked as such in the file.

### New theoretical result (not yet measured)

The equal-time response is a **delta function**, not merely a nonzero value.
Since `h(t) = u(t) + gamma_0*(memory over [0,t])`, the source enters `h`
directly, so `d h(t)/d u(s)` contains `delta(t-s)` and

    B(t,s) ⊃ gamma_0^{-1} <phi_ddot(h(t)) z(t)> delta_{mu,alpha} delta(t-s)

In discrete time the delta becomes `1/dt`, so the kernel diagonal exceeds its
neighbours by that factor — it *looks* like an outlier, which is exactly the
wrong instinct. After the `gamma_0 * dt * B` weighting it contributes at `O(1)`.

This sharpens F1 threefold: it is a distinct Onsager term with a different
`dt`-scaling, not a large-ish diagonal entry; an `O(1)` field contribution being
dropped explains the recorded 20–50% kernel error; and because it carries
`phi_ddot` it vanishes identically for linear `phi`, which is precisely F1b.
The minimal detector is nonlinear `L = 2`.

**Status: theory only.** Phase 4 must measure it.

## Part B — deep linear closure derived and certified

`equations.md` §3 named the causal operators `C`, `D` but never defined them,
and gave no response formulas. Both derived in `derivations/02-deep-linear.md`:

    C^l = mask * [A^{l-1} + Delta * H^{l-1}],   D^l = mask * [B^l + Delta * G^{l+1}]
    A^l = M_l^{-1} C^l,   B^{l-1} = Nt_l^{-1} D^l
    M_l = I - g0^2 C^l D^l,   Nt_l = I - g0^2 D^l C^l

with the exact correlator-rule prediction
`f = diag[ Nt_L^{-1} (G^{L+1} C^{LT} + D^L H^{L-1}) M_L^{-T} ]`.

`validate_deep_linear.py` — **13 checks, 0 failed**.

| Check | Measured |
|---|---|
| D1 exact scalar ODE (L=1) | 7.24e-3 at dt=0.02 |
| D1b error is O(dt) | ratio 2.03 under dt-halving |
| D1c orthogonal sector | **exactly 0** (no MC floor) |
| D2 reduces to certified L=1 MC solver | 1.13e-3 |
| D3 lazy limit, `K0 = (L+1)Kx`, L=3 | 6.86e-5 |
| D7 responses causal | exactly 0 outside the mask |
| D7b kernels symmetric | 2.2e-15 |
| D5 ablation inert at L=1 | exactly 0 |
| D5b responses matter, L=2 / L=3 | 24.6x / 71.3x worse without them |
| D5c resummation matters | 1.73e-2 shift at `gamma_0=3` |
| D8 annealing rescues F5 | cold start diverges, annealed converges |
| D6/D6b sim gap vs N, vs dt-floor | 3.9e-2 → 2.1e-2 → 6.5e-3; 1.2x floor |

### The response ablation (F17 control)

| L | sim gap, full | no-response | ratio |
|---|---|---|---|
| 1 | 3.91e-3 | 3.91e-3 | **1.0x** |
| 2 | 6.51e-3 | 1.60e-1 | 24.6x |
| 3 | 4.05e-3 | 2.89e-1 | 71.3x |
| 4 | 9.04e-3 | 4.01e-1 | 44.4x |

Inert at L=1 exactly as the boundaries `A^0 = B^1 = 0` require, and biting hard
at every depth above it. This is the first time in the program that the
response sector has been shown to matter against measurement rather than
asserted.

## Findings

**1. Mutation testing caught two blind spots, both now closed.** Reintroducing
plausible bugs and checking the battery notices:

| Mutation | Initially | Now |
|---|---|---|
| drop `dt` from the memory operator | caught (diverges) | caught |
| non-strict causal mask | caught | caught |
| `A = C` instead of `M^{-1}C` | **missed** | caught by D5c |
| swap `M` / `Nt` in the kernel recursion | **missed** | caught by D6b |

The `A = C` mutation shifts `f` by only 0.6% at `gamma_0 = 1` — a real
`O(gamma_0^2 C D)` effect that grows with richness (1.7% at `gamma_0 = 3`) but
slipped under a fixed absolute bar. D5c tests it at a `gamma_0` where it is
unambiguous.

**2. A fixed absolute bar for the simulation gap is the wrong instrument.**
The `M`/`Nt` swap produced a gap 4x worse than baseline that still passed.
Replaced by a *measured* bar: the algebraic solver has no sampling floor, so
the analogue of F8's "report the floor beside the gap" is the O(dt)
discretisation error, obtained by dt-halving. The widest-N gap must land within
3x of it. Baseline sits at 1.2x; the swap mutation at 5.2x.

This generalises the round-001 lesson. Both solvers now judge their sim gap
against a *measured* floor — sampling for the Monte-Carlo solver, discretisation
for the algebraic one. Only the identity of the floor changes.

**3. F5 stiffness mapped.** Convergence at L=3, dt=0.02, best of
`beta ∈ {0.3, 0.1, 0.03}`:

| `gamma_0` \ T | 31 | 51 | 101 |
|---|---|---|---|
| 1–2 | 0.30 | 0.30 | 0.30 |
| 3 | 0.30 | 0.30 | 0.10 |
| 4 | 0.30 | 0.10 | 0.10 |
| 6 | 0.10 | **diverges** | **diverges** |

The required damping falls as `gamma_0 * (T*dt)` grows, with the edge near
`gamma_0 * horizon ≈ 6` — consistent with F5's stated operator norm
`~ dt*lambda*T`. `solve_annealed()` applies both registry fixes adaptively
(anneal `gamma_0` upward with warm starts, drop damping on failure) and rescues
`gamma_0 = 6, horizon 1.0`, which diverges at every fixed damping tried.
It raises `Diverged` rather than returning a silently wrong answer.

## Registry changes

No new failure modes. F1 gained the delta-function sharpening (§Part A), and
F5 gained a measured stability map and a working adaptive fix.

## Not covered

- **The Onsager diagonal.** `phi_ddot = 0` for linear `phi`, so nothing here
  touches it. This is F1b and it is the single largest remaining gap.
- F6, F8, F15, F16 — Monte-Carlo artifacts that cannot occur in an algebraic
  solve.
- Single-site sampling, Gaussian source generation, forward-mode sensitivity
  code: all bypassed by the algebraic closure.
- HP transfer: no harness yet (Phase 3).
