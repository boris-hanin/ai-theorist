# Round 001 — two-layer solver rebuild

**Not pre-registered.** This round was executed before `rounds/` existed, so
there is no committed prediction set to score against. That is a real
limitation and it is the reason the directory starts at 001 with a caveat
rather than a clean example. Everything below was written after the fact.

Date: 2026-08-07
Skill exercised: `dmft-derivation` (L=1 branch)
Artifacts: `skills/dmft-derivation/scripts/`

## Goal

The program's original solvers were confirmed unrecoverable (the hyperbolic
repo is empty; the cited `hyp_dmft*.py` files exist nowhere). Rebuild the
simplest solver from the reconstructed documentation alone, and in doing so
test whether that documentation is sufficient to rebuild from — which is the
re-validation of the reconstruction, by construction.

## Outcome

Sufficient, with one blocking defect found first.

`numerics.md` as reconstructed prescribed Euler-marching the prediction ODE,
which registry F4 names as a failure mode; `dmft-master`, `algorithm.md` and
`solver-library.md` all prescribed the correlator rule instead. Building
against the pre-Step-0 text would have reproduced F4. Fixed in Step 0 before
the rebuild started. Same for response Jacobians: `numerics.md` said
autodiff-through-unrolled-solve where the other three said exact forward-mode.

After that, the documentation was enough. The single-site system, the matched
simulation parameterisation, and the variance-reduction patterns all
transferred without needing the lost code.

## Results

`python3 skills/dmft-derivation/scripts/validate.py` — 20 checks, 0 failed, 30s.
At S = 2^14, P = 3, tanh unless noted:

| Check | Measured | Bar |
|---|---|---|
| C1 exact scalar ODE (linear, whitened) | seed-rms 3.13e-3 | 1e-2 |
| C2 final kernel `H(inf)` | 1.79e-2, `Delta(T=8) = 2.3e-10` | 6.25e-2 |
| C3 quadrature vs analytic (linear, erf) | 4.4e-16, 2.2e-16 | 1e-10 |
| C3b MC vs quadrature (tanh, S=2^17) | 1.95e-3 | 2e-2 |
| C4 lazy limit at `gamma0 = 0.05` | 4.07e-4 | 2e-3 |
| C4b deviation is `gamma0`-driven | ratio 41.3 | >3 |
| C5 MC floor by sample-halving | 1.06e-3 | reported |
| C8 sim gap vs N (256/1024/8192) | 2.81e-2 / 8.28e-3 / 4.11e-3 | shrinking |
| C9 kernel movement vs `gamma0` (0.1→1.0) | theory 6.8e-3→4.1e-1; sim 6.9e-3→4.1e-1 | both grow |
| C10 F4 error, dt-halved | 1.92e-3 → 9.52e-4, ratio 2.0 | O(dt) |

## Findings

**1. At L=1 the finite-width network is exactly the solver at S = N.** In muP,
`h_i(0) = W0_i·x/sqrt(D)` is exactly Gaussian with covariance `Kx`, i.i.d.
across neurons; the discrete GD update closes on the population only through
`Delta(t)`. No approximation is involved. So C8 measures convergence in sample
count, not the correctness of a mean-field closure, and the `1/sqrt(S)` and
`1/sqrt(N)` errors are the same error. This weakens the "simulation match"
leg of the validation bar at L=1 specifically — the independent legs are the
exactly-solvable reductions. At L >= 2 the response sector appears and the
comparison regains its force.

**2. The battery was initially blind to F4.** Mutation testing — reintroducing
registered failure modes and checking that the battery catches them — showed
that Euler-marching at `gamma0 = 1, dt = 2e-3` moved the C1 error from 7.4e-4
to *6.7e-4*, i.e. undetectable. The effect is genuinely below the Monte-Carlo
floor there. C10 was added to compare both prediction paths on identical
sources so the noise cancels exactly; it then resolves cleanly and scales as
O(dt).

Without measuring the floor, C1 would have certified a solver with a registered
failure mode reintroduced. This is the strongest argument in the program for
F8's "report the floor beside the gap" rule, and it generalises: **a check
battery needs its own mutation test before it can be trusted.** Added to the
round checklist in `rounds/README.md`.

A within-step stale-read reordering was likewise undetectable at small dt. Both
blind spots are printed at the end of every `validate.py` run.

**3. The lazy-limit check was conflating two error sources.** Against a
quadrature NTK the gap is `gamma0`-independent (ratio 1.0 at S=2^12) — it is
dominated by common-mode sampling error in the solver's initial kernel, which
is identical across `gamma0` at fixed seed. Conditioning on the solver's own
`K(0)` isolates the genuine feature-learning deviation: ratio 41 at S=2^14, and
flat in S across 16x. Split into C4 (physics), C4b (`gamma0` trend), C4c
(sampling shrinks with S).

## Smaller things worth not rediscovering

- Gauss-Hermite quadrature is unreliable for ReLU (polynomial-exact, loses the
  kink): 1.6e-2 error in `G` against the analytic arccos kernel. Quadrature is
  used only for smooth activations; the analytic form is authoritative for
  ReLU.
- Same-seed runs at different S are nested, not independent — numpy fills
  row-major, so the `S/4` draw is a literal subset of the `S` draw. An
  S-scaling ratio reads 1.1 instead of 2.0 unless disjoint seed blocks are
  used. F10 in miniature.
- `H(inf)` is a `t -> inf` statement and needs a horizon where `Delta(T)`
  actually vanishes; at T=1.2 the check fails for reasons that look like a
  kernel error.

## Registry changes

No new failure modes. Two existing entries sharpened:

- **F1** gained the computation-order refinement recovered from
  `algorithm.md` (`Abar(t,t) = 0` for fields read before the backward pass,
  `Bbar(t,t) != 0` for drives that see the same step's forward pass).
- **F4** and **F15** are now recorded as the same channel: the readout
  correlator carries an explicit `1/gamma0`, so its MC floor is
  `O(1/(gamma0 sqrt(S)))`.

## Not covered

The response sector, in full: F1's diagonals, F5, F6, F17 are untested because
`A^0 = 0` and `B^1 = 0` identically at L=1. The alternating fixed-point solver,
damping, deep linear closure, depth-muP, attention and MoE are all
unimplemented. `dmft-master` remains **reconstructed, not re-validated** — this
round tested that its documentation is buildable-from, not that its nine steps
are correct on a novel architecture.
