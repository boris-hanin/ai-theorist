# Two-layer DMFT solver and Phase 5 battery

```
python3 validate.py                  # L=1 battery, ~30 s, 20 checks
python3 validate_deep_linear.py      # deep linear battery, 13 checks in quick mode
python3 validate_deep_nonlinear.py   # general-depth nonlinear P=1 battery
python3 validate_scaling.py          # scaling and transfer battery
#   --quick on either for a smaller/shorter run
```

Exit code is nonzero if any check fails. Every check prints its measured number
next to its bar.

| File | What it is |
|---|---|
| `dmft_two_layer.py` | L=1 solver: exact causal co-integration, correlator-rule predictions, antithetic readout pairs, optional joint-Sobol QMC |
| `dmft_deep_linear.py` | Deep **linear** solver: algebraic closure, live response sector, damped fixed point with γ₀-annealing (F5). No sampling floor |
| `dmft_deep_nonlinear.py` | General-depth **nonlinear**, P=1 single-site Monte Carlo solver with exact response sensitivities; strict-past endpoint by the round 003 measurement |
| `dmft_l2_nonlinear.py` | Independent L=2 nonlinear implementation retained as a cross-check and endpoint-coefficient instrument |
| `sim_two_layer.py` | Finite-width L=1 net in the matched parameterisation; updates `W0` explicitly rather than using the reduced `h`-recursion, so it is an independent implementation path |
| `sim_deep.py` | Finite-width depth-L net, same conventions, any activation |
| `exact.py` | Closed forms and Gauss-Hermite quadrature — the independent ground truth. Uses nothing from the solver |
| `validate.py`, `validate_deep_linear.py`, `validate_deep_nonlinear.py`, `validate_scaling.py` | Validation batteries |
| `activations.py` | `linear`, `tanh`, `relu`, `erf` with derivatives |

**Two floors, one discipline.** Neither battery judges a theory-vs-simulation
gap against a hand-picked number. Each measures its own floor and reports the
gap relative to it: sampling error (by S-halving) for the Monte-Carlo solver,
discretisation error (by dt-halving) for the algebraic one. Mutation testing
forced both — a fixed absolute bar let real bugs through in each case.

## Provenance

Written from `references/` and `skills/dmft-master/references/algorithm.md`
alone, after the program's original solvers were confirmed unrecoverable. The
reconstruction was therefore also a test of the reconstructed documentation:
the docs were sufficient to rebuild a working L=1 solver, but only after the
F4 correlator-rule prescription was restored to `numerics.md` (it had been
Euler-marching, contradicting the registry).

## Two structural facts worth knowing before reading results

**1. At L=1 the finite-width network IS the solver at S = N.** In muP,
`h_i(0) = W0_i · x / sqrt(D)` is exactly Gaussian with covariance `Kx`, and
`w_i(0) ~ N(0,1)`, both i.i.d. across neurons. The discrete GD update of
`(h_i, w_i)` closes on the population only through `Delta(t)`. So a width-N
network is *exactly* N samples of the single-site process — no approximation
anywhere.

Consequence: **C8 (theory vs simulation) is a convergence check in sample
count, not a test of a mean-field closure.** The `1/sqrt(S)` Monte-Carlo error
and the `1/sqrt(N)` finite-width error are the same error. The genuinely
independent checks are C1-C4, which compare against closed forms and
quadrature. Do not cite the sim match as closure validation at L=1; at
L >= 2 it becomes a real test, because the response sector appears.

**2. F4 is below the Monte-Carlo floor in this regime.** Expanding the exact
discrete update reproduces `f + dt*K*Delta` to `O(dt^2)` per step, so the
correlator rule and the Euler march differ by `O(dt*T)` overall. Measured at
`gamma0=3, T=3`: `1.9e-3` at `dt=0.02`, halving with `dt` (C10). At
`gamma0=1, dt=0.002` it is `~1e-4` — well under the `~1e-3` floor at
`S=2^14`. So F4 is real, but no theory-vs-sim comparison at accessible S can
see it here. C10 detects it by running both paths on identical sources so the
noise cancels exactly.

This is the F8 lesson in operation: without the floor next to the number, C1
would have "passed" a solver with F4 reintroduced.

## Independent re-derivation of the C1 ground truth (F14)

`equations.md` §3 gives the exactly solvable case. Re-derived here from the
single-site system rather than transcribed, so two routes agree.

For linear `phi`, `Kx = I`, `y = y*e`, write `a = h·e` and `w` for the readout:

    a' = gamma0 * w * Delta,    w' = gamma0 * Delta * a

Then `d/dt(a^2 - w^2) = 0`. With `S = <a^2 + w^2>` and `Q = <a w>`:

    S' = 4*gamma0*Delta*Q,   Q' = gamma0*Delta*S   =>   d/dt(S^2 - 4Q^2) = 0

At `t=0`, `<a^2> = <w^2> = 1` and `<aw> = 0`, so `S^2 - 4Q^2 = 4` throughout,
giving `S = 2*sqrt(1 + Q^2)`. Since `Q' = gamma0*Delta*S = gamma0*f'`, we get
`Q = gamma0*f = gamma0*(y - Delta)`, hence

    dDelta/dt = -S*Delta = -2*sqrt(1 + gamma0^2 (y-Delta)^2) * Delta

matching `equations.md`. Letting `Delta -> 0` gives
`<a^2> = sqrt(1+gamma0^2 y^2)` along `e` and `1` orthogonal to it, i.e. the
stated `H(inf)`. Both are used as ground truth in C1/C2.

## What the battery covers

Independent ground truth: C1 (exact scalar ODE), C2 (final kernel `H(inf)`),
C3 (init kernels, three routes: quadrature, analytic, Monte Carlo), C4 (lazy
limit vs closed-form NTK dynamics).

Failure-mode regression tests: C6 (F15 antithetic readout — also reproduces
the un-fixed signature, `|f(0)|` growing as `gamma0` shrinks), C7 (F16 — 
reproduces `|corr| = 0.98` between independently seeded Sobol streams, and
verifies one joint sliced stream fixes it), C10 (F4).

Convergence audits: C5 (MC floor by sample-halving, F8), C5b (`dt -> dt/2`),
C4c (init-kernel gap is sampling, seed-averaged per F10).

Physics trends: C8 (sim gap shrinks with N, reported against the floor),
C9 (kernel movement grows with `gamma0` in BOTH theory and simulation).

## Two caveats found while building this

* **Gauss-Hermite quadrature is unreliable for ReLU.** It is polynomial-exact
  and loses accuracy on the kink: measured disagreement with the analytic
  arccos kernel is `6.7e-4` in `Phi` and `1.6e-2` in `G`. The analytic form is
  authoritative for ReLU; quadrature is used only for smooth activations
  (`tanh`), where it agrees with the analytic `linear`/`erf` forms to machine
  precision.
* **Same-seed runs at different S are nested, not independent.** numpy fills
  row-major, so the `S/4` draw is a literal subset of the `S` draw at the same
  seed. A single-seed `S`-scaling ratio reads `1.1` instead of `2.0`. C4c uses
  disjoint seed blocks and seed-averaged RMS.

## Status of the response sector

`dmft_deep_linear.py` covers it for linear `phi`, exactly and with no sampling
floor. Measured: ablating the interior responses makes the simulation gap
**24.6x** worse at L=2, **71.3x** at L=3, **44.4x** at L=4, and is exactly inert
at L=1 where `A^0 = B^1 = 0`. F5 stiffness is mapped and handled by
`solve_annealed()`.

The nonlinear response sector is now implemented for general depth at P=1 in
`dmft_deep_nonlinear.py`. Its quick battery checks L=1–3, compares an independent
L=2 implementation, verifies response ablations at L=2/3, and reaches the
measured simulation/MC floor. The equal-time `1/dt` structure was measured in
round 003; the unit endpoint coefficient was falsified, so strict past is the
default. It has not been re-measured at L>=3.

Still missing: P>1 nonlinear coupling, residual/attention/MoE dynamics in this
core solver, real-data checks, and a general nonlinear fixed-point convergence
theory. New batteries still require mutation tests and measured floors.
