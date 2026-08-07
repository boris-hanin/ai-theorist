---
name: dmft-master
description: The master DMFT algorithm — a single zero-shot procedure for deriving the infinite-width (mean-field/μP) training-dynamics limit of a NOVEL architecture, synthesized from the validated instances (deep MLP, depth-μP ResNet, multi-head attention, MoE, and later hyperbolic Busemann networks). Every completed computation in the program is derivable as a trace of this algorithm. Use FIRST on any new-architecture DMFT request; fall back to the per-instance skills for their specific deltas.
---

> **PROVENANCE: RECONSTRUCTED.** Original validated skill tree (SKILL.md +
> references/algorithm.md, instances.md, failure-modes.md,
> solver-library.md) lost to workspace recycling; rewritten from the
> program record. The 9-step algorithm and edge classification below are
> as validated; the reference files' long-form content is partially
> reconstructed. Pending re-validation as a package. The failure-mode
> registry now lives at `registry/failure-modes.md` (F1–F17).

# The master algorithm (9 steps)

Given a novel architecture with explicit width dial(s) and a training
procedure, execute in order. The bar for any claimed result downstream:
derivation + independent numerics + finite-size simulations — never
symbolic plausibility alone. Pre-register the validation plan (targets,
tolerances, falsifiers) BEFORE computing.

**Step 0 — Scaling audit (empirical, mandatory).** Write every scale
factor explicitly; propose exponents for each parameter group (per width
axis, and per depth/sparsity/other dial); PIN them by measurement: per-
group update RMS (or induced feature velocity, for reused matrices) vs
each dial must be Θ(1)-flat at the correct exponent. Never derive against
unpinned exponents.

**Step 1 — Edge classification.** For every appearance of a random/
trained tensor in the computational graph, classify the edge:
  (a) **single-use** disorder → Gaussian source (CLT), covariance = the
      population kernel of what it multiplies;
  (b) **reused** disorder (same matrix forward + transposed backward, or
      multiple forward uses) → forward/backward field pair with response
      functions (Ā, B̄) and trained-part feature kernels;
  (c) **readout carrier** → correlator: the prediction f is an explicit
      function of population correlators each step (exact discrete time);
  (d) **bilinear order parameter** (logits, gates, norms) → close via
      (d1) deterministic recursion or (d2) Gaussian-sector cross-
      covariance, per its reuse structure.

**Step 2 — Populations and nesting.** Identify the exchangeable unit
populations (neurons, heads, experts, stream sites), what states attach
to each site, and how populations nest/condition (routing, heads within
layers). Single-site variables = per-site states + received fields.

**Step 3 — Exact update identities.** Integrate the parameter updates
exactly (discrete time preferred); express every field's evolution as
(frozen-disorder part) + (trained-part memory integral with explicit
learning-rate/richness prefactors). Keep γ₀ separate from γ = γ₀√N.

**Step 4 — Disorder average.** MSRDJ/cavity over the frozen disorder:
class-(a) edges give Gaussian sources; class-(b) edges give the response
pairs — including EQUAL-TIME diagonals where a field feeds the same
step's backward (F1).

**Step 5 — Closure.** Collect the self-consistent system: population
kernels C/D/Φ/G, responses Ā/B̄, deterministic order parameters, and the
prediction correlator. State the causal integration order; verify no
circular dependency at equal time.

**Step 6 — Simplify.** Apply the degenerate-case ladder before solving:
no-reuse ⇒ McKean–Vlasov (responses vanish); linear/gain-modulated ⇒
algebraic kernel closure; small-γ₀ lazy expansion; symmetry reductions.

**Step 7 — Solve.** Single-site Monte Carlo with exact forward-mode
sensitivities for responses (never finite differences in production;
FD only to TEST the sensitivity code), causal co-integration when
possible, damped fixed-point iteration otherwise; variance reduction:
antithetic pairs (F15), ONE joint QMC stream (F16), stratified heavy
components; kernel rows written before same-step reads (F17).

**Step 8 — Validate.** Anchors: lazy limit vs closed-form NTK dynamics;
exactly solvable reductions; degenerate-case collapse onto previously
certified skills; finite-size sims in the SAME parameterization and
discretization, seed-averaged (F10), across the width dial with gaps
shrinking at the predicted rate; MC floor by sample-halving (F8);
response ablation must matter or the test is reported under-powered.

# Instance traces (each validated computation as a run of the algorithm)

- **Deep MLP** (2205.09653): edges (a)+(b) per layer; alternating-MC
  solver; F1, F4, F5, F6 discovered here.
- **Depth-μP ResNet** (2309.16620): multi-block (b) sandwich at 1/√L;
  depth-collapse validation; F14.
- **Multi-head attention** (2405.15712): per-axis Step-0; class-(d)
  logits; F3 dichotomy corrected against data.
- **MoE** (2601.20205): routing-conditioned populations; √α movement-
  collapse audit; η_up ∝ α; F2-flavor bookkeeping fix.
- **Hyperbolic Busemann MLPs + horospherical residual** (novel, rounds
  1–3 of the hyperbolic program): centered hyperbolic μP invented via
  Step 0; L=1 McKean–Vlasov (Step 6 ladder); L=2 response sector; the
  residual net as a gain-modulated linear stream; depth+width transfer;
  F15–F17 discovered here.

Zero-shot claim (validated on the hyperbolic rounds, which postdate the
synthesis): a NEW architecture is handled by running Steps 0–8 without
consulting the per-instance skills, which serve as worked traces.
