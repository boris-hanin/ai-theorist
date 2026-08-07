# The master algorithm — long form

> RECONSTRUCTED from the program record; pending re-validation.

Expanded procedure for deriving the mean-field/μP training-dynamics limit
of a novel architecture. SKILL.md is the compact version; this file adds
the operational detail per step.

## Step 0 — Scaling audit

Write the forward pass with every width/depth/sparsity dial and scale
factor explicit; unit-Gaussian raw parameters. Propose exponent tables:
per parameter group, LR = η₀γ₀²·(dial)^p·dt; branch/output scales
similarly. Then PIN empirically before deriving anything:
- per-particle groups: grad or update RMS vs the dial, slope −p ⇒ Θ(1)
  updates at exponent p;
- reused matrices: the criterion is INDUCED FEATURE VELOCITY (apply a
  group-only update at candidate exponent, measure downstream feature RMS
  change) — row-norm criteria mislead here (a μP hidden matrix has
  O(dial^{−1/2}) rows but Θ(1) feature velocity at its exponent);
- depth/sparsity dials: few-step loss-drop flatness + per-block movement
  slopes (depth-μP: −1/2); movement-collapse under dial rescaling doubles
  as a bug detector (the MoE √α collapse).
Also audit init hygiene: limit value of f(0), variance slopes vs dial,
population heterogeneity scales (a needed centering/amplification shows
up here — e.g. the hyperbolic √m-centered features).

## Step 1 — Edge classification

For each tensor appearance in forward AND backward:
- (a) single-use disorder: appears once (per site) ⇒ in the limit its
  image is a Gaussian source; covariance = population kernel of the
  vector it contracts. No responses.
- (b) reused disorder: same matrix in forward and (transposed) backward,
  or multiple forward uses ⇒ forward field z and backward field ζ with a
  response pair Ā(t,s) = ⟨∂(forward-carried site variable)(t)/∂ζ(s)⟩,
  B̄(t,s) = ⟨∂(backward drive)(t)/∂(z-source)(s)⟩. Determine equal-time
  structure from computation order: fields read before the backward pass
  have Ā(t,t)=0; drives that see the same step's forward have B̄(t,t)≠0
  (F1). Trained parts of reused matrices are rank-P per step and give
  feature kernels θ(s)·C(s,t) (forward) and θ(s)·D(s,t) (backward), with
  θ carrying the exact LR/branch prefactors from Step 3.
- (c) readout carrier: express f exactly through population correlators
  each step (discrete-time-exact predictions; never Euler-march the
  theory, F4). Control variates on the sampled correlators.
- (d) bilinear order parameters (logits, gates, normalizers): close via
  deterministic recursion (d1) when self-averaging, or Gaussian-sector
  cross-covariance (d2) when they retain site randomness. Normalizer
  chains (RMS-type) enter as deterministic radial functions of second
  moments; collect ALL their derivative channels into per-sample
  deterministic coefficients (the F/Ψ-channel pattern) and verify the
  collected coefficient against the exact finite-size backward
  numerically (reduction checks) — hand expansions of normalizer radial
  terms are error-prone.

## Step 2 — Populations and nesting

Exchangeable units ⇒ site populations; list per-site states (embedding
particles, per-block scalars, readout components) and per-site fields
(one z/ζ pair per class-(b) matrix touching the site). Conditioning
structure (routing, heads) becomes mixture/conditioned single-site
measures; deterministic global scalars stay outside sites.

## Step 3 — Exact update identities

Integrate updates exactly in discrete time with the pinned exponents.
Keep γ₀ vs γ = γ₀√N distinct in every prefactor. Tangent projections and
normalizer renormalization drifts: carry exactly at finite size, then
show which terms vanish in the limit (order counting per coherent term).

## Step 4 — Disorder average

MSRDJ/cavity. Deliverables per class-(b) matrix: source covariances
(C for z, D for ζ) as population kernels; the response-term attachments
(which site-attached factor multiplies Ā/B̄ in each field equation); the
independence structure across matrices (independent sources given
populations; forward/backward sources of the SAME matrix asymptotically
independent — row vs column).

## Step 5 — Closure and causal order

Write the per-step schedule explicitly: population kernel rows → source
sampling (growing Cholesky) → forward fields (need same-step Ā rows —
WRITE THEM FIRST, F17) → prediction/error → backward drives → D rows,
B̄ rows → backward fields → updates + sensitivity propagation. Verify no
equal-time circularity; assert kernel rows nonzero at read time.

## Step 6 — Simplification ladder

(1) no reuse ⇒ McKean–Vlasov, exact causal co-integration, no responses;
(2) linear or gain-modulated-linear ⇒ kernels close deterministically up
to site-scalar heterogeneity; (3) γ₀ → 0 lazy expansion with quadrature
init kernels; (4) symmetry reductions (whitened data, angle-grid
populations in low input dimension).

## Step 7 — Solve

See solver-library.md.

## Step 8 — Validate

Pre-registered plan with tolerances and falsifiers. Minimum battery:
lazy anchor; exactly solvable reduction; degenerate-case collapse onto a
previously certified instance; seed-averaged finite-size sims across the
dial with gaps shrinking at the predicted rate; MC floor by S-halving;
response ablation (must matter, else report under-powered); order
parameters compared at final time, norm-relative. Report protocol
amendments (e.g. moving off a stability edge) explicitly.
