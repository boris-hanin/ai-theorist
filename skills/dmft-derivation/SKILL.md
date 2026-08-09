---
name: dmft-derivation
description: Derive and solve dynamical mean field theory (DMFT) descriptions of training dynamics in wide neural networks, following the Bordelon–Pehlevan self-consistent DMFT program. Use when a problem asks for the exact infinite-width training dynamics of a network in the feature-learning (mean-field/μP) regime — loss curves, kernel evolution (NTK/feature kernels), effects of the richness parameter γ₀, depth scaling, or lazy-vs-rich comparisons — on a fixed finite dataset under gradient flow/descent. Also use to set up the correct μP/depth-μP parameterization for hyperparameter-transfer questions.
---

# DMFT derivation skill (Bordelon–Pehlevan program)

This skill turns the self-consistent DMFT technique into an executable recipe.
Follow the phases in order. Do not skip Phase 0 (scoping) or Phase 5 (checks):
most silent failures are out-of-scope applications and unchecked sign/scaling
errors.

Reference files (load as needed):
- `references/equations.md` — complete closed equation systems (deep MLP,
  two-layer, deep linear, depth-μP residual) with every symbol defined.
- `references/numerics.md` — numerical solution algorithms (alternating MC
  fixed point; causal co-integration for L=1; algebraic solve for linear).
- `references/parameterizations.md` — μP / depth-μP / transformer scaling
  tables and the richness dial γ₀.
- `references/validation-checks.md` — mandatory sanity checks and known
  reproduction targets with exact settings.
- `scripts/` — runnable two-layer, deep-linear, and general-depth nonlinear P=1
  solvers; matched finite-width simulators; and four validation batteries.
  See `scripts/README.md` for the distinct certification and coverage bounds.

## Phase 0 — Scope check (mandatory)

The technique applies when ALL hold:
1. Width N → ∞ is taken FIRST, at fixed dataset size P and fixed time horizon
   t (both O(1) in N). If the problem needs P ∝ N, t growing with N, or
   online/one-pass data, STOP: base skill out of scope (data-averaged DMFT à
   la Mignacco et al. is a different skill; finite-width fluctuations are the
   2304.03408 delta).
2. Training is gradient flow (or small-step GD; discrete time is a recorded
   extension) on a differentiable loss over a fixed batch of P points.
3. The architecture's disorder is the random init of i.i.d. Gaussian weights,
   and every hidden unit is exchangeable within its layer (MLP; residual and
   transformer variants via the deltas in `references/parameterizations.md`).
4. The parameterization is (or can be converted to) the mean-field/μP form
   with explicit richness dial γ₀. If the problem is stated in NTK or standard
   parameterization, convert first (Phase 1); γ₀ → 0 recovers lazy/NTK.

Output of Phase 0: a scope verdict with the limit order written explicitly
(e.g. "N→∞ at P=10, t≤50, gradient flow, MSE").

## Phase 1 — Set up model and parameterization

1. Write the architecture with ALL width/depth scale factors explicit and unit
   Gaussian init: forward pass hˡ⁺¹ = (1/√N)·Wˡφ(hˡ), h¹ = (1/√D)·W⁰x, output
   f = (1/(γ√N))·wᴸ·φ(hᴸ), with W entries ~ N(0,1). For residual nets use the
   depth-μP table (1/√(LN) branches). See `references/parameterizations.md`.
2. Set γ = γ₀√N and learning dynamics dθ/dt = −γ²∇θL. This makes df/dt = O(1)
   and feature movement O(γ₀).
3. Define the error signal Δμ(t) = −∂ℓ/∂f|_{fμ(t)} (MSE: Δμ = yμ − fμ) and the
   O(1) backprop fields gˡ = γ√N·∂f/∂hˡ = φ̇(hˡ)⊙zˡ, zˡ = (1/√N)Wˡᵀgˡ⁺¹.
4. Declare the order parameters: feature kernels Φˡ_{μα}(t,s) = (1/N)φ(hˡμ(t))·φ(hˡα(s)),
   gradient kernels Gˡ_{μα}(t,s) = (1/N)gˡμ(t)·gˡα(s), and the NTK
   K_{μα} = Σₗ Gˡ⁺¹_{μα}Φˡ_{μα} with boundary conventions Φ⁰ = Kˣ = XXᵀ/D,
   G^{L+1} = 1.

## Phase 2 — Derivation (the six-step field-theory pipeline)

Execute these steps; each is a well-defined mathematical operation.

- **D1. Isolate disorder.** Integrate the weight dynamics so Wˡ(t) = Wˡ(0) +
  (learned rank-P update). Define the fields carrying random-init dependence:
  χˡ⁺¹μ(t) = (1/√N)Wˡ(0)φ(hˡμ(t)) and ξˡμ(t) = (1/√N)Wˡ(0)ᵀgˡ⁺¹μ(t). Rewrite
  hˡ, zˡ as (disorder field) + γ₀·(memory integral over kernels and Δ).
- **D2. MSRDJ generating functional.** Write the moment generating functional
  of {χ, ξ} over init disorder; enforce field definitions with Fourier
  delta-functionals, introducing conjugate fields χ̂, ξ̂.
- **D3. Disorder average.** Integrate out the i.i.d. Gaussian Wˡ(0) exactly
  (they appear linearly in exponents). CRITICAL: the same Wˡ(0) appears in
  both χˡ⁺¹ and ξˡ — this forward–backward coupling is what generates the
  response functions Aˡ, Bˡ. Introduce order parameters Φ, G (+ conjugates)
  and the response pair A, B via Hubbard–Stratonovich.
- **D4. Single-site factorization.** The action becomes N·S[Φ,Φ̂,G,Ĝ,A,B] with
  a single-site partition function: neurons decouple given order parameters.
- **D5. Saddle point.** N→∞: stationarity gives Φ̂ = Ĝ = 0 and the
  self-consistency Φˡ = ⟨φ(hˡ)φ(hˡ)ᵀ⟩, Gˡ = ⟨gˡgˡᵀ⟩ over the single-site
  measure; A, B are identified as linear responses to the Gaussian sources.
- **D6. Read off the single-site process.** Gaussian sources uˡ ~ GP(0, Φˡ⁻¹),
  rˡ ~ GP(0, Gˡ⁺¹); causal integral equations for hˡ, zˡ with γ₀-weighted
  memory kernels [A + ΔΦ] and [B + ΔG]; deterministic prediction dynamics
  dfμ/dt = Σα K_{μα}(t,t)Δα(t), f(0) = 0. Full system in
  `references/equations.md`.

Deliverable of Phase 2: the closed system (single-site processes +
self-consistency + prediction dynamics) with every symbol defined and
boundary conditions stated. Compare structurally against
`references/equations.md`; any mismatch in factors of γ₀, √N, or index
placement must be resolved before proceeding.

## Phase 3 — Simplify before computing

Always check, in order:
1. **L = 1 (two-layer):** A⁰ = 0, Bᴸ = 0 ⇒ ALL response functions vanish;
   zμ(t) = w(t) is the scalar readout; u is a static GP(0, Kˣ). The system
   becomes causally integrable with no fixed-point iteration (only equal-time
   kernels feed the prediction ODE). Use the co-integration algorithm in
   `references/numerics.md`.
2. **Linear activation:** kernels close algebraically — no Monte Carlo. Deep
   linear = matrix fixed-point equations; L=1 linear + whitened data + single
   output direction = scalar ODE ∂ₜΔ = −2√(1+γ₀²(y−Δ)²)·Δ.
3. **Small γ₀:** perturbative expansion Φ = Φ₀ + γ₀²Φ₂ + …, useful both as an
   analytic result and as a numerics check.
4. **Symmetry reductions:** whitened data (Kˣ = I), single output direction,
   permutation-symmetric targets — reduce the P×P×T×T kernel objects before
   discretizing.

## Phase 4 — Numerical solution

Follow `references/numerics.md`. Two invariants override the textbook forms:
- **Predictions by the correlator rule (F4):** f_μ(t) = γ₀⁻¹⟨w(t)φ(h^L_μ(t))⟩
  read off the sampled population each step. NEVER Euler-march
  df/dt = KΔ at SGD's step — that error masquerades as finite-width error.
- **Responses by exact forward-mode sensitivities (never finite differences in
  production).** FD is how you TEST the sensitivity code: correct agreement is
  ε-independent and O(1/S).

Summary:
- General deep case: alternating Monte Carlo fixed point (their Algorithm 1):
  sample S single-site trajectories given kernels → re-estimate kernels and
  responses → damped update (β ≈ 0.3–0.7) → iterate to convergence.
- Two-layer: forward causal co-integration (exact, no iteration): march S
  single-site samples t → t+dt, then read f off the updated population.
- Cost: O(P²T²) memory, O(P³T³) time. If N_sim·(cost of direct training) is
  cheaper than the DMFT solve (i.e. PT ≳ N), reconsider whether theory
  numerics are the right tool.
- Runnable implementation of the two-layer case, with the Phase 5 battery:
  `scripts/` (see `scripts/README.md`).

## Phase 5 — Mandatory checks (do not report results without these)

From `references/validation-checks.md`:
1. **Lazy limit:** γ₀ → 0 must freeze kernels at NNGP/NTK init values and
   reproduce linear NTK dynamics f(t) = (I − e^{−K₀t})y (MSE). Run your
   solver at γ₀ ≈ 0.05 and check against this closed form.
2. **t = 0 kernels** must match the standard NNGP/NTK recursions.
3. **Exactly solvable cases:** linear whitened 1D ODE; deep linear algebraic
   solution.
4. **Finite-width simulation match:** train an actual width-N network (N ≫ PT
   if possible, several seeds) in the SAME parameterization and discretization;
   compare loss curves, equal-time kernel trajectories, and final kernels
   (Frobenius alignment). Feature-learning effects must appear in BOTH (e.g.
   kernel movement growing with γ₀).
5. **Convergence audits:** results stable under dt → dt/2, S → 2S, and (for
   the fixed-point solver) different damping β and inits.
6. **Report the Monte-Carlo floor next to every theory-vs-sim gap (F8).**
   Certify it by sample-halving. A gap smaller than its own floor is not
   evidence of agreement OR disagreement.
7. **For hyperparameter transfer, hold normalized `eta` fixed (F25).** Compare
   common-seed trajectories across the scale dial and test whether absolute
   differences settle with the derived finite-size rate. Sweep the local
   finite-horizon optimum/largest stable rate separately under an
   `edge_of_stability` label; it is never the transfer verdict. Always record
   both normalized `eta` and the parameterization-derived raw optimizer LR.

`scripts/validate.py` runs checks 1–6 for the two-layer case and prints the
numbers; extend it rather than re-deriving the battery by hand.

## Failure modes to watch

Full entries in `registry/failure-modes.md` (repo-level, canonical).

- Applying the theory at P or t that scale with N (silently wrong) — Phase 0.
- **F1 — equal-time response diagonals can be nonzero and scale as `1/dt`.**
  Record them, but do not automatically include the endpoint in a causal memory
  sum. Computation order fixes both the diagonal and its integration weight; in
  round 003's L=2 solver the strict-past (weight-zero) endpoint fits and weight
  one is falsified. Linear cross-checks are blind to the nonlinear term (F1b).
- Dropping the response functions A, B for L ≥ 2 (they are O(1) and matter;
  only L=1 kills them). Corollary: L=1 agreement cannot certify response code.
- **F4 — never Euler-march theory prediction curves.** Correlator rule; see
  Phase 4.
- **F15 — the readout channel carries γ₀⁻¹**, so its MC floor is O(1/(γ₀√S))
  and GROWS as γ₀ shrinks, masquerading as theory failure in near-lazy runs.
  Fix with antithetic readout pairs.
- **F16 — independently seeded Sobol streams are NOT independent.** One joint
  stream, sliced per source family.
- **F17 — write response rows before same-step reads**, and assert nonzero at
  read time. An ablation that changes nothing is a red flag, not a pass.
- **F10 — seed-average before comparing**; f(0) = 0 holds only in the γ → ∞
  scaling (f(0) = O(1/(γ₀√N)) at finite width).
- Confusing γ (bare, = γ₀√N) with γ₀ in memory-kernel prefactors.
- Using finite differences for response sensitivities in production (noisy,
  unstable) instead of exact forward-mode propagation.
- Comparing against sims in a DIFFERENT parameterization (standard PyTorch
  init/LR does not match; convert first).
- **F5/F6 — undamped fixed-point iteration diverges** for rich γ₀ (the Δ-map
  has operator norm ~ dt·λ·T); response-noise rectifies into a positive kernel
  bias. Raise damping (extra damping on responses), anneal γ₀ from small
  values, or increase S.
- **F25 — never use width-wise `argmin_eta` drift as the primary transfer
  verdict.** It can track a moving finite-width stability edge even when fixed
  normalized-`eta` dynamics converge correctly. Derive the raw LR map in
  function space, test fixed-`eta` trajectories, and report the stability edge
  in a separate non-gating record.

## Extensions recorded (see references for the deltas)

- Depth-μP residual networks & joint N,L→∞ limit (layer-time Brownian term,
  NTK depth-ODEs, HP transfer across width AND depth) — arXiv:2309.16620.
- Finite-width 1/N fluctuations (propagator around the saddle) — 2304.03408.
- Multi-head transformer limits (per-axis mean fields, exponent audits) —
  2405.15712.
- Discrete-time GD, weight decay / L2, non-MSE losses — appendices of
  2205.09653.
