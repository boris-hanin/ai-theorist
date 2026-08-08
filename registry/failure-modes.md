# Failure-mode registry (DMFT derivation & solver program)

**This file is canonical.** Do not mirror it into skill trees — a byte
duplicate previously existed at `skills/dmft-master/references/failure-modes.md`
and was removed. Skills cite F-numbers and point here.

Named failure modes accumulated across validation rounds. F15–F17 are verbatim
from the session that discovered them; F1–F14 are summaries reconstructed from
the program record. The long-form originals were lost with the session archives
and are NOT recoverable — F7, F9 and F13 are permanent holes, not pending
restoration. By the program's own F14 standard, the reconstructed entries are
hints, not sources: where an entry conflicts with a derivation or a measurement,
the measurement wins and the entry gets rewritten.

Entry schema for new modes: **mechanism → detection signature → fix → guard.**
The reconstructed entries mostly lost their detection signatures; that is the
field that makes an entry actionable, so supply it for anything new.

- **F1 — equal-time response diagonal.** B̄(t,t) = ⟨φ̈·backward⟩ ≠ 0 for
  nonlinear φ (and for gain-modulated linear structures). Masking response
  kernels with strict lower-triangular masks silently drops it (was worth
  20–50% kernel error in the deep-MLP round). Corollary F1b: linear
  cross-checks are blind to this — they pass with the bug present.
  *Refinement (from `dmft-master/references/algorithm.md` Step 1b): the rule is
  set by computation order, not by nonlinearity alone — a field read BEFORE the
  backward pass has Ā(t,t) = 0; a drive that sees the SAME step's forward pass
  has B̄(t,t) ≠ 0. Determine both from the update graph rather than assuming.*
  *Sharpening (derived in `derivations/01-deep-mlp.md` §7, round 002): the
  equal-time response is a **delta function**, not merely a nonzero value. Since
  h(t) = u(t) + γ₀·(memory over [0,t]), the source enters h directly, so
  δh(t)/δu(s) ⊃ δ(t−s) and B̄ ⊃ γ₀⁻¹⟨φ̈(h(t))z(t)⟩δ_{μα}δ(t−s). **In discrete
  time the delta becomes 1/dt**, so the kernel diagonal exceeds its neighbours
  by that factor and looks like an outlier — "cleaning" it is the natural wrong
  instinct. **Status of the 1/dt structure: MEASURED AND CONFIRMED** (round 003,
  L=2 P=1 erf: the diagonal doubles as dt halves, 5.40 → 11.08 → 24.21).*
  *Correction (round 003): the follow-on claim that this term therefore
  contributes at O(1) — i.e. that it should be integrated against ∫₀ᵗds with
  weight 1 — was **FALSIFIED**. Against simulations extrapolated to N→∞, weight
  0 fits (4.4e-3, floor 2.6e-3) and weight 1 does not (1.07e-2, 4× the floor).
  A delta at the endpoint of the causal integral does not contribute; the
  correct discrete treatment uses the STRICT past. Scope: one solver, L=2, P=1,
  erf, γ₀=1. This does NOT retract F1 itself — the original deep-MLP round's
  20–50% error may have come from masking something other than the endpoint
  term — but the "worth 20–50%" figure is not reproduced here and should not be
  quoted as if it were general.*
- **F2 — backward-transpose scale.** The transposed-matrix backward field
  carries its own scale; mis-bookkeeping shows up as a collapse of
  measured parameter movements under the appropriate rescaling (used to
  catch an α_ffn error in the MoE round: movements collapsed under √α).
- **F3 — concentration vs freezing.** "This object stops moving as
  N grows" has two distinct mechanisms: concentration of a population
  average vs freezing of individual entries. Scaling claims must identify
  which; the attention round's ΔA=Θ(N^{-1/2}) pre-registration was
  falsified by data and corrected to the concentration branch.
- **F4 — Euler-marched theory curves.** Integrating the limit ODEs with
  the same step as SGD accumulates O(dt²λ²) per step and masquerades as
  finite-width error. Fix: exact discrete-time predictions via the
  correlator rule (f from population correlators each step) with control
  variates.
- **F5 — Δ-loop stiffness.** The residual map Δ ↦ y − f[Δ] has operator
  norm ~ dt·λ·T; naive fixed-point iteration diverges — needs inner
  damping. *Measured (round 002, deep linear, L=3, dt=0.02): the damping β
  needed falls as γ₀·(T·dt) grows, and the stable edge sits near
  γ₀·horizon ≈ 6. γ₀=3,T=101 needs β=0.1; γ₀=6,T=51 diverges at every fixed β
  in {0.3, 0.1, 0.03} but converges when annealed. Signature: the iteration
  either overflows or produces a singular I − γ₀²CD.* Fix: anneal γ₀ upward
  with warm starts AND drop damping on failure — `solve_annealed()` in
  `dmft_deep_linear.py` does both adaptively. Guard: raise rather than return
  a silently non-converged answer.
- **F6 — response-noise rectification.** MC noise in response kernels
  rectifies into a positive kernel bias inside self-consistent loops; fix
  with slow damping on the response updates.
- **F7 — LOST.** Entry existed in the archived long-form registry; content not
  recoverable. Do not reuse the number.
- **F8 — MC floor.** Solver Monte-Carlo floors scale as S^{-1/2} (or worse
  for heavy-tailed populations, cf. F12); certify by sample-halving and
  report the floor next to any theory-vs-sim gap.
- **F9 — LOST.** Entry existed in the archived long-form registry; content not
  recoverable. Do not reuse the number.
- **F10 — init-offset hygiene.** Finite-width init fluctuations (e.g.
  f(0) ≠ 0) contaminate comparisons; seed-average, and design
  parameterizations whose limit init is exactly known (centering,
  antithetic pairs).
- **F11 — quenched-randomness conditioning.** Condition on the realized
  disorder consistently between theory and simulation when comparing
  trajectories.
- **F12 — rate-amplified metrics.** Metrics multiplied by large rates
  (1/γ₀ readouts, heavy-tailed site weights) inflate MC variance and can
  invert apparent trends; use robust aggregation and variance-reduced
  estimators.
- **F13 — LOST.** Entry existed in the archived long-form registry; content not
  recoverable. Do not reuse the number.
- **F14 — garbled sources.** Fetched/transcribed formulas can be wrong
  (a ResNet-round transcription was 3.3× off). Derive independently; make
  two independent routes agree; simulations are ground truth.
- **F15 — 1/γ₀-amplified Monte-Carlo noise.** In near-lazy solves the
  readout channel multiplies sampled order parameters by 1/γ₀, so the MC
  floor is O(1/(γ₀√S)) — it GROWS as γ₀ shrinks and masquerades as theory
  failure. Fix: antithetic readout pairs (identical carrier states,
  opposite readout sign) make the amplified channel exactly zero at init;
  diagnosis signature: the gap's γ₀-trend flips sign after the fix.
- **F16 — cross-correlated "independent" QMC streams.** Independently
  seeded Owen-scrambled Sobol generators over the same dimensions are NOT
  independent (measured same-dimension |corr| up to 0.96,
  scipy.stats.qmc.Sobol). One stream per Gaussian-source family silently
  correlates sources across families/blocks. Fix: ONE joint Sobol stream
  of dimension (families × dims), sliced per family. Signature:
  init-time cross-moments that should vanish don't; seed scatter far
  above 1/√S.
- **F18 — one-step scale analysis is blind to correlation buildup.** A heuristic
  Θ(·) derivation labels each contraction in the chain coherent or incoherent
  *as it is at initialisation*, then takes one optimiser step. But correlations
  between weight matrices and backward signals **build over training**, flipping
  a contraction from incoherent to coherent — so the one-step answer can be
  right asymptotically and wrong at t=1, or vice versa. Discovered in the
  attention round: the update to the keys travels back through W_O, which is
  uncorrelated with ∂f/∂h̃ at init, so the first step is *suppressed*
  ((ΔA)² ~ N⁻² at t=1) and only reaches Θ(1) once that correlation develops.
  A one-step frame cannot see this **because the first step is the anomalous
  one**, not merely because one step is too few. Detection signature: the
  measured exponent drifts monotonically with training time toward the
  predicted one (measured −1.20 → −0.58 → −0.35 → −0.29 over t = 1 → 2500).
  Fix: enumerate *every* contraction in the chain, label each, **and state at
  what training time each label holds**. Guard: a heuristic derivation must
  declare whether it claims t=1 or t→large; if a claim spans both, it needs the
  DMFT track or a measurement. See `derivations/README.md` for the method
  taxonomy this failure motivates.
- **F19 — a derived coefficient with an unstated domain.** A derivation fixes a
  factor by counting *some* set of objects (here `d_L = L^{2 alpha - 1}`, fixed
  by accumulation over the `L` residual blocks), then writes it into a global
  rule (`theta_dot = -d_L gamma^2 grad L`) without saying **which parameters it
  applies to**. Every downstream implementation then resolves the ambiguity
  independently, and they need not agree. Instance: `derivations/05` §1 left the
  scope of `d_L` unstated; the *simulator* applied it to `W^0` and the *solver*
  applied it to the readout — opposite boundaries, same error. The first
  produced a fake falsification of the rule (slope `+1.55` vs a wanted `0`) and
  the second produced a fake solver-vs-sim gap (`1.92x`/`4.79x` over floor).
  Detection signature: a discrepancy whose size **grows with `L`** (or with
  whatever the coefficient counts) while the quantity it should govern is
  `L`-flat — the boundary path scales differently from the bulk path it was
  wrongly applied to. Fix: state the domain in the same sentence as the
  coefficient. Guard: for any factor fixed by counting `n` copies of a thing,
  **name the objects it multiplies**, and check every object that appears
  `O(1)` times rather than `n` times is excluded. Cross-check against the
  source's own table (CompleteP Table 1 gives Emb/Unemb no depth factor —
  that was on the page throughout and would have caught both).
- **F20 — the wrong floor for a common-random-number difference.** F8 says
  certify by sample-halving and quote the MC floor beside the gap. That recipe
  is for an **unpaired** comparison. When the two quantities being differenced
  share a random stream — an ablation solved twice from the same seed, ON vs
  OFF — the difference is a CRN estimate whose variance is far below either
  solve's own floor, because the common noise cancels. Quoting the individual
  MC floor then **over-rejects**: real signal is discarded as noise. Instance:
  the response-sector ablation in `derivations/05` §8d, where half the points
  read `0.3x`/`0.4x` of the individual floor and were rewritten as `5.6x`–`297x`
  of the correct one. Detection signature: an ablation whose measured effect is
  smooth and monotone in a control parameter, yet "inside the floor" — smooth
  monotone trends are not what noise looks like. Fix: floor the **difference**,
  i.e. recompute `d(S)` and `d(S/2)` at fixed seed and use `|d(S) - d(S/2)|`.
  Guard: whenever two runs share a seed, state which floor is being quoted.
  Note this cuts the opposite way from F8's usual conservatism, so it will not
  be caught by "was I strict enough?" — only by asking whether the runs were
  paired.
- **F21 — silent degenerate tie in a discrete selection.** An architecture with a
  hard `top_k` (routing, expert choice, any arg-selection) draws its selection
  from a score that is supposed to vary across the population. If *every* source
  of init variation in that score is zeroed, the scores are exactly equal and the
  selection collapses to whatever the kernel's tie-break is — `torch.topk`
  breaks ties by index, so the **same** units are chosen for every input, every
  layer, every step. The population is then degenerate and the mean-field average
  over it is averaging over one point, so an `E -> inf` limit is not the limit of
  that object. Instance: 2601.20205 has two conventions — the main text zeroes
  the biases and lets the router's `n^{-gamma}` noise carry diversity, App. E
  zeroes the router and lets random `b_k(0)` carry it. Taking `b = 0` *from one*
  and `r = 0` *from the other* silently destroys routing (measured cross-expert
  score spread `7.3e-13`). **Detection signature: the loss still goes down.**
  Nothing errors, gradients are finite, training curves look ordinary — the
  failure is only visible in the *selection statistics*, never in the objective.
  Fix: assert a nonzero spread of the selection score across the population at
  init, and refuse to construct the degenerate combination. Guard: when a paper
  offers two init conventions for the same mechanism, **identify which object
  carries the diversity in each** and never mix them; a "conveniently
  initialise at zero" remark is load-bearing on the rest of that convention.
  Corollary: an ablation that zeroes an init scale is not obviously benign — ask
  what population statistic it was the sole source of.
- **F17 — response-kernel write-order race in causal co-integration.**
  The response row Ā(t, s<t) is computable at time t and READ by the
  same-step field assembly; writing it after the read leaves the response
  sector silently OFF while spot-checks of Ā values pass. Diagnosis
  instrument: a theory-simulator pair with shared vs INDEPENDENT backward
  matrix (physical no-response control) isolates the base closure;
  lagged cross-moments ⟨z(t)w(s)⟩ flat in the solver but growing in the
  shared-matrix truth pinpoint the missing feedback. Guards: assert
  kernel rows nonzero at read time; an ablation that changes nothing is a
  red flag.
