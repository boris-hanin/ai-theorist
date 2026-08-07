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
  damping.
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
