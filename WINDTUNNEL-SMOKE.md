# The minimal end-to-end pipeline: proposal for round 012

Status: **PROPOSAL — no code below exists yet.** This document is the plan for
the first full-lifecycle smoke test: one command that pretrains a tiny model
from scratch, RL-post-trains it, and runs the inference-time scaling battery
on the result, at two widths, with the repo's certification discipline applied
end to end at reduced fidelity. It is the execution companion to
`WINDTUNNEL.md`: that file says *what* the full-lifecycle windtunnel should
eventually certify; this file says *what gets built first* and what a green
run does and does not mean. Per the repo's rules, the round's `prereg.md` —
not this file — is the preregistration; this file may be revised until the
prereg is committed, and after that corrections go in `results.md`.

This proposal was adversarially reviewed before being committed; the review
removed one wrong closed form, one control that could not bite under the
chosen optimizer, and two judged bars that failed by arithmetic. Where a
design choice below corrects `WINDTUNNEL.md`, the correction is named in
place.

## 0. Purpose and non-goals

**Purpose.** Establish, cheaply and honestly, that every piece of the
full-lifecycle windtunnel exists and composes: task suite with injectable
verifier error, tiny μP LM, pretraining sweep, GRPO loop, common-random-number
sampling pool, ρ(p) machinery, pass@k / vote / best-of-N predictors, floors,
controls, and mutation tests — all writing one artifact tree, at two widths,
from one command. Features and fidelity come later; what must be right *now*
is the plumbing and the statistics.

**Design property, stated up front:** every judged bar, control, and mutation
test below is one whose failure implicates the *implementation*, not nature.
The sole designed abort is M8 (a degenerate difficulty kernel), and it aborts
before the expensive legs run. Bars that would test statistics or phenomena
rather than code were demoted to report-only during review.

**What a green run certifies:** pipeline mechanics, artifact completeness,
estimator identities, floor discipline, and controls that bite.

**What it certifies about the science: nothing.** Transfer drifts are
*reported* with the full verdict machinery but preregistered as report-only
(§8): two widths cannot support an F22 settling verdict, horizons are tiny,
and round 010 says every transfer claim is horizon-limited until measured.
A green smoke run changes no existing certification row; it adds the pipeline
itself under a new SCAFFOLD state (§10, W4) defined in `PROGRAM.md` alongside
the existing three.

**Relation to `WINDTUNNEL.md` §IV sequencing.** Round 012 keeps §IV's number
but its scope expands: the smoke round *subsumes* item 1 (the vote/pass@k
windtunnel) — its Stage-3 leg is exactly that round's content — and threads
the pretrain and GRPO stages around it so the *interfaces* exist from day
one. Items 2–4 keep their existing numbers 013–015 unchanged. Similarly,
`WINDTUNNEL.md` §III's two-skill layout (`rl-post-training`,
`inference-scaling`) is deferred: v0 is one skill, `skills/windtunnel-e2e/`,
split later when the certification rounds need it. Dated pointer notes go
into `WINDTUNNEL.md` §III/§IV (annotated, not rewrites) so a reader of that
file finds this one.

## 1. The task: `modchain97`

Left-to-right arithmetic chain evaluation mod 97 with a step-by-step
scratchpad. Chosen over graph shortest-path (generator, serialization, and a
budget dial confounded with problem size would all have to be built),
sorting (ρ(p) collapses to a threshold in list length; vote degenerates over
permutation space), and non-modular multiplication (answer length grows with
difficulty — a built-in length-biased reward, the F23 pattern).

**Definition.** Evaluate `((a1 op1 a2) op2 a3) … op_k a_{k+1}) mod 97`,
strictly left-to-right, `a_i` uniform on 0..96, `op ∈ {+, −, *}`. The answer
is a residue emitted as exactly two digit tokens, zero-padded — fixed answer
length kills the length-bias confound, and a 97-way answer space keeps
majority vote and best-of-N informative (chance = 1%). The prime modulus
blocks digit-shortcut cheats that mod 100 would invite.

**Difficulty dial, honestly stated.** `k` = number of ops, 1..10; secondary
dial = fraction of `*` steps (multiplication has higher per-step error than
+/− at finite training). Success factorizes over steps,
`p ≈ Π_s (1 − ε(op_s, τ))`, so *within the pretraining support* (k ≤ 6, §
below) p at τ = 0.8 should sweep roughly [0.5, 0.98] at a fixed op mix, wider
across op mixes. The k ∈ 7..10 stratum is out-of-distribution by
construction; its p depends on length generalization, which tiny LMs can fail
abruptly — p ≈ 0 there is a possible outcome, not a failure. The endpoints
quoted in an earlier draft (0.98 down to 0.03) require confounding the op mix
with k and are not claimed. Consequently the non-degeneracy bar M8 (§8) must
be satisfiable from the k ≤ 6 strata alone, and the eval pool's per-k
multiplication fraction is on the dev-settable list (§8) so a too-easy or
too-hard mix can be repaired before the prereg freeze, never after.

**Format.** Vocabulary of 32 tokens: digits 0–9, ops `+ − *`, structural
`BOS EOS PAD = > ANS ERR @` (`OK CALC THINK` + 7 reserved-unused in v0).
Context 192.

```
Prompt:  BOS 1 7 + 4 2 * 3 - 8 8 =
Target:  > 1 7 + 4 2 = 5 9   > 5 9 * 3 = 8 0   > 8 0 - 8 8 = 8 9   ANS 8 9 EOS
```

(17+42 = 59; 59·3 = 177 ≡ 80; 80−88 = −8 ≡ 89 mod 97.) One `>` line per op —
so the thinking budget maps linearly onto solved chain depth, and the budget
dial is independent of the tokenized problem size (which shortest-path could
not offer). A direct-answer variant (`ANS 8 9 EOS`, no scratchpad) exists in
the corpus so that budget B = 0 is on-distribution.

**Answer parsing (the only metric).** One parser: the first `ANS` token after
the prompt; the following two tokens must both be digit tokens and are read
as the residue; anything else scores 0. The corpus emits `ANS d d` in 100% of
answer-bearing sequences — an earlier draft's 70/30 `ANS`/`=` format
ambiguity (to engineer a format-consolidation RL gradient, decomposed by a
second lenient parser) was cut in review as doing science, not plumbing: with
one format, GRPO gains are capability by construction. Format-vs-capability
decomposition is deferred to the round that studies reward channels.

**Verifier.** `verify_exact`: parse as above, compare residues — exact ground
truth. `verify_noisy(fp, fn)`: flips the exact verdict with the given rates,
seeded by `(task_id, completion_hash)` so injected error is a deterministic
function of (problem, answer) and therefore *paired across arms*. A
step-level verifier recomputes each `>` line and reports the index of the
first wrong one — this powers the corrupted-corpus construction and the
agentic stub.

**Pretraining corpus, with engineered RL headroom.** 300k sequences (~15M
tokens) from a seeded generator (self-contained; committed as generator code
+ seed). `k` restricted to 1..6 — **k ∈ 7..10 never appears in pretraining**
(held-out difficulty is the primary RL and inference headroom). Mixture: 80%
full correct scratchpads, 15% direct answers, 5% corrupted scratchpads
(exactly one intermediate line wrong, final answer still correct — an
imperfect procedure prior). Operands Zipf(1.2)-skewed in the corpus only,
leaving coverage gaps in the 97×97×3 op table. RL headroom mechanisms:
held-out k, the corruption prior, and the Zipf gaps.

**Problem-set inventory** (all deterministic in their seed tags; pairwise
disjoint by construction, enforced by the seed-tag schema):

| set | size | k-range | used by |
|---|---|---|---|
| pretraining corpus | 300k seqs | 1–6 | S1 training |
| pretrain val split | 2k seqs | 1–6 | S1 val loss (M1) |
| RL prompt set | 256 | 4–8 | S2 training |
| S3 eval pool | 250 (25/k) | 1–10 | M3, M8, all S-bars, Δρ |
| dev split | unbounded | any | W0–W3 development only |

## 2. The model

Decoder-only pre-LN transformer, causal mask, learned positional embeddings,
no weight tying, no dropout, fp32.

| preset | widths `D` | depth `L` | heads `H` | `D_h` | ctx | params |
|---|---|---|---|---|---|---|
| full | 64, 256 | 4 | 4 | 16, 64 | 192 | ~0.2M / ~3.2M |
| smoke (CI) | 16, 32 | 2 | 2 | 8, 16 | 96 | tiny |
| pico (pytest) | 16 | 2 | 2 | 8 | 96 | tiny |

**Parameterisation.** Repo convention: all weights stored `N(0,1)`, scale
carried by explicit forward prefactors. Hidden matmuls (`W_V`, `W_O`, MLP):
prefactor `1/sqrt(fan_in)`, AdamW lr `eta_0/sqrt(D)` — same as the
`dmft-graph` table's uncontested rows. Optimizer: **AdamW** throughout
(betas 0.9/0.95, grad-clip 1.0), with weight decay 0 in v0 — so the graph
table's `lambda = lambda_0 sqrt(D)` decay row is explicitly out of scope
(untested here, deferred with the β > 0 arm) and the optimizer is
functionally Adam. Real post-training is Adam-based, and a smoke test of an
SGD-only pipeline would smoke-test the wrong pipeline; the signGD-proxy debt
from round 011 is *not* inherited — this is real Adam, stated as such.

**Attention logits** scaled `q·k / D_h^{alpha_A}` with **`alpha_A = 1/2`**
(i.e. `1/sqrt(D_h)`). Honest basis: round 011 measured, *under SGD at
t ≤ 256 on graph attention*, that `alpha_A = 1/2` is the exponent giving
Θ(1) attention-pattern updates while `alpha_A = 1` (standard μP's `1/D_h`)
degenerated to uniform; but 011's own signGD leg points the other way at
`alpha_A = 1` (E8: the Q/K channel drifting toward coherence), and nothing
was measured under real Adam or on causal sequence attention. So the choice
is defensible but imported, and the W0 audit below measures the
attention-logit update channel under AdamW at the widths actually run,
rather than trusting the import.

**Three parameter rows are deliberately not copied from any table.**

1. *Embedding and unembedding.* The repo's graph-paper convention (encoder
   σ₀ = 1/√D, Adam lr η₀/√D) and the standard μP table (Yang & Hu, Tensor
   Programs V, Table 8 / the u-μP explicit-multiplier form: embedding
   multiplier 1, Adam lr Θ(1); unembedding multiplier 1/D, Adam lr Θ(1))
   genuinely disagree, are *not* abc-symmetry images of each other, and a
   token lookup is not a graph-feature encoder. Neither table's rows exist in
   `skills/dmft-derivation/references/parameterizations.md`, which has no
   embedding rows at all — W0 adds the derived rows there so the losing
   convention can be named in a file that exists.
2. *Q/K.* The graph table marks `sigma_QK` **contested and unmeasured**, with
   two live candidates at `alpha_A = 1/2` (`sqrt(D_h)` coherent vs
   `D_h^{1/4}` incoherent). Defaulting silently to `1/sqrt(fan_in)` would
   misquote the table this document cites.
3. The decision procedure for all three: the **W0 Step-0 scaling audit**
   (F23 discipline — enumerate every channel into the logit: embedding, every
   block, the attention-logit channel, unembedding) plus a **coordinate check
   with resolving power**: per-channel activation and per-step update RMS
   measured across `D ∈ {16, 64, 256}` and judged by fitted log-log slope,
   bar |slope| ≤ 0.1 (round-011 P1 style). A √D-wrong row shows as slope
   0.5 — unambiguous over three widths, where the earlier draft's
   two-width ratio band [1/3, 3] provably could not distinguish the
   candidate tables (a factor-2 effect inside a factor-3 band). The D = 256
   legs of this audit are step-0-plus-a-few-steps only — cheap, and listed in
   the prereg as exempt calibration measurements (§8).

## 3. Stage 1 — pretraining and the η mini-sweep

2000 steps, batch 24 × 192 tokens, next-token CE, linear warmup 100 then
cosine to 10%. **Val loss** = mean next-token CE over target positions only
(prompt and PAD masked); the M1 baseline is the CE of the empirical unigram
distribution of the corpus's target-position tokens, evaluated under the same
mask by the same eval function; both are logged in the artifact. Bar
(mechanism, M1): val loss ≤ 0.5× that baseline at both widths, all seeds;
plus greedy exact-match ≥ 0.9 at k ≤ 3 and ≥ 0.6 at k = 4..6 held-in. If
exact-match > 0.95 across all k ≤ 6, flag saturation risk (§10.1).

η₀ swept at D = 64 on a 7-point × 0.25-decade grid, **3 paired seeds**
(shared init draws and data order across grid points — CRN per F10/F20);
optimum by the existing quadratic refinement with edge detection. The D = 256
leg reruns a 5-point grid centered on the D = 64 optimum. Machinery: import
`transfer.verdict_from_optima` for the verdict; metric = fixed-horizon loss,
never best-during-training (F24); full loss-vs-η rows committed. The F24
**basin detector does not yet exist anywhere in the repo** (the registry
entry itself says a verdict function seeing only argmins cannot catch a
swapped basin), so it is a named new component: a routine in `pretrain.py`'s
sweep wrapper that counts resolved local minima in each seed-median
loss-vs-η row and downgrades the leg to SUSPECT when > 1. Drift in decades
is **reported, not judged** (§8 R1).

**Stage-1 deliverable (the S1→S2 interface):** per (width, seed), the
final-step checkpoint of a dedicated run at that width's refined η* (D = 256
uses its own 5-point grid's refined optimum), saved as
`rounds/012-e2e-smoke/ckpt/s1_D{D}_seed{s}.pt`. These 2 × 3 checkpoints are
the **base models**: S2 initializes from them and S3's base cells evaluate
exactly them.

## 4. Stage 2 — GRPO post-training

On-policy GRPO, deliberately minimal: G = 8 rollouts/prompt, 48 prompts/batch
from the 256-prompt RL set (k ∈ 4..8, straddling the p ∈ (0.2, 0.8) band so
group variance is nonzero), 150 updates, τ = 0.8, max 110 new tokens
(headroom above the k = 10 scratchpad, so truncation is not binding), single
gradient step per sampled batch. The token-level ratio clip 0.2 is therefore
**inert plumbing in v0** (on-policy single-step ⇒ ratio ≡ 1): it is carried
as interface and exercised only by the two-armed-bandit unit test with forced
off-policy ratios. Advantage `A_i = (r_i − mean)/(std + ε)` with degenerate
groups (all-equal rewards) yielding exactly `A = 0` and a logged
degenerate-group fraction — itself an effective-batch-size confound worth
recording. Reward: binary exact-match, nothing else; mean generated length
per reward bin logged every step (F23 watch). Fail-fast abort if > 80% of
groups have zero reward variance for 10 consecutive steps; the degraded path
this triggers is defined in §8.

KL to the frozen pretrained reference is **always measured** (both k1 and k3
estimators logged — their divergence is a preregistered watch-item) but
β = 0 in v0: the KL *budget* is recorded as the bridge unit against
best-of-N (§5), without adding a β tuning axis to the smoke.

**Reward-scale and the F12 mechanism, done for the optimizer actually used.**
An earlier draft imported `WINDTUNNEL.md` §I.1's prediction that reward ×16
with standardisation OFF shifts η*_RL by log₁₀16 ≈ 1.2 decades. That is SGD
algebra: Adam's update `m/(√v + ε)` is invariant to a global gradient
rescaling, so under AdamW *both* global-reward-scale legs are near-inert and
the control could never bite — it would have failed the round by
construction. Replacement (C3, §8): a preregistered **invariance identity
test** — one GRPO step with reward ×16, parameter update equal to the ×1
update within numerical tolerance (declared inert in advance, the F17
exemption) — plus the **biting demonstration at the level where the
mechanism is visible**: the exact two-armed-bandit GRPO ladder under plain
policy gradient in pytest, where the 1.2-decade shift is provable. What Adam
*does* see of the standardiser — the per-group 1/std reweighting across
prompts of different pass rates — is logged via the degenerate-group
fraction and left for round 014. No extra GRPO sweeps.

η_RL swept at D = 64 only (4 points, 3 seeds, rollout streams CRN-shared
across arms); D = 256 runs **once per seed at the transferred η_RL** — there
is deliberately no D = 256 RL sweep, so no η_RL* drift number exists and none
is reported (an earlier draft's R2 was cut as unmeasurable-without-a-sweep;
η_RL transfer is round 014's job). The plumbing test of the width interface
is that GRPO *runs and improves* at the transferred value (M3).

**Stage-2 deliverable (the S2→S3 interface):** the step-150 checkpoint —
never a best-during-training selection (F24) — per (width, seed), saved as
`ckpt/s2_D{D}_seed{s}.pt`. M3 (judged): paired Δpass@1 (post − pre) on the S3
eval pool (§5 — *the same pool*, so M3, Δρ, and every S-bar are functionals
of one object), ≥ +0.05 absolute and ≥ 5× the paired floor at both widths.

## 5. Stage 3 — inference-time scaling on frozen checkpoints

**The shared pool.** All pass@k / vote / BoN / Δρ observables are functionals
of one sample pool per (width, checkpoint, seed) cell — cells = {64, 256} ×
{base, post-RL} × 3 seeds, checkpoints from §§3–4. Pool: the 250-problem eval
set, n = 40 samples each at τ = 0.8, generation cap 110 tokens, sample j of
problem i seeded `seed_for('samp', task_id, j)` independent of batch
composition or arm — pass@k, vote@N, BoN@N are prefix-nested functionals of
the same pool and every cross-arm comparison is paired. The pool splits once,
in the prereg: **fit block = samples 1–8, held-out block = samples 9–40.**
Judged functionals are means over the 3 seeds, with seed spread reported.
Temperature pinned at 0.8 everywhere. The budget leg is *not* a pool
functional (truncation changes every subsequent token) and is a separate,
smaller generation job specified below.

**Scheduling and the abort gate.** The base D = 64 cells are sampled first,
before any S2 update and before any D = 256 sampling. M8 (judged) is
evaluated there: ≥ 20% of eval problems at p̂ = c/40 ∈ (0.1, 0.9),
**satisfiable from the k ≤ 6 strata alone** (150 problems), else the round
aborts — committing the partial artifact plus an ABORTED `results.md` and an
F-registry entry. This is the sole designed abort, and it fires before the
expensive legs.

**ρ(p) discipline** (settled math; the smoke wires it): primary artifacts are
the *unbiased functionals*, never a deconvolved density — from n draws
exactly the first n moments of ρ are identifiable (factorial-moment
identity), pass@k is a degree-k polynomial in p, hence unbiasedly estimable
for k ≤ n and for nothing beyond. The raw p̂ histogram is never reported as ρ
(it is ρ convolved with Binomial noise, piling fake mass at 0 and 1). A
spike-and-Beta empirical-Bayes fit (w₀δ₀ + w₁δ₁ + Beta(a,b) — the atoms are
real: verifiable tasks have p = 0 and p = 1 sectors) is fit on the fit block
for *prediction* and checked against the unbiased moments. **If the fit fails
its own moment check, S1's extrapolation legs return UNDER-POWERED
(fit-quality failure) — frozen now, in the prereg** — and the moment curve
stands in only for k ≤ 8.

**Floors, per functional** (this replaces a broken one-rule-for-everything
draft): sample-halving (F8, even/odd split of the relevant block) wherever
the functional is computable on the half-block — i.e. k, N ≤ 16 on the
held-out 32; **problem-level halving / paired problem bootstrap** (M = 250 →
125) for everything above that, stated per functional in the artifact.
Uncertainty on every judged gap = paired bootstrap over problems.

The judged science bars (all cells, none dropped):

- **pass@k extrapolation (S1).** Fit on the 8-sample block, predict pass@8
  and **pass@16**, verify on the held-out 32 samples with the unbiased
  estimator (stable product form `1 − Π_{j=n−c+1}^{n}(1 − k/j)`). Gap ≤ 3×
  floor. pass@32 sits at k = n where the estimator degenerates to the
  any-success indicator and no sample-halved floor exists — it is
  **report-only**, with a problem-bootstrap floor.
- **Plurality vote (S2).** vote@N = plurality over emitted 2-digit residues,
  ties broken uniformly at random (tie mass averaged analytically). Judged
  legs: (i) vote@5 predicted from the fit block by the exact U-statistic
  (unbiased: N ≤ n_fit) vs held-out measurement, within 3× floor; (ii)
  measured vote@N for N ∈ {1, 3, 5, 9, 17} (without-replacement U-statistics
  on the held-out block) never exceeds the **ceiling** — E[1{the correct
  answer is the strict mode of the answer distribution}] + tie terms — by
  more than 2× floor. An earlier draft's "frozen sector p < 1/2" ceiling is
  a *binary*-vote criterion and is wrong for a 97-way space (dispersed wrong
  mass is fixed by plurality voting even at p < 1/2); `WINDTUNNEL.md` §II.1's
  Binom(N, p) formula needs the same multiway generalization before it is
  wired into `kernels.py`. vote@17 *predicted* from 8-sample histograms is a
  with-replacement plug-in with a known optimism bias (the empirical mode's
  frequency is upward-biased) that floors cannot cover — it is
  **report-only**, predictor pinned as: per-problem plug-in multinomial over
  the 8 fit samples, exact plurality probability by enumeration over the ≤ 8
  distinct answers, no smoothing, expected bias direction stated.
- **Best-of-N under an imperfect verifier (S3).** N ∈ {1, 2, 4, 8, 16, 32}
  on the held-out block, prefix-nested; binary accept/reject verifier arms
  fp ∈ {0, 0.25} (fn = 0) scoring the *same* pool; selection = uniform among
  accepted, and **if none is accepted, uniform among all N** (the fallback
  rule is part of the prereg — the closed form depends on it). Closed form,
  per problem with accept probability `q = pT + (1−p)F`:
  `acc(N) = π(1 − (1−q)^N) + p(1−T)(1−q)^{N−1}`, `π = pT/q`, which with
  fn = 0 (T = 1) reduces to `acc(N) = π(1 − (1−q)^N)`; averaged over the
  per-problem joint (p, T, F), never over marginals. Anchors, wired as unit
  tests: acc(1) = pass@1 exactly; F = 0, T = 1 recovers pass@N exactly. (An
  earlier draft's formula failed both anchors — its fallback term assumed a
  fresh draw that a fixed pool cannot provide.) With a *binary* verifier the
  curve is monotone and saturates at the aggregated precision — it does
  **not** go non-monotone. **This corrects `WINDTUNNEL.md` §II.2**, whose
  "BoN degrades past a critical N" holds for heavy-tailed *scalar* scores
  (extreme-value classes: bounded ⇒ pass@N; exponential-scale ⇒ plateau in
  (p, 1); regularly-varying ⇒ rise-then-fall back to p), not for binary
  accept/reject; round 012's `results.md` records the correction per that
  file's own header rule, and the heavy-tail regime is exercised as a
  synthetic-pool unit test of the predictor. Bars: measured curves within 3×
  floor of the closed form; and control C4 — the fp = 0.25 curve sits below
  fp = 0 at N = 32 by ≥ 5× the paired floor (if injected FP does not visibly
  depress BoN, the harness cannot see verifier exploitation and the Stage-3
  battery is void).
- KL accounting (report-only R4): KL(BoN‖base) = log N − (N−1)/N logged as
  the budget *ceiling* — with massively tied binary scores the realized KL is
  strictly smaller and computable from the accept probability; both numbers
  recorded next to GRPO's measured KL-spent, the two ends of the bridge in
  the same units.
- Group-noise identity (demoted from a judged science bar to the identity
  layer): sd of the group-mean reward over nested groups of size
  G ∈ {2, 4, 8, 16}, *subsampled from the base-D64 pool* — zero new runs —
  log-log slope −0.5 ± 0.1, reported in the artifact beside the other
  identity checks.

**Thinking budget (M5, judged as mechanism only).** Two points, B ∈ {0, 64}:
after B generated tokens (wherever that falls, mid-line included), append
`ANS` to the context, sample exactly 2 tokens restricted to the 10 digit
tokens (renormalized softmax) at τ = 0.8, then force EOS; B = 0 forces `ANS`
immediately after the prompt. Separate generation job, n = 8 per problem per
B, seeded `seed_for('budget', task_id, B, j)`, run on all four (width,
checkpoint) cells per seed. Bar: paired Δpass@1(B=64 − B=0) ≥ 5× the paired
floor on the base D = 64 cells; same sign at D = 256; post-RL legs and any
budget *curve* are deferred (the 6-point grid and the sequential-vs-parallel
comparison were cut — they double Stage-3 sampling to power report-only
science).

**Agentic stub (M6, mechanism only — the science is cut).** One revise round:
propose → step-verifier feedback appended in-vocab (`ERR @ d d`, the
1-indexed first wrong line zero-padded, or `ERR ANS`) with round-2 context =
prompt + verified lines 1..s−1 + the feedback token span; generation resumes
to EOS or cap; halt early on `verify_exact` pass. Run on the base D = 64
cells, 1 episode per eval problem, seeded `seed_for('agent', task_id, r)`.
Bar M6: the loop runs to completion with full transcripts in the artifact and
the state advances on the model's own emissions, never teacher-forced.
**No effectiveness claim is possible and none is made:** the feedback grammar
never appears in the pretraining corpus, so the model has no reason to use
it — the geometric-null analysis, the equal-budget resample baseline, the
scrambled-feedback control, and the revision-transcript corpus slice that
would make feedback on-distribution are all deferred together to the agentic
round. The stub exists so the lifecycle's agentic *interface* (transcript
schema, environment step, seed tags) is exercised end to end.

## 6. The coupling observable (report-only, and committed as such)

Δρ = (post-RL minus base) difficulty-kernel functionals on the shared pool,
per width: the unbiased moment-curve difference, the p = 0 / p = 1 atom
shifts, and the signed pair (Δpass@1, Δpass@16) with paired floors — the
RLVR sharpening-vs-diversity tension of `WINDTUNNEL.md` §II.4 made
measurable. **No bar**, preregistered as bar-less so it cannot be promoted to
a claim post hoc; any diversity-collapse narrative goes in `results.md` as
observation.

## 7. Statistics and replicate topology, in one place

- **Replicates: 3, everywhere, keyed by pretrain seed.** One RL run per
  pretrain seed; the four-cell Stage-3 battery runs per seed (its pools are
  cheap next to training); judged functionals are seed means with seed spread
  reported.
- Every comparison that can be paired is paired: shared problem sets
  everywhere; shared init + data order across η grid points at fixed seed;
  shared rollout streams across η_RL arms; one pool across all Stage-3 arms;
  verifier noise seeded by (problem, completion). The independent axes are
  the 3 seeds and the half-splits used for floors.
- Honest pairing caveats, recorded in `SKILL.md`: a D = 64 and a D = 256 net
  cannot share a bitwise init stream, so cross-width comparisons pair by
  seed *label* and use seed spread, not CRN cancellation; post-RL token
  streams diverge from base at the first differing token, so pairing there
  holds at the (problem, sample-index) level — which is what the estimators
  average over.
- Uncertainty: per-problem differences, paired bootstrap over problems for
  every judged gap; discordant counts (n10, n01) logged for every 0/1 sign
  claim — they are the entire evidence at k = 1 (McNemar).
- Floors: per-functional mechanism as in §5; a bar whose floor exceeds it is
  UNDER-POWERED, neither pass nor fail.

## 8. Round 012 preregistration plan

Bar types, in the round-011 scoreboard style. **Round verdict = the judged
rows.** Per-control consequences are explicit: a must-bite control attached
to a judged bar (C2, C4) that fails to bite **fails the round** (F17 as
policy); the SP smoke arm (C1) and controls on report-only rows can only
mark their row UNDER-POWERED.

**Judged, mechanism:**

| id | claim | bar |
|---|---|---|
| M1 | pretraining works at both widths, artifacts complete | val CE ≤ 0.5× unigram baseline (§3 definition), all seeds; artifact schema complete |
| M2 | η sweep is F24-clean | interior optimum every cell; full rows committed; basin detector (§3) finds one basin, else SUSPECT |
| M3 | GRPO improves pass@1 | paired Δpass@1 ≥ +0.05 and ≥ 5× paired floor, both widths, on the S3 pool |
| M4 | every judged functional carries its floor | per-functional floor mechanism as declared in §5; unstable ⇒ that bar UNDER-POWERED |
| M5 | budget dial is alive | Δpass@1(B=64 vs 0) ≥ 5× paired floor at D=64 base; same sign at D=256 |
| M6 | agentic stub runs | transcripts complete; state advances on model emissions |
| M7 | smoke preset covers the pipeline | CI: all stages, both smoke widths, the named control + mutation (below), ≤ 5 min CPU |
| M8 | ρ(p) is non-degenerate | ≥ 20% of eval problems at p̂ ∈ (0.1, 0.9) on the base D=64 cell, satisfiable from k ≤ 6 alone; else abort (§5 path) |

**Judged, science (pure statistics — no scale claim):** S1 pass@8/pass@16
extrapolation, S2 vote@5 prediction + ceiling bound, S3 BoN closed form +
C4 — all as pinned in §5.

**Report-only (verdict machinery run where applicable, no round pass/fail):**
R1 pretrain η* drift across width (with the C1 caveat below); R3 coupling Δρ;
R4 KL bridge accounting; pass@32; vote@17 prediction; the group-noise slope.
(The former R2 — η_RL* drift — was cut: no D = 256 RL sweep exists, so no
drift number exists.)

**Controls:** C1 — SP parameterisation (global η, 1/√fan_in prefactors, no
width-scaled lr) run **at smoke widths only** as a plumbing check that the
harness ingests a second parameterisation; the full-preset SP sweep is
deferred to round 014 with the transfer claims it would power. C2 —
reward-sign flip (one GRPO run, D = 64, seed 0, CRN-paired): paired Δpass@1
≤ −5× floor; must bite. C3 — AdamW reward-scale invariance identity (§4):
one GRPO step at reward ×16 equals the ×1 update within tolerance, declared
inert in advance; the biting F12 demonstration lives in the bandit pytest.
C4 — injected verifier FP (§5); must bite.

**Mutation tests.** Split by where the signal lives — a live-run mutation
must have its signal guaranteed by construction, otherwise the round could
fail because nature declined to produce an artifact, which is the opposite
of what mutation testing certifies. *Live-run (each a flagged config, run
last, from the same commit):* naive plug-in pass@k must fail the identity
test; desynchronised CRN seeds must be detected by the paired-floor routine
(variance-ratio assertion) and refused; n = 4 ρ estimation must come back
UNDER-POWERED, not passed; τ = 0.05 must trip the degenerate-kernel
assertion. *Detector-level (pytest, synthetic inputs):* the F24 basin
detector must flag a planted second basin in a synthetic loss-vs-η row; the
reward-channel audit must flag a planted gold-vs-proxy divergence with
length drift; the BoN predictor must reproduce heavy-tail non-monotonicity
on synthetic pools. A missed mutation fails the round regardless of green
bars.

**Development runs, the prereg freeze, and what "pilot" means here.** The
prereg freezes bars, floors discipline, and the prediction list, plus a
*closed* dev-settable list: {η grid centers, RL prompt-mix k-range, GRPO
step count, the eval pool's per-k multiplication fraction, N/k grids}. The
values printed in §§3–5 are defaults; dev runs during W0–W3 may replace
items on that list and nothing else. Discipline replacing an earlier draft's
separate pilot apparatus: **dev runs are the pilot.** They are restricted to
the smoke/pico presets and D = 64 on the dev split; the only D = 256 work
before the prereg commit is the W0 calibration audit (§2), listed as exempt
in the prereg. No full-preset D = 256 training, no eval-pool sampling, and
no GRPO at full preset happens before the prereg commit — enforced by the
seed-tag schema (dev tags are distinguishable from preregistered tags in
every artifact), and the prereg records which defaults dev runs changed. Dev
data may never move a bar, a floor multiplier, or delete a prediction. If
any full run slips earlier, the round-007 honesty note is the fallback: say
so in `results.md`, never backdate.

**Degraded paths, defined now:** if S2 aborts (zero-variance monitor or
crash), S3 still runs the base cells; M3, M5's D=256 sign leg if unreached,
Δρ, and R4's RL side are reported N/A-S2-ABORT; the round verdict is FAILED
(M3 unmet) with the abort cause as the headline, per the round-011
failed-round template. M8's abort path is in §5.

**Not claimed (the list a green run must be read against):** no μP transfer
certification (two widths, tiny horizons); nothing about η_RL, β, G, or T
beyond the swept points; no reward-overoptimisation onset (C4 validates the
harness's ability to *see* verifier exploitation, not the phenomenon's
scaling); the coupling observable stays an observation; no DMFT content, no
theory-of-dynamics, hence no floors for comparisons that do not exist; one
synthetic task family — no number here compares to any benchmark; injected
verifier errors are iid coin flips — nothing about structured
learned-verifier errors; the agentic stub demonstrates an interface, not
agency; α_A = 1/2 and the audited parameter rows are pipeline choices
measured for sanity at three widths, not certified parameterisation claims.

## 9. Software plan

**Layout** — one new skill, matching existing idioms (plain scripts +
argparse, numpy/torch only, no new dependencies). LOC figures are estimates
after review, not aspirations:

```
skills/windtunnel-e2e/
  SKILL.md                    YAML frontmatter (name, trigger description) +
                              '> **Status: SCAFFOLD — ...**' blockquote per
                              repo convention; maps scripts to WINDTUNNEL.md
                              sections; names the audited parameter rows and
                              the losing tables; lists v0 cuts explicitly
  scripts/
    tasks.py       ~150 LOC   modchain97 generator, splits (§1 inventory),
                              corpus, verify_exact / verify_noisy / step
                              verifier, the single answer parser
    model.py       ~150 LOC   TinyLM, μP prefactors + param_groups(eta0),
                              coord_check (3-width log-log slope version)
    sampler.py     ~110 LOC   seed_for(*tags) via blake2b; CRN batched
                              sampling with per-(task,j) generators;
                              logprob_of; greedy
    pretrain.py    ~150 LOC   Stage 1 + lr_sweep + the F24 basin detector
                              (new component, §3); imports transfer.py for
                              verdicts
    grpo.py        ~160 LOC   Stage 2: rollout groups, update (clip carried
                              but inert in v0), degenerate-group guard,
                              k1/k3 KL, C3 invariance check; exact 2-armed
                              bandit update for tests
    kernels.py     ~250 LOC   pure numpy: unbiased pass@k, plurality vote
                              U-statistics + ceiling, BoN measured/predicted
                              (corrected closed form + heavy-tail synthetic
                              classes), spike-and-Beta fit with moment
                              check, per-functional floors
    agentic.py     ~60 LOC    the one-round revise stub + transcript schema
    e2e.py         ~350 LOC   THE driver; Suite class (overnight_suite_v2
                              idiom), stages S1 S2 S3 as positional args,
                              --smoke/--pico/--device/--output, incremental
                              atomic jsonio saves, per-stage try/except,
                              git-dirty assertion before non-smoke runs,
                              and the executable both-widths guard: refuse
                              to write a non-smoke artifact under
                              rounds/012-e2e-smoke/ unless every stage
                              block carries both width cells
rounds/012-e2e-smoke/         prereg.md (before any preregistered run),
                              ckpt/ (torch state_dicts + config + seed tags;
                              artifact records each cell's path + SHA256;
                              stage-subset invocations resolve inputs from
                              these and refuse on hash mismatch),
                              raw JSON, results.md
tests/test_windtunnel_e2e.py  the battery below (~250 LOC)
tests/conftest.py             add the new scripts dir to the path tuple
```

**Controls execution map:** C1 = `e2e.py S1 --param sp --smoke`; C2 =
`e2e.py S2 --flip-reward` (D = 64, seed 0); C3 = a flagged extra step inside
the S2 block; C4 = a scoring pass over the existing pool (no new
generation). Each lands in its stage's artifact block under `controls`.

**Smoke preset, fully pinned (these named dials are the permitted
config-diff deltas, asserted in CI):** widths {16, 32}, L = 2, H = 2,
ctx 96, corpus 5k sequences, pretrain 150 steps, η grid 3 points, 1 seed,
GRPO 20 updates, G = 4, 16 prompts with **k ∈ {1, 2}** (a barely-trained
model has nonzero group variance there; the zero-variance abort threshold is
also on the permitted-delta list and is exercised mechanically in pytest
with injected constant rewards), pool M = 30 (3/k) × n = 8, N ≤ 8,
B ∈ {0, 16}, agentic stub 8 episodes. Smoke asserts **plumbing invariants
only**: artifacts complete, both widths present, generation truncates at B
and the forced-ANS path executes (completion lengths differ across B; B = 0
emits no `>` tokens), nonzero reward attained on k = 1, and the two named
CI checks — the desynchronised-CRN mutation is caught, and C4 bites on a
synthetic pool. Never an accuracy assertion (an earlier draft's
"non-constant budget curve" smoke assert was accuracy-dependent by accident
and is folded into the full run's M5 instead).

**CLI.**

```bash
python skills/windtunnel-e2e/scripts/e2e.py --device cuda \
    --output rounds/012-e2e-smoke/e2e.json      # full: S1→S2→S3, D∈{64,256}
python skills/windtunnel-e2e/scripts/e2e.py --smoke --device cpu \
    --output /tmp/e2e.json                      # CI: ≤3 min target
python skills/windtunnel-e2e/scripts/e2e.py S1 S3 --device cpu ...  # subsets
```

**Artifact.** One JSON tree per run via `jsonio.dump` (atomic, allow_nan
false, nulls for non-finite): schema_version, preset, device, git SHA, the
full seed-tag schema, checkpoint paths + SHA256s, per-stage blocks (S1 loss
curves + per-seed optima + basin-detector output + coord-check slopes; S2
reward/KL/entropy/degenerate-fraction curves + C2/C3 blocks; S3 per-cell
moment curves, pass@k measured/predicted/floor + floor mechanism, vote, BoN
curves per fp arm, budget pairs, agentic transcripts), delta_rho, and
`S<k>_error` fields so a dead stage preserves partial work. Every per-seed
list keeps its seed index so reanalysis can re-pair (the round-011 lesson).

**CI.** One new step after the existing smoke tests, 3 min target / 5 min
cap; the pytest battery gains the tests below (pico run ~40 s included).

**Unit tests** (identity and guard layer; each detector-level mutation from
§8 anchors here): verifier ground truth + single-digit perturbation
rejection; noisy-verifier calibration + determinism (pairing); pass@k
unbiased-estimator identity on known discrete ρ; plurality-vote U-statistic
+ ceiling + tie handling on constructed histograms; BoN closed-form anchors
(acc(1) = pass@1; F=0 ⇒ pass@N) + binary plateau + heavy-tail non-monotone
synthetic classes; KL ceiling identity on continuous scores; floor halving
scales as 1/√2; sampler seed-pairing invariance (bitwise, across batch
compositions) and prefix nesting; GRPO advantage normalisation (degenerate
group → exactly 0, no NaN); GRPO matches the exact 2-armed-bandit update at
G = 4 over 20k groups, and the bandit F12 demonstration (reward-scale shift
under plain PG, invariance under Adam); zero-variance monitor trips on
injected constant rewards; basin detector flags a planted second basin;
reward-channel audit flags planted divergence; μP coordinate check (3-width
slope); pretrain CRN bitwise reproducibility; pico artifact completeness
(strict JSON, all blocks, both-widths guard fires on a mutilated artifact).

## 10. Milestones, compute, risks

**Milestones** (W-prefix — M-numbers are reserved for the bars in §8; walking
skeleton first, and everything before W4's prereg commit runs only under the
§8 dev-run restrictions):

- **W0** tasks + model + pretrain + e2e(S1) running at smoke widths + D=64
  dev; --smoke/--pico; CI green; **the Step-0 scaling audit + 3-width
  coordinate check that pins the embedding, unembedding, and Q/K rows**
  (§2 — the D=256 legs are step-0-scale calibration, exempt-listed in the
  prereg); derived rows added to `references/parameterizations.md`.
  ~1 session.
- **W1** sampler + kernels; S3 runs on a dev base checkpoint (S2 skipped via
  stage subset); identity tests green. Lifecycle shape 1→3 exists before RL
  does. ~1 session.
- **W2** grpo.py wired as S2 on dev; Δρ in the artifact (the stage interface
  measured); bandit-exactness + C3 invariance tests green. ~1–2 sessions.
- **W3** hardening: C1–C4, mutation tests (live + detector-level), agentic
  stub, smoke timed under CI budget. ~1 session.
- **W4** round 012 proper: prereg.md committed (bars frozen, dev-set values
  recorded) → full preset run → results.md scoreboard → F25+ entries **for
  anything that actually bit (possibly none)**, each with an executable
  guard in scripts/ → PROGRAM.md: define SCAFFOLD in the certification-state
  list ("pipeline mechanics smoke-tested end to end with mutation-tested
  guards; certifies no scientific claim; upgrade path is PARTIAL via a
  certification round") and add the row (Files: `skills/windtunnel-e2e/`,
  `rounds/012`) → README touch-list (round count, skills listing, the
  already-stale F1–F22 range) → dated pointer notes in `WINDTUNNEL.md`
  §III/§IV. ~1 session + one compute run.

**Compute, per component (full preset, one small GPU; generation is
overhead-bound at these sizes so figures are rough):**

| component | runs | est. |
|---|---|---|
| S1 η sweep D=64 (7 pts × 3 seeds) + η* runs | 24 | ~1–1.5 h |
| S1 η sweep D=256 (5 pts × 3 seeds) + η* runs | 18 | ~1.5–2.5 h |
| S2 η_RL sweep D=64 (4 pts × 3 seeds) + C2 + C3 | 14 | ~1–2 h |
| S2 D=256 at transferred η_RL (3 seeds) | 3 | ~0.5–1.5 h |
| S3 pools (4 cells × 3 seeds) + budget + stub + C4 | — | ~1–1.5 h |
| live mutation configs (n=4, τ=0.05, CRN-desync, plug-in) | — | ~0.5 h |

Total ≈ **5–9 GPU-hours — one overnight run** (round-010 posture), not the
earlier draft's 45–75 min, which priced only the primary arm. CPU-only is
viable for everything at D = 64 (~2–3 h) but not for the full preset; the
both-widths guard in `e2e.py` makes the D=64-only shortcut an executable
refusal, not prose. Smoke: 2–3 min on 2 CPU cores. Pico: < 30 s.

**Top risks and their guards** (each preregistered): (1) pretraining
saturates → all-zero-variance GRPO groups → silent no-op RL: prompt mix
pinned to k ∈ {4..8} + fail-fast monitor + defined degraded path. (2) ρ
collapse: M8 abort gate, early, k ≤ 6-satisfiable, with the op-mix repair
dial on the dev-settable list. (3) budget dial dead: 15% direct-answer
corpus share; M5 is judged on the full run only. (4) OOD cliff at k ≥ 7:
allowed for — no bar depends on the held-out stratum having p > 0. (5)
advantage standardisation as hidden lr (F12): C3 identity + bandit
demonstration, honestly scoped to what Adam can see. (6) vote trivial or
frozen: M8 + residue-entropy logging. (7) smoke-mode false confidence: smoke
asserts plumbing invariants only, never accuracy. (8) length bias
re-entering (F23): per-bin length logging + fixed 2-token answers. (9)
convention bug in the audited rows surviving to D = 256: the 3-width slope
check exists precisely because the 2-width band provably lacked the power.

## 11. Decisions taken here (flag any to reverse)

1. **Task = modchain97**, not graph/sorting/multiplication (§1 rationale).
2. **AdamW, not SGD or signGD-proxy** — smoke the pipeline that will be
   used; decay 0, so the λ-scaling row is out of scope.
3. **alpha_A = 1/2**, with the honest scoping in §2 and the W0 update-channel
   audit under Adam rather than an imported SGD verdict.
4. **Round numbering:** 012 keeps its WINDTUNNEL §IV number; scope expands to
   subsume item 1; 013–015 unchanged.
5. **One skill dir** (`windtunnel-e2e`); the §III two-skill split deferred.
6. **β = 0 with KL measured** (k1 + k3); β tuning deferred.
7. **τ = 0.8 pinned everywhere**; temperature axis deferred.
8. **Widths 64/256 full, 16/32 smoke**; depth pinned L = 4 (no depth claims).
9. **Embedding, unembedding, and Q/K μP rows decided by the W0 audit**
   (3-width slope check), not copied — the available tables disagree and one
   row is marked contested by the repo itself.
10. **Single answer format, single parser** — the format-ambiguity /
    lenient-parser apparatus was cut as science-in-a-smoke.
11. **Transfer legs report-only**; judged bars are mechanism +
    pure-statistics only; per-control failure semantics stated in §8.
12. **Agentic = stub** (interface only); its science deferred whole.

## 12. Deferred (v1+, in rough order of value)

β > 0 arm and (η, β) joint transfer; the D = 256 η_RL sweep and RL transfer
proper (round 014, with the full-preset SP control arm); horizon axis T
(P-RL3); temperature axis; the bandit/GRPO exact ladder as its own round
(013); the agentic round (revision-transcript corpus slice, geometric null,
equal-budget baseline, scrambled-feedback control, error autocorrelation);
the budget curve and sequential-vs-parallel optimum (R5 material); EVT tail
fits and scalar-score verifiers on model outputs; format-vs-capability
reward decomposition; off-policy k > 1 / replay (un-inerting the clip);
reward-model (learned-verifier) overoptimisation onset (P-RL4, round 015);
second task family (graph) to test task-generality of the ρ machinery; real
pretrained checkpoints (model-ladder rung above); splitting the skill per
WINDTUNNEL §III.
