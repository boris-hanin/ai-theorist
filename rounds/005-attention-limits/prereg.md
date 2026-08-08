# Pre-registration — round 005: multi-head attention limits (arXiv 2405.15712)

**Committed before running anything.** Rounds 001–004 were retrospective; this
is the first round to actually follow `rounds/TEMPLATE/prereg.md`.

Date: 2026-08-08
Skill exercised: `dmft-attention` (RECONSTRUCTED — this round is its
re-validation), plus the Leg-C harness from Phase 3.
Target: Bordelon, Chaudhry & Pehlevan, *Infinite Limits of Multi-head
Transformer Dynamics*, NeurIPS 2024 (arXiv:2405.15712v2).

## 0. Source handling (F14)

The paper text was read directly from the PDF, not from a summariser. This
matters: **two web-fetch summaries of this paper were confabulated.** One
reported the attention logit scaling as `1/sqrt(d_k)` (the paper requires
`1/N`, i.e. `alpha_A = 1`); another reported that the paper contains no
experiments (it contains Figures 2–5). Neither was used. All claims below are
quoted or paraphrased from the extracted PDF text.

Notation, from §2.1: `N` = key/query dimension per head, `H` = heads, `L` =
depth, `d_model = N*H`. Pre-attention `A^l_{h,s,s'} = N^{-alpha_A} k·q` with
`alpha_A ∈ [1/2, 1]`; residual branches scaled `beta_0 L^{-alpha_L}` with
`alpha_L ∈ [1/2, 1]`; readout `f = (gamma_0 N H)^{-1} w^L · mean_s h^L_s`.
SGD learning rate (Table 1): `eta = eta_0 * N * H * L^{2 alpha_L - 1}`, with a
first/last-layer rescale `L^{1/2 - alpha_L}`.

## 1. Scope verdict (Phase 0)

**In scope:** the paper's *scaling and structural* claims, tested empirically —
finite `N, H, L` ladders with fixed batch, fixed steps, taking one dial to
large values at a time. This is Leg A (vs prior theory) and Leg C (transfer).

**Explicitly OUT of scope this round:** the transformer DMFT solver itself
(the paper's Appendix E). The existing solvers in this repo are P=1 scalar MLP
solvers; a transformer DMFT needs per-head single-site processes, softmax
order parameters and layer-norm, and building it badly would be worse than not
building it. **So this round cannot claim to replicate the DMFT analysis** —
only the scaling results that analysis predicts. Leg B is not attempted.

**Known deviation:** the paper uses vision transformers on CIFAR-5M. This
round uses a synthetic sequence task. The claims under test are architectural
scaling laws that should not depend on the dataset — but if a prediction fails,
**dataset difference is a live candidate explanation** and must be reported as
such rather than as a refutation.

## 2. Predictions

| # | Observable | Prediction | Refuted if |
|---|---|---|---|
| P1 | Var over heads of `A^l_h` at fixed `(s,s')`, after training, vs `N`, at `alpha_A = 1` | slope `d log Var / d log N` ≈ **−2** (Fig 2b) | slope outside [−2.6, −1.4] |
| P2 | Same, at `alpha_A = 1/2` | slope ≈ **0** — variance does *not* decay (Fig 2b) | slope below −0.5 |
| P3 | Var over heads of `A^l_h` **at initialisation**, vs `N` | slope ≈ **1 − 2·alpha_A**, i.e. −1 at `alpha_A=1`, 0 at `alpha_A=1/2` (§3.3) | slope off predicted value by >0.6 |
| P4 | Optimal `eta_0` vs `N`, at `alpha_A = 1` | **transfers** (drift < 0.3 dec) | drift > 0.3 dec and statistically resolved |
| P5 | Optimal `eta_0` vs `N`, at `alpha_A = 1/2` | **also transfers** — Fig 2(a) states both exponents transfer similarly over finite `N` | drift > 0.3 dec and resolved |
| P6 | Residual-stream kernel `H^l` vs its large-`H` proxy, squared error vs `H` | slope ≈ **−1** (`O(H^{-1})`, Fig 3) | slope outside [−1.6, −0.4] |

**P4/P5 together are the interesting pair.** The paper's own position is that
hyperparameter transfer does **not** discriminate `alpha_A` at finite `N` — both
values transfer — and that the requirement `alpha_A = 1` comes instead from the
`N → ∞` limit existing at all (backward-pass stability over ≥2 SGD steps).
So a transfer sweep is the WRONG instrument for this question, and P2/P3 (the
variance scaling) is the discriminating observable. This is F3's rule applied
in advance: name the mechanism and the observable that separates it, rather
than reaching for the sweep that happens to be built already.

## 3. Tolerances and floors

- **Slopes** (P1–P3, P6) are least-squares fits of `log observable` on
  `log dial`, over at least four dial values. Reported with the across-seed
  standard error of the slope; a slope counts as measured only if its SE is
  below 0.25.
- **Transfer** (P4, P5) uses the existing verdict rule: optimum located per
  seed, drift counted only if it exceeds `2*sqrt(2)` times the pooled
  across-seed scatter AND exceeds 0.3 decades. UNDER-POWERED is a distinct
  outcome from a pass.
- **Init variance** (P3) is a pure initialisation statistic, so its floor is
  seed scatter alone; ≥8 seeds.
- No result is reported without its floor beside it (F8).

## 4. Ground truth

1. The paper's stated exponents (−2, 0, `1−2 alpha_A`, −1) — prior theory.
2. Independent re-derivation where cheap: `Var(A) = Theta(N^{1-2 alpha_A})` at
   init follows from `A = N^{-alpha_A} k·q` with `k, q` having `Theta(1)`
   entries in `N` dimensions, so `k·q = Theta(sqrt(N))` and
   `Var(A) = Theta(N^{1-2 alpha_A})`. This is checked by hand before measuring.
3. There is **no simulation-vs-theory leg** here, because no solver is built.

## 5. Ablations that must bite

- Setting `alpha_A = 1/2` must change P1's slope. If P1 and P2 give the same
  slope, the measurement is not sensitive to `alpha_A` and the round is
  under-powered — report that, not a refutation.
- Head-shuffling control: permuting head indices must leave across-head
  variance unchanged (it is a statistic over heads). A change means the
  estimator is picking up head *ordering*, which would be a bug.

## 6. Registry entries in scope

- **F3** (concentration vs freezing) — directly relevant. P1/P2 distinguish
  "heads collapse because the population concentrates" from "individual
  attention entries freeze". The program's own record says an earlier
  attention round pre-registered `ΔA = Θ(N^{-1/2})` and had it falsified. The
  paper's `N^{-2}` variance (i.e. `ΔA ~ N^{-1}`) is consistent with that
  correction, and P1 re-tests it against a source.
- **F8** — floors beside every number.
- **F10** — seed-average before comparing.
- **F17** — the `alpha_A` ablation must bite.

## 7. What would make this round a failure

- P1 and P2 giving indistinguishable slopes ⇒ the estimator cannot see
  `alpha_A`; round is under-powered, `dmft-attention` stays RECONSTRUCTED.
- P3 disagreeing with the hand derivation in §4.2 ⇒ the implementation of the
  parameterisation is wrong, and nothing downstream can be trusted.
- Any prediction refuted ⇒ report it as refuted, name the dataset deviation as
  a candidate cause, and do **not** quietly rescope. The skill is certified
  only if P1, P2, P3 all hold and at least one of P4/P5 is decidable.
