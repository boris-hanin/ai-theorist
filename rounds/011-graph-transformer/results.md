# Round 011 — results

Prereg: `prereg.md`, committed before any measurement (`9435a39`).
Derivations: `derivations/10-graph-transformer.md` (heuristic),
`derivations/11-graph-transformer-dmft.md` (cavity).
Model: `skills/dmft-graph/scripts/gt.py`. Runner: `.../experiments.py`.
Raw: `E1..E12-*.json` in this directory, with the console logs beside them.

## Overall verdict: FAILED

The preregistration makes any P1–P7 failure or any control that does not bite a
round failure. P3 was not established on all six legs; P3c2 and P3c3 did not
bite; P4, P4b, and P7 missed their bars. Later E8–E12 follow-ups sharpen the
mechanism and remove one signGD estimator artefact, but they do not retroactively
change that preregistered verdict. The static/scaling subclaims below remain
useful evidence; the full graph-transformer parameterisation is not certified.

## Headline

Three things happened, and only the first is a confirmation.

1. **The static sector is confirmed, sharply.** Feature and branch scales are
   flat in `D` and `H` and go as `1/L`; the corrected E9 probe verifies the
   `gamma` formula (G) as an algebraic identity on all six operators;
   `A_init ~ D_h^{1/2-alpha_A}` is right to `0.01`; the
   DMFT's LLN-vs-CLT prediction (S1/S2) is right to `0.03`; and the SGD width
   exponent is confirmed to `+1.041` against a predicted `+1` by a control that
   moves the optimum `0.94` decades.
2. **My attention derivation was incomplete, and the measurement found the
   missing piece.** `Delta A` has a channel `10` §3 never enumerated — the block's
   *input* moves, which moves the logits with no reference to the Q/K learning
   rate. It scales as `D_h^{1/2-alpha_A}`, confirmed on four configurations. P4
   at `alpha_A = 1` "passed" **for the wrong reason**: at that exponent both
   channels give `D_h^{-1/2}` and the total cannot distinguish them.
3. **The headline recommendation of `10` §5 is reversed by measurement.** The
   Q/K coherence flip I asserted (contraction C2 becoming coherent after one
   step) **does not happen within 256 SGD steps**. Consequently, at
   `alpha_A = 1` *every* channel of `Delta A` vanishes as `D_h^{-1/2}`, the
   attention matrix converges to uniform-over-neighbourhood at all times, and
   the MHSA branch degenerates into a second mean-aggregation MPNN branch with a
   learned value/output projection. `alpha_A = 1/2` is the only tested exponent
   at which the attention pattern receives `Theta(1)` updates.

And one thing that did **not** happen: **the learning-rate transfer sweeps did
not discriminate the attention parameterisation.** Two of four preregistered
controls failed to bite, one of them anti-bit, and the whole Adam-side leg is
unresolved. §4 is the honest accounting.

## Verdicts against the preregistered bars

| # | prediction | bar | measured | verdict |
|---|---|---|---|---|
| P1 | branch/feature RMS flat in `D` | ±0.10 | stream `+0.004`, MPNN `+0.004`, attn `−0.039`, MLP `+0.003` | **PASS** |
| P1 | flat in `H` | ±0.10 | `+0.001`, `0.000`, `+0.016`, `+0.001` | **PASS** |
| P1b | branch `~ 1/L` | `−1 ± 0.15` | MPNN `−1.065`, attn `−1.111`, MLP `−1.063` | **PASS** |
| P2 | `gamma_A` flat in `D` at `alpha_A = 1/2, 1` | \|slope\| ≤ 0.05 | `−0.010`, `−0.019` | **PASS** |
| P2c | control `alpha_A = 0`: `d_eff` slope ≤ −0.15, `gamma_A` slope ≥ +0.05 | — | `−0.230`, `+0.056` | **PASS — control bites** |
| P3 | `eta_0` transfers, SGD, `D` / `L` / `H` | TRANSFERS | **SUSPECT** (0.083, not settling) / TRANSFERS (0.103) / TRANSFERS (0.035) | **PARTIAL** |
| P3 | `eta_0` transfers, signGD, `D` / `L` / `H` | TRANSFERS | final-loss follow-up: UNDER-POWERED / TRANSFERS (0.390) / TRANSFERS (0.547) | **NOT ESTABLISHED; P3 FAILS AS A SIX-LEG CLAIM** |
| P3c1 | control `alpha_A = 0` breaks width transfer | FAILS/SUSPECT | SUSPECT, drift **0.065** — *less* than the treatment's 0.083 | **PASS by the literal bar, weak control** |
| P3c2 | control drift(qk-global) ≥ 2× drift(derived) | ≥ 2× | 0.122 vs 0.283 → **0.43×** | **FAIL — control anti-bites** |
| P3c3 | control `P = A` breaks width transfer | FAILS/SUSPECT | TRANSFERS, drift 0.144 | **FAIL — and my own §4 predicted this null** |
| P3c4 | control: SGD rate without `D` breaks transfer | FAILS | **FAILS**, drift 0.940 dec, implied exponent **+1.041** vs predicted `+1` | **PASS — control bites hard** |
| P4 | `Delta A\|_{t=1} ~ D_h^{-1/2}` (SGD) | `−0.5 ± 0.15` | `−0.456` | **PASS, but see §3 — right answer, wrong mechanism** |
| P4 | `Delta A\|_{t=1} ~ D_h^{0}` (signGD) | `0 ± 0.15` | `−0.323` | **FAIL** |
| P4b | `\|slope(t=8)\| < \|slope(t=1)\|` (SGD) | — | `0.4585` at `t=8` vs `0.4562` at `t=1` | **FAIL** |
| P5 | `A_init ~ D_h^{1/2-alpha_A}` | ±0.10 | `+0.499` / `−0.013` / `−0.513` at `alpha_A = 0, 1/2, 1` | **PASS** |
| P6 | `rho_l` rises with `l`; `gamma_P^l` rises with it | monotone | `rho` `0.003 → 0.023`; `gamma_P` `0.377 → 0.400`, monotone | **PASS, weak** (control does not bite; see §5) |
| P7 | pooled `C_ab` blows up with sparsity, node-level does not | ≥3× / ≤1.5× | `2.06×` / `0.40×` | **FAIL on the `C_ab` bar; PASS on the node-level bar** |
| S1 | across-seed sd of a logit `~ D_h^{-1/2}` (`alpha_A=1`), flat (`1/2`) | ±0.10 | `−0.531` / `−0.031` | **PASS** |
| S2 | across-head sd vanishes (`alpha_A=1`), not (`1/2`) | ±0.10 | `−0.522` / `−0.022` | **PASS** |

Not preregistered, added because P4 forced them: **E8** (channel decomposition +
horizon), **E9** (quantitative check of formula (G)), **E10** (power audit of the
transfer harness), **E11/E12** (two further attempts at the signGD legs). They
are reported as follow-ups, not as predictions.

## 1. What was confirmed sharply

**Formula (G), exactly on its stated inputs.** `10` §4 derives
`gamma^2 = <sum_v P_uv^2 + sum_{v!=v'} P_uv P_uv' rho_{vv'}>`. The original E9
paired each operator with a mismatched post-block stream and attributed its
1–9% gaps to node-norm heterogeneity. The audited probe now uses the actual
value/input tensor and enforces the formula's equal-node-norm assumption:

| operator | predicted | measured | ratio |
|---|---|---|---|
| softmax attention, `alpha_A = 0` | 0.9073 | 0.9073 | 1.000 |
| softmax attention, `alpha_A = 1/2` | 0.5322 | 0.5322 | 1.000 |
| softmax attention, `alpha_A = 1` | 0.4097 | 0.4097 | 1.000 |
| `P = D^{-1/2} A D^{-1/2}` | 0.3946 | 0.3946 | 1.000 |
| `P = D^{-1} A` | 0.4103 | 0.4103 | 1.000 |
| `P = A` (unnormalised) | 5.7966 | 5.7966 | 1.000 |

Two consequences worth stating:

- **`P = A` gives `gamma = 5.8` at mean degree 7.8**, on the same scale as the
  paper's scanned values (`gamma = 7` for PascalVOC-SP, `17` for
  MNIST-Superpixels). So
  their §2.5 hyperparameter is not free: (G) predicts it from the degree
  distribution and the neighbour correlation.
- **Degree-normalised `gamma` depends on the feature-correlation regime.** For
  row-normalised `P`, decorrelated neighbours give
  `gamma = <1/d_eff>^{1/2}` (about `0.42` in E9's graph), while perfectly aligned
  neighbours give `gamma = 1` exactly. The independent scan in
  `gamma-verification.md` confirms both endpoints. The earlier framing of
  `0.42` versus the paper's `1` as a disagreement is withdrawn.

**Probe alignment audit.** `attention_stats` had also paired each layer's
attention matrix with the post-MLP stream instead of the value tensor it
actually aggregates. E2 and E5 were rerun after `gt.py` began recording the
attention input, value, and output separately. P2/P2c/P6 keep the same verdicts.

**`alpha_A < 1/2` is fatal, by the derived mechanism.** At `alpha_A = 0`, over
`D = 32 → 512`: `d_eff` `2.33 → 1.21` (softmax saturating toward hard argmax) and
`gamma_A` `0.811 → 0.951` (rising toward the row-sum value 1), both still moving
at the largest width. At `alpha_A = 1/2` and `1`, `d_eff` is `4.5` and `7.8`
(the latter being essentially the full mean degree of 7.83, i.e. uniform
aggregation), and both are flat.

**The SGD width exponent, from a control.** Removing the `D` from
`eta_SGD = eta_0 D L` moves `log10 eta_0*` by `0.97 → 1.32 → 1.58 → 1.91` over
`D = 32 → 256`, a slope of **`+1.041`** against the predicted `+1` and a drift of
`0.940` decades. This is the round's cleanest quantitative confirmation of a
learning-rate exponent, and it doubles as proof that the harness *can* resolve a
mis-scaling.

**The DMFT's LLN-vs-CLT prediction (S1/S2).** `11` §M4b says
`A_uv = D_h^{1-alpha_A} x (population average over D_h width coordinates)`, so
`alpha_A = 1` is an LLN (attention concentrates, heads collapse) and
`alpha_A = 1/2` is a CLT (a Gaussian field survives, heads do not collapse).
Measured across-seed sd slopes `−0.531` / `−0.031` and across-head sd slopes
`−0.522` / `−0.022`. This is the one prediction in the round that only the cavity
route could make, and it is the sharpest.

## 2. The missing channel (E8) — and why P4 "passing" was not evidence

P4's signGD arm failed (`−0.323` against a predicted `0`). Chasing it produced
the substantive result of the round.

`10` §3 counted **one** way the attention logits move: the Q/K weights update.
There is a second, and it has nothing to do with the Q/K learning rate: **the
block's input moves by `Theta(1)` — that is the whole design — and the logits
move with it.** Since `q = (1/sqrt(D)) W_Q xt` with `W_Q` random,
`Delta q = Theta(1)` per coordinate, and `Delta(q.k)` is an incoherent sum over
`D_h`:

    Delta A |_stream = Theta( D_h^{1/2 - alpha_A} )     — the same order as A_init

E8 freezes one group at a time (`_freeze` in `experiments.py`) and sweeps the
horizon to `t = 256`:

| optimiser | `alpha_A` | channel | `t=1` | `t=8` | `t=64` | `t=256` | predicted |
|---|---|---|---|---|---|---|---|
| SGD | 1 | Q/K only | `−0.562` | `−0.561` | `−0.556` | `−0.544` | `−0.5` incoh. / `0` coh. |
| SGD | 1 | stream only | `−0.454` | `−0.446` | `−0.497` | `−0.518` | **`−0.5`** |
| SGD | 1/2 | Q/K only | `−0.562` | `−0.562` | `−0.561` | `−0.558` | `−0.5` incoh. / `0` coh. |
| SGD | 1/2 | stream only | `+0.080` | `+0.080` | `+0.089` | `+0.182` | **`0`** |
| signGD | 1 | Q/K only | `−0.151` | `−0.151` | `−0.149` | `−0.065` | `0` |
| signGD | 1 | stream only | `−0.482` | `−0.463` | `−0.520` | `−0.274` | **`−0.5`** |
| signGD | 1/2 | Q/K only | `−0.142` | `−0.139` | `−0.136` | `−0.169` | `0` |
| signGD | 1/2 | stream only | `+0.069` | `+0.058` | `+0.089` | `+0.150` | **`0`** |

Four readings:

- **The stream-channel formula is confirmed on all four configurations.** It is
  the piece the derivation was missing.
- **P4's SGD pass was a channel degeneracy.** At `alpha_A = 1` the Q/K channel
  and the stream channel both give `D_h^{-1/2}`, so the total (`−0.456`, E3)
  agreed with a derivation that had only counted one of them. At `alpha_A = 1/2`
  they separate (`−0.56` vs `+0.08`), the total follows the *larger* (`+0.076`),
  and the derivation is visibly wrong there. **A prediction that is right at one
  value of a structural exponent and wrong at another is a channel-counting
  error, not a coefficient error.** Registered as **F23**.
- **The Q/K coherence flip (contraction C2) is FALSIFIED under SGD at this
  horizon.** The Q/K-only slope is `−0.562` at `t=1` and `−0.544` at `t=256`: a
  drift of `0.018` over 256 steps. The coherent labelling predicts `0`; the
  incoherent labelling predicts `−0.5`. The incoherent labelling wins, and it
  wins at **both** `alpha_A` under the derived (`alpha_A`-compensating) rate,
  which is a two-point test of the formula rather than one.
  *Caveat, stated because it is the obvious escape hatch:* F18's own record
  (`03-attention.md` §D2b) has the analogous exponent drifting from `−1.20` to
  `−0.29` over `t = 1 → 2500`. I ran 256. What I can say is that the drift here
  is `0.018` over 256 SGD steps, an order of magnitude slower than that record,
  and that under **signGD** the same channel *does* drift (`−0.151 → −0.065`),
  so the instrument is not blind to the effect.
- **Under signGD the Q/K channel is nearly coherent already** (`−0.15`, versus
  `−0.56` for SGD) and improving with time. The sign operation removes the
  gradient-magnitude suppression, which is exactly what §3d's sign-factorisation
  argument says it should do; `−0.15` rather than `0` is the residual
  sign-cancellation over the `N d_bar ~ 190` node pairs, flagged as risk 3 in
  `10` §7.

**Consequence, and it reverses `10` §5's recommendation.** At `alpha_A = 1`,
`Delta A -> 0` as `D_h^{-1/2}` through *every* channel at every horizon tested.
Since `A_init` also vanishes as `D_h^{-1/2}`, the attention matrix converges to
**uniform over each graph neighbourhood, at all times** — the MHSA branch's limit
is a mean-aggregation MPNN branch with a learned value/output projection. The
parameterisation is perfectly stable and transfers; it just transfers a model
whose attention does not do anything. At `alpha_A = 1/2` the stream channel
alone delivers `Theta(1)` logit updates against a `Theta(1)` init pattern.

So the two routes now disagree about which exponent to prefer, and the honest
statement is that **both are right about different things**:

- `10`/E8 (measurement) prefer `alpha_A = 1/2`: it is the only tested exponent
  with a non-degenerate attention limit.
- `11` §9 (theory) prefers `alpha_A = 1`: at `alpha_A = 1/2` the closure needs
  `<softmax(Theta(1) Gaussian field)>` and drops a Jensen gap nobody in this
  program knows how to close.

That is not a resolution. It is a named open problem.

## 3. What the transfer sweeps did and did not establish (§4 of the honest accounting)

**Two registered controls failed to bite.** P3c2 anti-bit and P3c3 transferred.
P3c1 met its literal SUSPECT bar, but its drift was smaller than the treatment,
so it is weak evidence. Per F17, E10 audited the instrument's power by
mis-scaling quantities of known exponent:

| leg | verdict | drift (dec) | bites? |
|---|---|---|---|
| SGD width, treatment | SUSPECT (not settling) | 0.083 | — |
| CONTROL `alpha_A = 0` | SUSPECT | 0.065 | **literal bar passes; weak** |
| derived `alpha_A = 1/2` | TRANSFERS | 0.283 | — |
| CONTROL `qk-global`, `alpha_A = 1/2` | SUSPECT | 0.122 | **no** (0.43× treatment) |
| CONTROL `P = A` | TRANSFERS | 0.144 | **no** |
| CONTROL no-`D` SGD rate | **FAILS** | **0.940** | **yes** |
| POWER-CTL `W_V, W_O` rate × `sqrt(D)` | **FAILS** | **0.620** | **yes** |
| POWER-CTL paper §2.4 `sigma_{L+1} = 1` | TRANSFERS | 0.110 | **no** |

So the harness resolves a `D^{1}` mis-scaling (0.94 dec) and a `D^{1/2}`
mis-scaling **inside the attention sector** (0.62 dec, `W_V`/`W_O`). It does not
resolve the Q/K *logit* mis-scaling (0.12 dec) or `alpha_A` (0.065 dec).

**That is a result, not only a limitation.** The optimal `eta_0` is set by the
stability edge of the branches that carry the feature update — `W_V, W_O, W_1,
W_2, W~` — and the attention *pattern* sector is sub-dominant for it. A
learning-rate transfer sweep is therefore **not a test of the Q/K
parameterisation**, and reporting one as if it were would be exactly the
"transfer passed, so the parameterisation is right" inference this program keeps
having to retract. Round 011 does not establish `sigma_QK` empirically in either
direction.

All shared-seed transfer comparisons were also re-scored with the paired SEM of
the extrema rather than an unpaired combination of per-dial SEMs. The retained
analyses are `E4-transfer-paired.json` and `E10-power-audit-paired.json`; no P3
or control verdict is rescued by the correction.

Two more honest entries:

- **P3c3 (`P = A`) failing was predicted by my own derivation and I
  preregistered it anyway.** Formula (G) says `gamma_P` is a function of the
  degree distribution and `rho`, both `D`-independent, so an unnormalised
  operator rescales `eta_0` by a constant and cannot break *width* transfer. It
  breaks stability (every `lr >= 2.7e-2` diverges) and performance, which is what
  the paper's §2.5 actually reports. I wrote the bar from the paper's empirical
  headline rather than from my own §4, and it was wrong.
- **The paper's §2.4 typo does not present as a width-transfer failure.** `10`
  §1a claimed it would. Measured: drift 0.110 dec, TRANSFERS. The forward pass is
  unchanged (both `sigma_{L+1}` conventions give `z = Theta(D^{-1/2})`); only the
  nominal decoder scale moves by `sqrt(D)`, without producing the preregistered
  transfer failure. **`10` §1a is corrected accordingly** — the
  inconsistency in the paper is real, its consequence is milder than I claimed.

**The Adam/signGD side is not established.** Three grids were tried:

| attempt | grid | outcome |
|---|---|---|
| E4 v2 | `10^-3 .. 10^-0.3` | UNDER-POWERED, some seeds' argmin on the low edge |
| E10 | `10^-4.5 .. 10^-1` | UNDER-POWERED, some seeds' argmin on the high edge |
| E11 | `10^-3.5 .. 10^0` | "FAILS", 1.14 dec — **but the loss-vs-`eta_0` curve is bimodal** |
| E12 | same, **final** loss instead of best-so-far | unimodal; width UNDER-POWERED (`sem 0.264`), depth TRANSFERS (0.390), heads TRANSFERS (0.547) |

E11's 1.14-decade "failure" is an artefact of the estimator: the paper's
convention is the *best* train loss attained during training, and under a
sign-like optimiser that statistic rewards overshooting, creating a second basin
at large `eta_0` that the per-seed argmin jumps into at large `D`. Switching to
the loss at the horizon removes the second basin. Registered as **F24**. Even
after the fix the basin is shallow and `lr*` has `sem 0.22–0.26`, so the surviving
statement is only "no signGD drift larger than about 0.6 decades across `L` and
`H`, and nothing resolved across `D`".

**The SGD width leg is SUSPECT, not a pass.** Drift 0.083 dec (threshold 0.056,
so resolved), with `log10 eta_0* = −0.59, −0.57, −0.59, −0.65`: spread 0.08 over
the largest three against 0.03 over the smallest three. Per F22 that is not
settling and must not be read as a pass. Given the harness's measured resolution
(§3), a residual of this size is consistent with the `O(L^{-1})` and
`O(D^{-1/2})` finite-size corrections that E1 also shows (the attention branch
RMS drifts `−0.039` in `D`), but I have not shown that, so it stays SUSPECT.

## 4. Smaller findings

**P6 — oversmoothing (weak).** `rho_l` rises monotonically `0.003 → 0.023` over
8 layers and `gamma_P^l` rises with it `0.377 → 0.400`, in the direction formula
(G) requires. But the magnitude is tiny, and **the intended control (a denser,
decorrelating graph) rises too** — so it is not a control and P6 is a
directionally-correct observation, not a demonstration. With `1/L` branches at
`L = 8` the stream barely moves, exactly the "no room to rise" failure mode
declared in the prereg. Testing this properly needs many more layers or an
architecture without the `1/L` suppression.

**P7 — alignment factors.** The pooled `C_ab` (the paper's Eqn 25) and the
node-level Gram that the Q/K gradient actually reads out separate strongly with
feature sparsity:

| sparsity | `C_ab` (pooled) | `C` (node-level) | ratio |
|---|---|---|---|
| 0.00 | 14.61 | 2.80 | 5.2 |
| 0.50 | 14.42 | 2.62 | 5.5 |
| 0.90 | 16.48 | 1.43 | 11.6 |
| 0.97 | 30.04 | 1.12 | 26.8 |

The node-level factor goes to 1 while the pooled one grows — `10` §6b's
prediction, and the reason the attention sector should not inherit the encoder's
`C_ab`. But my preregistered bar wanted `C_ab` to grow ≥3× and it grew 2.06×, so
**the bar is failed and I am not moving it**. Random sparsity on Gaussian
features is not bag-of-words sparsity; the paper's Cora/Citeseer features are
sparse *and* structured, which is what makes their `M_ab` collapse. The
mechanism is demonstrated; their magnitudes are not reproduced.

## 5. Coverage, and what was never attempted

Synthetic random-geometric graphs only (`N = 24`, `B = 12`, mean degree 7.8),
one task type (graph-level scalar regression), `D <= 512`, `L <= 8`, `H <= 8`,
horizons `<= 256` steps, float64 CPU. No real datasets, no edge features, no
positional encodings, no LayerNorm, no AdamW/weight-decay leg, no real Adam in
the transfer sweeps (signGD proxy only), no `alpha_A` strictly between `1/2` and
`1`, and **no DMFT solver** — `11` derives the single-site system and nothing
evaluates it, so there is no theory-vs-simulation comparison of the dynamics and
no MC floor for one. The paper's Fig 2/3/6 performance claims, its citation-
network `C_ab` experiment and its `tau_epoch` result are untested here.

## 6. Registry

Two new entries, both with detection signatures:

- **F23 — incomplete enumeration of update channels.**
- **F24 — "best loss during training" makes the LR optimum bimodal under
  sign-like optimisers.**

Written to `registry/failure-modes.md`, with executable guards: `gt.py::train`
now takes `metric=` and documents the bimodality; `experiments.py::_freeze`
exists so channel decomposition is a one-liner rather than a rewrite.
