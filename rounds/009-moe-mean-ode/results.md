# Round 009 — residual MoE in the Neural Mean ODE scaling (NEW RESULT)

Derivation/prereg: `derivations/09-moe-mean-ode.md`, committed **before** any
measurement (`feac9b6`). Model: `skills/dmft-moe/scripts/moe_ode.py`.

Not a replication: merges the MoE parameterisation of **2601.20205** with the
joint `(L, M, D)` Mean-ODE analysis of **2509.10167** / **2603.18168**, which do
not treat sparse routing.

## The claim

Units are indexed by the triple `(i,j,l) in [E] x [M] x [L]`, but only **active**
experts contribute to a token's forward pass, so

    effective width  W_eff = L * a * M          (a = active experts, NOT E)
    residual scale   = sqrt(D)/(L M)
    rate             = O( 1/L + sqrt(D/(L a M)) + 1/sqrt(D) )

## Verdicts

| # | prediction | measured | verdict |
|---|---|---|---|
| N1 | init deviation slope `-1/2` in `L` | **-0.505** | **PASS** |
| N1 | slope `-1/2` in `a` | **-0.489** | **PASS** |
| N1 | slope `-1/2` in `M` | **-0.512** | **PASS** |
| N1/N3 | slope `0` in `E` at fixed `a` | **+0.030** | **PASS** |
| N1c | fan-in control: `M` drops out | **-0.013** (vs `-0.512` MLU) | **PASS** |
| **N2** | **invariant to the split at fixed `L a M`** | spread **1.10x** over 10 splits | **PASS** |
| N2c | fan-in not invariant | spread **3.90x** | **PASS** |
| N4 | optimal LR transfers | see below — needed **per-group** LRs | **PASS** after correction |
| N5 | Euler term `1/L`, `a`/`M`-independent | — | **not tested** |

## N2 — the test that mattered

`09` §7 named the risky step in advance: §2 treats the `a` active experts as
adding **incoherently** at initialisation, but they share the router-conditioning
event `q >= q*(kappa)`, which correlates them. If that correlation were
`Theta(1)`, the `a`-averaging would be coherent and `W_eff = L M`, not `L a M`.

At fixed `L a M = 256`, across ten splits spanning `a` = 2→16 and `L` = 2→64:

| (L, a, M) | MLU | fan-in |
|---|---|---|
| (4, 2, 32) | 2.17e-2 | 3.07e-2 |
| (8, 2, 16) | 2.26e-2 | 2.26e-2 |
| (16, 4, 4) | 2.31e-2 | 1.16e-2 |
| (64, 2, 2) | 2.22e-2 | 7.87e-3 |
| (2, 16, 8) | 2.28e-2 | 1.61e-2 |
| … | | |
| **spread** | **1.10x** | **3.90x** |

**The `a`-averaging is incoherent** and `W_eff = L a M` stands. Depth, active
experts, and expert width are mutually interchangeable to 10%.

The `(8,2,16)` row has `M = D = 16`, where the two initialisations are literally
the same model — the identity control landing inside the sweep, and agreeing to
every printed digit.

## N4 — HP transfer, and the two LR errors it caught

**Corrected result** (per-group learning rates, `lr_U = LMa/D`, `lr_W = LMaD`,
`lr_R = La sqrt(M)/D`):

| dial | range | drift, full | drift, largest 3 |
|---|---|---|---|
| embedding `D` | 8 → 128 | 0.24 | **0.04** |
| depth `L` | 4 → 64 | 0.16 | **0.01** |
| active experts `a` | 2 → 32 | 0.06 | 0.03 |
| expert width `M` | 8 → 128 | 0.08 | 0.07 |
| expert count `E` (`a` fixed) | 16 → 128 | **0.03** | 0.03 |

### The history, because the first "pass" was wrong

First pass: `D` drifted **0.65 decades** and pinned at the grid edge, while
`L`, `a`, `M` drifted 0.22–0.25. The `D` failure was a real gap: `09` states the
LR scaling for `L`, `M`, `a` but **never states its `D`-dependence**, and the
simulator used `lr = L M a eta`. Counting properly: the loss is per-coordinate
normalised so `b ~ 1/D`, hence `grad_w ~ 1/(L M a D)` and

    **lr = L * M * a * D * eta**

With that, `D` drift → 0.22, interior — **and I reported that as a pass. It was
not.** The curve was `+1.05, +1.13, +1.08, +0.91`: a monotone decline over the
last three points with *accelerating* increments (`+0.08, -0.05, -0.17`), where
`L` and `a` flatten. I had justified `L`/`a` as finite-size *because* they
flatten, then applied the same verdict to a curve doing the opposite, against an
arbitrary 0.25 bar.

**The real cause: a single global LR cannot be correct in `D`.** The three
parameter groups have different `D`-counting. Measured with one global LR, the
induced change in each group's own observable scales as

| group | observable | measured | required |
|---|---|---|---|
| `W` down-projection | block output | `D^{-0.18}` | `D^0` |
| `U` up-projection | `z = <u,h>` | **`D^{+1.28}`** | `D^0` |
| `R` router | logit `<r,h>` | **`D^{+1.37}`** | `D^0` |

The router and up-projection updates blow up with embedding dimension. Per-group
LRs derived from the coherent labelling (`09` §9) fix it, and the `D` curve
changes character: increments `+0.14, +0.06, +0.03, +0.01`, converging.

This is the second time this session a preregistered dial caught a scaling factor
a derivation left implicit (cf. `05` §1's unstated `d_L` domain, F19) — and the
first time this session I passed a dial that should have failed. The lesson is in
the *shape* of the drift, not its size: **a drift that accelerates is not a
finite-size transient**, whatever the bar says.

The residual `L`/`a` drift is **finite-size**, not a broken rule — it shrinks
monotonically as the smallest configs are dropped, which is what an asymptotic
transfer claim predicts:

| dial | drift over all 5 | over the largest 3 |
|---|---|---|
| depth `L` (4→64) | 0.26 | **0.11** |
| active `a` (2→32) | 0.25 | **0.06** |

**`E` at fixed `a` is the cleanest row in the table: 0.04 decades.** That is the
new prediction — the expert count is absent from the rate — and it transfers
better than any other dial.

## The fan-in control does not bite for HP transfer, and that is expected

Fan-in drifts 0.27 / 0.31 / 0.13 / 0.21 / 0.02 — indistinguishable from MLU.
This is **not** a failed control: round 007 established that the down-projection
init cannot move the update scale (`Delta E = M * (base/M) * Theta(1)`, free of
`sigma_down`), and the optimal LR is set by the update. So HP transfer is
**not a discriminating test of the down-projection init**; N1/N2 are, and there
the control bites hard (`-0.013` vs `-0.512`; 3.90x vs 1.10x).

Stated explicitly because a non-biting control is normally a red flag (F17), and
here it is a derived consequence instead.

## Figure

`../../moe-transfer.html` — HP transfer across all five dials, the `L a M`
invariance sweep, and the fixed-FLOP shape landscape. Underlying numbers in
`figure-data.json`. Published: https://claude.ai/code/artifact/d960549e-05dd-4ab8-8c5e-b7d4626eaed2

## What this buys — two budgets

    parameters  P = Theta(L E M D)      FLOPs  C = Theta(L a M D)

The rate depends on the FLOP budget. Minimising subject to `L a M D = C` with
`W = aM`:

    **D = Theta(C^{1/3})  binding;   L >= Theta(C^{1/6})  a floor, not an optimum;
      a M = C/(L D);   error = O(C^{-1/6})**

with the split of `aM` between expert count and expert width **free**.

**Correction, forced by the fixed-`C` sweep.** `09` §4 originally balanced all
three terms and reported `L = C^{1/6}` as *the* optimum. Substituting the
constraint, the CLT term is `D/sqrt(C)` — **independent of `L`**, because fixing
`C` and `D` pins `L * aM`. Measured directly: at `C = 8192`, `D = 16`, sweeping
`L` = 2 → 64 with `aM` = 256 → 8, the CLT term is flat (1.62e-2 → 1.49e-2). So
`L` enters only one of the three terms and can only help; `L = C^{1/6}` is the
**shallowest** shape achieving the optimal rate, not a unique optimum. The
binding requirement is `D = Theta(C^{1/3})`. `09` §4 has been corrected. So at
fixed FLOPs a model can buy capacity by raising `E` at fixed `a` — costing
parameters but neither FLOPs nor convergence rate. That is a scaling-theoretic
statement of why MoE works, and it locates the benefit of 2601.20205's Finding
3.1 (more, smaller experts) in the **limit object**, not in the approximation
rate — the rate cannot see `E` at all, which N1/N3/N4 now confirm three separate
ways (`+0.030`, and 0.04 decades of LR drift).

## The `C^{-1/6}` rate — MEASURED

Reference-free by construction: along the optimal-shape path
(`D ~ C^{1/3}`, `L ~ C^{1/6}`, `aM = C/(LD)`) **all three error terms are
`Theta(C^{-1/6})` regardless of their coefficients**, so the distance between the
`C` and `4C` ensembles carries the exponent directly:

    E_diff(C)^2 = |mu_C - mu_4C|^2 + var_C + var_4C   ~   C^{-1/3}

No limit reference is needed — and none is possible: `C^{-1/6}` is so slow that a
reference 10x better than `C = 1e7` would require `C_ref ~ 1e13`.

A100, `C` = 1e3 → 6.6e7 (4.8 decades), **192 seeds** per point:

| C → 4C | E_diff |
|---|---|
| 1.0e3 → 4.0e3 | 2.771e-1 |
| 4.0e3 → 1.6e4 | 2.309e-1 |
| 1.6e4 → 6.4e4 | 1.845e-1 |
| 6.4e4 → 2.6e5 | 1.489e-1 |
| 2.6e5 → 1.0e6 | 1.197e-1 |
| 1.0e6 → 4.1e6 | 9.443e-2 |
| 4.1e6 → 1.6e7 | 7.658e-2 |
| 1.6e7 → 6.6e7 | 6.137e-2 |

    slope (all)   -0.1571        slope (last 4)  -0.1597        predicted -0.1667

Local slopes after the first point are `-0.16 +/- 0.01`. At `S = 24` the fit gave
`-0.1494` / `-0.1227`; raising to `S = 192` moved **both** toward the prediction
and the tail from `-0.123` to `-0.160`, i.e. the estimate settles as precision
increases (the F22 criterion, applied to itself).

Data: `rate-C-one-sixth.json`; code `skills/dmft-moe/scripts/rate_gpu.py`.

## Trainable expert biases — the load-balancing rule, added

The largest fidelity gap has been closed. 2601.20205 Eq. (2):

    b_i  <-  b_i  -  eta_bias * (Load_i - kappa)

**Not a gradient step.** The biases enter only the hard top-`a`, receive no
gradient (the paper treats the activated set as no-grad), and are driven purely
by the measured batch load. Scaling Rule 2 sets `eta_bias = Theta(1)`.
Implemented in `moe_ode.py::balance_biases` and in the GPU rate script.

**It works, and it has a stability edge** — `max_i |Load_i - kappa|` over 25 steps:

| `eta_bias` | imbalance | |
|---|---|---|
| 0.0 | 0.750 → 0.750 | unchanged (control) |
| 0.3 | 0.750 → **0.250** | balances |
| 1.0 | 0.750 → **0.250** | balances |
| 3.0 | 0.750 → 0.750 | overshoots |

consistent with `eta_bias = Theta(1)` having a finite stable range.

**The `C^{-1/6}` rate is unchanged by it**, on the matched 7-point range
`C` = 1e3 → 1.6e7, 192 seeds:

| | slope (all) | slope (last 4) |
|---|---|---|
| balancing ON (`eta_bias = 1`) | **-0.1583** | **-0.1619** |
| frozen biases (control) | -0.1571 | -0.1597 |
| predicted | -0.1667 | -0.1667 |

The per-point `E_diff` values agree between the two arms to **~1% at every `C`**.

**Caveat, stated rather than glossed.** That near-identity is strong evidence the
rate is robust to balancing, but it is *also* consistent with balancing barely
biting over the 8 GD steps the rate test uses — the imbalance table above was
measured over 25 steps. What is **not** yet checked is how much the load actually
moves within the rate test's own horizon. Until that is measured, the honest
claim is "the rate is unchanged with the rule enabled", not "load balancing is
irrelevant to the rate".

## FIDELITY — what architecture this actually is

**Not the transformer of 2601.20205 §3.1, and not Chizat's model unmodified.**
It is a faithful merge of the two *scaling structures* on the **reduced MoE-only
model of 2601.20205 §4**. Deviations, in full:

*From Chizat 2509.10167 §2.1:*
- **added an embedding and a scalar readout** (his model is `R^D -> R^D` with
  `loss_i(h^L)`); needed because the rate test requires a `D`-comparable observable
- **per-group learning rates** instead of his single global `L M eta / alpha^2` —
  forced by convention: he bounds `||h|| = Theta(1)`, this program uses
  per-coordinate `Theta(1)` (`||h|| = sqrt D`), under which one global LR is
  provably wrong in `D` (round 009 N4)
- `alpha` fixed at 1

*From 2601.20205:*
- **no MHSA and no LayerNorm** — this is their §4 reduced model
- **plain GD**, not Adam / SignGD
- linear readout `<w, h^L>`, not `W_unembd phi(h^L)`
- ~~the expert biases are frozen~~ **RESOLVED** — the auxiliary-loss-free rule is
  now implemented and the rate re-measured with it on (see above). Remaining
  caveat: whether the rule bites within the rate test's 8-step horizon is
  unverified.
- the parameterisation is written in Chizat's convention (`1/(LM)` branch,
  `Theta(1)` weights) rather than theirs (`1/L` branch,
  `sigma_down = sqrt(D)/M`); algebraically identical, shown in `09` §2

So what is validated is the **scaling structure of the merged limit** —
`W_eff = L a M`, `E` absent from the rate, and the `C^{-1/6}` envelope — on the
reduced architecture. It is **not** a validation on a transformer with attention,
LayerNorm, Adam and live load balancing.

## Audit of every transfer plot in the program

Prompted by the `D` failure, all previously published transfer figures were
re-judged on the **shape** of the drift, not its size. Only the MoE one was
wrong.

| figure | arm | log10 lr* | drift | tail (largest 3) | shape |
|---|---|---|---|---|---|
| MLP width | muP L2 | -0.97 -0.91 -0.90 -0.87 | 0.10 | **0.04** | settling |
| MLP width | muP L3 | -0.99 -1.15 -1.03 -0.98 | 0.17 | 0.17 | non-monotone |
| MLP width | muP L4 | -1.17 -1.14 -1.14 -1.23 | 0.09 | 0.09 | non-monotone |
| MLP width | SP control | -2.10 -2.30 -2.49 -2.81 | 0.72 | 0.51 | fails, as it must |
| Attention | width `N`, `aA=1` | -0.49 -0.53 -0.34 -0.60 | 0.26 | 0.26 | non-monotone |
| Attention | width `N`, `aA=1/2` | -0.45 -0.54 -0.62 -0.62 | 0.17 | **0.08** | settling |
| Attention | heads `H` | -0.58 -0.53 -0.62 -0.60 | 0.10 | 0.10 | non-monotone |
| Attention | depth `L`, `aL=1` | -0.53 -0.58 -0.62 | 0.10 | 0.10 | settling |
| Attention | depth `L`, `aL=1/2` | *identical to its control* | — | — | identity, already flagged |
| Attention | all controls | — | 0.36–0.78 | — | bite |

**No published arm other than MoE `D` shows a non-settling drift.** The MLP and
attention figures stand as published.

Registered as **F22**, and `transfer.py::verdict` now returns **SUSPECT** rather
than a pass whenever `tail_drift > 1.3 * head_drift` and the drift is resolved.
Regression-tested against all five cases above.

## Settled after the audit

**1. The ~0.2 one-step residual (was: unexplained).** Every link in the chain was
measured separately and every one is on prediction — `h` `+0.004`, `b` `-0.995`,
`z` `+0.006`, gate `-0.005`, `<w,b>` `-0.490`, `<E,b>` `-0.478`, and the
gradients `grad_U -0.572`, `grad_W -1.068`, `grad_R -0.499` against `-1/2`,
`-1`, `-1/2`. So the residual is small deviations compounding, not a wrong
exponent — and it is **finite-D**: refitting the same observables on a larger
window settles all three onto their derived values.

| group | fit over `D` = 8–64 | over `D` = 64–512 | predicted |
|---|---|---|---|
| `W` block | -0.241 | **-0.029** | 0 |
| `U` (`z`) | -0.820 | **-0.526** | -1/2 |
| `R` (logit) | -0.678 | **-0.452** | -1/2 |

**2. The `1/sqrt(D)` term is NOT established, and three separate contaminations
were found.** This is the exponent the `C^{-1/6}` shape rests on, so it matters.

| attempt | result | what was wrong |
|---|---|---|
| round 008, loss vs `D` | `-0.645` | **task changes with `D`** — the regression problem itself differs, so this is not a property of the limit |
| fixed-task, vs a `D_ref = 512` reference | `-0.580` | **reference bias.** A `1/sqrt(D)` term is a *systematic bias*, not noise, so the reference carries one too: the gap is `E_D (1 - sqrt(D/D_ref))`, which biases the slope **steep**. With `D_ref = 512` that alone predicts an apparent `-0.70` |
| same, floor subtracted in quadrature | `-0.68` | matches the `-0.70` predicted by reference bias — i.e. **consistent with a true `-1/2`, but demonstrating nothing** |
| reference-free pairwise `\|f_D - f_2D\|` | `+2.17` | **below the seed floor** (ratios 0.6–1.2x, two points zeroed by subtraction) |
| pairwise at `L a M` = 2048 to cut CLT noise | `+3.15` | **the seed variance is not the CLT term.** Widening `LaM` 8x barely moved the floor, because the variance is dominated by the per-seed *embedding and readout init*, which `LaM` does not suppress |

### DEMONSTRATED — the instrument was the problem, not the exponent

The four failures above all tried to measure the `1/sqrt(D)` term as a **bias**
(a gap to some reference). That is what made them fragile: a bias needs a
reference, and any reference at finite `D_ref` carries the same bias.

**The term is not a bias — it is a fluctuation.** The large-`D` limit of the
Mean ODE is a mean-field over the `D` embedding coordinates, so a *coordinate
population average* fluctuates about its limit at `1/sqrt(D)`. Measuring the
**seed spread** of such an average needs no reference at all, and therefore
cannot suffer reference bias. Observable: the stream kernel `(1/D)||h^L||^2`.

Run on an A100, `D` = 16 → 4096, **256 seeds** per point
(`skills/dmft-moe/scripts/dfit_gpu.py`, data in `dfit-1oversqrtD.json`):

| configuration | slope, all `D` | slope, `D >= 256` |
|---|---|---|
| at init, `aM = 512` | **-0.4999** | **-0.5022** |
| at init, `aM = 4096` | -0.5077 | **-0.5003** |
| trained, `aM = 512` | -0.4990 | **-0.5013** |
| trained, `aM = 4096` | -0.5068 | **-0.4995** |

against a predicted `-0.5000`. Two controls come free and both hold:
**`L a M`-independence** (512 vs 4096 agree, so this is the `D`-coordinate
mean-field and not the `LaM` CLT) and **train ≈ init** (the rate is a property
of the limit, not of the optimiser).

*A bug caught on the way, which had produced a confident wrong answer.* The GPU
run chunks seeds to fit memory, and the per-chunk generator was seeded with a
constant — so every chunk drew **identical** randomness. Chunk size depends on
`D` and `M`, so the duplication varied along exactly the axes being measured,
giving slopes of `-0.53` to `-0.87` with a spurious `aM` dependence. Fixed by
offsetting the seed per chunk.

**Superseded:** the "not established" verdict below is the pre-A100 state, kept
for the record of how the instrument was wrong. The first two were reported as results before
this audit; both were wrong, in different ways, and the second was wrong *by
construction* — a bias cannot be measured against a reference that carries the
same bias.

**What it would take.** The signal is `E_D = C D^{-1/2}` with `C ~ 0.29`; the
per-seed spread is `~0.12`–`0.24` and falls only weakly with `D`. A 3x margin at
`D = 64` needs **`S ~ 100` seeds per point**, roughly 6x what was run. That is
the honest cost, and it is the first thing to spend GPU time on — before any
`C^{-1/6}` fit, which would otherwise rest on an unmeasured exponent.

**Consequence.** `C^{-1/6}` depends on three exponents and **all three are now
measured**: `1/L` (`-0.953`, and `M`-independent at `+0.012`),
`sqrt(D/(LaM))` (`-0.505`, `-0.489`, `-0.512`), and `1/sqrt(D)` (`-0.4999`).
The shape claim is now empirically grounded in every leg, and the `C^{-1/6}`
rate test is worth running.

A bug found on the way: the readout LR was written `eta * D` when the counting
gives `eta / D` (`f = <w,h>` is coherent over `D`, so `df = D dw` and `dw` must
be `Theta(1/D)`). Every fixed-task run diverged until it was inverted.

## Not done

- **N5** (Euler term `1/L`, independent of `a` and `M`) — not measured.
- The DMFT of `09` §5–§7 is derived but has **no solver**, so the response-sector
  claim (`O(1/(LaM))`, hence a memoryless limit) is untested. Same status as
  MoE in round 006: parameterisation and limit structure validated, dynamics not.
- The `C^{1/6}` shape inherits round 008's caveat: the `1/sqrt(D)` term it
  depends on is only suggestively measured.
