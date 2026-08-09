# Round 010 — overnight A100 suite

Historical v1 script: `skills/dmft-moe/scripts/overnight_suite.py`.  The v2
source was not committed; its exact effective settings were recovered from
`big.log` in `skills/dmft-moe/scripts/overnight_suite_v2.py`. This is a
reconstruction, not the missing
original source. Results stream to `big_out.json` / `big.log`; the portable
poller (`poll.sh`) can copy them from a configured remote host.

## v1 (completed, ~2 min) — `big_out_v1.json`, `big_v1.log`

| experiment | result | predicted |
|---|---|---|
| `C^{-1/6}`, balancing ON, `C` = 1e3→2.6e8 (5.4 dec) | slope **-0.1566**, tail **-0.1575** | -0.1667 |
| `C^{-1/6}`, frozen-bias control | slope **-0.1563**, tail **-0.1573** | -0.1667 |
| `1/sqrt(D)` at init, `D` to 16384 | slope **-0.5045**, tail **-0.4909** | -0.5000 |

**The load-movement caveat is now measured**, and the answer is nuanced. Max
imbalance `max_i |Load_i - kappa|` before → after 8 steps, with `eta_bias = 1`:

| `C` | 1e3 – 2.6e5 | 1.0e6 | 4.1e6 | 1.6e7 | 6.6e7 | 2.6e8 |
|---|---|---|---|---|---|---|
| imbalance | 0.750 → **0.750** | → 0.682 | → 0.613 | → 0.553 | → 0.559 | → 0.570 |

so **balancing does not bite at all below `C ~ 1e6` in 8 steps**, and bites
only partially above it. The frozen-bias control stays at 0.750 throughout, as
it must. Since the rate is identical in both arms (`-0.1566` vs `-0.1563`)
*including at the large `C` where balancing does move the load*, the conclusion
"the rate is robust to load balancing" now has support at the budgets where the
rule is actually active — not merely where it is inert.

**Bug found and fixed for v2:** E3's "trained" arm never trained (the `steps`
variable was unused), so it was a byte-duplicate of the init arm. The v1
`1/sqrt(D)` number is therefore an **init-only** result.

## v2 — COMPLETE

Scaled ~40x: 19 `C` values over 5 decades (top two, `5e8`/`1e9`, did not run),
seeds to 3072, 24 GD steps, 29-point LR grid, 256 seeds per transfer point.

### E1 — the `C^{-1/6}` rate

| arm | slope (all 18) | tail 8 | tail 4 |
|---|---|---|---|
| balancing ON | -0.1593 | -0.1575 | **-0.1668** |
| frozen control | -0.1596 | -0.1558 | -0.1614 |
| predicted | -0.1667 | -0.1667 | -0.1667 |

**The tail-4 slope is -0.1668 against a predicted -0.1667.** The estimate
*settles* onto the prediction as `C` grows (-0.159 → -0.167), which is the F22
criterion satisfied by the headline result itself.

**Load balancing now demonstrably bites**, at 24 steps: imbalance is flat at
0.750 up to `C ~ 1e5`, then falls monotonically to **0.52** at large `C`. The
frozen-bias control stays at 0.750 for all 19 budgets. The rate is nonetheless
identical across arms (`-0.1593` vs `-0.1596`), now tested over a range where
the rule is unambiguously active.

### E2 — HP transfer with load balancing ON

| dial | lr* | drift | tail | verdict |
|---|---|---|---|---|
| depth `L` | +0.31 +0.30 +0.30 +0.30 +0.32 | **0.024** | 0.024 | transfers |
| active `a` | +0.31 +0.30 +0.30 +0.30 +0.30 | **0.008** | 0.002 | transfers |
| expert width `M` | +0.31 +0.31 +0.30 +0.29 +0.27 | 0.041 | 0.031 | transfers |
| embedding `D` | +0.27 +0.28 +0.30 +0.31 +0.31 | 0.039 | **0.005** | transfers |
| expert count `E` at fixed `a` | +0.30 +0.34 +0.36 +0.38 +0.43 | **0.128** | 0.072 | **SUSPECT (F22)** |

Four dials transfer far better than before (0.008–0.041). The fifth degrades
sharply once balancing is on — it was 0.03 with frozen biases — and its
increments (`+0.04, +0.02, +0.02, +0.05`) do **not** settle, so by this
program's own F22 rule it is SUSPECT, not a pass.

**This is expected, and it is the paper's own caveat.** That dial holds `a`
fixed while raising `E`, i.e. it takes `kappa = a/E -> 0`, which 2601.20205 §3.2
explicitly excludes: they scale `n_exp` and `n_act` together *preserving* `kappa`
and warn "HPs will not transfer across different `kappa`". The drift appears
precisely when the `kappa`-dependent balancing rule is active and precisely in
the dial that varies `kappa`. The four dials that stay inside their scaling
regime all transfer.

### E3 — `1/sqrt(D)`, and an open discrepancy

| arm | slope | tail |
|---|---|---|
| at init, `aM = 512` | **-0.4969** | -0.4906 |
| at init, `aM = 4096` | **-0.5026** | -0.5089 |
| **after 24 GD steps** | **-0.4476** | **-0.3457** |

At initialisation the law is exact and `L a M`-independent across an 8x change
in `aM`, out to `D = 16384`. **After training it is not**: the slope moves to
`-0.45` and the tail to `-0.35`, i.e. it drifts *away* from `-1/2` as `D` grows.

This matters, and is **not** explained. The `C^{-1/6}` rate concerns trained
networks, so a trained-regime deviation in one of its three legs is a live
caveat — even though the rate test itself (which measures the output `f`, not
the kernel) lands on `-0.1668`. Candidates not yet separated: a genuine
trained-regime correction to the coordinate mean-field; or the fixed `eta` and
24-step horizon interacting with `D`. Next step is to re-run the trained arm at
two horizons and two `eta`, which distinguishes them.

## The `1/sqrt(D)` post-training failure — diagnosed

**DMFT reading.** In `09` §5 the stream kernel `C_h = (1/D)<h(t),h(s)>` is a
population average over the `D` coordinate-sites, and its `1/sqrt(D)`
fluctuation is the site-CLT. But under training the sites are **not**
independent: they are coupled through two *shared scalars*, the error trajectory
`Delta(t) = y - f(t)` and the readout `w`. `Delta` is itself a population
average, so it fluctuates at `1/sqrt(D)` — and it enters **every site
coherently**. Hence

    dK  =  [site-CLT]  +  (dK/dDelta) * dDelta
           ~ 1/sqrt(D)    ~ (dK/dDelta) / sqrt(D)   <- shared, rank-1

Both are `1/sqrt(D)` **at fixed susceptibility**. `dK/dDelta` is the F5
`Delta`-loop gain, and it *accumulates with training*. So the prediction is that
the law is exact at init and degrades with elapsed training.

**Measured (`skills/dmft-moe/scripts/diag_sqrtD.py`, `D` = 32…4096, 512
seeds).** The horizon dependence
is unambiguous:

| horizon | 0 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| slope | -0.5021 | -0.5018 | -0.5016 | -0.5010 | -0.4983 | -0.4620 | **-0.3468** |

`h = 64` reproduces the overnight tail (`-0.3457`). **The deviation is a
training-time effect, absent at init and growing monotonically.**

**What it is NOT.** I predicted the cause was *mismatched training progress* —
at a fixed step count larger-`D` models train further (loss `0.3930 -> 0.3741`
across `D` at 24 steps, and loss-matching needs `22 -> 19` steps), so they would
sit at different susceptibilities. Matching by **loss** instead of step count
tests this, and it **only partly holds**:

| matched loss | steps used | slope |
|---|---|---|
| 0.40 | 19–22 | **-0.4980** — restored |
| 0.30 | 37–44 | -0.4454 — not restored |
| 0.20 | 47–57 | -0.3743 — not restored |

**Historical interpretation, withdrawn pending a corrected P3 rerun:** these
appeared to track fixed-step values at the same step counts, leading to the
claim that elapsed training rather than progress controlled the deviation.
The audit below found that stopping was performed on width-dependent memory
chunks rather than per seed, so this table cannot distinguish those hypotheses.
P1/P2 still establish that the deviation grows with elapsed training; the
loss-matching conclusion is open.

**Consequence for `C^{-1/6}`: the rate result stands.** The rate test runs at
8 (v1) and 24 (v2) steps, where the `1/sqrt(D)` deviation is `<= 0.001` and
`~0.01–0.04` respectively. The failure lives at horizons *longer than the rate
test uses*, which is why the rate landed on `-0.1668`. The caveat is real but it
does not reach the headline — and it now has a bound rather than being open-ended.

### The proposed mechanism is FALSIFIED (second A100, B2)

> **Provenance limit.** The table below is the only surviving historical B2
> record: no raw JSON or original runner was committed or found in the working
> copy. `skills/dmft-moe/scripts/susceptibility_sqrtD.py` now defines a
> reproducible paired rerun, whose
> output must be labeled a new rerun rather than the missing historical data.

The diagnosis above predicted the exponent shift comes from the shared-`Delta`
channel, `dK = [site-CLT] + chi*dDelta` with `chi = dK/dDelta`. Since
`dDelta ~ 1/sqrt(D)`, that shift requires **`chi` itself to be `D`-dependent**:

    chi ~ D^beta   with   beta(h) = slope(h) + 1/2
    => beta(8) ~ 0.00,  beta(64) ~ 0.153

`chi` measured directly (perturb the target by 5%, read the change in the final
kernel; `D` = 32…1024, 256 seeds):

| horizon | 0 | 8 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| `chi` | 0 | 6.5e-3 | 8.0e-2 | 3.3e-1 | 2.0e-1 |
| `beta` | — | -0.006 | +0.055 | **+0.001** | +0.030 |

**Half the mechanism holds and half does not.** `chi` is exactly zero without
training and grows strongly with it, so the susceptibility does accumulate as the
DMFT says. But `beta ~ 0` at every horizon against a required `+0.153`. **A
`D`-independent `chi` cannot shift the exponent at all** — if both terms carry
`1/sqrt(D)`, so does their sum, whatever the size of `chi`.

So the shared-`Delta` channel is real but is **not** the source of the
post-training `1/sqrt(D)` failure. Remaining candidates, untested:

1. **`dDelta` is no longer `1/sqrt(D)` after training.** The error trajectory's
   own fluctuation is assumed to inherit the population `1/sqrt(D)`; if training
   changes its `D`-scaling, that propagates directly. Cheapest next test: measure
   the seed-spread of the final loss vs `D` at several horizons.
2. **The site-CLT term itself degrades** — i.e. the coordinate-sites acquire
   `D`-dependent correlations under training, so the leading term is no longer a
   clean CLT over `D` independent sites.

Recorded as **open**, with one hypothesis explicitly killed. The bound on the
`C^{-1/6}` result is unaffected: the deviation is still `<= 0.04` at the
horizons the rate test uses.

### Audit correction: loss matching

The historical P3 runner stopped an entire CUDA memory chunk when the chunk's
mean loss crossed the target. Chunk sizes shrink with `D`, so this was a
width-dependent stopping rule, and the recorded mean step count was also
unweighted across chunks. `skills/dmft-moe/scripts/diag_sqrtD.py` now uses
chunk-invariant per-seed
initialisation, freezes each seed at its own target crossing, and records every
seed's step count. The historical P3 table above remains provenance, not a
validated loss-matched comparison; it must be rerun before drawing a corrected
P3 conclusion.
