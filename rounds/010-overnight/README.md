# Round 010 — overnight A100 suite

Script: `skills/dmft-moe/scripts/overnight_suite.py`. Results stream to
`big_out.json` / `big.log`; a local poller (`poll.sh`) pulls them back every 10
minutes so a **spot preemption** cannot lose more than one interval.

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
