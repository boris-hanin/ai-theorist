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

## v2 (running) — scaled ~40x

- `C` ladder doubled in density and extended to **1e9** (6 decades), seeds up to
  3072 at small `C`, **24** GD steps instead of 8.
- E3 trained arm actually trains; adds a wide-`aM` arm as an independent control.
- E2 HP transfer across all five dials **with balancing on**, 256 seeds, 29-point
  LR grid.

Expected runtime several hours. Verdicts to be written when it lands.
