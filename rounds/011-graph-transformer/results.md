# Round 011 — results

Preregistration: `prereg.md` (commit `9435a39`, before measurement).
Derivations: `derivations/10-graph-transformer.md` and
`derivations/11-graph-transformer-dmft.md`.

## Overall verdict: FAILED, with one transfer leg incomplete

The preregistration says that **any** P1–P7 failure or any control that does not
bite makes the round a failure.  P3, P3c2, P3c3, P4, P4b, and P7 fail their
registered bars.  The third signGD sweep was interrupted before its heads leg,
but its completed width and depth legs already fail; the missing leg cannot
change the overall verdict.

## Prediction scoreboard

| prediction | measurement | registered bar | verdict |
|---|---|---|---|
| P1, flat in `D` and `H` | all stream/branch slopes `-0.039…+0.016` | within `0.10` of 0 | **PASS** |
| P1b, branches `~L^-1` | MPNN `-1.066`, attention `-1.111`, MLP `-1.063` | within `0.15` of `-1` | **PASS** |
| P2, `gamma_A` flat for `alpha_A=1/2,1` | corrected value-stream slopes `-0.010`, `-0.019` | absolute slope `<=0.05` | **PASS** |
| P2c, saturating control | `d_eff` slope `-0.230`, `gamma_A` slope `+0.056` | `<=-0.15`, `>=+0.05` | **PASS** |
| P3, all six transfer legs | SGD width **SUSPECT**; signGD width/depth under-powered twice, then **FAIL** on the straddling grid | all six TRANSFERS | **FAIL** |
| P3c1, `alpha_A=0` control | SUSPECT, drift `0.065` decades and not settling | FAIL or SUSPECT | **PASS** |
| P3c2, dropped Q/K correction | derived drift `0.283`; control drift `0.122` | control `>=2x` derived; derived transfers | **FAIL** |
| P3c3, unnormalised `P=A` | TRANSFERS, drift `0.144` | FAIL or SUSPECT | **FAIL** |
| P3c4, no-`D` SGD rate | FAILS, drift `0.940` | FAIL | **PASS** |
| P4, `Delta A` at `t=1` | SGD `-0.456`; signGD `-0.323` | `-0.5+-0.15`; `0+-0.15` | **FAIL** (signGD) |
| P4b, SGD moves toward 0 by `t=8` | `-0.456 -> -0.458` | absolute slope decreases | **FAIL** |
| P5, init logits | `-0.482` at `alpha_A=1`; `+0.018` at `1/2` | `-0.5+-0.10`; `0+-0.10` | **PASS** |
| P6, oversmoothing/gamma | geometric `rho: .0035 -> .0226`; `gamma_P: .3768 -> .4002` | rise; last four monotone | **PASS** |
| P7, alignment split | pooled ratio `30.035/14.605=2.056`; node ratio `1.119/2.805=0.399` | pooled `>=3`; node `<=1.5` | **FAIL** |
| S1/S2, seed/head concentration | `-0.031/-0.022` at `1/2`; `-0.531/-0.522` at 1 | within `0.10` of 0 / `-0.5` | **PASS** |

## Transfer audit and recovered artifacts

The original checkout contained completed artifacts that never reached the
public commit.  They are preserved here as `E4-transfer.json`,
`E10-power-audit.json`, and `E10.log`.  The historical `E11.log` is also kept,
but is explicitly incomplete: it ends after the width and depth legs.

The transfer harness formerly combined unpaired per-dial SEMs even though each
dial shares seeds.  `transfer.py` now uses the paired SEM of the extrema, and
the retained per-seed optima were re-scored without rerunning training in
`E4-transfer-paired.json` and `E10-power-audit-paired.json`.  The changes do not
rescue P3 or any failed control.

E10 also checks whether the harness is simply blind.  A deliberately
mis-scaled V/O rate fails sharply (drift `0.620` decades), so the harness has
power in the attention feature-update sector.  The decoder-typo control does
not reach the `0.3`-decade failure bar (`0.110`); it is not promoted as a
successful control.

## Probe corrections made during audit

Two implementation bugs affected auxiliary measurements:

1. `attention_stats` paired a layer's attention matrix with the post-MLP stream
   instead of the value tensor it actually aggregates.  E2 and E5 were rerun
   after recording the attention input/value/output tensors.  The registered
   P2/P2c/P6 verdicts are unchanged.
2. E9 compared formula (G), which assumes equal node norms, against
   heterogeneous post-block features.  It now normalises the actual input/value
   vectors to equal norm and is an algebraic identity: all six predicted/measured
   ratios are `1.0000`, including `P=sum`.  The old 1–9% discrepancy was a probe
   mismatch, not evidence about formula (G).

## What remains established

The step-0 branch scaling, stable attention regimes for `alpha_A>=1/2`, init
logit exponents, concentration/head-collapse result, and oversmoothing identity
survive their registered bars in this synthetic setup.  The stronger claim
that the complete graph-transformer parameterisation is validated does not.

Coverage remains exactly the preregistered synthetic bound: one graph family,
one task, small widths/depths, no real data, no LayerNorm, no positional or edge
features, and signGD rather than Adam in the transfer legs.  There is still no
graph-transformer DMFT solver or theory-vs-simulation dynamics comparison.
