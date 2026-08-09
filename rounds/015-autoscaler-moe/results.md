# Round 015 — sparse-MoE Autoscaler validation

## Status

**Pass for the declared reduced-product regime.**  This is a retrospective
product-validation round, not a formal certification of the full MoE DMFT
skill.  The final two-part routing gate and controller value were fixed before
the final two-worker campaign; the failed controller settings are retained.

## Declared regime

- Architecture: `Embed -> repeat(pre-norm top-1-of-4 MoE) -> Unembed`.
- Activation: GELU.
- Optimizer: Adam with the Table-1 parameter groups.
- Group rates: adapters/norms/readout bias `eta`; readout/router/up `eta/D`;
  down `eta/M`.
- Shapes: `(D,L,M) = (8,2,16), (18,3,24), (32,4,32), (50,5,40),
  (72,6,48), (98,7,56)`.
- Joint path: `LM/D=4` exactly at every scale.
- Fixed resources: 1,024 training examples, 256 validation examples, batch
  size 64, 320 optimizer steps.
- Replication: six common seeds and two independent A100-SXM4-80GB workers.
- Budget: 84 trials per worker.

## Final verdict

| gate | measurement | verdict |
|---|---:|---|
| reference tuning | `eta=0.1`, interior, non-flat | pass |
| fixed-eta transfer | penalty vs conservative probe `0.01526 ± 0.00830`; allowed `0.01659` | pass, narrowly |
| wrong single-global-rate control | paired loss increase `0.07808 ± 0.00845` | rejected (`9.24` SEM) |
| fit-only scaling law | `R^2=0.98810` | pass |
| held-out S6 | predicted `0.16703`, observed `0.17650`, error `5.36%` | pass |
| routing mean gate | worst scale mean deviation `0.20703 <= 0.25` | pass |
| routing hard gate | worst individual deviation `0.33203 <= 0.50` | pass |

The final all-scale fit has `R^2=0.98948`.  The issued adjacent-scale forecast
is final validation loss `0.17009` with 95% interval `[0.15993, 0.17748]` at
`1.8385x` the largest observed compute.

The transfer pass is intentionally reported as narrow: the selected fixed eta
is non-inferior to the conservative probe under the declared common-seed
two-SEM tolerance, but it is close to that boundary.  This is not evidence for
arbitrarily long extrapolation.  The product emits only the adjacent forecast.

## Exact two-worker reproducibility

After removing timestamps, durations, and tiny condition-number roundoff, the
complete result JSONs are identical.  All 84 trial losses, loss traces,
routing loads, raw group rates, and transfer/control records match exactly.
The checked-in `artifacts/result.json` is the canonical worker result.

## Router-controller falsification trail

Loss alone did not expose the controller failures:

| balance rate | scope | worst load deviation | routing verdict |
|---:|---|---:|---|
| 0.01 | 36-trial fixed-eta screen | 0.71484 | fail: under-correction/collapse |
| 0.03 | 36-trial fixed-eta screen | 0.40234 | fail: isolated collapse |
| **0.1** | 84-trial two-worker campaign | **0.33203** | **pass** |
| 0.3 | 84-trial two-worker campaign | 0.49609 | fail: mean gate |
| 1.0 | 84-trial two-worker campaign | 0.60547 | fail: oscillation/hard gate |

The final rule gates the mean across seeds of each run's worst-expert deviation
at 0.25 and separately caps every individual run at 0.50.  A single
worst-seed cutoff was rejected because it conflates reproducible imbalance
with finite-sample outliers; a mean-only cutoff was rejected because it can
hide an individual collapse.  Both channels are required.

## Scope boundary

This round validates one fixed-sparsity, fixed-data, fixed-horizon regime and
one adjacent loss forecast.  It does not certify other expert counts, active
fractions, datasets, horizons, Adam epsilon regimes, SGD, AdamW, Muon, or a
full MoE dynamics closure.  Changing any of those requires a new calibration.
