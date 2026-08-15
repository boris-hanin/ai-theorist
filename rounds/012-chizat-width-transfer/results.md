# Round 012 — Chizat width learning-rate transfer

> **Scope correction:** the measured model in this round is a residual-state
> map and does not train product-style embed/unembed boundaries.  Its verdict
> applies only to the Chizat block parameters.  End-to-end boundary transfer
> is a separate claim preregistered in round 013.

Date: 2026-08-09

## Provenance and status

This was a user-directed follow-up after the first width-wise LR sweep had
already been observed.  There is therefore **no valid preregistration** for
this round.  The derivation, executable guard, raw result schema, common-seed
design, and mutation control are retained, but this round does not upgrade the
reconstructed `dmft-resnet-depth` skill to certified status.

Derivation: `derivations/11-chizat-width-lr-transfer.md`.
Harness: `skills/dmft-resnet-depth/scripts/chizat_lr_transfer.py`.
Shard merger: `skills/dmft-resnet-depth/scripts/merge_chizat_lr_transfer.py`.
Merged standard-JSON artifacts: `artifacts/width-h80-merged.json`,
`artifacts/width-h640-merged.json`, and `artifacts/width-h1280-merged.json`.
Per-worker shards remain ignored runtime outputs; every duplicated trial was
compared structurally before the merged artifacts were written.

## Scope

Faithful non-MoE Chizat mean-ODE residual network:

```text
h^l = h^(l-1) + alpha/(L M) sum_j w_lj tanh(u_lj . h^(l-1))
lr_raw = L M eta / alpha^2
```

Fixed `D=32`, `P=64`, `L=8`, `alpha=1`, full-batch plain GD, float64, final
training MSE, and common finite horizons.  Widths were
`M = 64,128,256,512,1024,2048`.  Five unique seeds were split over two
independent A100-SXM4-80GB workers; seed 0 was duplicated on both workers.

The primary transfer coordinate was fixed at `eta=79.43282347242814`.  The
80- and 640-step campaigns also swept 17 rates from `39.8107` through
`1584.8932`.  The 1280-step stress run evaluated only the fixed eta and its
omitted-M control, so it makes no stability-edge claim.

## Correct verdict definition

Transfer requires both:

1. absolute common-seed trajectory gaps settle toward large `M`, with the
   leading finite-width fit recorded against `M^-1/2`; and
2. fractional learning progress from step 0 has an `M^0` log-log slope.

The second condition prevents a vanishing-update rule from looking flat merely
because no width learns.  The per-width finite-horizon LR optimum and largest
finite rate are recorded separately as `edge_of_stability` diagnostics and do
not gate transfer.

## Results

| horizon | fixed-eta verdict | progress slope vs `M` | omit-M slope | control rejected | duplicate trials exact |
| ---: | --- | ---: | ---: | --- | ---: |
| 80 | **PASS** | +0.0002577 | -0.394240 | yes | 108 |
| 640 | **PASS** | +0.00000230 | -0.387197 | yes | 108 |
| 1280 | **PASS** | +0.000000108 | -0.368673 | yes | 12 |

Every one of the ten recorded fixed-eta checkpoints passed at every horizon.
At 640 steps, the adjacent absolute loss gap fell from `5.6675e-6` between the
two smallest widths to `1.0439e-9` between the two largest.  At 1280 steps it
fell from `3.1712e-7` to `3.9960e-16`.

Final mean loss by width:

| horizon | M=64 | M=128 | M=256 | M=512 | M=1024 | M=2048 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 80 | 8.1432e-3 | 7.8971e-3 | 7.5971e-3 | 7.7382e-3 | 7.6048e-3 | 7.5287e-3 |
| 640 | 6.4252e-6 | 7.5767e-7 | 2.6542e-8 | 5.7600e-9 | 4.3033e-9 | 3.2594e-9 |
| 1280 | 3.2037e-7 | 3.2475e-9 | 1.8794e-11 | 1.7348e-15 | 9.0923e-16 | 5.0963e-16 |

The large relative separation at late time is not an LR-transfer failure:
losses are near zero and relative/log ratios are ill-conditioned.  The
absolute gaps shrink by orders of magnitude, while learning progress remains
`M^0` to numerical precision.

## Stability edge at the final full-grid horizon

All 640-step local optima were interior to the 17-point grid:

| M | local best normalized eta | largest finite normalized eta |
| ---: | ---: | ---: |
| 64 | 199.526 | 199.526 |
| 128 | 316.228 | 316.228 |
| 256 | 398.107 | 398.107 |
| 512 | 630.957 | 630.957 |
| 1024 | 199.526 | 794.328 |
| 2048 | 316.228 | 1000.000 |

The non-monotone finite-horizon local optimum at the two largest widths is
exactly why it is not used as the transfer observable.  The largest finite
rate continues to increase with width, while the fixed-eta dynamics transfer.

## Verdict

For this Chizat setup, the first-principles rule

```text
lr_raw = L M eta / alpha^2,  with eta fixed across M
```

has strong width-transfer evidence through 1280 steps and `M=2048`.  No
additional fitted `M^0.6` factor is supported.  The omitted-M mutation bites,
the cross-worker identity checks are exact, and the stability edge is cleanly
separated from the transfer claim.
