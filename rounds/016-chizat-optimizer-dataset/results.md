# Round 016 results — Chizat optimizer-by-dataset matrix

## Decision

The broad nine-cell product claim **fails** under the committed gates.  No cell
earns a validation-loss forecast.  The failure is informative rather than an
execution failure: all 1,330 trials across the two A100 replicas are finite and
complete, and the two workers reproduce every scientific verdict.

The narrower optimizer-transfer conclusions are:

- Fixed-eta final-loss non-inferiority passes in all nine cells.
- Adam's semantic rates pass the complete transfer package—interior tuning,
  settled trajectories, non-inferiority, and the constant-unembed control—on
  all three datasets.
- Muon passes that complete transfer package on linear and
  sinusoid-plus-quadratic tasks.  Its tanh-teacher final loss transfers and its
  wrong-W-rate control bites, but the validation trajectories do not settle.
- SGD does not pass the complete package on any dataset: the linear and tanh
  controls do not bite, while tanh and sinusoid-plus-quadratic select a
  boundary eta.
- The current fixed-data, 600-update `LM/D=8` ladder does not support a fully
  accepted loss forecast for any optimizer/task cell.  The gates are retained.

## Cell verdicts

`Transfer` below is the preregistered largest-scale final-loss non-inferiority
check.  `Trajectory`, `control`, `fit`, and `holdout` remain independent gates.

| Cell | selected eta | interior | trajectory | transfer | control | fit R2 | holdout error | holdout | forecast |
|---|---:|:---:|:---:|:---:|:---:|---:|---:|:---:|:---:|
| SGD / linear | 0.09 | yes | yes | yes | **no** | 0.9886 | 0.029% | yes | no |
| SGD / tanh | 0.27 | **no** | yes | yes | **no** | 0.7386 | 1.197% | yes | no |
| SGD / sin+quad | 0.27 | **no** | yes | yes | yes | 0.3798 | 0.105% | yes | no |
| Adam / linear | 0.01 | yes | yes | yes | yes | -0.0964 | 9.052% | **no** | no |
| Adam / tanh | 0.01 | yes | yes | yes | yes | -0.1388 | 6.180% | yes | no |
| Adam / sin+quad | 0.003 | yes | yes | yes | yes | 0.9191 | 0.398% | yes | no |
| Muon / linear | 0.03 | yes | yes | yes | yes | -0.0140 | 3.734% | yes | no |
| Muon / tanh | 0.01 | yes | **no** | yes | yes | -0.0028 | 2.439% | yes | no |
| Muon / sin+quad | 0.03 | yes | yes | yes | yes | 0.0725 | 0.387% | yes | no |

Adam/linear's point error is below the absolute 10% ceiling but falls outside
its uncertainty tolerance, so the held-out gate correctly rejects it.
Adam/sin+quad and SGD/linear have usable fit-only short-range laws, but the
former fails after incorporating the holdout into the final refit and the
latter fails its negative control.  Neither issues a forecast.

## Negative controls

Seven of nine optimizer-specific controls are rejected.  The control-minus-
primary paired increases for the three Muon cells are `0.0003124` (linear),
`0.0004710` (tanh), and `0.0120347` (sin+quad), all above their committed
paired thresholds.  Adam rejects constant-unembed mutations on all tasks.
SGD rejects the wrong-global-LMD mutation only on sin+quad; on tanh the control
is actually better by `0.001598`, so that cell cannot support the proposed SGD
semantic rule.

## Reproduction

Both A100-SXM4-80GB workers ran commit
`1858f2cecdc66606f6b63cfb338948926f84d1c6`.  Each passed all 81 repository
tests and an identical CUDA Chizat-Muon smoke before the matrix.  Each matrix
contains 665 trials, or 399,000 optimizer steps at the fixed 600-step horizon.

The matrix manifests, matrix summaries, trial losses, traces, selected rates,
raw semantic rates, controls, held-out values, and verdicts are identical.
Three ill-conditioned fits differ only in the last digits of the reported
condition number; removing timestamps, wall durations, and that diagnostic
scalar makes all nine structured results identical.  These tiny condition-
number differences do not change a gate.

The byte-identical manifest SHA-256 is
`508201acde3943b2d1614d7a03a39892734c71e5b1463a6e2188c5eb27c23cf3`.
The byte-identical matrix-summary and worker-log SHA-256 is
`4ed18d7a520d792c1e6e9cd4bea6ac6b1ddd83903850f62dcaab17c225b76305`.

## Next experiment

Do not relax the gates or retrofit the eta grids.  The next clean study should
separate the already-successful optimizer-transfer question from the failed
fixed-horizon loss-law question: preserve the committed semantic rates, add a
freshly preregistered trajectory screen for Muon/tanh and SGD controls, then
design a capacity path/horizon whose fit scales are demonstrably outside the
optimization floor before spending another full matrix.
