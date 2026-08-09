# Round 016 preregistration draft — Chizat optimizer-by-dataset matrix

## Provenance status

This is a **draft protocol, not yet a preregistration**.  It must be committed
together with the product implementation and matrix manifest before any round-
016 A100 cell begins.  Results produced before that commit are exploratory and
cannot be relabeled as formal evidence.

## Question

Does one normalized learning-rate coordinate transfer across the fixed Chizat
`(L,M,D)` ladder, and does the resulting fixed-horizon loss law predict a fully
held-out largest model, for each cell of:

```text
optimizer = SGD, Adam, Muon
dataset   = linear, tanh teacher, sinusoid-plus-quadratic
```

Each cell is an independent claim.  The broad optimizer-by-dataset claim passes
only if all nine cells pass; a failure is retained and does not erase passing
cell-level claims.

## Architecture and immutable boundary contract

Every cell uses the bias-free trained model

```text
h0 = x E
hl = h(l-1) + 1/(L M) tanh(h(l-1) Ul) Wl
f  = hL R
```

with `E ~ N(0,1/d0)`, `U ~ N(0,1/D)`, `W ~ N(0,1)`, and
`R ~ N(0,1/D^2)`.  Embed and unembed are trained in every trial.  Semantic
optimizer routing—not tensor rank—determines their optimizer and rate.

The six shapes keep `LM/D = 8`:

| Scale | D | L | M | Role |
|---|---:|---:|---:|---|
| S1 | 8 | 2 | 32 | fit |
| S2 | 18 | 3 | 48 | fit |
| S3 | 32 | 4 | 64 | reference/fit |
| S4 | 72 | 6 | 96 | fit |
| S5 | 128 | 8 | 128 | fit |
| S6 | 200 | 10 | 160 | sealed holdout |

## Optimizer coordinates

At one normalized `eta`, raw group rates are:

| Optimizer | embed | U | W | unembed |
|---|---:|---:|---:|---:|
| SGD | `D eta` | `L M eta / D` | `L M D eta` | `eta / D` |
| Adam | `eta` | `eta` | `sqrt(D) eta` | `eta / D` |
| Muon + auxiliary Adam | `eta` | `eta` | `sqrt(D) eta` | `eta / D` |

Muon applies only to U/W.  It uses momentum `.95`, Nesterov, five float32
Newton--Schulz steps, RMS matching, and zero weight decay.  Embed/unembed use
auxiliary Adam with betas `(.9,.95)` and epsilon `1e-10`.  Adam uses betas
`(.9,.999)` and epsilon `1e-8`; SGD uses zero momentum.

These rows are preregistered transfer hypotheses.  In particular, the Adam
and Muon semantic rates are not treated as certified before their fresh matrix
cells pass the gates below.

## Data and horizon

- Generator version: 1.
- Train/validation examples: 16,384 / 4,096.
- Dataset seed: 4242; model/training seeds: `101,211,307,419,523`.
- Updates: 600; batch size: 256; microbatch size: 128.
- Data count and update horizon remain fixed across every scale.
- Data-size and horizon scaling are explicitly out of scope.

## Tuning and held-out protocol

Eta is tuned only on S3.  The numerical optimum must be interior after at most
two factor-three edge expansions.  Among candidates within one SEM of the
numerical minimum, select the lowest eta.  Freeze it before evaluating the
remaining scales.  S6 is excluded from eta selection and scaling-law fitting.

The target is final validation loss at update 600, never best-so-far loss.
Validation loss is also recorded at fixed checkpoints to test trajectory
settling without consulting per-scale eta optima.

## Cell acceptance

A cell passes only if all are true:

1. The S3 numerical eta optimum is interior.
2. All selected-eta runs are finite and complete exactly 600 updates.
3. Fixed-eta validation trajectories settle across the scale ladder after
   step 1.
4. At S6, transferred eta is non-inferior to the preregistered lower
   conservative probe under common-seed paired uncertainty.
5. The optimizer-specific negative control is rejected by a paired S6 final-
   loss increase exceeding `max(2 SEM, 1% of primary loss)`:
   - SGD: one global `LMD` block rate;
   - Adam: omit the `1/D` unembed factor;
   - Muon: replace `sqrt(D) eta` for W by `D eta`.
6. At least four fit scales support the fixed-horizon law and its diagnostics
   clear the existing monotonicity, noise-floor, conditioning, and `R^2 >= .9`
   gates.
7. S6 point-prediction error is at most 10% and falls inside the declared
   uncertainty tolerance.

Failure of transfer and failure of the loss forecast are reported separately.
A valid transfer result cannot rescue a bad scaling law, and a good loss fit
cannot rescue failed transfer.

## Reproducibility and execution

The immutable manifest is
`configs/autoscaler/a100_chizat_optimizer_dataset_matrix.json`.  Its nine cells
must execute sequentially per GPU; concurrent cells on one accelerator are
forbidden.  A second A100 repeats common-seed cells for hardware replication.
CPU smoke runs, if any, use separate output paths and are not round-016
evidence.
