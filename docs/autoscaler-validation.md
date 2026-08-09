# Autoscaler MVP validation contract

## Supported product slice

The MVP is intentionally not a general neural-network DAG editor.  It accepts
three typed residual architectures:

`DatasetAdapter -> Embed -> repeat(pre-norm residual MLP) -> Unembed`

`DatasetAdapter -> Embed -> repeat(Chizat particle block) -> Unembed`

`DatasetAdapter -> Embed -> repeat(sparse MoE block) -> Unembed`

The Chizat block uses tanh and independently declared representation width D,
depth L, and particle width M.  It supports SGD, Adam, and the locally tested
Muon-plus-auxiliary-Adam candidate contract.  Muon is deliberately rejected
for the standard MLP and MoE because no transfer rule has been validated
there.  All
scale levels share one named dataset task, batch size, and update horizon.  The
only tuned hyperparameter is the normalized learning-rate coordinate `eta`.
The raw optimizer LR is derived from the declared parameterization at every
scale.  Chizat always reports four semantic rates for trained embed, U, W, and
unembed; it never compresses them into an ambiguous global raw LR.  The three
versioned task families are linear, tanh-teacher, and sinusoid-plus-quadratic
regression.  AdamW, attention, arbitrary skip edges, data scaling, and token-
horizon scaling remain out of scope.

The optimization target is final validation loss at the declared horizon, not
best-so-far loss.

This contract is schema version 2.  Its tuning key is
`normalized_learning_rates`; schema-v1 `learning_rates` is rejected rather
than silently reinterpreted.

## Forecast acceptance gates

A study may issue a next-scale forecast only when all of these are true:

1. The reference normalized-`eta` optimum is interior to the adaptively
   expanded search.
2. Every LR comparison uses the same random seeds.
3. The run at fixed normalized `eta` is finite and is non-inferior, under a
   common-seed paired test, to a lower conservative-`eta` probe.  A more
   aggressive probe performing better does not fail transfer; it is reported
   separately as width-dependent stability headroom.
4. Fixed-eta validation-loss trajectories settle across the scale ladder.
5. An optimizer-specific negative control is rejected.  Chizat uses a global
   LMD block rate for SGD, constant unembed rate for Adam, and `D eta` W rate
   for Muon.
6. At least four non-holdout scales support
   `L(C) = L_inf + A C^(-alpha)`.
7. Loss decreases with compute, improvement clears the noise floor, the fit has
   `R^2 >= 0.9`, and the floor/exponent are not pinned or ill-conditioned.
8. The completely held-out largest scale falls inside the uncertainty
   tolerance and has at most 10% point-prediction error.

Any failed gate produces explicit refusal reasons and no next-scale number.
Wide uncertainty cannot turn a point-prediction miss into a pass.

## Automated validation layers

- Schema/compiler: strict unknown-field rejection, supported-block and
  optimizer checks, increasing-compute ladder, fixed horizon, and model/plan
  parameter-count agreement.
- Optimizers: one-step parity for SGD and bias-corrected Adam; Muon square,
  rectangular, transpose, momentum/Nesterov, RMS-adjustment, semantic-routing,
  and checkpoint-continuation tests.
- Training: deterministic final-loss replay, exact checkpoint continuation,
  accumulation/full-batch equivalence, divergence reporting, and strict final
  horizon accounting.
- Tuning/transfer: normalized-to-raw LR conversion, edge expansion,
  divergent-candidate rejection, common-seed paired SEM, fixed-`eta`
  non-inferiority, a separate edge-of-stability report, and negative control.
- Scaling: exact synthetic-law recovery, flat/non-monotone refusal, bootstrap
  interval stability, and held-out calibration.
- API: health, strict compile, asynchronous launch, polling, result persistence,
  CORS for the local workbench, and invalid-spec errors.
- UI: component click and drag, trained-boundary inspection, invalid-ladder blocking,
  immutable controls during a run, real API progress, refusal rendering,
  desktop/mobile layout, and browser-console cleanliness.
- Hardware: the same CUDA canary runs on both A100s, repeated GPU runs must be
  deterministic, and CPU/GPU loss drift is measured before full campaigns.

Run the local suite with `pytest`.  The browser build is validated from
`apps/web` with lint, TypeScript, the production build, and rendered-HTML tests.

## A100 campaigns

The checked-in campaigns are:

- `configs/autoscaler/a100_adam.json`
- `configs/autoscaler/a100_sgd.json`
- `configs/autoscaler/a100_chizat_optimizer_dataset_matrix.json`

The standalone Adam and SGD studies use six increasing width/depth levels,
16,384 fixed training examples, 600 fixed updates, batch size 256, four common
seeds, adaptive reference LR tuning in normalized `eta`, largest-scale
fixed-`eta` probes, a wrong-scaling control, 400 bootstrap fits, and a fully
held-out sixth scale.  SGD and Adam are separate studies; a
passing optimizer does not certify the other.

The Chizat matrix expands strictly to nine cells: three optimizers by three
versioned datasets.  It uses five common seeds and otherwise preserves the
same fixed data count, update horizon, batch size, bootstrap count, and sealed
sixth scale.  Cells execute sequentially on each accelerator to avoid
contention.  Its Adam and Muon semantic-rate rules remain hypotheses to test,
not certified conclusions.  Its round-016 protocol is a draft until both it
and the implementation are committed; no matrix result should be treated as
formal evidence before that point.

The Adam ladder grows depth more aggressively.  The first SGD screen showed
that the same 2/3/4/6/8/12 depth ladder becomes optimization-limited at the
fixed horizon even when the horizon is tripled; its checked-in ladder therefore
uses 2/3/4/4/5/6 repeats while preserving strictly increasing compute.  This is
an optimizer-specific validated regime, not a claim that arbitrary depth/width
paths scale smoothly under SGD.

Example:

```bash
ai-theorist-autoscale run configs/autoscaler/a100_adam.json \
  --device cuda --output runs/autoscaler/a100-adam-v1 \
  --summary --progress-jsonl
```

Preview the matrix without training:

```bash
ai-theorist-autoscale matrix-plan \
  configs/autoscaler/a100_chizat_optimizer_dataset_matrix.json
```

The result directory contains an immutable input manifest and strict JSON
result.  Raw campaign outputs remain runtime artifacts unless deliberately
promoted into a research round with provenance and an acceptance decision.
