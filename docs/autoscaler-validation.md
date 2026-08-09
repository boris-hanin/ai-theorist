# Autoscaler MVP validation contract

## Supported product slice

The MVP is intentionally not a general neural-network DAG editor.  It accepts
one typed architecture:

`DatasetAdapter -> Embed -> repeat(pre-norm residual MLP) -> Unembed`

The user may choose ReLU or GELU, width/depth scale levels, and either SGD or
Adam.  All scale levels share one dataset, batch size, and update horizon.  The
only tuned hyperparameter is the global base learning rate.  Adam transfers
that rate unchanged; SGD applies the fixed `sqrt(D_ref / D)` parameterization.
AdamW, Muon,
attention, arbitrary skip edges, data scaling, and token-horizon scaling are
out of scope until this slice is calibrated.

The optimization target is final validation loss at the declared horizon, not
best-so-far loss.

## Forecast acceptance gates

A study may issue a next-scale forecast only when all of these are true:

1. The reference LR optimum is interior to the adaptively expanded search.
2. Every LR comparison uses the same random seeds.
3. The transferred LR is no more than 0.3 decades from the largest-scale local
   probe optimum, with no significant paired loss penalty.
4. An optimizer-specific negative control is rejected: square-root LR growth
   for Adam, or constant LR for SGD.
5. At least four non-holdout scales support
   `L(C) = L_inf + A C^(-alpha)`.
6. Loss decreases with compute, improvement clears the noise floor, the fit has
   `R^2 >= 0.9`, and the floor/exponent are not pinned or ill-conditioned.
7. The completely held-out largest scale falls inside the uncertainty
   tolerance and has at most 10% point-prediction error.

Any failed gate produces explicit refusal reasons and no next-scale number.
Wide uncertainty cannot turn a point-prediction miss into a pass.

## Automated validation layers

- Schema/compiler: strict unknown-field rejection, supported-block and
  optimizer checks, increasing-compute ladder, fixed horizon, and model/plan
  parameter-count agreement.
- Optimizers: one-step parity for SGD and bias-corrected Adam.
- Training: deterministic final-loss replay, exact checkpoint continuation,
  accumulation/full-batch equivalence, divergence reporting, and strict final
  horizon accounting.
- Tuning/transfer: edge expansion, divergent-candidate rejection, common-seed
  paired SEM, boundary reporting, and negative control.
- Scaling: exact synthetic-law recovery, flat/non-monotone refusal, bootstrap
  interval stability, and held-out calibration.
- API: health, strict compile, asynchronous launch, polling, result persistence,
  CORS for the local workbench, and invalid-spec errors.
- UI: component click and drag, fixed-node inspection, invalid-ladder blocking,
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

Each uses six increasing width/depth levels, 16,384 fixed training examples,
600 fixed updates, batch size 256, four common seeds, adaptive reference LR
tuning, largest-scale transfer probes, a wrong-scaling control, 400 bootstrap
fits, and a fully held-out sixth scale.  SGD and Adam are separate studies; a
passing optimizer does not certify the other.

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

The result directory contains an immutable input manifest and strict JSON
result.  Raw campaign outputs remain runtime artifacts unless deliberately
promoted into a research round with provenance and an acceptance decision.
