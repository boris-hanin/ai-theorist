# Autoscaler MVP validation contract

## Supported product slice

The MVP is intentionally not a general neural-network DAG editor.  It accepts
two typed architectures:

`DatasetAdapter -> Embed -> repeat(pre-norm residual MLP) -> Unembed`

`DatasetAdapter -> Embed -> repeat(pre-norm top-k MoE residual block) -> Unembed`

The user may choose ReLU or GELU and declare width/depth scale levels.  MLP
blocks support SGD and Adam.  Sparse MoE blocks expose depth `L`, expert width
`M`, stream width `D`, expert count, and active-expert count; this product slice
supports Adam only.  The default MoE ladder grows all three size axes while
holding the theory-motivated joint invariant `LM/D = 4` exactly.

All scale levels share one dataset, batch size, and update horizon.  The only
tuned hyperparameter is the normalized learning-rate coordinate `eta`.  The
raw optimizer rates are derived from the declared parameterization at every
scale.  For MLPs, Adam uses `lr_raw = eta` and SGD uses
`lr_raw = eta / sqrt(D)`.  MoE Adam uses group rates: adapters, norms, and
readout bias use `eta`; readout, router, and expert-up weights use `eta/D`;
expert-down weights use `eta/M`.  Chizat mean-field SGD uses
`lr_raw = L M eta / alpha^2`.  None of these rules is inferred by fitting a
width-wise finite-horizon LR optimum.

AdamW, Muon, attention, arbitrary skip edges, data scaling, and token-horizon
scaling are out of scope until this slice is calibrated.

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
4. An optimizer/architecture-specific negative control is rejected:
   square-root LR growth for MLP Adam, constant LR for MLP SGD, or one global
   raw rate matched to the reference MoE router/up rate.
5. At least four non-holdout scales support
   `L(C) = L_inf + A C^(-alpha)`.
6. Loss decreases with compute, improvement clears the noise floor, the fit has
   `R^2 >= 0.9`, and the floor/exponent are not pinned or ill-conditioned.
7. The completely held-out largest scale falls inside the uncertainty
   tolerance and has at most 10% point-prediction error.
8. For MoE, each scale's mean across seeds of the per-run worst-expert load
   deviation is at most 0.25, and no individual run exceeds 0.50.  Routing
   health is a forecast gate, not an informational dashboard metric.

Any failed gate produces explicit refusal reasons and no next-scale number.
Wide uncertainty cannot turn a point-prediction miss into a pass.

## Automated validation layers

- Schema/compiler: strict unknown-field rejection, supported-block and
  optimizer checks, independent MoE `L/M/D` axes, increasing-compute ladder,
  fixed horizon, and model/plan parameter-count agreement.
- Optimizers: one-step parity for SGD and bias-corrected Adam.
- Training: deterministic final-loss replay, exact checkpoint continuation,
  accumulation/full-batch equivalence including router-load aggregation,
  divergence reporting, and strict final-horizon accounting.
- Tuning/transfer: normalized-to-raw LR conversion, edge expansion,
  divergent-candidate rejection, common-seed paired SEM, fixed-`eta`
  non-inferiority, a separate edge-of-stability report, and negative control.
- Scaling: exact synthetic-law recovery, flat/non-monotone refusal, bootstrap
  interval stability, and held-out calibration.
- API: health, strict compile, asynchronous launch, polling, result persistence,
  CORS for the local workbench, and invalid-spec errors.
- UI: component click and drag, fixed-node inspection, invalid-ladder blocking,
  MLP/MoE block selection, `L/M/D` editing with an `LM/D` warning, invalid
  optimizer/ladders blocked, immutable controls during a run, real API
  progress, refusal rendering, desktop/mobile layout, and browser-console
  cleanliness.
- Hardware: the same CUDA canary runs on both A100s, repeated GPU runs must be
  deterministic, and CPU/GPU loss drift is measured before full campaigns.

Run the local suite with `pytest`.  The browser build is validated from
`apps/web` with lint, TypeScript, the production build, and rendered-HTML tests.

## A100 campaigns

The checked-in campaigns are:

- `configs/autoscaler/a100_adam.json`
- `configs/autoscaler/a100_sgd.json`
- `configs/autoscaler/a100_moe_adam.json`
- `configs/autoscaler/a100_moe_adam_extensive.json`

The MLP campaigns use six increasing width/depth levels, 16,384 fixed training
examples, 600 fixed updates, batch size 256, four common seeds, adaptive
reference tuning in normalized `eta`, largest-scale fixed-`eta` probes, a
wrong-scaling control, 400 bootstrap fits, and a fully held-out sixth scale.
SGD and Adam are separate studies; a passing optimizer does not certify the
other.

The extensive MoE campaign uses six `LM/D=4` levels, 1,024 fixed training
examples, 256 validation examples, 320 updates, batch size 64, six common
seeds, six normalized-eta candidates, a held-out sixth scale, a wrong-global-
rate control, 400 bootstrap fits, and router-load gates.  The smaller MoE file
is the fast A100 smoke campaign; it is not the release-evidence substitute.

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

For a one-coordinate transfer screen without tuning, `screen` now selects the
architecture's parameterization and records the MoE group rates and routing
imbalance rather than silently treating every block as a standard MLP.

The result directory contains an immutable input manifest and strict JSON
result.  Raw campaign outputs remain runtime artifacts unless deliberately
promoted into a research round with provenance and an acceptance decision.
