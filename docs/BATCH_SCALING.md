# Batch scaling and constant-TPP validation

The batch subsystem is deliberately separate from the forecast path. It may
recommend a static optimizer transform before it may recommend a dynamic batch
schedule, and it never treats `B_opt`, a steps-to-target critical batch, a
checkpoint-continuation transition, and a gradient-noise scale as synonyms.

## Canonical run contract

Every trial emits a versioned `BatchRunRecord` containing:

- model family, width, depth, parameter count, seed, and device metadata;
- total non-padding tokens, global batch tokens, microbatch tokens,
  accumulation, replicas, and optimizer steps;
- learning rate, momentum or Adam moments, epsilon, weight decay, and their
  implied half-lives in token units;
- validation checkpoints, target-loss crossing steps, estimated FLOPs, and
  wall time.

The invariant

`global batch tokens = microbatch tokens × accumulation × data-parallel replicas`

is checked when a record is constructed and again when it is reloaded.

## Transfer-rule registry

Inspect the registry:

```bash
ai-theorist-autoscale batch-rules
```

The registry contains the fixed negative control, SGD linear batch scaling,
the Adam SDE square-root rule, the Complete(d)P joint batch/duration rule, an
exact-beta token-half-life control, and a fitted horizon power law. Every
result includes formula multipliers, assumptions, a citation label, and
refusal reasons. Invalid beta transforms are refused rather than clipped.

## Campaign order

Run the inexpensive mechanistic calibration first:

```bash
ai-theorist-autoscale batch-quadratic \
  configs/autoscaler/batch_quadratic_smoke.json \
  --output runs/autoscaler/batch-quadratic-smoke/result.json
```

Then run the matched SGD/Adam Transformer census:

```bash
ai-theorist-autoscale batch-census \
  configs/autoscaler/batch_census_smoke.json \
  --output runs/autoscaler/batch-census-smoke/result.json
```

Finally run the held-out constant-TPP transfer test:

```bash
ai-theorist-autoscale batch-tpp \
  configs/autoscaler/batch_tpp_smoke.json \
  --output runs/autoscaler/batch-tpp-smoke/result.json
```

The Transformer campaigns write every completed trial atomically to their
configured cache directory. Re-running the same command reloads those trials;
an interrupted campaign resumes instead of retraining completed cells.

All three neural campaigns are also first-class persistent web jobs. The Batch
scaling workspace launches them, polls their progress, and renders qualified
critical-batch consensus without bypassing refusal reasons. The same job
identity resumes the same per-trial cache after a service interruption.

For A100 work, use `a100_batch_census.json` and `a100_batch_tpp.json` with
`--device cuda`. The largest TPP scale is excluded from fitting. Its oracle
grid is retained only to measure regret; the transfer rule sees no held-out
loss before it is evaluated.

## Critical-batch qualification

The three estimators are:

1. A Zhang-style `S(B) = a + b/B` fit with a bracketed 20% overhead point.
2. A direct common-checkpoint continuation assay at matched token budgets.
3. A debiased gradient-noise scale in per-token units.

Each estimator has its own coverage and fit checks. Consensus requires at
least two qualified estimators and a default maximum disagreement ratio of
2×. The loss-optimal batch is reported separately after independent learning-
rate tuning.

## Seesaw safety gate

`batch-seesaw` consumes a qualified consensus estimate and a piecewise
baseline learning-rate schedule. It remains locked unless:

- critical-batch consensus has qualified;
- late training has been identified as variance dominated;
- the initial and proposed batches remain below 80% of the consensus estimate;
- no single learning-rate cut exceeds the configured staged limit.

The compiler also emits an intentionally aggressive `batch × cut²` negative
control. That control is labeled and is never returned as a recommendation.

```bash
ai-theorist-autoscale batch-seesaw \
  configs/autoscaler/batch_seesaw_example.json
```

## Interpretation

Constant TPP means `total tokens / parameter count` is held fixed. Rounding to
whole optimizer steps is allowed only when the realized TPP spread remains
inside the configured tolerance. The joint Adam coordinate is

`q = (target batch / base batch) / (target tokens / base tokens)`.

The Complete(d)P transform implemented here is

`eta' = eta sqrt(q)`, `epsilon' = epsilon / sqrt(q)`, and
`1 - beta_i' = q (1 - beta_i)`.

The normalized synthetic campaigns intentionally run SGD and Adam. The
separate real-text GPT baseline uses AdamW by default and can also run Adam or
SGD. See `PRETRAINING_RUNTIME.md` for its tokenizer, bf16, FlashAttention,
single-node FSDP, and operational-boundary contract.

For the fixed-model composition of empirically calibrated horizon effects with
full Adam batch transforms, including a separate composition cross-check and a
doubly held-out corner, see `JOINT_HORIZON_BATCH_TRANSFER.md`.
