# From transfer assays to real-text scaling-law forecasts

## Decision summary

The completed A100 campaigns establish two useful but narrower facts:

1. The normalized Transformer transfers its peak learning rate across the tested
   token horizons with the `T^(-1/3)` rule for cosine, linear warmup/decay, and
   warmup-stable-decay schedules.
2. The Jiang-MHSA + Chizat-FFN model transfers the CompleteP per-parameter-group
   learning-rate parameterization across the tested model ladder at fixed
   duration, but its source-faithful half-warmup/constant schedule does **not**
   inherit the normalized Transformer's `T^(-1/3)` horizon rule.

These results are enough to justify a larger, carefully staged scaling-law
campaign. They are not yet enough to publish reliable loss predictions at 1B,
5B, or 10B parameters. The immediate engineering target should be a real-text
ladder up to roughly 100M parameters, followed by at least one genuinely hidden
200M--300M validation rung before certifying a 1B forecast. Predictions at 5B
and 10B should remain explicitly labeled scenario bands until additional
intermediate anchors make their extrapolation auditable.

The main obstacle is not whether either architecture can instantiate a 100M
model. Both can. The obstacles are the training path, data and token budget,
efficient runtime, and uncertainty contract needed to distinguish a useful
pretraining law from a clean but severely undertrained pilot.

## What the A100 evidence actually says

Both token-horizon assays used FineWeb-Edu sample-10BT fingerprint
`666710b377c444e7c0354dc3496d4375adcb482a5d65d086db86a8cfa61315e1`,
16,384 frozen training windows, 1,024 frozen validation windows, and seeds
`11, 29, 47`. Candidate rules were fit on 65,536 through 524,288 presented
tokens and frozen before evaluating the hidden 1,048,576-token horizon.

### Normalized Transformer

The 1,121,028-parameter model completed 351 A100 trials. The `T^(-1/3)` rule
passed both the 2% transfer-regret gate and the mechanism-discrimination gate
for every tested schedule:

| Schedule | Frozen-rule loss | Held-out grid-oracle regret | Flat-control loss |
| --- | ---: | ---: | ---: |
| Cosine | 1.74412 | 0.400% | 1.90235 |
| Linear warmup/decay | 1.73327 | 0.000% | 1.82388 |
| Warmup-stable-decay | 1.73637 | 0.202% | 1.83352 |

The fitted exponents were `0.460`, `0.494`, and `0.432`, respectively. They are
diagnostics, not replacements for the preregistered one-third rule. The old
`0.32` sensitivity point is retained only for provenance: this horizon span and
seed count cannot meaningfully distinguish `0.32` from `1/3`, so the product no
longer presents them as competing hypotheses.

### Jiang-MHSA + Chizat-FFN

The 306,688-parameter model completed 114 A100 trials while retaining all seven
CompleteP groups: embeddings, norms, attention QKV, attention output, FFN up,
FFN down, and other biases. Their respective calibrated multipliers were `1`,
`1`, `0.0625`, `0.5`, `1`, `0.0625`, and `1`; the duration candidate scaled the
already-parameterized groups together rather than replacing their distinct
model-scale formulas.

At the hidden horizon, the grid oracle was `eta=0.03` with loss `2.07072`. The
flat rule achieved loss `2.07174`, only `0.049%` above that oracle. The
`T^(-1/3)` rule achieved loss `2.13133`, or `2.927%` regret, and failed the 2%
gate. A fitted exponent of `-0.652` had only `R^2=0.353` and was correctly
refused because it predicted increasing optimal learning rate with horizon.

This is a qualified negative result, not a broken CompleteP implementation.
CompleteP model-scale transfer at fixed duration and duration transfer under a
particular schedule are different claims. The first passed at the tested
scales; the second did not.

## What has not yet been established

The current evidence does not establish any of the following:

- that model-size, token-horizon, and batch rules compose on real text;
- that either family follows one stable loss law from sub-million models to
  100M parameters;
- that a law fit below 100M remains calibrated at 1B, 5B, or 10B;
- that fixed-token, constant-tokens-per-parameter, and compute-optimal scaling
  answer the same question;
- that the present byte tokenizer and context length produce a representative
  modern pretraining loss.

Hyperparameter transfer removes a major retuning confound. It does not, by
itself, imply that validation loss follows a particular power law.

## Why the present pilot is too small for frontier-style forecasts

The materialized corpus contains 67,156,577 training byte tokens and 8,395,537
validation byte tokens, using `byte_v1` at context length 64. The existing
FineWeb Jiang model-scale run presented only 307,200 tokens to every model. At
its 6.48M-parameter top rung, that is about `0.047` presented tokens per
parameter. It is a good LR-transfer assay and a poor proxy for a trained
language-model scaling curve.

For illustration, a constant `T/N = 20` path requires about 2B presented tokens
at 100M parameters. A 1B, 5B, or 10B target on the same path corresponds to
20B, 100B, or 200B tokens. Extrapolating from a 100M/2B endpoint is therefore a
joint extrapolation in model size and token horizon, not merely a prediction
for a wider model. Repeating the current 67M-token slice roughly thirty times
at the 100M rung would also make repetition a dominant experimental variable.

The product must therefore ask which scaling path a forecast means:

- **Fixed `T`:** a useful diagnostic, but larger models become increasingly
  undertrained and the curve is not a frontier-pretraining forecast.
- **Constant `T/N`:** the clean first product for studying transfer and loss
  scaling together.
- **Compute-optimal `N,T`:** the eventual pretraining planner, requiring a
  two-dimensional allocation law rather than a one-dimensional size fit.

Constant `T/N` should be the default for the next campaign, with fixed `T`
retained as a labeled control.

## Existing scale capacity

With the current 260-token byte vocabulary, the Jiang ladder that preserves
`L*M/D = 4` already reaches a natural approximately 100M canary:

| `L` | `M` | `D` | Parameters |
| ---: | ---: | ---: | ---: |
| 2 | 128 | 64 | 87,808 |
| 2 | 256 | 128 | 306,688 |
| 4 | 256 | 256 | 1,666,560 |
| 4 | 512 | 512 | 6,478,848 |
| 8 | 384 | 768 | 23,901,696 |
| 8 | 544 | 1,088 | 47,787,136 |
| 8 | 768 | 1,536 | 94,989,312 |

The current normalized Transformer can likewise instantiate the following
engineering ladder:

| Depth | Width | Parameters |
| ---: | ---: | ---: |
| 4 | 128 | 1,121,028 |
| 4 | 256 | 4,338,948 |
| 6 | 384 | 14,381,060 |
| 8 | 512 | 33,865,988 |
| 8 | 640 | 52,818,180 |
| 8 | 768 | 75,964,676 |
| 8 | 896 | 103,305,476 |

These counts are engineering canaries, not final campaign definitions. A
modern 50k-class tokenizer changes the embedding contribution substantially:
the normalized Transformer has separate input and output embeddings, while the
Jiang decoder ties them. All target parameter counts must therefore be
recomputed after freezing the tokenizer and context contract.

A 100M model is plausible on one 80GB A100. Runtime throughput, the token
budget, and the number of seeds and LR probes are the binding costs. The current
efficient pretraining path supports bf16, PyTorch SDPA/FlashAttention, and
single-node FSDP, but the theory-faithful normalized and Jiang runners do not
yet use that path. They still need bf16/SDPA integration, activation
checkpointing, gradient accumulation, resumable optimizer checkpoints, and a
measured tokens-per-second benchmark. The two available one-GPU hosts cannot
exercise the existing single-node FSDP implementation as a multi-GPU job, and
multi-node rendezvous is not implemented.

## Required product and statistical work

The current generic study path restricts normalized-Transformer scaling studies
to synthetic Markov data. The real-text horizon runners hold model size fixed,
and the Jiang real-text model ladder is a specialized campaign rather than a
first-class web study. These paths must be unified before the web app can launch
and audit the proposed campaign.

The current loss fitter uses estimated compute `C = 6NT`, fits
`L(C) = E + A C^(-alpha)`, and forecasts one adjacent compute step. Its bootstrap
mostly propagates seed noise. It does not yet represent model-form uncertainty
or support explicit 1B/5B/10B parameter targets. Long-range forecasts need:

- target coordinates `(N,T,B,architecture,optimizer,schedule)` rather than a
  bare parameter count;
- rolling upper-rung holdouts and at least one completely hidden larger model;
- an explicit maximum extrapolation factor;
- comparison of plausible floor-power, pure-power, and broken-power fits;
- prediction intervals that widen with extrapolation distance and include
  model-form disagreement, not just seed error;
- refusal when fit families disagree, a target is too far beyond the largest
  observation, or transfer/control gates fail.

## Staged campaign

### Stage 0: freeze the production contract

The tokenizer portion is implemented: `olmo2_1124` is pinned to an immutable
revision, verified assets and encoding canaries, and document-delimited uint32
shards whose packing identity enters every job. The remaining Stage 0 choices
are the larger disjoint corpus snapshot, context length, optimizer, schedule,
and scaling path. Record unique corpus tokens, presented tokens, repetition,
updates, and tokens per parameter separately. Preserve the Jiang CompleteP
group formulas and the normalized Transformer's own parameterization in full.

### Stage 1: unify and benchmark the runtime

Move both theory-faithful models onto the real-text bf16 SDPA/FlashAttention
path without changing their parameterization. Add gradient accumulation,
activation checkpointing, mid-trial resume, and throughput/memory telemetry.
Numerically compare FP32 reference and bf16 accelerated updates on small models,
including every Jiang parameter group, before accepting the fast path.

### Stage 2: qualify composition below 25M

Run model-size, token-horizon, and batch composition on real text with at least
three seeds. Use a modest constant-`T/N` ladder first, reserve the largest rung,
and retain flat-LR, schedule, and fixed-`T` controls. The normalized Transformer
may use the qualified one-third horizon rule as a candidate. Jiang must treat
flat duration scaling as the present evidence-backed default and test new
duration rules rather than importing one-third.

### Stage 3: extend the observed ladder to about 100M

Use six to eight model rungs spanning at least 30x, with one or two upper rungs
hidden from all fitting and selection. Tune at the declared reference scale and
only use preregistered diagnostic probes elsewhere. The campaign should not
advance if optima hit LR-grid boundaries, transfer controls fail, validation
loss is non-monotone beyond uncertainty, or repetition becomes material.

### Stage 4: validate the extrapolator

Fit all candidate laws below the hidden rung and score their predictive loss,
interval coverage, rank ordering, and calibration at that rung. Repeat as
rolling backtests. A successful 100M endpoint should trigger a hidden
200M--300M confirmation run, not an immediate 10B claim.

### Stage 5: publish bounded forecasts

After the 200M--300M holdout passes, publish a 1B forecast with its full
training coordinate, interval, extrapolation factor, and model-family spread.
Publish 5B and 10B only as clearly labeled scenarios until additional anchors
reduce their distance and the competing fits agree. Never display a single
loss number without those qualifications.

## Go/no-go contract

A target forecast is certifiable only if all of the following hold:

1. the architecture-specific HP parameterization passes its transfer gates;
2. the chosen horizon and batch rules pass real-text composition tests;
3. the observed ladder spans at least 30x in parameters and has multiple
   training-loss improvements larger than seed noise;
4. every reserved upper-rung backtest passes a preregistered error and coverage
   threshold;
5. corpus repetition remains below the declared limit;
6. fit families agree within the reported interval;
7. the target lies within the declared extrapolation factor.

Until then, the web app should show the fit as an exploratory scenario and say
exactly which gate prevents certification.

## Bottom line

There is no architectural or memory barrier to starting a real-text ladder up
to approximately 100M parameters for either family. There is a scientific
barrier to treating the current pilot as evidence for an accurate 1B, 5B, or
10B loss. A credible 1B forecast is a realistic next milestone after the
runtime/data integration, 100M ladder, and hidden 200M--300M backtest. The 5B
and 10B forecasts require more anchors or must remain broad scenario bands.
