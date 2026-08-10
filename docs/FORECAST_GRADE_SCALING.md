# Forecast-grade real-text scaling

`real_text_scaling_ladder` is the product path for training an exact Jiang +
Chizat or νGPT ladder and deciding whether a larger-model loss prediction is
scientifically reportable. Finishing a run does not imply that its forecast is
accepted. Every gate is explicit and the app shows withheld predictions.

## Frozen coordinate

One campaign freezes all of the following before reference tuning:

- immutable tokenizer revision, encoding canaries, vocabulary, packing, and
  verified token shards;
- architecture family, context length, head geometry, depth at every rung, and
  exact parameter-count target;
- constant presented tokens per parameter, global batch, schedule, optimizer
  moments, epsilon, seeds, and validation sample;
- one reference LR grid and every architecture-specific parameter-group rule;
- held-out rung count, fit-family spread, backtest error, corpus-repetition,
  and maximum-extrapolation limits.

Jiang ladders preserve the declared `rho = L*M/D`, tied embeddings,
`QK^T/d_head`, `1/L` residual branches, and all seven CompleteP Adam groups.
νGPT ladders preserve separate input/output embeddings, normalized hidden and
matrix geometry, iteration-aware LR groups, and the post-step sphere projection.
The latter is why νGPT accepts replicated DDP but refuses FSDP.

## Execution and recovery

Both models use PyTorch SDPA in math, automatic, or explicitly required flash
mode. Runtime options include fp32/bf16, activation checkpointing, gradient
accumulation, single-node DDP, and—where definition preserving—FSDP. Atomic
checkpoints restore model, optimizer, RNG, sampling position, validation
history, and elapsed time. FSDP resumption requires the same topology.

Public-corpus acquisition has a streaming Parquet backend. Downloads resume by
HTTP byte range, source Parquet inventory and revision are frozen, document
provenance is recorded, and raw-text plus token-shard materialization resumes
only at atomically committed boundaries. The forecast preset uses FineWeb-Edu
and the pinned OLMo 2 tokenizer.

## Forecast decision

The LR grid is tuned only at the declared reference rung. Every scale then uses
that frozen scalar coordinate through the full theory-specific parameter-group
formulas. The largest rung is hidden from fitting, and a wrong single-global-LR
run is retained as a negative control.

The fitter compares pure-power, floor-power, and broken-power laws. It runs
rolling upper-rung backtests and combines family disagreement with bootstrap
seed uncertainty. A target receives `prediction = null` when any of these fail:

- the reference LR optimum lies on the grid edge;
- the observed parameter span is below the preregistered minimum;
- any hidden upper-rung error exceeds its limit;
- the wrong-global-LR control is not worse;
- corpus repetition exceeds its limit;
- fewer than two loss-law families qualify or they disagree too much;
- a rolling backtest fails; or
- the target exceeds the maximum extrapolation factor.

The exploratory fit is retained for diagnosis but is never presented as a
certified forecast.

## Operator flow

Materialize the corpus (the operation safely resumes):

```bash
ai-theorist-autoscale corpus-materialize \
  configs/autoscaler/fineweb_edu_olmo2_forecast_corpus.json \
  --output-root runs/autoscaler/public-corpora \
  --progress-jsonl
```

Copy the completed token-stream manifest path into one of the ladder configs,
then compile it before allocating GPUs:

```bash
ai-theorist-autoscale forecast-plan \
  configs/autoscaler/jiang_olmo2_100m_ladder.json
```

Launch the resumable campaign:

```bash
ai-theorist-autoscale forecast-ladder \
  configs/autoscaler/jiang_olmo2_100m_ladder.json \
  --device cuda \
  --output runs/autoscaler/jiang-olmo2-100m \
  --progress-jsonl
```

The equivalent νGPT preset uses DDP. The web app exposes the same immutable
fields, public-corpus job, progress, hidden-rung calibration, and forecast
refusal reasons. The supplied 7M–100M stage intentionally retains the 30×
span gate: it can qualify transfer and the hidden 100M rung, but it cannot by
itself certify the displayed 1B target. Extend the frozen ladder through an
observed roughly 200M rung and reserve a 300M rung before asking that gate to
pass.

## Remaining scale boundary

This is a forecast-grade single-node campaign engine, not yet a frontier
trainer. Multi-node training, elastic world-size changes, packed-document
attention masks, fused optimizer kernels, and sequence/tensor/pipeline
parallelism remain outside the current contract. A serious 1B forecast still
requires actually running and passing the 100M ladder and its hidden upper
rung; implementation alone supplies no empirical certificate.
