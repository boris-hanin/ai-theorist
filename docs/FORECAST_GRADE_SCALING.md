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
and the pinned 32,768-token Mistral v0.3 tokenizer. An already verified raw
snapshot can be retokenized without reacquiring source data.

Long trials may checkpoint by optimizer-step cadence, wall-clock cadence, or
both. The production presets use a 15-minute timer instead of checkpointing
every 100 steps; the latter would spend an unacceptable fraction of a large
run rewriting optimizer state. The exact timer is part of the trial identity.

## Independent GPU fleets

The forecast campaign has a two-phase physical trial DAG. Phase `tune` runs
every reference learning-rate/seed probe. Only after those exact cache records
are complete does the controller select the preregistered mean-loss optimum.
Phase `ladder` then runs every remaining rung/seed pair and the largest-rung
wrong-global-LR controls. The selected reference trials are reused rather than
trained twice, so the supplied three-seed, five-rate, seven-rung campaign has
36 physical trials and 39 logical analysis records.

Tasks are assigned by deterministic longest-processing-time balancing using
the declared `6*N*T` FLOP estimate. Each shard is a single-process worker with
an immutable trial cache, so it works on independent one-GPU machines and does
not require a multi-node process group. Interrupted shards safely replay their
assignment and immediately reuse completed trials.

Compile the balanced assignments without allocating a GPU:

```bash
ai-theorist-autoscale forecast-tasks CONFIG \
  --phase tune --shard-count 2
```

Run tuning shards independently, gather their `trials/*.json` caches, and
select the frozen learning rate:

```bash
ai-theorist-autoscale forecast-shard CONFIG \
  --phase tune --shard-index 0 --shard-count 2 \
  --device cuda --output RUN/tune-0 --progress-jsonl

ai-theorist-autoscale forecast-select CONFIG \
  --cache-directory RUN/tune-0/trials \
  --cache-directory RUN/tune-1/trials \
  --output RUN/reference-selection.json
```

Use the selected rate for every ladder shard. `forecast-aggregate` first
proves that every expected cache filename and record contract is present; it
refuses incomplete or stale fleets before invoking the canonical analysis.

The checked-in 100M preset targets independent GPU workers: explicit bf16
FlashAttention, fused Adam, one process per trial, vectorized memory-mapped
token sampling, activation checkpointing, and 15-minute atomic checkpoints.
Single-node DDP/FSDP remains available through the web builder when one trial
actually benefits from model sharding.

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
  configs/autoscaler/fineweb_edu_mistral_forecast_corpus.json \
  --output-root runs/autoscaler/public-corpora \
  --progress-jsonl
```

Bind the completed manifest to a checked-in template. Binding verifies every
token shard and tokenizer asset, compiles all forecast gates, and atomically
writes a deployable config; a placeholder or undersized corpus never reaches a
GPU:

```bash
ai-theorist-autoscale forecast-bind \
  configs/autoscaler/jiang_mistral_100m_forecast.json \
  runs/autoscaler/public-corpora/CORPUS/token-streams/manifest.json \
  --output runs/autoscaler/jiang-mistral-100m.json
```

Launch the resumable campaign:

```bash
ai-theorist-autoscale forecast-ladder \
  runs/autoscaler/jiang-mistral-100m.json \
  --device cuda \
  --output runs/autoscaler/jiang-mistral-100m \
  --progress-jsonl
```

Before trusting a multi-GPU topology, compare the same short campaign on one
GPU and two-GPU DDP. The qualification preserves the global sample draw and
partitions it by rank, then checks task coordinates, per-group learning rates,
validation coordinates, selected reference LR, and loss parity:

```bash
scripts/qualify_forecast_2gpu_ddp.sh \
  configs/autoscaler/jiang_mistral_100m_forecast.json \
  runs/autoscaler/mistral-corpora/CORPUS/token-streams/manifest.json \
  runs/autoscaler/ddp-qualification
```

On one node with eight GPUs, production uses one independent trial worker per
GPU rather than data-parallelizing small models. The controller freezes the
plan and balanced assignments, completes all tuning workers, selects the LR,
then completes the ladder workers and refuses partial aggregation:

```bash
scripts/run_forecast_8gpu_fleet.sh \
  runs/autoscaler/jiang-mistral-100m.json \
  runs/autoscaler/jiang-mistral-8gpu
```

For the current two-independent-A100 topology,
`scripts/launch_two_a100_forecast_pair.sh` waits for corpus completion, checks
destination capacity, streams only the immutable token shards and tokenizer to
the second worker, proves dataset-fingerprint parity by binding on both hosts,
and launches Jiang + Chizat and νGPT under nonduplicating file locks. It does
not copy credentials to either worker, and both remote campaigns remain
resumable after the launcher exits.

The equivalent νGPT preset uses the same independent-worker execution. The web app exposes the same immutable
fields, public-corpus job, progress, hidden-rung calibration, and forecast
refusal reasons. The default Jiang ladder compiles exact rungs from 2.2M to
100M parameters; its fit rungs span more than 30× before the hidden 100M rung.
The displayed 1B target is a bounded roughly 10× extrapolation, but it remains
withheld until the real run passes LR, transfer-control, backtest,
family-agreement, and repetition gates.

## Remaining scale boundary

This is a forecast-grade independent-trial fleet and single-node model engine,
not yet a frontier trainer. Multi-node model training, elastic world-size
changes, packed-document attention masks, and sequence/tensor/pipeline
parallelism remain outside the current contract. A serious 1B forecast still
requires actually running and passing the 100M ladder and its hidden upper
rung; implementation alone supplies no empirical certificate.
