# Real-text Transformer pretraining runtime

This is the operational baseline for batch-scaling experiments on real token
streams. It is deliberately separate from the normalized-Transformer theory
runner: the baseline is an ordinary GPT-style pre-norm decoder, while the
νGPT runner remains an explicit mechanistic control.

## Model contract

`StandardTransformer` has learned token and position embeddings, repeated
pre-norm causal multi-head self-attention and GELU MLP blocks, a final layer
norm, and a tied unembedding by default. Attention always goes through PyTorch
scaled-dot-product attention (SDPA). The model/configuration fingerprint,
corpus fingerprint, exact token counts, batch geometry, optimizer state, loss
checkpoints, estimated FLOPs, wall time, and peak CUDA memory are recorded for
every trial.

The initial optimizer is AdamW with a linear warmup and cosine decay. The
underlying trial runner also accepts Adam and SGD so comparative campaigns do
not require another training stack.

## Dataset formats

Two deterministic formats are supported:

- `byte_v1`: UTF-8 `.txt` files or newline-delimited JSON with a configurable
  text field. The fixed vocabulary contains 256 bytes and four control tokens.
- `uint16_bin_v1`: a headerless little-endian `uint16` token stream. It is
  memory mapped and intended for a production tokenizer prepared outside the
  app. Set the model vocabulary large enough for the greatest token ID.

Training and validation must be separate files. Each pair is content hashed,
and the hash is part of every cache key. The bundled files under
`data/pretraining/` are only a deterministic end-to-end fixture; they are not
a research corpus. A serious run should point to a versioned pretokenized
corpus and retain its tokenizer vocabulary/configuration beside the run.

## Runtime modes

The precision contract is `fp32` or `bf16`. CUDA bf16 is rejected below compute
capability 8.0. CPU bf16 is useful only for functional validation.

The attention contract is:

- `math`: force the SDPA math backend for deterministic smoke tests;
- `auto`: let PyTorch choose a compatible SDPA kernel;
- `flash`: explicitly require the SDPA FlashAttention backend. This requires
  CUDA, bf16, and a head dimension divisible by eight. A missing compatible
  kernel is an error, not a silent fallback.

`fsdp` is single-node data parallelism launched through `torchrun`. Each
Transformer block is an FSDP wrapping unit, model states synchronize from rank
zero, parameters/gradients/buffers use the configured mixed-precision policy,
and global batch size must divide the process count. Rank zero performs the
definition-preserving checkpoint-continuation and per-example gradient-noise
assays while peer ranks wait; these assays intentionally do not inherit
data-parallel gradient averaging.

## Commands

Validate a campaign without training:

```bash
ai-theorist-autoscale pretrain-plan \
  configs/autoscaler/pretraining_text_smoke.json
```

Run the complete real-text critical-batch census locally:

```bash
ai-theorist-autoscale pretrain-census \
  configs/autoscaler/pretraining_text_smoke.json \
  --output runs/autoscaler/pretraining-text-smoke \
  --progress-jsonl
```

Run the supplied two-GPU A100 runtime profile:

```bash
ai-theorist-autoscale pretrain-census \
  configs/autoscaler/a100_pretraining_text_fsdp.json \
  --device cuda \
  --output runs/autoscaler/a100-pretraining-text-fsdp \
  --progress-jsonl
```

Replace the fixture paths before treating the A100 profile as a scientific
campaign.

`a100_pretraining_text_single_gpu.json` is the conservative one-GPU canary.
The adjacent `*_target3.json` and `*_target28.json` manifests are retained
threshold/optimizer stress cases for AdamW, Adam, and SGD; they should not be
pooled as independent scaling-law observations because they reuse the same
corpus, seeds, model ladder, and token budgets.

## Web jobs and resumption

The Batch scaling workspace can launch all three neural campaigns:

- `standard_pretraining_census` for real text;
- `transformer_census` for the normalized synthetic control;
- `constant_tpp` for the held-out constant-tokens-per-parameter test.

For the real-text census the web plan exposes AdamW, Adam, and SGD, along with
the target validation loss and checkpoint cadence.  These fields are part of
the immutable job identity because the steps-to-target estimator is only
meaningful when crossings have both adequate coverage and measurable dynamic
range.  Loading a persisted job restores the exact optimizer, target, runtime,
tokenizer, and corpus paths used by that job.

`POST /api/batch/jobs` accepts `campaign`, `config`, and `device`. `GET
/api/batch/jobs/{id}` returns persisted progress and results. Job IDs are a
hash of that immutable request. Reposting an identical completed or active
request returns the same job; reposting after interruption restarts the worker
and reuses each atomically completed trial. Manifests, job state, trial caches,
and final results live under the API's `batch-jobs/` run root.
The identity also contains an explicit interpretation version.  That version
is bumped whenever an estimator or qualification change would make an older
completed result unsafe to reuse for the same request.

## Current boundary

This runtime is intentionally single-node. It does not yet implement
multi-node rendezvous, activation checkpointing, gradient accumulation,
sequence/tensor/pipeline parallelism, fused optimizers, data-loader workers,
packed-document masks, checkpoint consolidation, or elastic/preemptible
recovery inside a single trial. Job-level trial resumption is implemented;
mid-trial resumption is not. Those are required before the app represents a
full modern large-scale pretraining stack.
The web selector targets the machine hosting the Autoscaler API; it is not an
SSH scheduler or remote-cluster control plane.  Run the API on the intended GPU
node (or add a separately authenticated dispatcher) before selecting CUDA.
