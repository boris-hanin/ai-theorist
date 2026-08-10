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
- `uint32_bin_v1`: the equivalent legacy 32-bit local stream. Like the uint16
  path, it pins content but not the tokenizer that created it.
- `sharded_uint32_le_v1`: a manifest-backed stream created by an allow-listed
  pinned tokenizer. The manifest verifies every asset, canary, special-token
  ID, packing choice, and token shard before a job receives an identity.

Training and validation must be separate files. Each pair is content hashed,
and the hash is part of every cache key. The bundled files under
`data/pretraining/` are only a deterministic end-to-end fixture; they are not
a research corpus. A serious run should point to a versioned pretokenized
corpus and retain its tokenizer vocabulary/configuration beside the run. Raw
binary formats are reported as unpinned and cannot support a certified result.
The complete approved-tokenizer and sharding contract is in
`docs/TOKENIZER_CONTRACT.md`.

The web campaign workspace can now prepare a frozen FineWeb-Edu sample-10BT or
OpenWebText snapshot without manually entering server paths.  The materializer
uses allow-listed dataset identities, disjoint source-row ranges, a recorded
Hub revision, per-file SHA-256 checks, and a final token-stream fingerprint.
See `docs/REAL_TEXT_TRANSFER.md` for the acquisition and Jiang-transfer
contract.

## Runtime modes

The precision contract is `fp32` or `bf16`. CUDA bf16 is rejected below compute
capability 8.0. CPU bf16 is useful only for functional validation.

The attention contract is:

- `math`: force the SDPA math backend for deterministic smoke tests;
- `auto`: let PyTorch choose a compatible SDPA kernel;
- `flash`: explicitly require the SDPA FlashAttention backend. This requires
  CUDA, bf16, and a head dimension divisible by eight. A missing compatible
  kernel is an error, not a silent fallback.

`ddp` and `fsdp` are single-node data-parallel modes launched through
`torchrun`. DDP keeps a complete replica on each GPU. FSDP wraps each
Transformer block and shards its state. Global batch size must divide
`num_processes * gradient_accumulation_steps`. Activation checkpointing is
applied at the block boundary when requested.

An atomic runtime checkpoint can be written every declared number of optimizer
steps, every declared number of wall-clock seconds, or whichever cadence fires
first. It contains the model, optimizer, CPU/CUDA and sampler RNG states, loss
checkpoints, and elapsed time. DDP writes one complete state per rank; FSDP
writes one sharded state per rank and therefore requires the same world size on
resume. Completed trials still use the independent immutable trial cache.

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

The Batch scaling workspace can launch the neural campaigns, including:

- `standard_pretraining_census` for real text;
- `transformer_census` for the normalized synthetic control;
- `constant_tpp` for the held-out constant-tokens-per-parameter test;
- `real_text_scaling_ladder` for exact Jiang or νGPT constant-T/P ladders with
  hidden upper rungs and forecast refusal gates.

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

This model runtime is intentionally single-node. It now implements activation
checkpointing, gradient accumulation, fused Adam for the forecast path, and
same-topology mid-trial recovery,
but it does not yet implement multi-node rendezvous, sequence/tensor/pipeline
parallelism, data-loader workers, packed-document attention
masks, FSDP checkpoint consolidation, or elastic world-size changes. Those are
required before the app represents a full frontier-scale pretraining stack.
The web selector targets the machine hosting the Autoscaler API; it is not an
SSH scheduler or remote-cluster control plane.  Run the API on the intended GPU
node (or add a separately authenticated dispatcher) before selecting CUDA.
