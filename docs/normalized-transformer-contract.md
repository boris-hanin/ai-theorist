# Normalized Transformer v1 contract

Status: implementation contract for the first normalized-Transformer autoscaler slice.

## Scientific question

At fixed dataset, context length, token horizon, batch size, optimizer family, and
learning-rate schedule, does the 2026 νGPT mid-alignment parameterization preserve
one tuned base Adam learning rate across joint increases in depth and embedding width,
and does fixed-horizon validation cross-entropy admit a calibrated held-out scaling
law?

This study does not yet claim batch-size or token-horizon transfer, even though νGPT
provides a token-horizon prescription. It also does not claim Muon transfer or the
separate 2026 Training nGPT optimizer recipe.

## Architecture

The model is a decoder-only causal language model with:

- untied input and output token embeddings;
- standard rotary position embeddings;
- a fixed even head dimension, so width scaling grows the number of attention
  heads as in the paper's primary width-transfer intervention;
- no LayerNorm or RMSNorm;
- unit-normalized hidden states after embedding and after each attention/MLP update;
- unit-normalized embedding, output, Q/K/V, attention-output, MLP-input, and
  MLP-output vectors along their embedding dimension;
- normalized queries and keys, followed by the nGPT `sqrt(head_dim)` attention
  softmax scale;
- a 4x SwiGLU MLP;
- learned coordinate-wise attention and MLP interpolation rates, initialized at
  `0.05 * (L/L_ref)^-1`, with trainable representations initialized at 0.03;
- learned query/key scales initialized effectively at one with trainable
  representations initialized at 0.03;
- learned logit scales initialized effectively at `(D/D_ref)^1/2`, with trainable
  representations initialized at 0.03;
- learned MLP scales initialized at one.

Every constrained matrix is projected to the unit sphere before training and after
every optimizer step. The forward pass also re-normalizes hidden, query, and key
vectors to prevent numerical drift.

## Training protocol

- Objective: next-token cross-entropy.
- Initial data source: one deterministic noisy second-order Markov language shared
  by every scale and seed.
- Optimizer: Adam with beta1 0.9, beta2 0.95, epsilon `1e-16`, zero weight decay.
- Learning-rate schedule: cosine decay from the tuned peak to 10% of peak; no warmup.
- Tuned coordinate: one base peak learning rate `eta` on the reference scale.
- At width multiplier `m_D = D/D_ref`, input embeddings use `eta*m_D^-1/2`,
  hidden matrices use `eta*m_D^-3/4`, output weights use
  `0.5*eta*m_D^-3/4`, and scalar/vector rescalers use `eta`.
- Gradient clipping: the existing safety ceiling, held fixed across scale.
- Batch size, context length, number of update steps, and presented tokens remain
  fixed across the scale ladder.

The synthetic task is an integration and transfer test, not evidence about natural
language. A real token corpus becomes a separate, preregistered validation stage.

## Required diagnostics

Each trial records:

- final validation cross-entropy;
- maximum constrained-matrix norm error;
- maximum hidden-state norm error;
- mean causal-attention entropy;
- mean learned attention and MLP interpolation rates;
- mean learned logit scale;
- all raw optimizer learning rates, seed, duration, and memory use.

The forecast gate rejects a run if matrix or hidden-state norm error exceeds `1e-5`,
in addition to the existing learning-rate transfer, held-out calibration, tuning
interiority, and negative-control gates.

## Validation ladder

1. Algebraic unit tests: RoPE shapes, causal masking, parameter count, exact unit
   norms, untied embeddings, and optimizer projection.
2. CPU integration tests: deterministic data/training, microbatch equivalence,
   checkpoint resume, finite gradients, and strict JSON artifacts.
3. Transfer smoke test: five small scales, paired seeds, fixed base `eta`,
   conservative and aggressive held-out probes, and baseline nGPT's incorrect
   single-global-learning-rate rule as a negative control.
4. A100 confirmation: repeat the preregistered ladder in bfloat16 and float32
   reference modes; compare loss curves and sphere diagnostics across devices.
5. Natural-language confirmation: freeze a tokenizer/corpus split and repeat the
   same held-out procedure before treating the model family as product-certified.

## Deferred work

The separate 2026 Training nGPT additions—logit-gradient preconditioning, logarithmic
learning-rate decay, GatedAdamW, angular-update control, and exploration—are a
second parameterization. They must not be mixed into this baseline because doing so
would confound architecture transfer with optimizer and schedule transfer.

Primary references:

- `nGPT: Normalized Transformer with Representation Learning on the Hypersphere`
  (arXiv:2410.01131)
- `Learning Rate Transfer in Normalized Transformers` (arXiv:2604.27077), the
  governing νGPT transfer parameterization
- NVIDIA `ngpt` reference implementation
- `Training nGPT` (arXiv:2608.01284), deferred recipe
