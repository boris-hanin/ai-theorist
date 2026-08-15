# Round 017 preregistration — Jiang attention with Chizat FFNs

## Scope and source contract

This round tests the dense, interleaved core of the decoder architecture in
Jiang–Bordelon–Pehlevan–Hanin (arXiv:2601.20205v3, §3.1 and Table 2), replacing
each MoE by one dense mean-field FFN.  It is not a claim about sparse routing.
The fixed-data, fixed-token-horizon limit is taken while depth `L`, FFN particle
width `M`, and residual width `D` vary.  Per-head dimension is fixed and the
head count is `D / d_head`.

For every layer, in this order,

```text
h_attn = h + (1/L) MHSA(LN(h))
h_next = h_attn + (1/L) FFN(LN(h_attn))
```

MHSA uses causal masking and `Q K^T / d_head`, as in the paper.  The dense FFN
uses

```text
u = GELU(h W_up)
FFN(h) = u W_down
std(W_up) = D^(-1/2)
std(W_down) = sqrt(D) / M
```

so its hidden-width sector is mean-field/Chizat rather than fan-in.  Token
embedding and unembedding are tied; learned absolute positions and pre-LayerNorm
match the paper's experiment setup.

## Primary optimizer coordinate

The first implementation uses Adam with `(beta1, beta2) = (0.9, 0.95)`, zero
weight decay, and one normalized base coordinate `eta`.  Relative to a declared
reference shape `(L0, M0, D0)`, Table 2 gives these peak learning rates:

```text
embedding/unembedding, LayerNorm: eta
attention Q/K/V/O:               eta * (D/D0)^(-1)
FFN up:                           eta * (D/D0)^(-1)
FFN down:                         eta * (M/M0)^(-1)
```

The per-group Adam epsilons retain the paper's scale exponents, normalized to a
declared `epsilon0` at the reference shape:

```text
embedding: epsilon0 * (D/D0)^(-1)
LayerNorm: epsilon0
attention: epsilon0 * (D/D0)^(-1) * (L/L0)^(-1)
FFN up:    epsilon0 * (M/M0)^(-1) * (L/L0)^(-1)
FFN down:  epsilon0 * (D/D0) * (M/M0)^(-2) * (L/L0)^(-1)
```

No SGD transfer claim is made by this preregistration.  A plain-GD composite
requires an independent induced-feature-velocity audit for the attention
matrices before it can inherit the older Chizat block rates.

## Primary joint path

The primary joint ladder holds

```text
rho = L M / D
```

constant.  This is preregistered as the cleanest path based on round 014.  Pure
`L`, pure `M`, pure `D`, and a nonconstant-rho joint ladder are secondary arms.

## Measurements and falsifiers

All measurements use common seeds, an identical synthetic autoregressive task,
fixed datapoints, fixed context, fixed batch size, and fixed optimizer steps.

1. Initialization: logits, residual-stream RMS, attention entropy, and loss are
   finite and remain `O(1)` across every pure axis.
2. Step-0 audit: group-only induced feature velocity is fit against `L`, `M`,
   and `D`.  A primary exponent is accepted only when every intended flat slope
   has absolute value at most `0.15` after the smallest shape is excluded.
3. Transfer: at one reference-tuned fixed `eta`, common-seed loss trajectories
   must show nontrivial progress at every shape and final log-progress slope
   versus the declared dial must have absolute value at most `0.30`.
4. The constant-`rho` joint arm must be at least as flat as the nonconstant-rho
   arm within two paired standard errors; otherwise the preferred-path claim is
   rejected.
5. Controls must bite: standard fan-in `W_down`, omitted attention width factor,
   omitted FFN hidden-width factor, omitted depth branch factor, and disabled
   attention are each reported separately.  A control that is empirically
   identical makes the associated test under-powered, not confirmed.
6. Attention movement distinguishes individual-logit freezing from population
   concentration by reporting per-entry movement and seed-to-seed movement of
   the head-averaged attention kernel separately after steps 1, 2, and final.
7. Largest-scale validation is held out until all rate choices and tolerances
   above are fixed.

Local CPU/CUDA smoke runs establish only implementation correctness.  They are
not evidence for transfer.
