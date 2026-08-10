# Per-parameter learning-rate transfer contracts

This document is the launch gate for transfer experiments. A campaign is
eligible only when its forward parameterization, initialization, optimizer
groups, optimizer constants, schedule, manual updates, and transfer axes match
one of the contracts below. Every runtime manifest must also prove that every
trainable tensor belongs to exactly one named optimizer group.

## Supported matrix

| Architecture | Optimizer | Transfer axes | Base-scale tuning | Raw group-rate law |
|---|---|---|---|---|
| Fixed-depth μP residual MLP | Adam | width only | one normalized `eta` | input/vector/output `eta`; hidden matrices `eta / m_width`; MuReadout forward factor `1 / m_width` |
| Fixed-depth μP residual MLP | SGD | width only | one normalized `eta` | input/vector/output-weight `eta * m_width`; square hidden matrices `eta`; output bias `eta` |
| Chizat equation-(22) 2LP ResNet | full-batch GD | joint `L`, `M`, `D` | independent `eta_u`, `eta_v` | `lr_U = eta_u L M D`; `lr_V = eta_v L M D` |
| νGPT normalized Transformer | Adam | width and depth at fixed token horizon | global `eta`; cited output multiplier `1/2` | input `eta m_width^-1/2`; hidden `eta m_width^-3/4`; output `(eta/2) m_width^-3/4`; rescalers `eta`; no depth LR factor |
| Jiang sparse MoE decoder | Adam | `L`, `D`, `M`, `E` at fixed `A/E` | global `eta` plus the Appendix-D.1 base constants | Table 2 of arXiv:2601.20205v3, including group-specific Adam epsilon |
| Dense Jiang-attention/Chizat-FFN hybrid | Adam | joint `L`, `M`, `D` | global `eta` plus the applicable Appendix-D.1 base constants | dense specialization of Jiang Table 2; explicitly a derived architecture, not the published sparse MoE |

`m_width`, `m_depth`, and similar symbols are target/base ratios. A
scale-independent fitted multiplier is part of the base model; it is not a
replacement for the theoretical scale factor.

For every campaign, the reference is the smallest model in the scaling ladder.
The deployable reference learning rate is the smallest fully finite grid point
whose paired-seed mean validation loss is within 10% of the reference model's
grid oracle. This conservative near-optimal choice is made from reference-model
data only. Transfer succeeds only if that fixed choice remains within 10% of
each larger model's separately swept oracle; exact discrete argmin equality is
reported as a diagnostic rather than treated as a hard gate.

## Fixed-depth μP residual MLP

Source: Tensor Programs V, arXiv:2203.03466v2, and the official
`microsoft/mup` optimizer and `MuReadout` implementation.

- The depth, dataset, batch size, residual multiplier, and training horizon are
  fixed. There is no MLP depth-transfer claim.
- The readout divides its input by `width / reference_width`.
- MuAdam and MuSGD use different tensor-type rules; a single global LR is a
  negative control.
- The runtime protocol is Adam `(0.9, 0.95, epsilon=1e-8)` or momentum-free
  SGD, zero weight decay, constant LR, and no gradient clipping.

Implementation: `skills/dmft-resnet-depth/scripts/mup_mlp_transfer.py`.

## Exact Chizat 2LP mean-ODE ResNet

Source: *The Hidden Width of Deep ResNets*, arXiv:2509.10167v2, equations
(22)-(23), plus the authors' Julia notebook.

- `h_0 = W_in x`, with fixed `W_in`.
- `h_l = h_(l-1) + (1/(L M)) sum_j v_(j,l)
  tanh(u_(j,l)^T h_(l-1) / D)`.
- `f = W_out^T h_L / D`, with fixed `W_out`.
- `U` and `V` are initialized iid entrywise with standard deviation `sqrt(D)`
  in the critical MLU regime.
- `eta_u` and `eta_v` are tuned independently at the reference shape. Their
  raw full-batch GD rates are `eta_u L M D` and `eta_v L M D`.
- There is no momentum, weight decay, schedule, warmup, adaptive
  preconditioning, or gradient clipping in the finite-network equation-(23)
  experiment.
- The tested joint domain enforces `D = O(L M)`. The preferred comparator holds
  `L M / D` constant. Separate controls omit `L`, `M`, or `D`, or hold the raw
  reference rates fixed.

This object is a mean-ODE residual-particle model. It is not an MLP depth
extension of the μP contract.

Implementation: `src/ai_theorist/autoscaler/chizat_resnet.py` and
`skills/dmft-resnet-depth/scripts/chizat_equation22_transfer.py`.

## νGPT

Source: *Learning Rate Transfer in Normalized Transformers*,
arXiv:2604.27077v2, Table 1 and the experimental protocol.

- Untied input/output embeddings; no LayerNorm or RMSNorm.
- Hidden states, attention/MLP branch outputs, and the prescribed matrix axes
  are projected to the unit sphere.
- Causal RoPE attention uses normalized Q/K and the `sqrt(d_head)` softmax
  multiplier, with fixed `d_head` and head count proportional to width.
- Each block is attention LERP followed by SwiGLU LERP. The initial attention
  and MLP interpolation strengths are `0.05 m_depth^-1`; their raw rescaler
  coordinates initialize at `0.03`.
- Q/K rescalers initialize with effective value `1`; MLP rescalers with `1`;
  the logit rescaler initializes with effective value `sqrt(m_width)` and raw
  scale `0.03`.
- Adam uses `(0.9, 0.95, epsilon=1e-16)`, zero weight decay, no warmup, cosine
  decay from peak to 10%, and no gradient clipping.
- The current certified campaigns keep iteration count/token horizon fixed, so
  `m_data=1`. Horizon transfer must additionally apply `m_data^-1/3` and is not
  part of the present launch.

The web trial result stores the complete group audit and optimizer protocol.

## Jiang sparse MoE

Source: *Hyperparameter Transfer with Mixture-of-Experts Layers*,
arXiv:2601.20205v3, Sections 3.1-3.3, Table 2, Appendices A and D.1.

- Tied token embedding/unembedding, learned absolute positions, pre-LN,
  causal `QK^T/d_head` attention, fixed head dimension, and interleaved MHSA
  and MoE residual branches each scaled by `1/L`.
- Router: sigmoid gates, hard no-gradient top-`A` set, fixed
  `kappa=A/E`, and mixing divided by `A`.
- Expert up weights initialize at `D^-1/2`; down weights at
  `(1/4) sqrt(D)/M` after applying the reported constant-scale multiplier;
  expert biases initialize at zero. Router initialization obeys
  `D^-gamma`, `gamma >= 1/2`.
- Attention Q/K initialize at `D^-1/2`, while V uses the reported `1/16`
  initialization multiplier. QKV and output weights are distinct optimizer
  groups because only QKV receives the reported LR multiplier.
- Adam groups are embeddings, norms, attention QKV, attention output, router,
  expert up, expert down, and other biases. Both LR and epsilon use the exact
  Table-2 ratios.
- The Appendix-D.1 LR constants are QKV `1/16`, router `1/16`, and expert down
  `1/16`, with all other group constants equal to one. Reference probes are
  relative checks around these source values; the values are never silently
  replaced by an all-one setup.
- The non-Adam expert routing bias is a separate zero-initialized buffer with
  update `b_i <- b_i - eta_bias (Load_i-kappa)` and constant `eta_bias` at
  fixed sparsity.
- Adam uses `(0.9,0.95)`, base epsilon `1e-12`, zero weight decay, no gradient
  clipping, linear warmup for the first half of the fixed-token run, then a
  constant peak LR.
- Scale-independent group constants are verified or refined only on the
  reference model. Scaling exponents are never fitted.

Implementation: `src/ai_theorist/autoscaler/jiang_moe.py` and
`skills/dmft-moe/scripts/jiang_moe_transfer.py`.

## Dense Jiang-attention/Chizat-FFN hybrid

This is a preregistered derived architecture used to isolate interleaved MHSA
and dense mean-field FFN behavior. It must never be described as the published
Jiang sparse-MoE architecture.

- Boundaries are tied token embedding/unembedding, learned absolute positions,
  and a final affine LayerNorm.
- Each block is affine pre-LN, causal MHSA, then affine pre-LN and a dense GELU
  FFN. Linear biases use the dedicated Table-2 bias group. Both residual
  branches use `1/L`.
- Attention uses fixed `d_head`, `QK^T/d_head`, and `D^-1/2` matrix
  initialization, with the reported `1/16` V initialization multiplier. FFN
  up uses `D^-1/2`; FFN down uses `(1/4) sqrt(D)/M`.
- Adam groups and epsilons are the dense specialization of Jiang Table 2. QKV
  and FFN-down LR constants are `1/16`, all other constants are one, and QKV
  is separated from the attention output projection. The constants are checked
  only at the reference scale; the schedule and Adam constants match the Jiang
  fixed-token protocol.

Implementation: `src/ai_theorist/autoscaler/jiang_chizat.py` and
`skills/dmft-attention/scripts/jiang_chizat_tuned_transfer.py`.

## Explicitly unsupported or demoted results

- Any standard residual-MLP campaign that varies depth under the μP label.
- Any campaign that transfers a single raw global LR across parameter groups.
- The legacy simplified `pre_norm_moe` web model as evidence for the full Jiang
  sparse-MoE theory.
- The legacy Chizat scripts whose trained boundaries or reparameterized rates
  do not implement equations (22)-(23) literally.
- Partial block transplantation without the source architecture's boundaries,
  initialization, residual normalization, optimizer grouping, epsilon rules,
  schedule, and manual updates.
