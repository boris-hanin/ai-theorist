# Jiang–Chizat interleaved transformer contract

## What is inherited, and what is new

Jiang et al. (2601.20205v3) actually train a decoder with alternating pre-LN
MHSA and MoE residuals.  Their theoretical reduction removes attention.  The
repo's earlier Chizat rounds likewise use residual FFNs without MHSA.  The
interleaved model is therefore a new composite and must not be described as a
theorem from either source.

The inherited pieces are exact:

- From Jiang §3.1: MHSA then FFN, separate residual additions, each `1/L`.
- From Jiang Appendix Table 2: attention and dense-FFN Adam group exponents.
- From Jiang Appendix A: fixed head dimension, head count proportional to `D`,
  `QK^T/d_head`, pre-LN, tied embeddings, learned absolute positions, GELU.
- From Chizat and round 014: treat FFN hidden units as a mean-field population
  and preregister `rho = L M / D` as the preferred joint shape coordinate.

The new claim to test is that these pieces remain compatible when MHSA and FFN
are interleaved and trained together.

## Forward normalization

For `H = D/d_h` heads and sequence positions `s,t`,

```text
q_hs, k_hs, v_hs = LN(h_s) W_QKV,h
A_hst = <q_hs, k_ht> / d_h + causal_mask(s,t)
MHSA_s = concat_h sum_t softmax(A_hs)_t v_ht W_O
```

All attention matrices have entry scale `D^(-1/2)`.  Unlike standard
scaled-dot-product attention, division is by `d_h`, not `sqrt(d_h)`.  Since
`d_h` is fixed, this preserves `O(1)` logits as `D` grows by increasing the
number of exchangeable heads.

The dense FFN is

```text
FFN(h) = GELU(h W_up) W_down,
std(W_up) = D^(-1/2),
std(W_down) = sqrt(D)/M.
```

The down-projection is below fan-in by `sqrt(D/M)`.  At fixed `M/D` this is a
constant-scale difference, but under an independent `M` dial it is essential:
the trained mean-field part can stay `O(1)` while the random part decays.

## Per-axis audit requirement

Attention contains residual width `D`, head count `H`, and key dimension
`d_h`; the FFN contains `D` and `M`; both share depth `L`.  Consequently no
single row-norm measurement can validate the optimizer table.  The primary
observable is the change in post-block residual features produced by updating
one semantic group while freezing the others.  The audit is run independently
along every pure axis before any joint-transfer verdict.

For attention, individual attention-logit movement and head-averaged kernel
movement are both retained.  Suppression of the latter can be concentration
rather than freezing, so reporting only a head average is insufficient.

## Exact reductions

- Disabling MHSA must reduce to the dense Jiang/Chizat FFN stack.
- Setting `M = D` removes only the independent hidden-width dial; it does not
  turn the FFN into standard fan-in because the down-projection remains the
  mean-field choice.
- `L = 1` remains an attention residual followed by an FFN residual; neither
  sublayer may disappear or be reordered.
- Replacing `sqrt(D)/M` by `M^(-1/2)` is the preregistered standard-fan-in
  control.

## Status

This document specifies an implementation and validation target.  It is not a
completed derivation of the composite DMFT and carries no certification until
round 017's preregistered audits and finite-model studies pass.
