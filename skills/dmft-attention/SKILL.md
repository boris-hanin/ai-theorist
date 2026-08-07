---
name: dmft-attention
description: Extend the DMFT derivation skill to multi-head self-attention in the infinite-width/infinite-head limit (arXiv 2405.15712). Use for attention-layer mean-field limits, per-axis scaling audits (heads, key/query dimension, model width), attention-entropy/kernel dynamics, and μP-style transfer for transformer blocks.
---

> **PROVENANCE: RECONSTRUCTED.** Original validated skill file lost to
> workspace recycling; rewritten from the program record. Faithful in
> substance, NOT verbatim; pending re-validation. Cited results were
> obtained in the original round.

# DMFT for multi-head attention (delta on dmft-derivation)

Prerequisite: `dmft-derivation` Phases 0–1. Attention adds two new
structural ingredients relative to MLPs:

1. **Multiple width axes.** Model width N, number of heads H, per-head
   key/query dimension d_k: each gets its own scaling exponent and its own
   limit. Write EVERY axis's exponent explicitly (per-axis version of the
   Step-0 scaling audit) before deriving. The audit is empirical:
   per-parameter-group update RMS vs each axis, holding others fixed;
   pin exponents by Θ(1)-update (or Θ(1)-feature-velocity) flatness.
2. **Bilinear/softmax order parameters.** The attention logits
   A_{μν} ∝ q·k/√d_k are bilinear in trained tensors — master-algorithm
   class (d): they close through deterministic recursions or
   Gaussian-sector cross-covariances rather than through response pairs,
   depending on reuse structure. Softmax makes the attention matrix a
   deterministic function of these order parameters in the limit.

## The concentration-vs-freezing dichotomy (F3 — discovered here)

The central lesson of this round. "The attention matrix stops moving as
width grows" has two inequivalent mechanisms:
- **Freezing:** individual logit entries stop moving (lazy behavior of the
  q/k parameters at their exponent);
- **Concentration:** entries move Θ(1) but the POPULATION average defining
  the attention order parameter concentrates — the limit object moves
  deterministically.

A pre-registered claim in the original round (ΔA = Θ(N^{−1/2}) at
α_A = 1/2, read as freezing) was FALSIFIED by the measured scaling and
corrected: the observed suppression was concentration, not freezing. Any
scaling claim about attention movement must state which mechanism it
asserts and test the discriminating observable (per-entry movement vs
order-parameter movement, seed-resolved).

## Validated claims (original round)

- Reproduced the paper's infinite-head/width mean-field limits for the
  attention block, with per-axis exponent audit matching the derived
  table; solver-vs-sim agreement on training dynamics of the attention
  kernels; μP-transfer of the learning rate across the audited axes.

## Checks specific to this skill

- Per-axis audits BEFORE derivation; never assume one width axis's
  exponent applies to another.
- Discriminate F3 mechanisms explicitly whenever a Θ(N^{−a}) movement
  claim is made.
- Softmax linearization is NOT valid at Θ(1) logit spread — keep the full
  softmax of the limiting logit Gaussian/order parameters.
- Reduce to the MLP skill when attention is frozen (identity-attention
  control) — the pipeline must reproduce the plain-MLP answers.
