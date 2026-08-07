---
name: dmft-moe
description: Extend the DMFT derivation skill to Mixture-of-Experts architectures (Jiang–Bordelon–Pehlevan–Hanin, arXiv 2601.20205) — expert-count scaling, routing-aware μP tables, and MoE training-dynamics limits. Use for questions about scaling laws/HP transfer in E experts with sparsity α, expert-stream backward scaling, and router-conditioned mean-field limits.
---

> **PROVENANCE: RECONSTRUCTED.** Original validated skill file lost to
> workspace recycling; rewritten from the program record. Faithful in
> substance, NOT verbatim; pending re-validation. Cited results were
> obtained in the original round.

# DMFT for Mixture-of-Experts (delta on dmft-derivation)

Prerequisite: `dmft-derivation` Phases 0–1. MoE adds: an expert axis E, a
sparsity/activation fraction α (each token sees N = E/α... active-set
structure), and routing — a per-sample discrete conditioning of which
experts' streams carry signal.

## Scaling audit with routing (the α-bookkeeping trap)

The round's registered error (F2 flavor): the backward carrier through the
expert stream was mis-scaled as O(1) when its true scale is α^{−1} — the
backward signal sums over the active expert-stream neurons (a set of size
set by α), so the up-projection learning rate must scale η_up ∝ α to keep
Θ(1) per-parameter updates. The error was DIAGNOSED empirically before it
was explained: measured per-group movements collapsed onto a single curve
only when rescaled by √α — the collapse exponent pinpointed the wrong
bookkeeping term. Protocol: whenever a routing/sparsity dial exists, run
the movement-collapse audit in that dial FIRST; a clean power-law collapse
in the dial is both the bug detector and the exponent measurement.

## DMFT structure

- Experts are exchangeable populations conditioned on routing: the
  single-site measure acquires a router-conditioned mixture structure
  (which samples a given expert serves), with population averages over
  experts entering the shared-stream order parameters.
- Router logits: bilinear order parameters (master class d); with frozen
  or slowly-adapting routers the conditioning is quenched per sample
  (F11: condition theory and simulation identically).
- Expert weight matrices: reused disorder within their stream (class b)
  → per-expert response pairs, population-averaged into the stream
  kernels with routing weights.

## Validated claims (original round)

- Reproduced the paper's MoE μP table including the η_up ∝ α rule;
  movement-collapse audit across α; solver-vs-sim agreement for the
  training dynamics at the audited exponents; HP transfer across the
  expert-count/sparsity dials at fixed compute per token.

## Checks specific to this skill

- √α-collapse audit before any derivation is trusted.
- Dense limit α → 1 must reduce to the plain-MLP (or per-expert-MLP)
  skill's answers.
- Router-frozen control: with routing quenched at init, the theory must
  reduce to a mixture of conditioned MLP limits.
- Report per-expert load statistics next to any transfer claim (imbalance
  is a rate-amplified metric — F12).
