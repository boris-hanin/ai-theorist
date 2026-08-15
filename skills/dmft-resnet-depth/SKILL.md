---
name: dmft-resnet-depth
description: Extend the DMFT derivation skill to depth-μP residual networks (Bordelon–Pehlevan–et-al., arXiv 2309.16620) — the joint width-and-depth limit with 1/√(LN) branch scaling. Use for depthwise hyperparameter transfer questions, depth-collapse of training curves, and the large-L ODE/SDE structure of residual-stream DMFT.
---

> **PROVENANCE: RECONSTRUCTED.** The original validated skill file was lost
> to cloud-workspace recycling; this version was rewritten from the program
> record. Faithful in substance, NOT verbatim, and pending re-validation
> before being treated as certified. The validation results it cites were
> obtained and recorded in the original round.

# DMFT for depth-μP residual networks (delta on dmft-derivation)

Prerequisite: run `dmft-derivation` Phases 0–1 first; this skill supplies
the residual-specific deltas.

## Architecture and scaling (the depth-μP table)

Residual stream h^{l+1} = h^l + (β_L/√N)·W^l φ(h^l) with branch scale
β_L = β₀/√L, entries W ~ N(0,1), readout f = (1/(γ₀√N... γ))·w·φ(h^L),
γ = γ₀√N. The 1/√L branch scale is the unique choice for which, as L→∞:
- the forward stream stays Θ(1) (branch contributions accumulate as a CLT
  sum over layers → layer-time diffusion, not blowup or decay);
- per-block feature updates are Θ(1/√L) while the COLLECTIVE feature
  velocity and NTK stay Θ(1) — depth-μP's "collective learning";
- block learning rates need NO extra depth exponent (s = 0) in the
  γ²-scaled dynamics.

Empirical Step-0 audit protocol (always run before trusting the table):
measure per-block update RMS and few-step loss-drop vs L ∈ {1..16} at the
candidate exponents; correct choice ⇒ loss dynamics collapse across L
(slope ≈ 0 in log L) and per-block updates scale L^{−1/2}. Wrong choices
show stream-variance blowup (unscaled) or lazy-in-depth decay (1/L).

For the Chizat mean-ODE parameterization
`h^l = h^(l-1) + alpha/(L M) sum_j w_lj sigma(u_lj.h^(l-1))`, plain GD uses
the normalized coordinate `eta` with raw LR `L M eta / alpha^2`.  Transfer in
`M` or `L` is tested by holding `eta` fixed and comparing trajectories.  The
per-dial finite-horizon optimum is a separate stability-edge diagnostic (F25),
not the transfer verdict.  Use
`scripts/chizat_lr_transfer.py`, which enforces this output separation.

End-to-end Chizat transfer studies must train the boundary maps rather than
silently freezing them.  Use a fan-in embed
`E_ad ~ N(0,1/d0)` and a mean-field scalar unembed
`R_d ~ N(0,1/D^2)`.  Their plain-GD rates are `D eta` and `eta/D`,
respectively.  `scripts/chizat_lmd_transfer.py` implements those boundaries by
default and retains explicit freeze/mis-scaling controls.

For the Chizat-only Muon research path, use
`scripts/chizat_muon_transfer.py` and the contract in
`derivations/12-chizat-trained-boundaries-muon.md`.  It routes only the `U/W`
particle matrices through RMS-matched Muon; the 2D embed and unembed remain on
auxiliary Adam by semantic role.  The candidate raw rates are
`(eta, eta, sqrt(D) eta, eta/D)` for `(embed,U,W,unembed)`.  This is
tested in round 013: its fixed-eta and negative-control gates passed on CPU
joint and pure-axis screens, a 160-step follow-up, and two independent A100
replicas.  Treat that result as strong exploratory evidence rather than formal
certification because the preregistration was not committed before execution.

## DMFT structure

Each block's W^l is reused disorder (class-b): per block a forward field
χ^l and backward field ξ^l with a response pair (A^l, B^l), plus rank-P
trained-matrix kernels — the multi-block sandwich. Sources are
conditionally independent across blocks given the population kernels
(independent matrices). At finite L this is L coupled copies of the
deep-MLP single-site system; as L→∞ with 1/√L scaling, per-block response
corrections are O(1/L) and the stream kernels obey depth-ODEs in layer
time τ = l/L (the paper's large-L limit; NTK depth-ODE).

## Validated claims (original round; settings in the lost report)

1. Reproduced the paper's limiting equations for the residual DMFT and
   the depthwise-transfer prediction: with (β_L, LR) per the depth-μP
   table, the optimal η₀ is L-independent and training curves collapse
   across depths; mis-scaled controls drift and destabilize with L.
2. Solver-vs-finite-sim agreement across widths at fixed L, and the
   depth-collapse of loss curves at fixed width.
3. **F14 lesson (registered here):** a fetched transcription of the
   paper's Eq. 7 was 3.3× off; the discrepancy was resolved by
   independent re-derivation + simulation, not by trusting the source.
   Simulations are ground truth for formula disputes.

## Checks specific to this skill

- Depth-collapse: loss curves at L ∈ {2,4,8,16} overlay after the audit's
  exponents are applied; report the collapse metric (mean |Δlog loss|).
- Width-collapse: at fixed `eta`, common-seed trajectory differences across a
  pure `M` dial must settle compatibly with the `M^-1/2` empirical-measure
  correction. Report absolute gaps; relative/log gaps near zero are unstable.
- Stability edge: report the local optimum and largest finite eta separately;
  neither is allowed to change the fixed-eta transfer verdict.
- Boundary audit: report embed and unembed initialization and update RMS;
  include a frozen or deliberately mis-scaled boundary control.
- Optimizer routing: assign parameters by mathematical role.  Matrix rank is
  not sufficient because trained embed/unembed matrices are not Muon groups.
- Per-block movement slope vs L ≈ −1/2 (collective learning signature).
- L=1 must reduce exactly to the two-layer causal co-integration case.
- Response ablation at moderate L must degrade the fit (F17 guard: an
  ablation that changes nothing is a red flag, not a pass).
