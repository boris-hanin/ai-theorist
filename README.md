# ai-theorist

Skills and infrastructure for a validated "mathematical assistant" that
derives dynamical mean-field theory (DMFT) descriptions of neural network
training dynamics (Bordelon–Pehlevan program), built and certified through
an iterative validate-then-extend process.

Validation bar used throughout: **derivation + independent numerics +
finite-size simulations** — never symbolic plausibility alone. Plans are
pre-registered before computing; paper transcriptions are treated as hints
(simulations are ground truth); failures are reported honestly and
registered as named failure modes.

## Contents

- `skills/dmft-derivation/` — the core skill (verbatim, account-synced):
  phased recipe (scope check → parameterization → single-site derivation →
  closure → numerics → mandatory checks) with complete reference equation
  systems, numerical algorithms, muP/depth-muP tables, and validation
  targets.
- `registry/failure-modes.md` — the failure-mode registry accumulated
  across all validation rounds (F1–F17).
- `PROGRAM.md` — history and status of the full skill program, including
  the four companion skills validated in prior sessions (ResNet depth-muP,
  multi-head attention, Mixture-of-Experts, and the master zero-shot
  algorithm) whose files currently live in session archives pending
  restoration.

## Validated instances (chronological)

1. Deep MLPs (arXiv 2205.09653) — exact limiting equations reproduced;
   solver vs sims across widths.
2. Depth-muP ResNets (arXiv 2309.16620) — depthwise hyperparameter
   transfer.
3. Multi-head attention (arXiv 2405.15712) — concentration-vs-freezing
   dichotomy corrected against data.
4. Mixture-of-Experts (arXiv 2601.20205) — routing-aware scaling audit.
5. Synthesis: one master algorithm from which all completed computations
   are derivable as traces.
