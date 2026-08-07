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

- `skills/dmft-derivation/` — the core skill (**certified**): phased recipe
  (scope check → parameterization → single-site derivation → closure →
  numerics → mandatory checks) with complete reference equation systems,
  numerical algorithms, muP/depth-muP tables, validation targets, and a
  runnable two-layer solver + Phase 5 check battery under `scripts/`.
- `skills/dmft-master/` — the 9-step zero-shot algorithm for deriving the
  limit of a NOVEL architecture, with long-form steps, instance traces, and
  the solver-pattern library. **Reconstructed, pending re-validation.**
- `skills/dmft-{resnet-depth,attention,moe}/` — per-instance deltas.
  **Reconstructed, pending re-validation.**
- `registry/failure-modes.md` — canonical failure-mode registry (F1–F17)
  accumulated across all validation rounds.
- `rounds/` — one directory per validation round: pre-registration committed
  before results, then results and raw numbers beside it.
- `PROGRAM.md` — program history, per-skill certification status, and the
  method invariants.

## Validated instances (chronological)

1. Deep MLPs (arXiv 2205.09653) — exact limiting equations reproduced;
   solver vs sims across widths. Origin of F1, F4, F5, F6.
2. Depth-muP ResNets (arXiv 2309.16620) — depthwise hyperparameter
   transfer. Origin of F14.
3. Multi-head attention (arXiv 2405.15712) — concentration-vs-freezing
   dichotomy corrected against data. Origin of F3.
4. Mixture-of-Experts (arXiv 2601.20205) — routing-aware scaling audit.
5. Synthesis: one master algorithm from which all completed computations
   are derivable as traces.
6. Hyperbolic Busemann networks (novel, rounds 1–3) — executed AFTER the
   synthesis, from the algorithm alone, as a **zero-shot test** of it.
   Origin of F15, F16, F17. See `skills/dmft-master/references/instances.md`;
   the round's solver code was lost and is not being recovered.

## Status, honestly

Instances 1–6 were validated in sessions whose artifacts were largely lost to
workspace recycling. What survives here is the *method*: the recipe, the
algorithm, the registry. Of the numerical work, only the two-layer solver in
`skills/dmft-derivation/scripts/` is present and re-certified against the
exactly-solvable cases — everything in the response sector (L ≥ 2) is specified
but unimplemented. Claims about instances 1–6 rest on the program record, not
on anything runnable in this repo. Treat them accordingly.
