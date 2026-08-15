# Round 013 preregistration — Chizat Muon transfer with trained boundaries

## Freeze point

This preregistration is written before any A100 result from
`chizat_muon_transfer.py` is interpreted.  Local CPU smoke runs establish only
that the runner executes; they are not evidence for the transfer claim.

## Primary claim

For the bias-free end-to-end Chizat network with trained fan-in embed and
mean-field scalar unembed, one normalized eta transfers jointly in L, M, and D
under the following hybrid optimizer:

```text
embed:   Adam, lr = eta
U:       RMS-matched Muon, lr = eta
W:       RMS-matched Muon, lr = sqrt(D) eta
unembed: Adam, lr = eta/D
```

Muon uses momentum 0.95, Nesterov, five Newton--Schulz steps, float32
orthogonalization, and zero weight decay.  Auxiliary Adam uses betas
`(0.9,0.95)`, epsilon `1e-10`, and zero weight decay.  Weight decay is not tuned
in this round.

## Primary design

- Shapes: a strictly increasing joint L/M/D path with at least five points.
- Reference: the middle nonterminal shape, declared in the command manifest.
- Eta grid: at least five log-spaced values with up to two preregistered edge
  expansions; the numerical optimum must be interior.
- Seeds: at least five common model/data seeds in increasing order.
- Horizon: fixed across every shape and rule.
- Task: the same nested fixed scalar teacher at every shape.
- Target: final loss at the declared horizon, never best-so-far loss.

The first production command will be retained verbatim with its environment
fingerprint and output path.  A second worker may receive disjoint seeds plus
at least one duplicated seed for cross-worker determinism.

## Acceptance rule

The primary claim passes only if all are true:

1. The reference numerical eta optimum is interior.
2. Every selected-eta primary run is finite.
3. Fixed-eta trajectory differences settle at every checkpoint after step 1.
4. Final fractional learning progress is at least `1e-3` and its absolute
   log-progress slope against the joint dial is at most `0.3`.
5. `wrong_W_D`, `wrong_sgd_LMD`, and `wrong_constant_unembed` are all rejected
   by at least one of: trajectory settling, learning progress, or a common-seed
   paired final-loss increase at the largest shape exceeding
   `max(2 SEM, 1% of primary loss)`.
6. A repeated common-seed trial on the second A100 is identical within the
   separately recorded hardware tolerance.

Failure of one gate yields `PARTIAL` or `FAILED`; it is not repaired by a
favorable per-shape eta optimum.

## Secondary diagnostics

- Constant-W, frozen-embed, and frozen-unembed controls.
- First-step update RMS and update/weight RMS for all four roles.
- Per-shape eta optima and largest finite probes, labeled diagnostic-only.
- The original Muon aspect-ratio adjustment as an implementation ablation.
- A second horizon after the primary decision, labeled confirmation rather
  than part of the preregistered primary test.

## Mutation requirements

Before A100 execution, the local suite must catch missing RMS adjustment,
wrong transpose handling, omitted Nesterov behavior, non-matrix Muon routing,
double-routed boundaries, missing parameters, and broken optimizer-state
continuation.

## Amendment after exploratory CPU screen, before A100 execution

The first 160-step CPU confirmation showed that a wrong unembed rule could be
many times worse at the largest shape while still passing a settling-only
control test: its absolute adjacent gaps stopped growing even though it had
settled to the wrong loss.  Item 5 now includes the paired largest-shape loss
test already used by the autoscaler study harness.  This amendment is informed
by exploratory data and is not retroactively presented as preregistered CPU
evidence; it is frozen before the first Chizat-Muon A100 run.
