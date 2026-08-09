# Validation rounds

One directory per round: `NNN-short-name/`.

## The rule that matters

**This repository is the working directory of a round, not an export of one.**

The program has lost its primary artifacts four times — the original session
archives, then two git bundles left sitting in a working tree, then the
hyperbolic round's solver code, which no longer exists anywhere. Every loss had
the same shape: the real work lived somewhere that was not a commit on a
branch, and then that somewhere went away. F7, F9 and F13 are permanently blank
registry entries because of it.

So: commit as you go, on a branch, from the start. A round that produces a
downloadable archive has produced nothing.

## Lifecycle

1. **Pre-register.** Copy `TEMPLATE/prereg.md` into the round directory, fill
   it in, and **commit it before running anything**. The commit timestamp is
   the evidence. A pre-registration written after seeing results is worth less
   than none, because it looks like the real thing.
2. **Run.** Code goes in the repo. Raw numbers go in the round directory.
3. **Report.** Write `results.md` next to the pre-registration, stating for
   each pre-registered prediction whether it held. Predictions that failed stay
   in the record — F3 exists because a falsified pre-registration was reported
   honestly rather than quietly revised.
4. **Register failures.** Anything that bit gets an entry in
   `registry/failure-modes.md` with the schema
   *mechanism → detection signature → fix → guard*. Add the guard as an
   assertion or a check in the relevant `scripts/`, not only as prose. The
   registry ratchets only if the guards are executable.
5. **Update status.** Move the skill's row in `PROGRAM.md` if its certification
   state changed.

## What a round must report

- The Monte-Carlo floor next to every theory-vs-simulation gap (F8). A gap
  below its own floor is not evidence in either direction.
- Seed-averaged comparisons (F10), with the number of seeds.
- Any coverage bound: top-N, sampling, a check that was skipped, a regime not
  swept. Silent truncation reads as full coverage.
- For any new check battery: a mutation test. Reintroduce the failure modes the
  battery claims to catch and confirm it catches them. The two-layer battery
  was initially blind to F4 and only mutation testing revealed it.

## Existing rounds

| Round | What | Pre-registered |
|---|---|---|
| `001-two-layer-rebuild` | Rebuild the L=1 solver from the reconstructed docs; make Phase 5 runnable | No — retrospective, see its `results.md` |
| `011-graph-transformer` | Extend 2607.05017 to graph transformers: parameterisation, DMFT, HP transfer | Yes — `9435a39`, before any measurement |

Rounds 1–6 of the original program (deep MLP, depth-muP ResNet, attention, MoE,
synthesis, hyperbolic) predate this directory and their artifacts were lost.
`skills/dmft-master/references/instances.md` holds what survives of them.
