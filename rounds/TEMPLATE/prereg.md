# Pre-registration — round NNN: <name>

**Commit this file before running anything.** Do not edit it afterwards;
corrections and outcomes go in `results.md`.

Date: <YYYY-MM-DD>
Skill(s) exercised: <e.g. dmft-derivation, dmft-master>
Certification intent: <re-validate a reconstructed skill / extend a certified
one / novel architecture>

## 1. Scope verdict (Phase 0)

Write the limit order explicitly, e.g. "N→∞ at P=10, t≤50, gradient flow, MSE".
State which of the four Phase 0 conditions could fail and why you believe they
do not.

## 2. Predictions

Numbered, falsifiable, stated before computing. For each: the observable, the
predicted value or scaling, and **what measurement would refute it**.

| # | Observable | Prediction | Refuted if |
|---|---|---|---|
| P1 | | | |
| P2 | | | |

If a prediction is about something "stopping moving" as a dial grows, say which
mechanism you are claiming — concentration of a population average, or freezing
of individual entries — and name the discriminating observable (F3).

## 3. Tolerances and floors

- Monte-Carlo floor: how it will be measured (sample-halving, F8) and the S at
  which the comparison runs.
- Discretisation: the dt, and the dt-halving check.
- Seeds: how many, and that comparisons are seed-averaged (F10).
- **The pass bar for each prediction, stated relative to the floor.** A bar
  tighter than the floor cannot be met; a bar much looser than the floor is not
  testing anything. If a prediction's effect size is below the achievable
  floor, say so now and either raise S or drop the prediction.

## 4. Ground truth

What plays the role of truth, in order of preference:
1. Exactly solvable reductions (closed form).
2. Independent derivation by a second route (F14 — transcriptions are hints;
   a published formula is not ground truth).
3. Finite-size simulation in the SAME parameterisation and discretisation.

State explicitly whether the simulation is an independent test of the closure
or only a convergence check. At L=1 it is only a convergence check, because a
width-N network is exactly N samples of the single-site process.

## 5. Ablations that must bite

List the controls, and for each, what change you expect. An ablation that
changes nothing is a red flag, not a pass (F17) — unless you can say in advance
why it should be inert, as for the response sector at L=1.

## 6. Known failure modes in scope

Which registry entries could plausibly fire here, and the guard for each.
Guards should be assertions or checks in `scripts/`, not intentions.

## 7. What would make this round a failure

State it now. "We would conclude the skill is not certified if ..."
