# Round 004 — residual branch exponent α: an INCONCLUSIVE test

**Not pre-registered.** And the headline is a negative result about the
*experiment*, not about the physics.

Date: 2026-08-07
Skill exercised: `dmft-derivation` (Leg C only — no residual DMFT was derived)
Artifacts: `scripts/nets.py` (α free), `scripts/audit.py::block_linearity`

## What was being tested

CompleteP ([2505.01618](https://arxiv.org/abs/2505.01618)) argues that the
depth-μP residual scale `L^{-1/2}` asymptotically linearises residual blocks:
the first-order term scales as `Θ(L^{-1})` and the second-order as
`Θ(L^{α-2})`, so their ratio goes as `L^{α-1}` — vanishing at `α = 1/2`,
`Θ(1)` at `α = 1`. It further reports that depth-μP's optimal LR *shifts* with
depth for multi-transformation (block depth `k ≥ 2`) blocks, i.e. transfer
fails outright.

Two measurements, both with `α` free rather than hardcoded:

## Results

**Depth transfer** (N=128, L ∈ {2,4,8}, 3 seeds, 20 steps, synthetic teacher):

| parameterisation | k | verdict | drift | lr* by L |
|---|---|---|---|---|
| depth-μP α=0.5 | 1 | TRANSFERS | 0.089 | −1.17 −1.26 −1.26 |
| depth-μP α=0.5 | 2 | TRANSFERS | 0.114 | −1.30 −1.18 −1.25 |
| CompleteP α=1.0 | 1 | TRANSFERS | 0.068 | −1.02 −0.95 −1.02 |
| CompleteP α=1.0 | 2 | TRANSFERS | 0.079 | −1.00 −1.01 −1.08 |

**Block linearity** (ratio of the nonlinear part of the block update to its
linearisation):

| α | k | ratios at L = 2, 4, 8 | slope d log ratio / d log L |
|---|---|---|---|
| 0.5 | 1 | 0.3511 0.3690 0.3709 | +0.039 |
| 0.5 | 2 | 0.3225 0.3466 0.3516 | +0.062 |
| 1.0 | 1 | 0.3703 0.3628 0.3435 | −0.054 |
| 1.0 | 2 | 0.3373 0.3499 0.3387 | +0.003 |

Neither effect appears: `α = 0.5` transfers at `k = 2`, and the linearity ratio
is flat in L at every setting (predicted slope at `α = 1/2` is ≈ −0.5).

## This does NOT refute CompleteP. The test is mis-specified.

The most likely defect is in my own estimator, and it is specific enough to
name. `block_linearity` computes

    ratio = || full block-output change - its linearisation || / || linearisation ||

Both numerator and denominator carry the same `L^{-α}` residual prefactor, so
**the ratio is scale-free in L by construction** — the very `L`-dependence
CompleteP's argument is about divides out. Their `L^{-1}` vs `L^{α-2}` counting
is over terms in the expansion in the *weight update* `ΔW`, whose own scaling
with L is what carries the effect. My ratio cannot see it. A flat slope is
therefore the expected output of this estimator whether or not the physics is
there, which makes it useless as evidence either way.

Four further reasons the setup is under-powered even with a correct estimator:

1. **L ∈ {2,4,8} is a very short lever.** A genuine `L^{-1/2}` would be a factor
   of 2 across the whole range.
2. **N = 128, and the claim is about the joint N,L limit.**
3. **The architecture is not theirs.** No LayerNorm, no attention, a plain
   residual MLP with a μP readout — CompleteP is stated for pre-LN transformers.
4. **The task is 20 steps on a 64-point synthetic regression.** Their 12–34%
   compute-efficiency gains are from LLM pretraining on Cerebras CS-3s. A
   mechanism that governs long-horizon training need not show in 20 steps.

Per the program's own rule — an ablation that changes nothing is a red flag,
not a pass; report under-powered as under-powered — **this round establishes
nothing about α.** It is recorded so the next attempt does not repeat it.

## What the round DID produce

- `nets.py` carries `alpha`, `block_k` and `lr_depth_exp` as free parameters,
  so the depth question can be asked properly rather than assumed. The repo's
  previous depth story hardcoded `α = 1/2` inherited from a paper with no
  audit — which is a Step-0 violation by the master algorithm's own rule
  ("never derive against unpinned exponents").
- A correctness catch: the first attempt gave `α = 0.5` an `L^{-1/2}` LR factor
  (CompleteP's general rule `L^{α-1}`) while depth-μP prescribes *no* depth LR
  factor. That is a different parameterisation from the one under test. The
  numbers above use the corrected pairing — both with no depth LR factor, so
  `α` is the only difference.

## What a real test needs

1. **Fix the estimator.** Measure the `ΔW`-expansion terms directly, or measure
   the block's contribution to the network function without dividing out
   `L^{-α}`.
2. **Longer lever:** L up to 32–64, and larger N.
3. **Longer horizon:** train to convergence or a realistic step budget.
4. **Closer architecture:** at minimum add normalisation; ideally a pre-LN
   transformer block.
5. **Derive the residual DMFT at general α first**, so the linearisation ratio
   is a *predicted* quantity to test rather than a heuristic to eyeball. That
   derivation was not attempted this round.

## Registry

No F18 entry yet. The inherited-exponent failure mode is real and worth
registering, but the evidence for it here is the repo's own history
(hardcoding `α = 1/2` without an audit), not this measurement. Registering it
on the back of an inconclusive experiment would be exactly the kind of thing
F17 warns about.
