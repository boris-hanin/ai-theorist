# Round 005 results — HP transfer across N, H and L for transformers

Pre-registered in `prereg.md`, committed at `348939a` **before** any of this ran.
Derivation in `derivations/03-attention.md`, corrected at `f12808e`.

Date: 2026-08-08
Target: Bordelon, Chaudhry & Pehlevan, arXiv:2405.15712v2, Table 1 and §3.2.
Artifacts: `skills/dmft-attention/scripts/{attention.py, sweep.py}`

## Headline

**All three dials transfer under the Table 1 scaling, and every control that
should bite does.** `eta = eta_0 * N * H * L^{2 aL - 1}`, derived independently
in D3 and matching Table 1 exactly.

| Sweep | Full scaling | Control (predicted factor removed) |
|---|---|---|
| **N** = 4→32, `aA = 1` | TRANSFERS, drift 0.260 | **FAILS**, drift 0.771 |
| **N** = 4→32, `aA = 1/2` | TRANSFERS, drift 0.166 | **FAILS**, drift 0.779 |
| **H** = 2→16 | TRANSFERS, drift 0.097 | **FAILS**, drift 0.770 |
| **L** = 2→8, `aL = 1` | TRANSFERS, drift 0.098 | **FAILS**, drift 0.359 |
| **L** = 2→8, `aL = 1/2` | TRANSFERS, drift 0.110 | inert (0.110) — *expected* |

Drift is movement of the optimal `eta_0` in decades across the dial range;
"transfers" requires it to be under 0.3 decades **and** the optimum to be
located per seed with its across-seed scatter measured.

The `aL = 1/2` row is not evidence — `L^{2 aL - 1} = L^0` is the identity there,
so removing it changes nothing by construction. It is in the table as a
consistency check on the harness, and the control reproducing the full result
to three decimals is the check passing.

**Control slopes.** Removing a factor should force `eta_0` to compensate with
slope 1 in that dial. Measured: `N` 0.85, `H` 0.85, `L` 0.60 (per decade of the
dial). Directionally right and large; the shortfall is plausibly the
mis-scaled model already sitting near instability at the top of each ladder,
which clips the optimum. Not investigated further.

## P4/P5 — the prediction that mattered

**P4 and P5 both hold: `aA = 1` and `aA = 1/2` transfer equally well** (drift
0.260 and 0.166, both under bar). This was pre-registered as the *expected*
outcome, following the paper's Fig 2(a), and it is the entry with real content:

> D2 said that at the level of the forward pass and one step, any `aA` can be
> made to give `Theta(1)` attention updates by rescaling the learning rate — so
> a transfer sweep should **not** discriminate `aA`. The discriminating
> observable is the across-head variance scaling, not the transfer sweep.

Written down before measuring, and it held. Had the round reached for the
transfer harness as the instrument for `aA` — the tool that was already built —
it would have measured the wrong thing and concluded the exponent does not
matter.

## Implementation verification, before any sweep

The architecture is §2.1 with `aA`, `aL` free. Key/query weights are stored as
`W_K = N^{1-aA} What_K` with `What ~ N(0,1)`, which moves `aA` out of the
forward pass (where `k` is `Theta(1)` for every `aA`) and into the learning rate
on `What_K`, as `eta / N^{2(1-aA)}`. Algebraically identical to the paper's
convention.

**Var A(0) reproduces Table 3 exactly** — the check that the forward
parameterisation is right:

| | measured slope in `N` | Table 3 |
|---|---|---|
| `aA = 1` | **−1.06** | `Theta(N^{-1})` |
| `aA = 1/2` | **−0.06** | `Theta(1)` |

## An unresolved discrepancy, reported as such

The **`(delta k)^2` half of Table 3 is not cleanly reproduced at this scale.**

First attempt measured the *total* change in `k` and found no `aA`-dependence
at all. That was a measurement error: `k = c_k W_K hbar` moves for two reasons —
`W_K` changing (where `aA` acts) and `hbar` changing (the residual stream,
governed by the bulk rate) — and the second dominates. Isolating the
weight-driven part by holding `hbar` at its initial value separates them.

Isolated, the **difference** between exponents comes out right: slope gap 1.01
at `t = 20`, against Table 3's predicted gap of 1 (0 vs −1). But both curves
carry a common offset, and it moves with training time:

| `aA` | t=1 | t=50 | t=500 | t=2500 | Table 3 |
|---|---|---|---|---|---|
| 1 | −1.20 | −0.58 | −0.35 | **−0.29** | 0 |
| 1/2 | −2.20 | −1.61 | −1.95 | **−1.99** | −1 |

The `aA = 1` row climbing monotonically toward 0 is **direct confirmation of the
D2b mechanism**: the first SGD step is suppressed because `W_O` is uncorrelated
with the backward gradient, and the suppression lifts as that correlation
builds. The `t = 1` value of −1.20 is the suppressed regime; the paper's Fig 12
shows the same thing at `t = 1` vs `t = 200`.

But neither row reaches its Table 3 asymptote here. Candidate causes, in order:
this model is small (`N <= 64`, `H = 4`, `L = 2`, `d_model <= 256`) and the
asymptotic regime may need larger `N`; the task is a synthetic sequence
regression rather than CIFAR-5M; and Table 3's "update to k, q entries" may be
defined as the total rather than the weight-driven part isolated here.
**Unresolved. Not claimed as a replication of Table 3's second column.**

## Scope

- The transformer **DMFT solver** (Appendix E) was not built, as pre-registered.
  This round replicates scaling results, not the DMFT analysis.
- P1/P2/P3 (across-head variance vs `N`, trained and at init) were **not run**.
  P3's init half is effectively covered by the Var A(0) check above; the trained
  half (Fig 2b's `N^{-2}`) remains open.
- Dataset deviation from CIFAR-5M stands as pre-registered.

## Status of `dmft-attention`

Still **RECONSTRUCTED**, not certified. Its four substantive claims all check
out against the paper, and Table 1's learning-rate scaling is now verified by
independent derivation *and* by measurement on all three dials. But the round's
own certification bar required P1–P3, and only part of P3 was run.
