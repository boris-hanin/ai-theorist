# Round 007 — the down-projection init and its consequences

Targets: **arXiv 2509.10167** (Chizat) and **arXiv 2603.18168**
(Chaintron–Chizat–Maass). Derivation: `derivations/08-chizat-mlu.md`.
Model: `skills/dmft-resnet-depth/scripts/mean_ode.py`.

**Prereg honesty note.** Q1–Q5 were fixed in `08` §6 *before* any measurement,
but `08` was committed *after* the first run rather than before it. So this is a
weaker guarantee than rounds 005/006, where the prereg landed in git first. Not
claimed as a committed prereg.

## The one line

In the convention `||h|| = Theta(sqrt D)`, for a 2LP block
`h^l = h^{l-1} + (1/L) W_down phi(W_up^T LN(h^{l-1}))` with `W_down in R^{D x M}`:

    fan-in / NTP (CompleteP) :  sigma_down = M^{-1/2}
    MLU / mean-field         :  sigma_down = sqrt(D) / M

    ratio sqrt(M/D)    ->    identical iff M = Theta(D)

## Verdicts

| # | prediction | measured | verdict |
|---|---|---|---|
| Q2 | ratio is `sqrt(M/D)`, `=1` at `M=D` | exact to 4 d.p. at `M/D` = 1..64 | **PASS** |
| Q3 | MLU init deviation `~ sqrt(D/(LM))` | `L` **-0.502**, `M` **-0.501**, `D` **+0.510** | **PASS** |
| Q3c | fan-in init deviation `~ 1/sqrt(L)`, no `M` or `D` | `L` **-0.502**, `M` **-0.002**, `D` **+0.009** | **PASS** |
| Q4 | at fixed `LM`, MLU deviation invariant to the `L`/`M` split | spread **1.03x** over `(4,64) -> (256,1)` | **PASS** |
| Q4c | fan-in must *not* be invariant | spread **7.85x**, monotone in `L` | **PASS** |
| Q1 | fan-in loses stability as `M/D` grows | ceiling slope **+0.000** vs predicted **-0.50** | **FALSIFIED** |

## Q1 was mine, and it was wrong for an instructive reason

I predicted that inflating `sigma_down` must inflate the update and so
destabilise. It does not. Under SignGD the block's **absolute** update is

    Delta E = M * eta_down * Theta(1) = M * (base/M) * Theta(1) = base

which contains **neither `sigma_down` nor `M`**. The down-projection
initialisation cannot move the update scale at all — it is dimensionally unable
to. I had reasoned about "the init is bigger, so the dynamics are more violent"
without writing down which quantity the init actually multiplies.

The identity control is what makes this verdict trustworthy: at `M = D` the two
arms are the *same model*, and the measured ceiling difference was `+0.00`
decades. So the flat slope is a real null, not a dead instrument. (The
instrument *was* also coarse — the ceiling took only two distinct grid values —
which is a second, independent reason not to have read anything into it.)

## What the init actually does — the real consequence

It changes the **initial** contribution, not the update:

    block contribution per coordinate at init  =  sigma_down * sqrt(M)

    fan-in :  M^{-1/2} * sqrt(M)     = 1            <- M-independent
    MLU    :  (sqrt D / M) * sqrt(M) = sqrt(D/M)    <- vanishes in M

Accumulating `L` blocks incoherently against the `1/L` multiplier:

    init deviation  =   fan-in: 1/sqrt(L)      MLU: sqrt( D/(L M) )

**So the MLU down-init is precisely what makes the block width `M` enter the
mean-field average.** Under fan-in the effective width of the network is `L`;
under MLU it is `L*M`. Widening the block does *nothing* for convergence to the
Neural Mean ODE under fan-in — only depth helps.

This is the mechanism behind Chizat's headline ("infinite depth behaves as
infinitely wide"), and Q4 is that headline in its purest form: at fixed `LM`,
sweeping the split from `(L,M) = (4,64)` all the way to `(256,1)` moves the MLU
init deviation by **3%**, while the fan-in arm moves by **7.85x**. `M = 1` with
large `L` sits on the same curve as `M = 64` with small `L`.

The `(L,M) = (8,32)` row of that sweep has `M = D = 32`, and the two arms agree
there to every printed digit — the identity control landing inside the sweep.

## Connection to round 006 (MoE)

The MLU boundary and the MoE down-projection init are **the same condition**,
reached by two independent routes: Chizat's necessary-and-sufficient phase
analysis, and the Stein computation in `06-moe.md` §1c
(`sigma(W_down) = alpha_ffn^{-1} n^{-1/2} = sqrt(D)/M` with `alpha_ffn = M/D`).
Round 006's measurements — init slope `-0.500`, fan-in control `-0.000`, over
`alpha_ffn = M/D` from 1 to 512 — are therefore *also* a validation of the phase
boundary of 2509.10167, in a different architecture.

**Caveat on comparing the two papers' formulas directly.** Chizat bounds
`||h||_2 = Theta(1)`; this program and the MoE paper use per-coordinate
`Theta(1)`, i.e. `||h||_2 = Theta(sqrt D)`, which is what pre-LN enforces. Their
printed "residual scale" numbers are therefore not comparable term by term
without matching conventions, and an earlier draft of `08` asserted an exact
identity that had not done that matching. The convention-independent statement is
the `sqrt(M/D)` ratio above, which is what is claimed and measured.

## Not done

- **Q5** (lazy branch, `alpha* = (ML)^{1/4}`) — not measured.
- The **trained** error to the limit, i.e. the `1/L` Euler term and the
  constants `a`, `b` in their Figure 2. Everything above is at initialisation,
  which is why it is clean but also why it says nothing about the
  discretisation half of the bound.
- The `P^{-1/6}` optimal shape (`L = P^{1/6}, M = P^{1/2}, D = P^{1/3}`,
  derived in `08` §5) — derived, not tested.
