# The Neural Mean ODE, the MLU residual scale, and why depth acts as width

> **Method: heuristic scale analysis + CLT accounting.** Targets:
> **arXiv 2509.10167** (Chizat, *The Hidden Width of Deep ResNets*) and
> **arXiv 2603.18168** (Chaintron–Chizat–Maass, *ResNets of All Shapes and
> Sizes*). These are rigorous-probability papers; this file derives their
> *rates* by scale counting and checks the exponents, it does not reproduce
> their proofs.

## 0. The model (their §2.1)

    h^0 = x in R^D
    h^l = h^{l-1} + (alpha / (L M)) sum_{j=1}^{M} phi(h^{l-1}, Z^{j,l})
    GD:   Z^{j,l}_{k+1} = Z^{j,l}_k - (L M eta / alpha^2) grad_{Z^{j,l}} Lhat

`M` is the block's hidden width (for a 2LP block, `phi(h,z) = w sigma(<u,h>)`
with `z = (u,w)`, so the block *is* a 2LP with `M` hidden units); `D` the
embedding dimension; `L` the depth. The limit is the **Neural Mean ODE**

    d/ds h(s,x) = alpha * E[ phi(h(s,x), Z(s)) ]

## 1. Where the two error terms come from

This is the whole content, and both terms are elementary once the sum is written
with its indices exposed. Write `phibar(h,s) = E[phi(h, Z(s))]`. Then

    h^L - h^0 = (alpha/(LM)) sum_{l=1}^{L} sum_{j=1}^{M} phi(h^{l-1}, Z^{j,l})

              = alpha * (1/L) sum_l phibar(h^{l-1}, s_l)          <- Riemann sum
                + (alpha/(LM)) sum_{l,j} [ phi(h^{l-1},Z^{j,l}) - phibar ]   <- fluctuation

**Term 1 — the Riemann sum** is the Euler discretisation of the Mean ODE on a
grid of `L` points, so it carries the standard first-order error

    O(1/L)

**Term 2 — the fluctuation** is a centred sum of **`L * M` independent terms**,
each `O(1)`, against a prefactor `alpha/(LM)`. By the CLT its norm is

    (alpha/(LM)) * sqrt(L M) * ||phi - phibar||  =  alpha * ||phi - phibar|| / sqrt(L M)

and since `phi - phibar` is a vector in `R^D` with `O(1)` coordinates, its norm
carries a `sqrt(D)`:

    O( alpha * sqrt(D) / sqrt(L M) )  =  O( alpha * sqrt( D / (L M) ) )

Adding them gives their Theorem 1 / Theorem 3 bound

    O( 1/L + sqrt( D / (L M) ) )

## 2. "Infinite depth behaves as infinitely wide" — the mechanism

The units of the network are indexed by the **pair** `(j,l) in [M] x [L]`, and
the mean-field average that defines the limit runs over **all `L M` of them**,
not over the `M` units of a single layer. So the CLT sees `LM`. That is the
entire "hidden width" claim, and three consequences follow immediately:

1. **`M` need not diverge.** `M = 1` with `L -> inf` gives fluctuation
   `1/sqrt(L)` and converges to the *same* limit. Depth alone supplies the
   averaging.
2. **`L` and `M` are interchangeable in term 2, but not in term 1.** At fixed
   product `LM` the fluctuation is invariant while the Euler error `1/L` is not.
   **This is the sharpest available test of the claim** and it is testable *at
   initialisation*, with no training at all — see §5.
3. Depth therefore buys two things at once (discretisation *and* averaging)
   while width buys only one, which is why the bound is asymmetric in `L` and `M`.

## 3. The MLU residual scale, and an identity with the MoE paper

They define the **residual scale** as (branch multiplier) x (init scale of the
block's output layer), and prove `O(sqrt(D)/(L M))` is **necessary and
sufficient** for maximal local feature updates. Deriving it: the branch
multiplier is `alpha/(LM)`, so with output-layer init `s_w` the condition
`alpha * s_w = Theta(sqrt D)` gives, at `alpha = Theta(1)`,

    s_w = Theta( sqrt(D) )   per unit, i.e. residual scale = sqrt(D)/(L M)

**A normalisation warning before comparing papers.** Chizat bounds
`||h||_2 <= R = Theta(1)`; the MoE setup (and this program generally) uses
per-coordinate `Theta(1)`, i.e. `||h||_2 = Theta(sqrt D)`, which is what pre-LN
enforces. The two "residual scale" numbers are therefore **not** directly
comparable term by term, and an earlier draft of this file asserted an exact
identity without matching the conventions. What *is* convention-independent is
the statement below, so that is what is claimed and tested.

**The invariant statement.** Write the block as
`h^l = h^{l-1} + (1/L) W_down phi(W_up^T h^{l-1})` with `W_down in R^{D x M}` and
per-entry init `sigma_down`, in the convention `||h|| = Theta(sqrt D)`. Then

    fan-in / NTP :   sigma_down = M^{-1/2}
    MLU / mean-field: sigma_down = sqrt(D) / M

    ratio = sqrt(D/M)          equal  <=>  M = Theta(D)

`06-moe.md` §1c derives the second by a Stein computation on the
sign-correlation (`sigma(W_down) = alpha_ffn^{-1} n^{-1/2}` with
`alpha_ffn = M/D`, which is `sqrt(D)/M`). Chizat obtains the same boundary by a
necessary-and-sufficient phase analysis, in his own normalisation, and proves it
for general `(L, M, D)`. **Two papers, two independent routes, one boundary** —
an F14-grade cross-check.

**Why this is not just CompleteP.** CompleteP uses fan-in, so it is the
`M = Theta(D)` slice, exactly as Chizat says. Off that slice fan-in is wrong by
`sqrt(M/D)`. Real FFNs run `M = 4D`, so fan-in is off by a factor 2 — small and
systematic; MoE with a large expert-width multiplier, or any wide-block ResNet,
pushes `M/D` far higher.

## 3a. Consequences of the new down-projection init — the practical content

Everything below follows from the single factor `sqrt(D/M)` and is what actually
bites in practice. `alpha_ffn = M/D` throughout.

| # | consequence | status |
|---|---|---|
| **C-a** | fan-in and MLU **coincide iff `M = Theta(D)`**; all of CompleteP's validation lives on that slice and says nothing about `M != Theta(D)` | derived |
| **C-b** | under MLU the block's *init* contribution vanishes as `alpha_ffn^{-1/2}` while its *update* stays `Theta(1)` — so the wide-block limit is genuinely feature-learning, and `alpha_ffn`-independent | **measured**: init `-0.500` vs `-0.50`; update `+0.011` vs `0.00` (round 006 P3) |
| **C-c** | under fan-in the block's *update* scale instead **grows** as `alpha_ffn^{+1/2}`, so the optimal LR must fall as `alpha_ffn^{-1/2}` and **HP transfer across the FFN ratio breaks** | **measured**: control term B `+0.512` vs `+0.50`; optimal-LR slope `-0.193 -> -0.308` toward `-0.5` (round 006 P3c/P6c) |
| ~~C-d~~ | ~~at fixed LR, fan-in destabilises as `M/D` grows~~ | **FALSIFIED** (round 007 Q1): fan-in's stable-LR ceiling slope is `+0.000` against my predicted `-0.50` |
| **C-d'** | **the MLU init is exactly what makes the block width `M` enter the mean-field average at all.** Under fan-in the effective width is `L`; under MLU it is `L*M` | **measured** (round 007 Q3/Q4) — see below |

**Why C-d was wrong.** I argued fan-in inflates the update and so must destabilise.
It does not: under SignGD the block's *absolute* update is
`Delta E = M * (base/M) * Theta(1) = base`, **independent of `sigma_down` and of
`M`**. The down-projection init cannot move the update scale at all. What it moves
is the *initial* contribution:

    block contribution per coordinate at init  =  sigma_down * sqrt(M)

    fan-in :  M^{-1/2} * sqrt(M)      = 1              -> M-independent
    MLU    :  (sqrt(D)/M) * sqrt(M)   = sqrt(D/M)      -> vanishes in M

so, accumulating `L` blocks incoherently against the `1/L` multiplier,

    init deviation ||h^L - h^0||  =  fan-in: 1/sqrt(L)      MLU: sqrt(D/(L M))

**That is the whole consequence.** Widening the block does *nothing* for
convergence to the Mean ODE under fan-in — only depth helps. Under the MLU init,
`L` and `M` are perfectly interchangeable through the product `L M`, which is
precisely Chizat's "infinite depth behaves as infinitely wide". The new
down-projection init is not a tuning refinement; it is the thing that buys the
`M`-averaging.

Measured (round 007, at initialisation, no training):

| | slope in `L` | slope in `M` | slope in `D` |
|---|---|---|---|
| MLU, predicted | `-1/2` | `-1/2` | `+1/2` |
| MLU, measured | **-0.502** | **-0.501** | **+0.510** |
| fan-in, predicted | `-1/2` | `0` | `0` |
| fan-in, measured | **-0.502** | **-0.002** | **+0.009** |

and at fixed `L M = 256`, sweeping the split from `(L,M) = (4,64)` to `(256,1)`:
**MLU spread 1.03x** (invariant), **fan-in spread 7.85x**.

## 4. The lazy regime and the optimal `alpha`

Their Theorem 2: for `1 << alpha << sqrt(ML)` the ResNet approaches the **Neural
Tangent ODE** (the linearisation of the Mean ODE's drift about its init) with

    O( 1/alpha + 1/L + alpha/sqrt(M L) )

`alpha` is a richness dial, and it is the analogue of this program's `gamma_0`:
small `alpha` = rich/MLU, large `alpha` = lazy/linear. The two `alpha` terms
trade off, and minimising `1/alpha + alpha/sqrt(ML)` gives

    alpha* = (M L)^{1/4},     error at alpha* = 2 (M L)^{-1/4}

so the *best possible* lazy approximation degrades only as `(ML)^{-1/4}` — worse
than the MLU rate `(ML)^{-1/2}`. **Prediction P4 below.**

## 5. The optimal shape at fixed parameter budget

Paper 2 adds the large-`D` limit of the Mean ODE at rate `1/sqrt(D)`, giving

    O( 1/L + sqrt( D/(L M) ) + 1/sqrt(D) )

With `P = Theta(L M D)` parameters, balance all three terms:

    1/L = 1/sqrt(D)          =>  D = L^2
    1/L = sqrt(D/(LM))       =>  M = D L = L^3

    P = L * M * D = L * L^3 * L^2 = L^6

    =>  **L = P^{1/6},  M = P^{1/2},  D = P^{1/3},  error = O(P^{-1/6})**

which is their `O(P^{-1/6})`. The *shape* is the checkable part: at a fixed
budget the optimum is `M >> D >> L`, i.e. **wide blocks, moderate embedding,
shallow depth** — the opposite of the "make it deeper" instinct.

## 6. Registered predictions (`rounds/007-mean-ode/prereg.md`)

Ordered by what the down-projection init actually buys.

| # | prediction | control |
|---|---|---|
| **Q1 (primary)** | **C-d**: at fixed LR, fan-in loses stability as `alpha_ffn = M/D` grows; MLU does not. The stable-LR ceiling falls as `alpha_ffn^{-1/2}` under fan-in and is flat under MLU | the two must **coincide at `alpha_ffn = 1`** (`M = D`) — if they differ there, the comparison is confounded |
| **Q2** | **C-a**: the two inits agree exactly at `M = Theta(D)` and diverge as `sqrt(M/D)` off it | measure the ratio directly; must be `1.0` at `M = D` |
| **Q3** | init deviation `||h^L - h^0|| ~ sqrt(D/(LM))`: slope `-1/2` in `L` and in `M`, `+1/2` in `D` | width-only and depth-only rows must both bite |
| **Q4** | **at fixed `LM`, the init deviation is invariant to the `L`/`M` split** — "infinite depth behaves as infinitely wide", in its purest form | a quantity depending on `M` alone must *not* be invariant |
| **Q5** | lazy branch: error minimised at `alpha* = (ML)^{1/4}` with value `~ (ML)^{-1/4}` | — |

Q3/Q4 are measurable **at initialisation** — no training, no optimiser
confounds, nearly free. Q4 is the one that would falsify the papers' headline.
Q1 is the one a practitioner would act on.
