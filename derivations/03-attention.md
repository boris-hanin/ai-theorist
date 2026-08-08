# Multi-head attention: first-principles derivation, then checked against 2405.15712

> **Method: MIXED.** D4 is DMFT (cavity, class-(b) reused edge). D1, D2, D3, D5, D6 are **heuristic one-step scale analysis** — they establish the parameterisation, not the dynamics. An earlier version of this header claimed the whole file used the cavity method; it does not.

Derived from the architecture in §2.1 of the paper **before** reading the paper's
results. Note the method split flagged above: only D4 uses the cavity method of
`00-method.md`; the rest is heuristic scale counting, then compared against
its Table 1, §3.3 and Result 1 (Eq. 8). Each item below records what was
derived independently and whether it matched.

**Scoreboard: 4 derived and confirmed; 1 error found in my own one-step
analysis (D2b), which propagated into two wrong conclusions (D5, D6) that are
now corrected.** The paper is consistent throughout; the discrepancy was mine.

The error in one line: **the coherence bookkeeping has to be done twice** — once
for `delta k . q`, and once for the backward path through `W_O` that sets the
size of `delta k`. I did the first and assumed the second, which makes a
one-step calculation look sufficient when the first step is precisely the
anomalous one.

## 0. Setup (paper §2.1, transcribed)

`N` = key/query dim per head, `H` = heads, `L` = depth, `d_model = N H`.

    h^1_s      = (1/sqrt(D)) W^0 x_s                        in R^{NH}
    ht^l_s     = h^l_s  + beta0 L^{-aL} MHSA(h^l_s)
    h^{l+1}_s  = ht^l_s + beta0 L^{-aL} MLP(ht^l_s)
    MHSA(h^l_s)= (1/sqrt(NH)) sum_h W^l_{Oh} vsig^l_{hs}
    vsig^l_{hs}= sum_{s'} sig^l_{h,ss'} v^l_{h,s'}
    v^l_{hs}   = (1/sqrt(NH)) W^l_{Vh} hbar^l_s ,  hbar = LN(h)
    A^l_{h,ss'}= N^{-aA} k^l_{hs} . q^l_{hs'}
    k^l_{hs}   = (N^{aA - 3/2} H^{-1/2}) W^l_{Kh} hbar^l_s      (same for q)
    f          = (gamma0 N H)^{-1} w^L . mean_s h^L_s

`W_{Vh}, W_{Kh}, W_{Qh} in R^{N x NH}`, `W_{Oh} in R^{NH x N}`;
`W_O, W_V` have Theta(1) entries, `W_K, W_Q` have entries Theta(N^{1-aA})`.
`aA, aL in [1/2, 1]`.

## D1. Scale of the attention logits at initialisation — CONFIRMED

Layer norm makes `hbar` have Theta(1) entries in `NH` dimensions. Then
`(W_{Kh} hbar)_i` sums `NH` terms of size `Theta(N^{1-aA})` with random signs,
so it is `Theta(N^{1-aA} sqrt(NH))`. With the stated prefactor,

    k_i ~ N^{aA-3/2} H^{-1/2} * N^{1-aA} * sqrt(NH) = N^0 = Theta(1)

so keys and queries have Theta(1) entries, as the paper states. Now `k` and `q`
come from *different* weight matrices, so at initialisation they are
uncorrelated and `k.q` is an incoherent sum of `N` Theta(1) terms:

    k.q = Theta(sqrt(N))   =>   A = N^{-aA} k.q = Theta(N^{1/2 - aA})

    **Var(A) = Theta(N^{1 - 2 aA})**

**Check:** paper §3.3 — "as `N -> inf` the distribution of `A` will approach a
Gaussian with variance `Theta(N^{1-2 aA})`." **Match.**

At `aA = 1`, `Var(A) = Theta(1/N)`: the logits *vanish* at init, so softmax
gives uniform attention. At `aA = 1/2`, `Var(A) = Theta(1)`.

## D2. What one SGD step does to `A` — CONFIRMED

The gradient of the loss w.r.t. `k` runs through `A`, and
`dA_{ss'}/dk_{s,i} = N^{-aA} q_{s',i}`. So the update `delta k` is *proportional
to* `q` — it is **aligned**, not random. Therefore `delta k . q` is a
**coherent** sum of `N` terms, not an incoherent one:

    delta A ~ N^{-aA} * N * delta k = N^{1-aA} delta k

Setting `delta A = Theta(1)` gives

    **delta k_i = Theta(N^{aA - 1})**

**Check:** paper §3.2 — "it is possible to obtain Theta(1) updates to the
attention variable for alternative values of `aA` if we choose the change to
key entries after gradient descent to be `delta k_i ~ Theta(N^{-1+aA})`."
**Match.**

At `aA = 1` this is `delta k = Theta(1)`, which is exactly Yang et al.'s muP
assumption that key/query entries move by `Theta(1)`.

**Consequence worth stating in advance:** D1 and D2 together say that at the
level of the forward pass and one step, *any* `aA` can be made to give `Theta(1)`
attention updates by rescaling the learning rate. So a hyperparameter-transfer
sweep across `N` should **not** discriminate `aA`. The paper's Fig 2(a) says
exactly this. Reaching for the transfer harness here — the instrument already
built — would have measured the wrong thing.

## D3. Learning-rate scaling — CONFIRMED

*Width.* `f = (gamma0 N H)^{-1} w . mean_s h`, so `df/dw_i = h_i/(gamma0 N H)`.
An SGD step `delta w = eta * Delta * h/(gamma0 N H)` changes the output by

    delta f = (gamma0 N H)^{-1} delta w . h = eta Delta |h|^2 / (gamma0 N H)^2

and `|h|^2 = Theta(NH)`, giving `delta f ~ eta Delta / (gamma0^2 N H)`.
For `delta f = Theta(1)`: **`eta ∝ N H`.**

*Depth.* Block `l` contributes `beta0 L^{-aL} F_l` to the stream. Its weight
update carries one factor `L^{-aL}` from the backward pass through the branch,
and its effect on `h` carries another from the forward branch scale, so each
block moves the stream by `~ eta beta0^2 L^{-2 aL}`. Summing `L` blocks:

    total delta h ~ eta beta0^2 L^{1 - 2 aL}   =>   **eta ∝ L^{2 aL - 1}**

Together: `eta = eta_0 N H L^{2 aL - 1}`.

**Check:** paper Table 1, SGD row — `eta_0 N H L^{2 aL - 1}`. **Match, exactly.**
The same computation gives a per-block stream update of `beta0^2 L^{-1}`, matching
the paper's footnote 1 (`O(gamma0 beta0^2 / L)`).

## D4. Single-site structure of the key/query process — CONFIRMED

Apply M3 of `00-method.md`. `W_{Kh}` appears in the forward pass (making `k`)
and transposed in the backward pass (the gradient w.r.t. `hbar` runs back
through `W_K^T`). That is a **reused edge**, class (b) — so by M4 its frozen
part is not Gaussian, and the cavity expansion must produce a Gaussian source
plus an Onsager reaction term. The partner field that `k` responds to is the
one sharing the matrix, i.e. `q` (through `A = N^{-aA} k.q`). So the single-site
process must read

    k^l_{hs}(x,t) = u^l_{Khs}(x,t) + sum_{t',s'} int dx' C^{k,l}_{ss'}(x,x',t,t') q^l_{hs'}(x',t')

with `u_{Kh}` a Gaussian source whose covariance is the residual-stream kernel
`H^l` (that being the population kernel of what `W_K` multiplies, per M3).

**Check:** paper Eq. (8), second line — identical in form, and it states
`u^l_{Kh}` has covariance `H^l`. **Match.** Likewise the residual-stream
equation in Eq. (8) is a Gaussian source plus a memory integral over kernels
against backward fields, weighted by `eta_0 gamma0 beta0^2 L^{-1}` — the same
shape as the MLP case, with that prefactor playing the role of `gamma_0`.

## D2b. THE ERROR IN D2 — the first step is suppressed

D2's algebra is a correct *consistency relation*: `delta A = Theta(1)` requires
`delta k = Theta(N^{aA-1})`, and the paper's §3.2 and Table 3 agree. But I then
used it as a description of **what one SGD step actually does**, and that is
wrong.

Figure 12's caption states it directly:

> "At the first step of SGD, the updates to the keys and attention variables are
> **suppressed due to a lack of correlation between `W_O` and the gradient
> `df/dht`**. After training for multiple steps, this correlation increases and
> non-negligible updates to the attention variables occur."

and Fig 12(b) measures `(delta A)^2 ~ N^{-2}` at `t = 1`, i.e.
**`delta A|_{t=1} = Theta(N^{-1})`**, not `Theta(1)`.

**Where my argument broke.** I wrote `delta k ∝ (backward signal) x q` and
treated the backward coefficient as `Theta(1)`. It is not, at initialisation.
The gradient reaching `k` travels back through the attention output path, i.e.
through `W_O`. At init `W_O` is random and independent of `df/dht`, so that
contraction is an **incoherent** sum and carries its own `1/sqrt(.)`
suppression. Only after a step does `W_O` acquire a component aligned with the
backward signal, at which point the contraction becomes coherent and
`delta A` reaches `Theta(1)`.

So the coherence bookkeeping has to be done **twice** — once for `delta k . q`
(which I did) and once for the backward path that sets the size of `delta k`
(which I silently assumed away). Getting only the first one right is what made
a one-step calculation look sufficient.

This is not a small correction. It is the entire reason the paper's `aA = 1`
argument needs **two or more** steps, and it is why D5 and D6 below were wrong.

## D5. Why `aA = 1` is required — CORRECTED

The init entries of `W_K` are `Theta(N^{1-aA})`. The backward pass into the
residual stream runs through `W_K^T`, so it is amplified by `||W_K||`, which is
larger by `N^{1-aA}` than at `aA = 1`. At `aA = 1` the init variance is
`Theta_N(1)` and the backward pass is controlled; below it, the backward
signal carries a growing factor of `N`.

With D2b in hand this is no longer mysterious. The one-step update is
suppressed because `W_O` is uncorrelated with the backward signal at init. What
builds up over subsequent steps is exactly that correlation — and the size of
what it builds is set by `||W_K||`, whose init entries are `Theta(N^{1-aA})`.
At `aA = 1` that is `Theta_N(1)` and the accumulated backward signal stays
controlled; below it, each additional step compounds a factor growing with `N`,
so the backpropagation diverges as `N -> inf`. That is precisely the paper's
statement, and it is inaccessible to a one-step calculation **because the first
step is the anomalous one** — not merely because one step is "not enough".

The detailed two-step calculation is Appendix E.1.2; I have the mechanism, not
the full computation.

## D6. Head collapse — MY N^{-1} CLAIM WAS WRONG

Previous version of this section claimed the derivation predicts trained
across-head `Var(A) ~ N^{-1}` against the paper's `N^{-2}`, and filed the
difference as "a dynamical effect this derivation is missing". **That was my
error propagating, not a gap in the paper.**

The `N^{-1}` came from assuming the head-specific part of the update enters at
the naive scale — the same assumption D2b just destroyed. Once the first-step
suppression is included, the head-dependent contribution enters at the
suppressed scale `delta A ~ Theta(N^{-1})` (Fig 12b, `(delta A)^2 ~ N^{-2}` at
`t=1`), which is `Var ~ N^{-2}` — the paper's Fig 2(b) rate. The two figures
are consistent with each other; they were only inconsistent with my arithmetic.

What the paper's own saddle point says (Appendix E.2.1, Eq. 69–71), and which I
should have read as the primary source rather than reconstructing:

- At init, `A^l_{h}(x,0) = 0` and `Q_h = K_h = V_h = H^l` for **every** head —
  the MHSA kernels are identical across heads at `t=0` in the limit.
- The keys and queries obey `k_h = u_{Kh} + C^k * q_h`,
  `q_h = u_{Qh} + C^q * k_h`, where **`C^k` and `C^q` carry no head index** —
  the text is explicit that they "only involve deterministic head-averaged
  kernels". Head dependence enters solely through the Gaussian sources.
- The induction then runs: identical kernels ⇒ identical response functions ⇒
  identical kernels at later times. Collapse is exact in the limit, for every
  `aA`, `aL`, `L`.

So the structure is exactly the reused-edge / response-pair form of `00-method.md`
(D4 stands), and the collapse is a statement about the limit, with `N^{-2}` the
finite-`N` fluctuation around it.

**Round 005's P1/P3 still stand as written**, but their status changes: they are
now a check on a resolved picture (init `N^{-1}`, trained `N^{-2}`, both from
the paper and both consistent), not a probe of a suspected gap. If P1 returns
`-1` rather than `-2`, the first suspect is my implementation or the dataset
deviation — not the paper.

## Summary

| # | Claim | Status |
|---|---|---|
| D1 | `Var(A)_init = Theta(N^{1-2 aA})` | derived, matches §3.3 |
| D2 | `delta k = Theta(N^{aA-1})` for `Theta(1)` attention updates | derived, matches §3.2 |
| D3 | `eta = eta_0 N H L^{2 aL - 1}` | derived, matches Table 1 exactly |
| D4 | `k = u_K + C^k * q`, `u_K` covariance `H^l` | derived from M3/M4, matches Eq. 8 |
| D2b | **first SGD step is suppressed**: `W_O` uncorrelated with the backward signal at init, so `delta A|_{t=1} = Theta(N^{-1})` | **MY ERROR** — found via Fig 12 caption; invalidated D5 and D6 as first written |
| D5 | `aA = 1` required for a stable `N -> inf` limit | mechanism derived once D2b is included; full two-step computation is App E.1.2 |
| D6 | heads collapse; trained across-head `Var(A) ~ N^{-2}` | consistent with the paper once D2b is included; my earlier `N^{-1}` was the error propagating |

## What I got wrong and why it matters

I recorded D6 as "a result in the paper this derivation does not reproduce",
which framed my own arithmetic slip as a gap in someone else's physics. It was
not. The registry has a name for the general shape of this — F14 says
transcriptions can be wrong and the source must be checked — but the failure
here is narrower and worth its own note: **when a derivation and a checked
source disagree, the prior should be on the derivation being wrong**, and the
first thing to re-examine is whichever step quietly assumed an `O(1)`
coefficient. In this case that was the backward path through `W_O`.

A concrete guard for next time: in any coherence argument, enumerate *every*
contraction in the chain and label each one coherent or incoherent. I labelled
`delta k . q` and stopped.
