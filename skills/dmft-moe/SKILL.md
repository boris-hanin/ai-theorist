---
name: dmft-moe
description: Parameterisation and mean-field structure for Mixture-of-Experts layers (Jiang-Bordelon-Pehlevan-Hanin, arXiv 2601.20205) — scaling width, depth, expert count and expert size at fixed sparsity, the non-fan-in down-projection init, the alpha_* ODE/SDE criterion, and the routing quantile threshold. Use for MoE HP transfer and MoE scaling limits.
---

> **REWRITTEN from round 006 (`rounds/006-moe/`), replacing a RECONSTRUCTED
> file.** The previous version's central claim — a rule `eta_up ∝ alpha` with
> `alpha` the *sparsity* — appears nowhere in 2601.20205 and contradicts the
> derivation: it conflated `kappa` (sparsity) with `alpha_ffn` (expert width
> multiplier) and the up projection with the down projection. Per F14 the source
> and the measurement win. Everything below is derived in
> `derivations/06-moe.md` and measured in `rounds/006-moe/results.md`.

# MoE parameterisation (delta on `dmft-derivation`)

Prerequisite: `dmft-derivation` Phases 0–1, and `05-completep-dmft-sgd.md` for
the depth sector — the MoE residual multiplier is `1/L`, i.e. **CompleteP
`alpha = 1` unchanged**. MoE adds three dials: expert count `E`, expert width
`m = alpha_ffn * n`, and sparsity `kappa = a/E`.

## The single most important structural fact

**Sparsity `kappa` is held FIXED, not scaled** (paper §3.2). It appears in *none*
of the scaling rules. Scaling `E` with `a` fixed (`kappa -> 0`) is a different
problem and these rules do not apply to it. If a question involves `kappa`
changing, stop and say so.

## The parameterisation (their Table 1), with what is and isn't standard

| group | init std | LR (Adam/SignGD) |
|---|---|---|
| router | `n^{-gamma}`, `gamma >= 1/2` | `n^{-1}` |
| expert bias | `0` (main text) / **nonzero** (App. E) — see F21 below | `Theta(1)` |
| expert up `W_up` | `n^{-1/2}` | `n^{-1}` |
| expert down `W_down` | **`alpha_ffn^{-1} n^{-1/2}`** | `alpha_ffn^{-1} n^{-1}` |

- **`W_up` is blind to `alpha_ffn`.** Both its forward sum and its update sum run
  over the *input* index `n`, never over `m`. Do not attach `alpha_ffn` to it.
- **`W_down`'s LR is ordinary muP at fan-in `m`** (`eta = 1/m`). Nothing exotic.
- **`W_down`'s init is `alpha_ffn^{-1/2}` BELOW fan-in.** This is the one
  non-standard entry and the one worth checking any implementation against. It
  is *mean-field*, not NTP: it makes `E_k(init) = Theta(alpha_ffn^{-1/2})` while
  `Delta E_k = Theta(1)`, so the trained part dominates and the limit is
  `alpha_ffn`-independent. Measured: init slope **-0.500** vs `-0.50`; fan-in
  control **-0.000** vs `0.00`.
- **The readout obeys the same condition one level up**: init `n^{-1}`, not
  fan-in `n^{-1/2}`. Getting this wrong leaves a random `Theta(1)` function in
  the width limit and breaks width transfer (measured: 11.5 s.e. -> 2.0 s.e. on
  the fix). This is the most likely bug in a fresh implementation.

## The three-level hierarchy is one statement, applied three times

Trained-coherent beats init-incoherent, at three aggregations:

| level | average over | init | trained | ratio |
|---|---|---|---|---|
| within expert | `m` units | `alpha_ffn^{-1/2}` | `Theta(1)` | `alpha_ffn^{-1/2}` |
| over experts | `a` active | `(a alpha_ffn)^{-1/2}` | `Theta(1)` | `a^{-1/2}` |
| residual stream | `L` blocks | `(L a alpha_ffn)^{-1/2}` | `Theta(1)` | `L^{-1/2}` |

## `alpha_*` is not an assumption

    alpha_* = n_embd / (n_hid n_exp L)  =  (1/kappa) * Var[ init contribution to the stream ]

Square the last row above. `alpha_* = 0` -> neural ODE; `alpha_* > 0` -> neural
SDE; any joint scaling with the same `alpha_*` gives the same limit. Verified
dial by dial: `-1.005` (`L`), `-1.000` (`E`), `-0.988` (`alpha_ffn`), and
`-0.056` in `n`, which must not appear. Use this as the first check on any
proposed MoE joint-scaling rule: compute `alpha_*` and see what it says.

## Routing in the limit

Selection becomes a **deterministic quantile threshold** on `q = sigma(r) + b`,
because top-`a`-of-`E` on an exchangeable population is thresholding at the
`(1-kappa)` quantile. At `gamma = 1/2` the logits are standard normal so the
threshold has a closed form with no free parameters:

    q*(kappa) = sigmoid( Phi^{-1}(1 - kappa) )

and in the paper's **Appendix E** convention (router zero, diversity from random
biases) it is a bare Gaussian order statistic:

    q*(kappa) = 1/2 + b_std * Phi^{-1}(1 - kappa)

verified at `kappa` = 1/8, 1/4, 1/2, 3/4 (worst 1.43 s.e.). Threshold fluctuation
falls as `E^{-1/2}` (measured `-0.559`).

**The bias init is load-bearing (F21).** There are two conventions and they must
not be mixed: the main text zeroes the biases and lets the router's `n^{-gamma}`
noise carry init diversity; App. E zeroes the router and lets random `b_k(0)`
carry it. Take `b = 0` from one and `r = 0` from the other and every expert has
an identical gate, `top_k` breaks the tie by index, and the same experts serve
every token in every layer — a fixed subnetwork, not sparse routing. **The loss
still goes down**, so this is invisible unless you look at selection statistics.
Token-dependent routing does develop from the App. E init (`0.000 -> 0.189`
over 30 steps), as `Delta r = Theta(1)` requires.

## The crossover — quote the asymptotic exponent only where it applies

The coherent and incoherent parts of `(W_down s)_k` are in ratio
`sqrt(alpha_ffn)`, so they are **comparable at `alpha_ffn ~ 1`**. Fitted
crossover `c^2 = 0.551`, i.e. the asymptotic regime needs `alpha_ffn >> ~1.8`.
Consequences:

- Do not test the `alpha_ffn` exponents at `alpha_ffn` of a few and call a
  mismatch a failure — measure at `alpha_ffn >= 16`.
- Transfer across `alpha_ffn` is weakest exactly at `alpha_ffn = 1`, which is
  the paper's base config.

## Constant `rho = L M / D` (Jiang--Chizat compatibility)

Jiang's raw expert-down scale is `sqrt(D)/M`, and the residual branch supplies
`1/L`. Therefore `sqrt(D)/(L M)` is the **effective residualized scale**. Do
not put it into the raw down matrix or depth is counted twice.

At constant `rho`, `M=rho D/L`, so the residualized down initialization becomes
`1/(rho sqrt(D))` and the residualized down LR becomes `1/(rho D)`: both are
depth-independent in the normalized width coordinate. The full substitution,
including every Adam epsilon and the tied embedding/unembedding boundary, is in
`derivations/15-jiang-moe-constant-rho.md` and executable with
`scripts/constant_rho_compatibility.py`.

Two qualifications are mandatory:

- `alpha_* = D/(MEL) = 1/(rho E)`. Fixed `E` preserves a finite neural-SDE
  sector; only growing `E` at fixed `kappa` drives the family to the universal
  neural ODE.
- `alpha_ffn=M/D=rho/L`. Fixed rho with unbounded depth eventually crosses
  below the measured expert-down crossover. A finite ladder can be compatible;
  an unlimited constant-rho depth limit is not already validated.

## Checks specific to this skill

- **Down-init check first.** Confirm `sigma(W_down) sqrt(m) ~ alpha_ffn^{-1/2}`
  before trusting anything else; it is the load-bearing non-standard entry.
- **Readout-init check.** `f(init)` must vanish with width, not be `Theta(1)`.
- **`alpha_*` dial-by-dial**, never a single joint fit — a joint fit can hide
  compensating errors across `L`, `E`, `alpha_ffn`.
- **Enumerate the two contractions of `Delta E` separately** (term A
  `dW_down phi(hup)`; term B `W_down d[phi(hup)]`). They have *different*
  `alpha_ffn` scaling under fan-in, so the total is not a clean probe (F18).
- **Pair the comparisons.** Dial sweeps share seeds and data, so the floor is the
  paired one, not the seed-to-seed spread (F20).
- **Assert nonzero spread of the gating score across experts at init** before
  anything else (F21). Cheap, and it catches a degeneracy nothing else reveals.
- Report per-expert load beside any transfer claim (F12).
- Dense limit: `E = a = 1`, `kappa = 1` must reduce to the plain residual-MLP
  results of `05-completep-dmft-sgd.md`.

## Not established here

The full DMFT (single-site processes, response kernels) for MoE — see
`derivations/07-moe-dmft.md` for status. This file is scale counting plus the
structural part of the limit, so by F18 it claims the `t -> large` labelling.
