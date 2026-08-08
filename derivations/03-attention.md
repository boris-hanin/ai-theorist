# Multi-head attention: first-principles derivation, then checked against 2405.15712

Derived from the architecture in §2.1 of the paper using the cavity method of
`00-method.md`, **before** reading the paper's results, then compared against
its Table 1, §3.3 and Result 1 (Eq. 8). Each item below records what was
derived independently and whether it matched.

**Scoreboard: 4 derived and confirmed, 1 derived qualitatively only, 1 result
in the paper that this derivation does NOT reproduce.** That last one is the
interesting entry.

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

## D5. Why `aA = 1` is required — DERIVED ONLY QUALITATIVELY

The init entries of `W_K` are `Theta(N^{1-aA})`. The backward pass into the
residual stream runs through `W_K^T`, so it is amplified by `||W_K||`, which is
larger by `N^{1-aA}` than at `aA = 1`. At `aA = 1` the init variance is
`Theta_N(1)` and the backward pass is controlled; below it, the backward
signal carries a growing factor of `N`.

**This is the mechanism, but I have not derived the divergence.** The paper is
specific that it takes **two or more** gradient steps to appear (§3.2: "After
performing two or more gradient descent steps, we demonstrate that the
backpropagation signals will diverge as `N -> inf` unless initial key and query
weight matrices are downscaled to have variance of order `Theta_N(1)`"), and
that the reason is *not* key/query correlation. A one-step argument therefore
cannot reach it, and D2 above is a one-step argument. Recorded as **taken from
the paper, not independently derived** (Appendix E.1.2).

## D6. Head collapse — PARTIALLY DERIVED, and one factor of N unexplained

At `aA = 1`, D1 gives `Var(A) = Theta(1/N) -> 0` at init, so every head's
attention matrix approaches the same uniform-softmax value; head-to-head
differences in `A` are `O(N^{-1/2})`. The learned part `delta A = Theta(1)` from
D2 is driven by the loss, which is shared, so to leading order it is
head-independent. That gives head collapse, and predicts

    Var over heads of A, after training  ~  Theta(1/N)

**The paper's Fig 2(b) reports `O(N^{-2})`, not `O(N^{-1})`.** So training
suppresses the across-head spread by an *additional* factor of `N` beyond what
the initialisation and the leading-order update account for. This derivation
does not explain that.

This is the sharpest thing to measure in round 005, and the prediction set is
written to separate the two: **P3** measures the init exponent (derived here:
`1 - 2 aA`, so `-1` at `aA = 1`) and **P1** measures the trained exponent
(paper: `-2`). If both come out as stated, the extra factor of `N` is real and
is a dynamical effect this derivation is missing — a concrete gap to close,
not a discrepancy to paper over. If the trained exponent comes out at `-1`,
either the paper's figure is measuring a different quantity than I think, or
the replication differs (a different dataset is a live candidate — see the
round's pre-registration).

## Summary

| # | Claim | Status |
|---|---|---|
| D1 | `Var(A)_init = Theta(N^{1-2 aA})` | derived, matches §3.3 |
| D2 | `delta k = Theta(N^{aA-1})` for `Theta(1)` attention updates | derived, matches §3.2 |
| D3 | `eta = eta_0 N H L^{2 aL - 1}` | derived, matches Table 1 exactly |
| D4 | `k = u_K + C^k * q`, `u_K` covariance `H^l` | derived from M3/M4, matches Eq. 8 |
| D5 | `aA = 1` required for a stable `N -> inf` limit | mechanism only; divergence taken from Appendix E.1.2 |
| D6 | heads collapse; trained across-head `Var(A) ~ N^{-2}` | collapse derived; the `N^{-2}` (vs `N^{-1}`) is **not** derived |
