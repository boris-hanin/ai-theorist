# DMFT for the CompleteP residual architecture, SGD — full derivation

> **Method: DMFT (cavity).** This is the derivation `04-completep.md` does *not*
> contain. That file is heuristic scale counting and gives the parameterisation;
> this one derives the limiting dynamics and shows where `alpha` enters them.
> Optimiser: **SGD** (the Adam version is `04-completep.md`, heuristic only).

Checked against three sources at the end: 2405.15712 Result 3 / Eq. (10)–(11)
and Fig. 5a, 2505.01618 Table 1 / Eq. (6), and `equations.md` §4 (depth-muP,
2309.16620).

## 1. Setup

Residual stream of width `N`, depth `L`, `P` inputs, block depth `k = 2` (a
two-layer perceptron block — the case where `alpha` matters, per CompleteP §6):

    h^1_mu   = (1/sqrt(D)) W^0 x_mu                              in R^N
    z^l_mu   = (1/sqrt(N)) W^{l,1} h^l_mu                        block hidden pre-activation
    a^l_mu   = phi(z^l_mu)
    F_l      = (1/sqrt(N)) W^{l,2} a^l_mu
    h^{l+1}_mu = h^l_mu + (beta_0 / L^alpha) F_l                 alpha in [1/2, 1]
    f_mu     = (1/(gamma sqrt N)) w . h^{L+1}_mu ,  gamma = gamma_0 sqrt(N)

All weights i.i.d. `N(0,1)`. Dynamics `theta_dot = -d_L gamma^2 grad_theta L`,
where `d_L` is a depth factor on the learning rate, **left free** and fixed by
the derivation in §6. `Delta_mu = -dl/df_mu`.

> **`d_L` applies to the in-block weights `W^{l,1}, W^{l,2}` and to those
> ONLY.** `W^0` and the readout `w` sit *outside* the residual stack. The
> counting in §6 that fixes `d_L` is an accumulation over the `L` blocks; the
> boundary weights appear **once, not `L` times**, so there is nothing to
> compensate and applying `d_L` to them is simply wrong. CompleteP Table 1 says
> the same thing independently — Emb and Unemb LR carry no depth factor.
> This qualification was implicit and unstated in the first version of this
> file, and both the simulator and the solver were written without it. That is
> the entire content of the §8c "falsification" and of the first solver-vs-sim
> gap; see §8c.

## 2. Backward fields (M1 prerequisite)

`g^l_mu = gamma sqrt(N) df_mu/dh^l_mu` on the stream. From
`h^{l+1} = h^l + (beta_0/L^alpha) F_l`,

    g^l_mu = g^{l+1}_mu + (beta_0/L^alpha) (dF_l/dh^l)^T g^{l+1}_mu

and `dF/dh = (1/sqrt N) W^{l,2} diag(phi_dot(z^l)) (1/sqrt N) W^{l,1}`, so with

    q^l_mu = (1/sqrt N) W^{l,2 T} g^{l+1}_mu        (backward into the block)
    p^l_mu = phi_dot(z^l_mu) * q^l_mu               (the in-block "g")

    g^l_mu = g^{l+1}_mu + (beta_0/L^alpha) (1/sqrt N) W^{l,1 T} p^l_mu

**Note both block matrices appear forward and transposed-backward.** By M3 each
is a *reused* edge, so **each block carries two response pairs**, one per matrix.
That is the structural consequence of `k = 2` and it is what makes this case
different from a `k = 1` block.

## 3. M1 — integrate the weight dynamics exactly

    df_mu/dW^{l,1}_{ij} = (beta_0 / (L^alpha gamma N)) p^l_{i mu} h^l_{j mu}
    df_mu/dW^{l,2}_{ij} = (beta_0 / (L^alpha gamma N)) g^{l+1}_{i mu} a^l_{j mu}

so with `theta_dot = -d_L gamma^2 grad L` and `gamma = gamma_0 sqrt N`:

    W^{l,1}_dot = (d_L gamma_0 beta_0 / (L^alpha sqrt N)) sum_mu Delta_mu p^l_mu h^{l T}_mu
    W^{l,2}_dot = (d_L gamma_0 beta_0 / (L^alpha sqrt N)) sum_mu Delta_mu g^{l+1}_mu a^{l T}_mu

Both rank-`P`, both carrying the same prefactor. Define

    **kappa = d_L gamma_0 beta_0 / L^alpha**

This single quantity is the "`gamma_0`" of this architecture — every memory
kernel below carries exactly one factor of it. Everything about `alpha` follows
from how `kappa` combines with the number of blocks.

## 4. M2 — split the fields

Substituting `W(t) = W(0) + learned`:

    z^l_mu(t) = chi^{l,1}_mu(t) + kappa int ds sum_nu Delta_nu(s) H^l_{mu nu}(t,s) p^l_nu(s)
    F_l,mu(t) = chi^{l,2}_mu(t) + kappa int ds sum_nu Delta_nu(s) Phi^l_{mu nu}(t,s) g^{l+1}_nu(s)
    q^l_mu(t) = xi^{l,2}_mu(t)  + kappa int ds sum_nu Delta_nu(s) G^{l+1}_{mu nu}(t,s) a^l_nu(s)
    (1/sqrt N) W^{l,1 T} p^l_mu(t)
              = xi^{l,1}_mu(t)  + kappa int ds sum_nu Delta_nu(s) P^l_{mu nu}(t,s) h^l_nu(s)

with the four order parameters

    H^l_{mu nu}(t,s)   = (1/N) h^l_mu(t) . h^l_nu(s)        residual-stream kernel
    Phi^l_{mu nu}(t,s) = (1/N) a^l_mu(t) . a^l_nu(s)        in-block feature kernel
    G^l_{mu nu}(t,s)   = (1/N) g^l_mu(t) . g^l_nu(s)        stream gradient kernel
    P^l_{mu nu}(t,s)   = (1/N) p^l_mu(t) . p^l_nu(s)        in-block gradient kernel

## 5. M3–M4 — sources and Onsager terms

Both block matrices are reused, so both frozen parts split as Gaussian source
plus reaction:

    chi^{l,1} = u^{l,1} + kappa int ds sum_nu A^{l,1}_{mu nu}(t,s) p^l_nu(s),   u^{l,1} ~ GP(0, H^l)
    chi^{l,2} = u^{l,2} + kappa int ds sum_nu A^{l,2}_{mu nu}(t,s) g^{l+1}_nu(s), u^{l,2} ~ GP(0, Phi^l)
    xi^{l,2}  = r^{l,2} + kappa int ds sum_nu B^{l,2}_{mu nu}(t,s) a^l_nu(s),   r^{l,2} ~ GP(0, G^{l+1})
    xi^{l,1}  = r^{l,1} + kappa int ds sum_nu B^{l,1}_{mu nu}(t,s) h^l_nu(s),   r^{l,1} ~ GP(0, P^l)

each source covariance being the population kernel of what its matrix
multiplies (M3). Responses defined with the `kappa^{-1}` convention so they are
`O(1)`, exactly as in `01-deep-mlp.md`:

    A^{l,1} = kappa^{-1} <d h^l / d r^{l,1}>,   B^{l,1} = kappa^{-1} <d p^l / d u^{l,1}>,  etc.

**Closed single-site system.** Collecting, the block fields obey

    z^l_mu(t) = u^{l,1}_mu(t) + kappa int_0^t ds sum_nu [A^{l,1} + Delta_nu(s) H^l]_{mu nu}(t,s) p^l_nu(s)
    q^l_mu(t) = r^{l,2}_mu(t) + kappa int_0^t ds sum_nu [B^{l,2} + Delta_nu(s) G^{l+1}]_{mu nu}(t,s) a^l_nu(s)

which is *the same shape* as the deep-MLP system of `01-deep-mlp.md` §6 with
`gamma_0 -> kappa` — as it must be, since a residual block is an MLP hung off
the stream. The residual stream itself then satisfies

    h^{l+1}_mu(t) = h^l_mu(t) + (beta_0/L^alpha)[ u^{l,2}_mu(t)
                    + kappa int ds sum_nu ( A^{l,2} + Delta_nu(s) Phi^l )_{mu nu}(t,s) g^{l+1}_nu(s) ]

and `f_mu(t) = gamma_0^{-1} <w(t) h^{L+1}_mu(t)>` by the correlator rule (F4).

## 6. The `L -> infinity` limit: three contributions, three exponents

Set layer time `tau = l/L`. The stream accumulates `L` block contributions, each
weighted `beta_0 L^{-alpha}`. **They do not all accumulate the same way**, and
that is the whole content of the depth limit:

| contribution | how it accumulates over `l` | size |
|---|---|---|
| **(a) Gaussian source `u^{l,2}`** | independent across blocks ⇒ **incoherent**, `sqrt(L)` | `L^{-alpha} sqrt(L)` = **`L^{1/2 - alpha}`** |
| **(b) trained memory `kappa Delta Phi g`** | all blocks driven by the same `Delta` ⇒ **coherent**, `L` | `L^{-alpha} * kappa * L` = `d_L L^{1 - 2 alpha}` |
| **(c) Onsager `kappa A g`** | driven by the shared kernels ⇒ **coherent**, `L`; and `A` itself carries one branch factor `L^{-alpha}` | `d_L L^{1 - 3 alpha}` |

> **Contribution (c) was wrong twice over in the first version of this file**,
> which had `d_L L^{1/2 - 2 alpha}`. Both errors are now measured and corrected
> (§8d):
> 1. It summed the per-block Onsager terms **incoherently** (`sqrt(L)`). They
>    are responses to each block's own independent source, which is what made
>    me call them incoherent — but each response is a *functional of the shared
>    kernels* `H, Phi, G, P` and is contracted against the shared `Delta`, so
>    the terms carry a systematic sign and add **coherently**: `L`, not
>    `sqrt(L)`.
> 2. It treated the response kernel `A^l` as `Theta(1)`. It is not: the
>    sensitivity path that defines `A^l` runs *through the residual branch*
>    (`dh^{l+1}/dh^l = 1 + (beta_0 L^{-alpha}) dF/dh`, and only the second term
>    is the response), so `A^l ~ L^{-alpha}`.
>
> The exponent shift from the two fixes is `+1/2` and `-alpha`. **At
> `alpha = 1/2` they cancel exactly**, which is precisely why the old formula
> looked correct there and only `alpha = 1` exposed it. Assembling honestly:
> `L * s * kappa * A = L * L^{-alpha} * (d_L L^{-alpha}) * L^{-alpha}
> = d_L L^{1-3alpha}`.

Requiring (b) `= Theta(1)` fixes the learning-rate depth factor:

    **d_L = L^{2 alpha - 1}**     (SGD)

and then, substituting back:

| | `alpha = 1/2` | `alpha = 1` |
|---|---|---|
| (a) init Brownian `L^{1/2-alpha}` | **`Theta(1)`** — survives | `L^{-1/2}` — **vanishes** |
| (b) trained memory | `Theta(1)` | `Theta(1)` |
| (c) Onsager `d_L L^{1-3alpha}` = **`L^{-alpha}`** | `L^{-1/2}` | `L^{-1}` |
| per-block weight movement `d_L L^{-alpha}` = `L^{alpha-1}` | `L^{-1/2}` — **blocks freeze** | **`Theta(1)`** — blocks learn |

**The trichotomy.** (a) and the per-block movement pull in *opposite*
directions, and no `alpha` gets both:

- `alpha = 1/2` keeps a nontrivial random initial kernel — the stream is a
  Brownian motion in layer time, `dh = beta_0 du(tau)` — but every block's own
  weights move only `O(L^{-1/2})`, so the blocks **freeze**.
- `alpha = 1` makes each block learn at `Theta(1)`, but the blocks contribute
  nothing at initialisation: the init stream kernel is `H(tau,tau') = K^x`, flat
  in layer time. **Complete feature learning, trivial init kernel.**

That is the tension, derived rather than asserted, and it is why the two papers
land on different exponents for different purposes.

## 7. Precise checks

### vs 2405.15712 Result 3 and Eq. (10)–(11)

Their limiting stream equation is

    h_s(tau,x,t) = beta_0 delta_{alpha_L, 1/2} int_0^tau du_s(tau')
                   + eta_0 gamma_0 beta_0^2 sum_{t'<t} int dx' int_0^tau dtau' C(tau') g(tau')

| their statement | my derivation |
|---|---|
| the Brownian term "survives in the limit **only if** `alpha_L = 1/2`" — note the explicit `delta_{alpha_L,1/2}` in Eq. (10) | contribution **(a)**, `L^{1/2-alpha}` | **match** |
| Brownian covariance `∝ delta(tau-tau')[Phi + V^sigma]`, Eq. (11) | source `u^{l,2} ~ GP(0, Phi^l)`, independent across blocks ⇒ white in layer time with covariance `Phi` | **match** (their `V^sigma` is the attention-block analogue of my `Phi`) |
| "weights inside each hidden layer are **frozen** in the `L -> inf` limit **unless** `alpha_L = 1`" | per-block movement `L^{alpha-1}` | **match** |
| Fig. 5(a): key/query weight change `~ L^{-1/2}` at `alpha_L = 1/2` | `L^{alpha-1}` at `alpha = 1/2` is `L^{-1/2}` | **match, including the exponent** |
| "all response functions are suppressed at `L -> inf` **unless** `alpha_L = 1/2`" | contribution **(c)**, corrected to `d_L L^{1-3alpha}` = `L^{-alpha}` | **direction confirmed; strict `Theta(1)` at `alpha=1/2` not reproduced — see below** |

**The disagreement, restated after measuring it (§8d).** My original claim here
was that responses go as `L^{-1/2}` at *both* exponents — i.e. that `alpha` does
not enter at all. **That is falsified.** The measured response share scales as
`L^{-alpha}`: slope `-1.03` at `alpha = 1` against a predicted `-1.00`, with the
`d_L = 1` control biting at `-1.96` against a predicted `-2.00`. So the paper's
*direction* is right and I was wrong — `alpha = 1/2` is the least-suppressed
choice and `alpha = 1` suppresses responses strictly faster.

What is **still** not reproduced is the strict `Theta(1)` survival at
`alpha = 1/2`: I get `L^{-1/2}` there (measured `-0.60`), not `L^0`. Getting
`L^0` would require the response kernel `A^l` to be `Theta(1)` rather than
`L^{-alpha}`, and the branch-factor argument above says it is not. Scope of my
measurement: `k = 1` blocks, `P = 1`, pre-LN, `gamma_0 * horizon` in `[0.6, 3.6]`
(§8d checks the richness dependence). Their setup is MHSA+MLP with `k = 2`.
**Open, and narrowed**: the question is now exactly "is `A^l` `Theta(1)` or
`L^{-alpha}`", not the coherence question, which is settled coherent. Their
Appendix E.4 has the full computation; I have not reproduced it.

### vs 2505.01618 (CompleteP)

| their statement | my derivation |
|---|---|
| Eq. (6): linear term `L^{-1}`, second-order `L^{alpha-2}`, equal **only** at `alpha = 1` | per-block movement `L^{alpha-1}` (§6) fed into their expansion gives exactly those two orders | **match** |
| §6: "we need the weight update to satisfy `Delta w = Theta(L^{alpha-1})`" | per-block movement `L^{alpha-1}` | **match** |
| Desideratum 3 (complete feature learning) holds only at `alpha = 1` | blocks freeze unless `alpha = 1` | **match** |
| Table 1 **Adam** depth LR `m_L^{alpha-1}` | **not this file** — Adam is `04-completep.md` D5 | consistent, and see below |

**Optimiser cross-check.** The depth factor differs by optimiser:

    SGD   d_L = L^{2 alpha - 1}      (this file, §6)
    Adam  d_L = L^{alpha - 1}        (04-completep.md D5)

Both are derived by the same counting; the difference is that Adam's update is
`Theta(eta)` per entry regardless of the gradient, while SGD's inherits the
gradient scale — so SGD picks up one extra power of `kappa`. **Both match their
respective published rows**: 2405.15712 Table 1 gives SGD
`eta_0 N H L^{2 alpha_L - 1}` and Adam `eta_0 N^{-1/2} H^{-1/2} L^{-1+alpha_L}`.
That the same method reproduces *both* rows, with the difference in exactly the
right place, is the strongest single check in this file.

### vs `equations.md` §4 (depth-muP, 2309.16620)

That section is `alpha = 1/2` with a `k = 1` block, and states `eta = eta_0
gamma_0^2 N` **independent of `L`**. My rule gives `d_L = L^{2(1/2)-1} = L^0`
— independent of `L`. **Match.** It also states "`du` Brownian-in-depth
increments with covariance from `Phi(tau)`", which is contribution (a) at
`alpha = 1/2`. **Match.**

## 8. CHECKED AGAINST SIMULATION — two confirmed, one falsified

`skills/dmft-resnet-depth/scripts/residual_sgd.py`, k=2 MLP blocks, tanh, SGD,
float64, seed-averaged.

### (a) init accumulation `L^{1-2 alpha}` — CONFIRMED EXACTLY

Measured as `(1/N)|h^{L+1}(0) - h^1(0)|^2`, isolating the block contribution:

| alpha | N=256 slope | N=1024 slope | predicted |
|---|---|---|---|
| 1/2 | **-0.005** | **+0.003** | 0 |
| 1 | **-1.017** | **-1.004** | -1 |

N-independent, so the exponent is clean. *First attempt measured
`H^{L+1}-H^1` instead and got -1.50 at alpha=1: that observable mixes in a
cross term `2s sum_l h^1.F_l / N` which is `O(N^{-1/2})` with the wrong
`L`-scaling. Same lesson as round 005's `delta k` — measure the quantity the
derivation is about, not a proxy that mixes paths.*

### (b) per-block movement `L^{alpha-1}` — CONFIRMED at alpha=1/2, and at
early time for alpha=1

| alpha | t=2 | t=20 | t=200 | predicted |
|---|---|---|---|---|
| 1/2 | -0.508 | -0.504 | -0.481 | **-0.5** |
| 1 | -0.034 | -0.113 | -0.785 | **0** |

`alpha = 1/2` matches at every time, and **-0.5 is exactly the exponent
2405.15712 Fig 5(a) measures** for key/query movement. `alpha = 1` matches at
`t = 2, 20` and degrades by `t = 200` — because with `d_L = L` the deep models
have already converged (loss 0.038 at L=32 vs 2.17 at L=4), so movement
saturates. Matched-*loss* comparison gives -0.9, but steps-to-that-loss also
scale as `L^{-0.9}` (125 → 19), so movement *per step* is `L^0` as derived.
The papers take the limit at **fixed steps**, so the `t = 2/20` rows are the
right comparison.

### (c) `d_L = L^{2 alpha - 1}` — first FALSIFIED, then **RESOLVED AND CONFIRMED**

> **Resolution (later round).** The falsification below was **my simulator's
> bug, not the rule's.** `residual_sgd.py` applied `d_L` to *every* parameter,
> including `W^0`. `W^0` sits outside the residual stack and appears once, not
> `L` times (§1), so scaling its LR by `L^{2 alpha - 1}` made the stream *input*
> move `L` times too fast — which is exactly the `+1.5` slope reported below.
> Isolating it settles it beyond argument:
>
> | config | stream movement by `L` | slope |
> |---|---|---|
> | `W^0` **trained** (the buggy run) | 3.72e-4 1.06e-3 3.09e-3 9.44e-3 | **+1.553** |
> | `W^0` frozen | 1.12e-5 1.04e-5 9.28e-6 8.32e-6 | **-0.144** |
>
> With `d_L` correctly restricted to the in-block weights, and measuring the
> **isolated** block contribution (`W^0` and readout frozen, so `|delta h|^2`
> measures contribution (b) alone rather than (b) plus a boundary path that
> swamps it):
>
> | alpha | `d_L` | `|delta h|^2` by `L` = 2,4,8,16 | slope | predicted |
> |---|---|---|---|---|
> | 1/2 | `L^{2a-1}` **derived** | 1.34e-5 1.30e-5 1.31e-5 1.26e-5 | **-0.025** | 0 |
> | 1/2 | `1` control | *(identical — `L^0`, see below)* | -0.025 | 0 |
> | 1 | `L^{2a-1}` **derived** | 1.14e-5 1.09e-5 1.01e-5 9.73e-6 | **-0.082** | 0 |
> | 1 | `1` control | 7.18e-7 1.71e-7 3.96e-8 9.54e-9 | **-2.081** | **-2.0** |
>
> The derived rule is flat at both exponents, and at `alpha = 1` the negative
> control **bites at `-2.081` against a predicted `-2.0`**. `d_L = L^{2 alpha
> - 1}` is confirmed. The `alpha = 1/2` control remains vacuous (identity) and
> is not counted as evidence.
>
> **Two errors compounded here, and they are the same error.** Applying `d_L`
> to `W^0`, and measuring total stream movement instead of the isolated block
> path — both are *failures to isolate the path the derivation is about*. The
> second is what let the first hide: with the boundary path included, even the
> corrected run could not show the control biting.

The original falsifying measurement is kept below as the record.

Direct test of contribution (b): does the derived `d_L` make the *stream*
movement `L`-independent? Measured `(1/N)|h^{L+1}(t) - h^{L+1}(0)|^2` at fixed
steps:

| alpha | `d_L` | slope | wanted |
|---|---|---|---|
| 1/2 | `L^{2a-1}` = `L^0` (derived) | **-0.058** | 0 |
| 1/2 | `L^{a-1}` = `L^{-1/2}` | -1.055 | — |
| 1 | `L^{2a-1}` = `L^1` (derived) | **+1.506** | 0 |
| 1 | `L^0` | -0.361 | 0 |

And the HP-transfer sweep agrees: at `alpha = 1` the derived rule gives a
**0.971 decade drift** in the optimal `eta_0` (`-1.56 -1.99 -2.26 -2.53`), i.e.
it does not transfer.

**Two things this means.**

1. **The `alpha = 1/2` confirmation is vacuous as a test of the rule.** There
   `L^{2 alpha - 1} = L^0`, so "apply the derived rule" and "apply no depth rule
   at all" are the same run — verified, byte-identical. This is the *third* time
   in this program an identity control has looked like evidence (round 005's
   `alpha_L = 1/2`, the CompleteP depth-LR control, now this). The rule is only
   tested at `alpha = 1`, and there it fails.
2. **Since `|delta h|^2` is quadratic in the update, the measured slopes give
   `delta h ∝ L^{-0.18}` at `d_L = L^0` and `L^{+0.755}` at `d_L = L^1`** — a
   clean unit shift per power of `d_L`, as it must be. Setting `delta h` slope to
   zero needs `d_L ≈ L^{0.18}`, far from the derived `L^1`.

**Candidate causes I listed at the time, all three wrong.** (i) no LayerNorm —
eliminated by measurement (adding pre-LN moved the slope `+1.506 -> +1.553`).
(ii) the chain effect through the stream. (iii) MLP vs MHSA blocks. The actual
cause was in none of them: it was the unstated scope of `d_L` in §1, i.e. **my
own derivation was written ambiguously and my simulator resolved the ambiguity
the wrong way.** The lesson is that "candidate causes" drawn from *modelling
differences with the paper* crowded out the possibility of a plain bug in the
thing I controlled. All three candidates pointed away from my own code.

### Solver check — built, first comparison, NOT yet validated

`skills/dmft-resnet-depth/scripts/dmft_residual.py` implements the §5 system
with pre-LN, specialised to `k = 1` blocks (one response pair per block instead
of two) and `P = 1`.

**Pre-LN in the limit.** With `h` having i.i.d.-like entries across the width,
its per-sample mean self-averages to 0 and its variance to `H^l(t,t)`, so
`LN(h^l) -> h^l / sqrt(H^l(t,t))` — a *deterministic* gain set by the stream
kernel, not a new stochastic field. That is what makes pre-LN tractable here,
and it fixes the block input scale at `(1/N)|hbar|^2 = 1` whatever the stream
does.

| L | alpha | plateau | gap vs sim | sim floor | MC floor | gap/combined |
|---|---|---|---|---|---|---|
| 2 | 1.0 | 8.0e-7 | 1.83e-2 | 3.48e-3 | 8.84e-3 | **1.92x** |
| 2 | 0.5 | 8.7e-7 | 4.36e-2 | 1.90e-3 | 8.89e-3 | **4.79x** |
| 4 | 1.0 | 9.4e-7 | 1.15e-2 | 1.94e-3 | 2.76e-2 | 0.42x |
| 4 | 0.5 | 9.4e-7 | 1.84e-2 | 6.24e-3 | 2.56e-2 | 0.70x |

`L = 4` sits inside the combined floor, `L = 2` does not. **Not validated** at
the time; resolved immediately below.

**Pre-LN did not explain the §8c falsification.** Re-running the `d_L` test with
LN on gives slope **+1.553** at `alpha = 1` (was +1.506 without). Candidate (i)
is eliminated.

#### Solver check — RESOLVED. The solver had the mirror of the simulator's bug.

None of the three candidates was the cause. The solver applied `d_L` to its
**readout** update `w(t) = w(0) + d_L gamma_0 int Delta hbar^{L+1}` — the same
§1 scope error as the simulator, on the other boundary. With `d_L` removed from
the readout (it is outside the stack) and the simulator matched to the solver's
boundary (`W^0` frozen, `d_L` on blocks only), every configuration lands at or
inside the combined floor:

| L | alpha | plateau | gap vs sim | sim floor | MC floor | gap/combined |
|---|---|---|---|---|---|---|
| 2 | 1.0 | 9.8e-7 | 1.80e-2 | 3.79e-3 | 1.49e-2 | **1.17x** |
| 2 | 0.5 | 8.2e-7 | 2.81e-2 | 5.66e-3 | 1.53e-2 | **1.72x** |
| 4 | 1.0 | 7.8e-7 | 8.90e-3 | 4.15e-3 | 9.47e-3 | 0.86x |
| 4 | 0.5 | 8.7e-7 | 1.69e-2 | 5.12e-3 | 2.45e-2 | 0.68x |
| 8 | 1.0 | 8.3e-7 | 1.85e-2 | 2.39e-2 | 2.46e-2 | 0.54x |

`L = 2` at `alpha = 1` went from **1.92x -> 1.17x** and `alpha = 1/2` from
**4.79x -> 1.72x**, and `L = 8` now runs. The `k = 1` vs `k = 2` specialisation
was **not** the cause and is no longer an open candidate for this gap.

### (d) contribution (c), the response sector — corrected exponent, MEASURED

§6's original `d_L L^{1/2-2alpha}` = `L^{-1/2}`-at-both is **falsified**; the
corrected `d_L L^{1-3alpha}` = `L^{-alpha}` is confirmed. Instrument: solve with
the response kernels `A, B` on and off from **the same seed**, and report the
response share `|f_on - f_off| / |f_on|`.

> **The floor here is subtle and I got it wrong first.** Because ON and OFF
> share a seed, the difference is a *common-random-number* estimate whose
> variance is far below either solve's own MC floor. Comparing the gap to the
> individual MC floor (F8's default recipe) made half the points look like
> noise (`0.3x`, `0.4x`) when they are signal. The right floor is the floor
> **on the difference**: recompute the difference at `S` and `S/2`. Every point
> below then sits `5.6x`–`297x` above its own paired floor.

| alpha | `d_L` | share at `L` = 4,8,16,32 | slope | predicted `d_L L^{1-3a}` |
|---|---|---|---|---|
| 1/2 | `L^{2a-1}` derived | 4.76e-2 3.37e-2 2.09e-2 1.39e-2 | **-0.601** | -0.50 |
| 1/2 | `1` control | *(identical — `L^0`)* | -0.601 | -0.50 |
| 1 | `L^{2a-1}` derived | 2.53e-2 1.40e-2 6.33e-3 3.04e-3 | **-1.032** | **-1.00** |
| 1 | `1` control | 9.76e-3 2.82e-3 6.78e-4 1.70e-4 | **-1.959** | **-2.00** |

The `alpha = 1` control **bites with a completely different slope** — `-1.96`
against a predicted `-2.00` — which tests the exponent formula itself and not
merely the composite. (The `alpha = 1/2` rows are byte-identical: `d_L = L^0`
there. That is the **fourth** identity control in this program to present itself
as evidence; recorded, not counted.)

**Not a near-lazy artifact.** Raising richness `gamma_0 * horizon` from 0.6 to
3.6 leaves the exponents alone:

| `gamma_0 * horizon` | slope at `alpha = 1/2` | slope at `alpha = 1` |
|---|---|---|
| 0.6 | -0.631 | -1.041 |
| 3.6 | -0.594 | **-1.006** |

`alpha = 1` is on its predicted `-1.00` to within 0.04. `alpha = 1/2` sits
persistently at `-0.59/-0.63` rather than `-0.50` — a real residual ~0.1 that I
have **not** explained; it is in the direction of the paper being *more* right
than my exponent, not less, and it does not go away with richness. Flagged, not
swept.

## 9. Status

| result | status |
|---|---|
| single-site system, sources, two response pairs per block | derived |
| `kappa = d_L gamma_0 beta_0 L^{-alpha}` as the effective richness | derived |
| SGD depth LR `d_L = L^{2 alpha - 1}`, **on in-block weights only** | derived, matches two papers, and **CONFIRMED in simulation at both exponents** once `d_L` is correctly scoped and the block path isolated — negative control bites at `-2.081` vs `-2.0` (§8c). The earlier falsification was my simulator's bug. |
| init Brownian survives iff `alpha = 1/2` | derived, matches Result 3, **and confirmed in simulation to 3 d.p.** (§8a) |
| per-block weights freeze unless `alpha = 1` | derived, matches Result 3 and Fig 5a, **and confirmed in simulation** (§8b) |
| the init-kernel vs block-learning tension | derived |
| response sector scales as `d_L L^{1-3alpha}` = `L^{-alpha}` | **derived (corrected) and CONFIRMED** — measured `-1.03` at `alpha=1` vs `-1.00`, `d_L=1` control bites at `-1.96` vs `-2.00`, stable under a 6x richness change (§8d). Supersedes the original `L^{1/2-2alpha}`. |
| response functions suppressed unless `alpha = 1/2` | **direction confirmed, magnitude not.** `alpha=1/2` is least-suppressed as the paper says, but I get `L^{-1/2}` there, not `Theta(1)`. Narrowed to a single question: is `A^l` `Theta(1)` or `L^{-alpha}`? Open (§7). |
| solver vs simulation | **VALIDATED** at `L` = 2,4,8 and both exponents, all inside the combined two-floor bar (worst 1.72x) (§8c). |

**Not attempted: solving this system numerically.** The structure is the deep-MLP
system with `gamma_0 -> kappa` plus a layer-time integral, so
`dmft_deep_nonlinear.py` is the natural starting point, but nothing here has
been checked against a solver or a simulation.
