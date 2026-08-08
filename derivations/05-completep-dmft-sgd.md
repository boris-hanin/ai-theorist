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
| **(c) Onsager `kappa A g`** | response to each block's *own* source ⇒ **incoherent** | `d_L L^{1/2 - 2 alpha}` |

Requiring (b) `= Theta(1)` fixes the learning-rate depth factor:

    **d_L = L^{2 alpha - 1}**     (SGD)

and then, substituting back:

| | `alpha = 1/2` | `alpha = 1` |
|---|---|---|
| (a) init Brownian `L^{1/2-alpha}` | **`Theta(1)`** — survives | `L^{-1/2}` — **vanishes** |
| (b) trained memory | `Theta(1)` | `Theta(1)` |
| (c) Onsager `d_L L^{1/2-2alpha}` | `L^{-1/2}` | `L^{-1/2}` |
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
| "all response functions are suppressed at `L -> inf` **unless** `alpha_L = 1/2`" | contribution **(c)**, which I get as `L^{-1/2}` at *both* exponents | **MISMATCH — see below** |

**The one disagreement, stated plainly.** I derive the Onsager contribution as
incoherent across blocks, giving `d_L L^{1/2-2alpha} = L^{-1/2}` at both
`alpha = 1/2` and `alpha = 1`. The paper says responses survive at `alpha = 1/2`
and are suppressed at `alpha = 1`. Getting `Theta(1)` at `alpha = 1/2` requires
the response contribution to accumulate **coherently** (`L`, not `sqrt(L)`),
i.e. `d_L L^{1-2alpha} = L^0`.

Which is right turns on whether the per-block Onsager terms are correlated
across blocks. They are responses to each block's own independent source, which
is what made me call them incoherent — but the responses are functionals of the
*shared* kernels `H, Phi, G, P`, which may make them coherent after all. **I have
not settled this.** The paper's Appendix E.4 has the full computation; I have
not reproduced it. Recorded as an open disagreement rather than resolved in my
favour, and by the program's own rule the prior is on my derivation being wrong.

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

### (c) `d_L = L^{2 alpha - 1}` — **FALSIFIED at alpha = 1**

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

**Candidate causes, in order.** (i) My model has **no LayerNorm**, while both
2405.15712 and CompleteP use pre-LN throughout; LN rescales every block's input
and can change this counting. (ii) The blocks are chained through the stream, so
a change in block `l` propagates to all later blocks — I argued this
amplification is `Theta(1)` at `alpha = 1` but did not carry it. (iii) My blocks
are MLP blocks, not MHSA+MLP. **Unresolved. `d_L = L^{2 alpha - 1}` is withdrawn
as a confirmed result of this file** — it agrees with 2405.15712 Table 1 on
paper and disagrees with my own simulation, and by the program's rule the prior
is on my derivation or my model, not on the paper.

### Not done: solver check

No residual DMFT solver was built, so the **single-site system of §5 is
unchecked** — only its scaling consequences were tested. The open response-
function question of §7 therefore also remains open.

## 9. Status

| result | status |
|---|---|
| single-site system, sources, two response pairs per block | derived |
| `kappa = d_L gamma_0 beta_0 L^{-alpha}` as the effective richness | derived |
| SGD depth LR `d_L = L^{2 alpha - 1}` | derived and matches two papers on paper, but **FALSIFIED against my own simulation at `alpha = 1`** (§8c). Withdrawn pending LayerNorm and the chain effect. |
| init Brownian survives iff `alpha = 1/2` | derived, matches Result 3, **and confirmed in simulation to 3 d.p.** (§8a) |
| per-block weights freeze unless `alpha = 1` | derived, matches Result 3 and Fig 5a, **and confirmed in simulation** (§8b) |
| the init-kernel vs block-learning tension | derived |
| response functions suppressed unless `alpha = 1/2` | **NOT reproduced** — I get `L^{-1/2}` at both. Open. |

**Not attempted: solving this system numerically.** The structure is the deep-MLP
system with `gamma_0 -> kappa` plus a layer-time integral, so
`dmft_deep_nonlinear.py` is the natural starting point, but nothing here has
been checked against a solver or a simulation.
