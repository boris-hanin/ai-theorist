# MoE parameterisation and mean-field structure — first-principles derivation

> **Method: heuristic scale analysis (one-step), plus the structural part of the
> mean-field limit.** Per `README.md` this answers *the parameterisation* and the
> *structure* of the limit; it is not yet the full DMFT (single-site processes,
> response kernels). The DMFT proper is `07-moe-dmft.md`.
> Target paper: **arXiv 2601.20205**, Jiang–Bordelon–Pehlevan–Hanin,
> *Hyperparameter Transfer with Mixture-of-Experts Layers* (ICML 2026).

**Discipline for this file.** Derived from the model definition in the paper's
§3.1/§4 and from nothing else. Their Table 1, their Scaling Rules 1–3, and their
five structural claims in §4 were read *before* deriving (they are in the body of
the paper, unavoidable), so this file is **not** a blind rederivation of the
answers. What it *is*: an independent derivation of every rule by a **different
route** from theirs, plus derivations of things they state without proof in the
main text. Where my route and theirs agree, that is two independent routes (F14).
**Appendix E was deliberately not read before writing this file** — the DMFT
check in `07-moe-dmft.md` is the blind one.

## 0. Notation

| symbol | meaning |
|---|---|
| `n` | `n_embd`, residual stream width |
| `m` | `n_hid = alpha_ffn * n`, expert hidden width |
| `E` | `n_exp`, number of experts |
| `a` | `n_act`, experts activated per token |
| `kappa` | `a/E`, **sparsity — held FIXED** as `E -> inf` (their §3.2) |
| `L` | depth (number of MoE blocks) |

The reduced model they analyse (their §4), which is what I derive for:

    h^0     = W_embed x                                   in R^n
    h^{l+1} = h^l + (1/L) f_MoE^l(h^l)                    l = 0..L-1
    f(x)    = W_unembd phi(h^L)                           in R

    f_MoE(h) = (1/a) sum_{i in A(h)} g_i(h) E_i(h)
    g_i(h)   = sigma(r_i),  r_i = W_router^{(i)} . h
    A(h)     = top_a ({ g_i(h) + b_i })
    E_i(h)   = W_down^{(i)} phi( (W_up^{(i)})^T h ),   W_up, W_down in R^{n x m}

Note the residual multiplier is `1/L`, i.e. **CompleteP `alpha = 1`** in the
sense of `05-completep-dmft-sgd.md`. Everything in `05` about the depth sector
carries over unchanged; this file derives only what is *new*, namely the
`m`, `E`, `a` sectors.

## 1. The expert MLP — where `alpha_ffn` enters

This is the interesting part, and the one the paper flags as non-standard.
Convention here is **theirs**: no explicit prefactors, the scale lives in the
init std `sigma` and the LR `eta`. Optimiser is **SignGD**, their stated
Adam proxy — so every weight entry moves by exactly `eta` per step,
`Delta W_{kj} = -eta * sign(dL/dW_{kj})`.

Write `g = dL/dE in R^n` for the backward signal arriving at the expert output,
`h_up = W_up^T h in R^m`, and take `phi' > 0` and `Theta(1)` (they assume this).

### 1a. `sigma(W_up) = n^{-1/2}` and `eta(W_up) = n^{-1}`

Forward: `h_up,j = sum_k (W_up)_{kj} h_k` is an **incoherent** sum of `n` terms
(the init matrix is independent of `h`), so `h_up,j = Theta(sigma_up sqrt(n))`.
Pre-LN gives `|h| = Theta(sqrt n)`. Demanding `h_up,j = Theta(1)`:

    sigma_up = n^{-1/2}                                            [fan-in]

Update: `Delta(W_up)_{kj} = -eta_up sign(h_k phi'(h_up,j) (W_down^T g)_j)
= -eta_up sign(h_k) s_j` with `s_j = sign((W_down^T g)_j)`. Then

    Delta h_up,j = sum_k Delta(W_up)_{kj} h_k = -eta_up s_j sum_k sign(h_k) h_k
                 = -eta_up s_j |h|_1 = -eta_up s_j Theta(n)

a **coherent** sum over `n` (the sign structure aligns it exactly). Demanding
`Delta h_up,j = Theta(1)`:

    eta_up = n^{-1}

**Neither carries `alpha_ffn`**, because both sums run over the *input* index
`k = 1..n`. The up-projection is blind to the expert width. ✓ Table 1.

### 1b. `eta(W_down) = alpha_ffn^{-1} n^{-1}`

    Delta(W_down)_{kj} = -eta_down sign(g_k) sign(phi(h_up,j))
    Delta E_k = sum_j Delta(W_down)_{kj} phi(h_up,j)
              = -eta_down sign(g_k) sum_j |phi(h_up,j)| = -eta_down sign(g_k) Theta(m)

**coherent** over the `m` hidden units. Demanding `Delta E_k = Theta(1)`:

    eta_down = Theta(1/m) = alpha_ffn^{-1} n^{-1}                  ✓ Table 1

So the *learning rate* is just the ordinary muP rule at fan-in `m`. Nothing
exotic. The exotic part is next.

### 1c. `sigma(W_down) = alpha_ffn^{-1} n^{-1/2}` — mean-field, not fan-in

Fan-in would give `sigma_down = m^{-1/2}`. The paper gives
`alpha_ffn^{-1} n^{-1/2} = alpha_ffn^{-1/2} m^{-1/2}` — a factor
`alpha_ffn^{-1/2}` **below** fan-in. Their route is an operator-norm alignment
argument. Here is an independent route via **Stein's lemma**, which does not
assume alignment but computes it.

The condition is that the *other* term in `Delta E`, namely
`W_down diag(phi'(h_up)) Delta h_up`, is also `Theta(1)`. From §1a,
`Delta h_up = -eta_up |h|_1 s` with `s_j = sign((W_down^T g)_j)`. The vector `s`
is **not** independent of `W_down` — it is a function of it — so this contraction
must be computed, not labelled. For `X = (W_down)_{kj} ~ N(0, sigma_down^2)` and
`Y = sum_{k' != k} (W_down)_{k'j} g_{k'} ~ N(0, sigma_down^2 |g|^2)`:

    E[ X sign(g_k X + Y) ]  =  sigma_down^2 g_k * 2 * p_{g_k X + Y}(0)      [Stein]
                            =  sigma_down^2 g_k * 2 / sqrt(2 pi sigma_down^2 |g|^2)
                            =  sqrt(2/pi) * sigma_down * g_k / |g|

Summing the `m` hidden units (each contributing this mean — so this *is* coherent,
and now demonstrably so):

    (W_down s)_k = Theta( m sigma_down g_k / |g| ) = Theta( m sigma_down n^{-1/2} )

using `g_k/|g| = Theta(n^{-1/2})`. Therefore

    [W_down diag(phi') Delta h_up]_k = eta_up |h|_1 * Theta(m sigma_down n^{-1/2})
                                     = n^{-1} * Theta(n) * Theta(m sigma_down n^{-1/2})
                                     = Theta( m sigma_down n^{-1/2} )

Demanding `Theta(1)`:

    sigma_down = n^{1/2} / m = alpha_ffn^{-1} n^{-1/2}             ✓ Table 1

**Two independent routes agree** (their `||.||_op` alignment; my Stein
computation of the sign-correlation). Per F14 that is the bar for trusting a
formula I did not generate blind.

### 1d. What `1c` *means*: the expert is mean-field in its own width

With `sigma_down = alpha_ffn^{-1} n^{-1/2}`, read off the init size of the expert
output:

    E_k(init) = Theta( sigma_down sqrt(m) ) = Theta( alpha_ffn^{-1/2} )

against `Delta E_k = Theta(1)` from §1b. So **the trained part dominates the
initial part by `alpha_ffn^{1/2}`**. That is exactly the mean-field (not NTP)
condition one level down: had we used fan-in init, `E_k(init)` would be
`Theta(1)` and the init would never wash out.

**This is the mechanism behind their Observation 1** (`alpha_ffn`-independence of
the limiting dynamics, hence HP transfer across `alpha_ffn`, their Figure 2 last
column): as `alpha_ffn -> inf` the only surviving contribution to `E` is the
trained one, which is normalised to `Theta(1)` and carries no `alpha_ffn`. The
fan-in control does **not** have this property — which is why their Figure 11
ablation bites, and it is the sharpest available test of `1c`.

## 2. Router and biases

`r_i = W_router^{(i)} . h`, init std `n^{-gamma}`.

- **Init**: `r_i(init) = Theta(n^{1/2 - gamma})` — incoherent over `n`.
  `gamma = 1/2` puts the logits at `Theta(1)`; `gamma = 1` (their default) sends
  them to `Theta(n^{-1/2})`, so gates start uniform and *selection at init is by
  the ordering of an `O(n^{-1/2})` field*. Selection is still well-defined; only
  the gate magnitudes degenerate to `sigma(0)`.
- **Update**: `Delta r_i = sum_k Delta(W_router)_{ki} h_k = -eta_r |h|_1
  = -eta_r Theta(n)`, coherent. Demanding `Theta(1)`:

      eta_router = n^{-1}                                          ✓ Scaling Rule 1

  Note this makes `Delta r_i = Theta(1)` swamp `r_i(init) = Theta(n^{-1/2})` at
  `gamma = 1` after a single step — the router is *fully* feature-learning, and
  `gamma` only sets a transient. That is consistent with their empirical remark
  that `gamma` has no practical effect.
- **Biases**: `b_i <- b_i - eta_bias (Load_i - kappa)`. For the bias to be able
  to move an expert across the selection threshold it must move on the same
  scale as the spread of `q = sigma(r) + b`, which is `Theta(1)` once the router
  has trained. So `eta_bias = Theta(1)`, and `b = 0` at init is the
  load-balanced choice. ✓ Scaling Rule 2.

## 3. The three-level mean-field hierarchy, derived

Their §4 states a three-level structure. Here is where each level comes from —
each is the *same* mean-field statement (trained coherent part beats independent
init part) applied at a different aggregation:

| level | average over | init size | trained size | ratio |
|---|---|---|---|---|
| within expert | `m` hidden units | `alpha_ffn^{-1/2}` | `Theta(1)` | `alpha_ffn^{-1/2}` |
| over experts | `a` active experts | `(a alpha_ffn)^{-1/2}` | `Theta(1)` | `a^{-1/2}` |
| residual stream | `L` blocks | `(L a alpha_ffn)^{-1/2}` | `Theta(1)` | `L^{-1/2}` |

Row 2: `f_MoE = (1/a) sum_{i in A} g_i E_i`. At init the `E_i` are **independent**
across experts (independent `W_up^{(i)}, W_down^{(i)}`), so the sum is incoherent:
`(1/a) * sqrt(a) * Theta(alpha_ffn^{-1/2})`. After a step every `Delta E_i` is
driven by the **same** backward signal `g`, so the sum is coherent:
`(1/a) * a * Theta(1) = Theta(1)`.

Row 3: the `L` block outputs are independent at init, so they accumulate
incoherently against the `1/L` multiplier: `sqrt(L) * (1/L) * (a alpha_ffn)^{-1/2}`.

## 4. `alpha_*` — the ODE/SDE criterion, derived

Square the last row of §3. The variance of the initialisation's contribution to
the residual stream is

    Var[ h^L(init) - h^0 ]  =  Theta( 1 / (L a alpha_ffn) )
                            =  Theta( 1 / (L kappa E alpha_ffn) )
                            =  (1/kappa) * n / (m E L)

Their stated criterion is

    alpha_*  =  lim  n_embd / (n_hid n_exp L)

**These are the same object up to the fixed constant `kappa`.** So `alpha_*` is
not an assumption — it is *precisely the surviving variance of the initial
(Brownian) contribution to the residual stream*, and their trichotomy follows
immediately:

- `alpha_* = 0` — init contribution vanishes, stream is deterministic in layer
  time ⇒ **neural ODE** (their Observation 3).
- `alpha_* > 0` — a `Theta(1)` Brownian term survives ⇒ **neural SDE**
  (Observation 3). This is the exact analogue of contribution (a) in
  `05-completep-dmft-sgd.md` §6, where `L^{1/2-alpha}` survived only at
  `alpha = 1/2`; here the `1/L` multiplier is fixed and the *other three*
  dimensions supply the compensating factor instead.
- Any joint scaling with the same `alpha_*` gives the same limit
  (Observation 2) — because `alpha_*` is the only combination of `n, m, E, L`
  that survives.

**Observation 4 also follows**: at *fixed* `L` and *fixed* `m`, sending
`n, E -> inf` makes row 2 of §3 vanish as `a^{-1/2} = (kappa E)^{-1/2} -> 0`, so
each block's output concentrates and the evolution is deterministic — while `m`
finite means each expert's internal init still matters, so the limit depends on
`m` and **not** on `nu = E/n`, which never appears. ✓

## 5. Registered predictions (see `rounds/006-moe/PREREG.md`)

Everything above is scale counting, so by F18 it claims the `t -> large`
labelling, not `t = 1`. The falsifiable content:

| # | prediction | control that must bite |
|---|---|---|
| P1 | loss curves collapse across `alpha_ffn` under Table 1 | fan-in `sigma_down = m^{-1/2}` must **not** collapse |
| P2 | loss curves collapse across `E` at fixed `kappa` | — |
| P3 | `E_k(init) ~ alpha_ffn^{-1/2}`, `Delta E_k ~ alpha_ffn^0` | fan-in gives `alpha_ffn^0` and `alpha_ffn^{+1/2}` |
| P4 | init stream variance `~ 1/(L a alpha_ffn)`, i.e. slope `-1` in each of `L, a, alpha_ffn` separately | — |
| P5 | the top-`a` selection threshold converges to a deterministic quantile `q*(kappa)` as `E -> inf`, with fluctuation `~ E^{-1/2}` | — |
| P6 | optimal LR transfers across `n, E, alpha_ffn` | fan-in control must drift in `alpha_ffn` |

## 6. Status of the pre-existing `skills/dmft-moe` skill

That skill is marked `PROVENANCE: RECONSTRUCTED` and its central claim is a rule
`eta_up ∝ alpha` with `alpha` defined there as the *sparsity/activation
fraction*. **No such rule exists in 2601.20205**, and it disagrees with this
derivation:

- Sparsity `kappa` is *held fixed* in the paper's scaling (§3.2) and appears in
  **none** of the Table 1 rules.
- The `alpha_ffn` factor attaches to the **down** projection (init *and* LR),
  not the up projection — §1a shows the up projection is blind to expert width.

The reconstruction appears to have conflated `kappa` (sparsity) with
`alpha_ffn` (expert width multiplier) and `up` with `down`. Per the registry's
own F14 standard — reconstructed entries are hints, not sources, and the
measurement/source wins — the skill is **superseded by this file** and is
rewritten in `rounds/006-moe`. Its "validated claims" section describes a round
whose artifacts no longer exist and could not be checked.
