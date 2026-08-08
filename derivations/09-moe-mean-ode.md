# Residual MoE in the Neural Mean ODE scaling — a new joint limit

> **Status: NEW.** Not a replication. This file merges the MoE parameterisation
> of **2601.20205** (Jiang–Bordelon–Pehlevan–Hanin) with the joint `(L, M, D)`
> Mean-ODE analysis of **2509.10167** (Chizat) and **2603.18168**
> (Chaintron–Chizat–Maass), which do not treat sparse routing. Heuristic
> derivation in §1–§4, DMFT in §5–§7, predictions in §8.

## 0. The model

Five scale dimensions: depth `L`, embedding `D`, expert count `E`, active
experts `a`, expert hidden width `M`. Sparsity `kappa = a/E` **fixed** (2601.20205
§3.2). Each expert is a 2LP with `M` hidden units — Chizat's block, made
expert-local.

    h^0     = x in R^D
    h^{l+1} = h^l + c_L * (1/a) sum_{i in A(h^l)} g_i(h^l) * E_i^l(h^l)

    E_i^l(h) = sum_{j=1}^{M} w^{i,j,l} phi( <u^{i,j,l}, h> )     (2LP, M units)
    g_i(h)   = sigma(r_i),  r_i = <W_router^{i,l}, h>
    A(h)     = top_a( { g_i(h) + b_i } )        b_i(0) random, nonzero  (F21)

    u^{i,j,l} ~ N(0, I_D/D)          so <u,h> = Theta(1) with ||h|| = Theta(sqrt D)
    w^{i,j,l} ~ N(0, s_w^2 I_D)
    c_L      = branch multiplier

## 1. The central question: what does the average run over?

Chizat's insight is that the units are indexed by `(j,l)` so the mean-field
average sees `L*M` of them, not `M`. Here the units are indexed by the **triple**

    (i, j, l)  in  [E] x [M] x [L]

but only the **active** experts contribute to any given token, so per layer
`a*M` units fire, and across the network

    **effective width  W_eff  =  L * a * M**

Note it is `a`, not `E`. Inactive experts contribute exactly zero to that
token's forward pass, so they cannot participate in its CLT. **The expert count
`E` does not appear in the approximation rate at all** — it changes the *limit
object* (capacity), not the *rate of approach* to it. That separation is the
main conceptual content of this file and is what §8's tests probe.

## 2. The MLU residual scale

Split the block into mean and fluctuation, exactly as `08` §1:

    c_L * (1/a) sum_{i in A} g_i E_i
      =  c_L * M * Ebar(h, s)                                   <- coherent, the drift
       + c_L * (1/a) sum_{i in A, j} [ g_i w^{ij} phi_ij - mean ]  <- a*M-term CLT

where `Ebar(h,s) = E[ g w phi | active ]` is a **conditional** expectation, the
conditioning being the routing event `q >= q*(kappa)` (round 006 §2a; the
threshold is a deterministic quantile in the `E -> inf` limit).

- **Drift**: the `M` units inside one expert add coherently through their shared
  mean, giving `c_L * M * Ebar`. For a nontrivial `L -> inf` ODE this must be
  `Theta(1/L)`, so `c_L * M = Theta(1/L)`, i.e. `c_L = Theta(1/(L M))`.
- **Fluctuation**: `(1/a) * sqrt(a M) * s_w sqrt(D) = sqrt(M/a) * s_w sqrt(D)`,
  times `c_L`, accumulated over `L` layers incoherently:

      sqrt(L) * c_L * sqrt(M/a) * s_w * sqrt(D)
        = (1/(L M)) * sqrt(L) * sqrt(M/a) * s_w * sqrt(D)
        = s_w * sqrt( D / (L a M) )                      at c_L = 1/(LM)

So with `s_w = Theta(1)`:

    **residual scale  =  c_L * ||w||  =  (1/(L M)) * s_w sqrt(D) = sqrt(D)/(L M)**

    **CLT error term  =  sqrt( D / (L a M) )**

**Consistency checks.**
- `a = 1` (dense, one expert): reduces to Chizat's `sqrt(D)/(LM)` and
  `sqrt(D/(LM))`. ✓
- 2601.20205: their branch is `1/L`, their MoE normalisation `1/n_act = 1/a`,
  and their down-projection init is `sigma_down = sqrt(D)/M` (round 006 §1c).
  Per-unit total scale `= (1/L)(1/a)(sqrt(D)/M) * a = sqrt(D)/(LM)`. ✓ **The MoE
  paper's parameterisation is already exactly the MLU scale of the Chizat
  program**, which neither paper states, because neither treats the other's
  scaling dimensions.

## 3. The rate

Adding the Euler term and the large-`D` term of 2603.18168:

    ||ResNet-MoE - limit||  =  O( 1/L + sqrt( D / (L a M) ) + 1/sqrt(D) )

Three of the five dimensions enter, and they enter **only through the two
combinations `L` and `L a M`**. Consequences:

1. **`a` and `M` are exchangeable.** Doubling the active-expert count and
   halving expert width leaves the rate unchanged. This is the MoE analogue of
   Chizat's depth/width exchange, and it is new.
2. **`E` is absent.** At fixed `a` (hence, at fixed `kappa`, fixed `E` — but see
   §8) the expert count does not affect the rate.
3. **Depth is still special**: it appears in both `1/L` and `1/(LaM)`, so it buys
   discretisation *and* averaging, while `a` and `M` buy only averaging.

## 4. Optimal shape — two budgets, not one

MoE has the property that parameters and FLOPs scale differently:

    parameters  P = Theta( L E M D )         all experts stored
    FLOPs       C = Theta( L a M D )         only active experts computed

The rate depends on `L a M D`-type combinations, i.e. on the **FLOP** budget.
Minimising `1/L + sqrt(D/(L a M)) + 1/sqrt(D)` subject to `L a M D = C`, and
writing `W = a M`:

    substituting W = C/(L D) into the CLT term:
        sqrt( D/(L W) ) = sqrt( D / (L * C/(L D)) ) = D / sqrt(C)

so the three terms are

    T1 = a/L        T2 = b * D/sqrt(C)        T3 = c/sqrt(D)

> **CORRECTION (round 009, caught by the fixed-`C` sweep).** An earlier version of
> this section "balanced all three terms" and reported `L = C^{1/6}` as *the*
> optimum. That is wrong: **`T2` does not depend on `L` at all** once `C` and `D`
> are fixed, because `L * W = C/D` is then pinned. So `L` appears only in `T1` and
> can only help. The measurement is direct — at `C = 8192`, `D = 16`, sweeping
> `L` = 2 → 64 with `aM` = 256 → 8, the measured CLT term is **flat**
> (1.62e-2 → 1.49e-2), exactly because `L aM = C/D = 512` throughout.

Minimising properly: balance `T2` against `T3` over `D`,

    b/sqrt(C) = (c/2) D^{-3/2}   =>   **D = Theta( C^{1/3} )**,  T2 = T3 = Theta( C^{-1/6} )

and then require `T1 = a/L` not to dominate, i.e. `L >= Theta(C^{1/6})`:

    **D = Theta(C^{1/3}),   L >= Theta(C^{1/6}),   a M = C/(L D),   error = O(C^{-1/6})**

At the shallowest admissible depth `L = C^{1/6}` this gives `aM = C^{1/2}`, which
is the shape quoted before — but it is the **shallowest shape achieving the
optimal rate, not a unique optimum**. Deeper is not worse (until `aM` hits its
floor of 1). The genuinely binding requirement is `D = Theta(C^{1/3})`.

**The split of `aM` between expert count and expert width is free.** At fixed FLOPs the model is then free to
buy capacity by raising `E` at fixed `a` — which costs parameters but neither
FLOPs nor rate. **That is a scaling-theoretic statement of why MoE works**, and
it is consistent with 2601.20205's Finding 3.1 (more, smaller experts is better
at fixed parameter count) while locating the benefit in the *limit*, not in the
approximation rate.

## 5. DMFT — the single-site system

Cavity method, moves M1–M6 of `00-method.md`. Fields per token, per layer.

**M1 — integrate the parameter dynamics.** Under gradient flow with the LR
scaling of §2, for an active expert `i` and unit `j`:

    d/dt w^{ij}(t) = eta_w  int ds  Delta(s) g_i(s) phi(<u^{ij}(s), h^l(s)>) 1_{i in A(s)}
    d/dt u^{ij}(t) = eta_u  int ds  Delta(s) g_i(s) phi'(.) <w^{ij}(s), b^{l+1}(s)> h^l(s)

**M2 — split the fields.** Write `z^{ij} = <u^{ij}, h^l>`:

    z^{ij}(t) = chi^{ij}(t)  +  gamma_0 int ds  C_h(t,s) 1_{i in A(s)} g_i(s) p^{ij}(s)

with `chi^{ij}` the frozen-disorder Gaussian source (covariance `C_h`, the stream
kernel) and `p^{ij}` the in-expert backward field.

**M3 — classify the disorder edges.** Three reused matrices, hence **three
response pairs** per expert per layer:
  (i) `u^{ij}` reused forward/backward -> response `R_u`,
  (ii) `w^{ij}` reused forward/backward -> response `R_w`,
  (iii) `W_router^i` reused in `g_i` and in the routing gradient -> `R_r`.
This is one more than the dense 2LP case, and it is where the coherent/incoherent
labelling has an extra place to go wrong (`05` §6 got that labelling wrong twice).

**M4 — cavity-expand.** The Onsager reaction for the `w`-edge:

    Onsager_w  =  c_L * (1/a) * sum_{i in A} g_i^2 * int ds R_w(t,s) phi(z(s))

and its counting is the *same* as §2's fluctuation counting, so it carries
`1/(L a M)`, i.e. it is **suppressed by the effective width** — the MoE analogue
of `05` §6 contribution (c). At `L a M -> inf` the response sector drops out and
the limit is the deterministic Mean ODE. **This is the DMFT statement of why the
Mean ODE has no memory kernel.**

**M5 — close on population kernels.** Three levels of averaging (round 006 §3),
now with the routing conditioning explicit:

    C_h(t,s)     = (1/D) < h^l(t), h^l(s) >                       stream
    Phi_i(t,s)   = (1/M) sum_j phi(z^{ij}(t)) phi(z^{ij}(s))      within expert
    Mbar(t,s)    = E_i[ 1_{i in A} g_i(t) g_i(s) Phi_i(t,s) ]     over experts

`Mbar` is the object that replaces Chizat's `E[phi(h,Z)]`: a **routing-conditioned
population average**, and the only place `kappa` enters the limit.

**M6 — boundaries.** `h^0 = x`; `b^L = grad loss`. The limit ODE is

    d/ds h(s,x) = Ebar[ g * w * phi( <u, h(s,x)> ) | q >= q*(kappa) ]

i.e. **Chizat's Mean ODE with the expectation taken over the conditioned expert
population.** Sparsity enters the limit only through that conditioning.

## 6. What the DMFT adds over the heuristic

The heuristic (§1–§4) gives the rate. The DMFT gives:
- the *conditioning* structure — the limit is an expectation over the
  `q >= q*(kappa)` slice, so `kappa` survives in the limit even though it is
  absent from the rate;
- the **response sector is `O(1/(LaM))`**, hence the limit is memoryless — a
  statement the heuristic cannot make because it has no response functions;
- three response pairs per expert-layer, including a router one with no
  counterpart in the dense analysis.

## 7. Where this could be wrong

By the program's record, the coherent/incoherent labelling in §2 is the risky
step (it was wrong twice in `05` §6, cancelling exactly at one exponent). The
specific hazard here: I treat the `a` active experts as contributing
**incoherently** at init. They share the router-conditioning event, which
correlates them. If that correlation is `Theta(1)` rather than vanishing, the
`a`-averaging is coherent and the effective width is `L M`, not `L a M`.
**Test N2 below is designed to catch exactly that**, and it is the test I expect
to be most informative.

## 8. Predictions

| # | prediction | control |
|---|---|---|
| **N1** | init deviation `~ sqrt(D/(L a M))`: slope `-1/2` in each of `L`, `a`, `M` | `E` at fixed `a` must give slope **0** |
| **N2** | **at fixed `L a M`, the deviation is invariant to how it is split among `L`, `a`, `M`** | if the `a`-average is coherent instead, the `a`-split will *not* be invariant — see §7 |
| **N3** | `E` absent from the rate: at fixed `a`, varying `E` (hence `kappa`) leaves the deviation unchanged | — |
| **N4** | optimal LR transfers across `L`, `a`, `M`, `E`, `D` under this parameterisation | fan-in `sigma_down = M^{-1/2}` must drift |
| **N5** | Euler term `1/L`, independent of `a` and `M` | — |

## 9. Measured (round 009) — all preregistered predictions confirmed

| claim | measured |
|---|---|
| `W_eff = L a M`: slopes `-1/2` in each | `L` **-0.505**, `a` **-0.489**, `M` **-0.512** |
| `E` absent from the rate | **+0.030** at fixed `a`; 0.04 decades of LR drift |
| **invariance at fixed `L a M`** (§7's risky step) | spread **1.10x** over 10 splits, `a` = 2..16, `L` = 2..64 |
| fan-in control kills the `M`-averaging | `M` slope **-0.013** vs `-0.512`; split spread **3.90x** vs 1.10x |
| HP transfer | `L` 0.11, `a` 0.06, `M` 0.14, `D` 0.22, `E` **0.04** decades |

**§7's hazard did not materialise**: the `a` active experts add *incoherently*
despite sharing the routing-conditioning event, so `W_eff = L a M` stands.

**One gap this file had**, caught by the HP-transfer test: §2 fixes the LR
scaling in `L`, `M`, `a` but never states its **`D`-dependence**. Since the loss
is per-coordinate normalised, `b ~ 1/D` and `grad_w ~ 1/(L M a D)`, so the
learning rate must carry a factor `D`:

    **lr = L * M * a * D * eta**

Without it the optimal LR drifts 0.65 decades across `D`; with it, 0.22.
