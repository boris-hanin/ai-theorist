# Graph transformers: parameterisation from first principles, then checked against 2607.05017

> **Method: HEURISTIC one-step scale analysis.** No cavity method anywhere in
> this file. Per `README.md` §A this answers *the parameterisation* — a table of
> powers — and nothing about the dynamics. The DMFT is `11-graph-transformer-dmft.md`.
> Target paper: **arXiv 2607.05017**, DeZoort & Hanin, *Hyperparameter Transfer in
> Graph Neural Networks* (7 Jul 2026).

**Provenance discipline.** The paper was obtained as PDF and extracted with
`pypdf` (never a web-fetch summary — F14; two summaries in this program's history
were confabulated). Every formula quoted below is quoted from that extraction.
Their §2.2, Table 1, §3.2–3.4 (Props 1–3, Remarks 4–5) were read before this file
was written — they are in the body of the paper and unavoidable. What is
*independent* here is (a) the route (direct coherent/incoherent counting of every
contraction, rather than their Frobenius-norm recursion), and (b) **the entire
attention sector, which the paper does not derive at all** — its §4 names
attention as an explicit limitation: "modern GNNs often include richer components
like edge features, attention, Laplacian encodings ... Extending this analysis to
these additional architectural features is an important next step." That
extension is what this file is.

**Scoreboard.** 6 of the paper's entries re-derived and confirmed; 1 internal
inconsistency found in the paper (a typo in §2.4 against its own Table 1 and
Prop. 3); 4 new entries derived for the attention sector; 1 entry (`sigma_QK`)
whose "control" is an **algebraic identity** at the recommended exponent and
therefore cannot be tested there — recorded because getting this wrong is the
program's most repeated error.

---

## 0. The architecture

The paper's base transfer GNN (their Eqns 4–7) permits an MHSA module — §2.2:
"Each of these encoder, MPNN, MHSA, and decoder modules can be chosen flexibly,
so long as they do not significantly grow or shrink their inputs as `D`
increases" — but every equation, proposition and experiment in the paper uses
only MPNN + MLP. The **base transfer graph transformer** is that architecture
with the MHSA branch actually written down:

    X^(0)      = (1/sigma_0) f_x^(0)(X, A)                         encoder      (E)
    Xt^(l+1)   = X^(l)    + (1/L) MPNN^(l+1)(X^(l), A)             message pass (M)
    Xh^(l+1)   = Xt^(l+1) + (1/L) MHSA^(l+1)(Xt^(l+1), A)          attention    (A)
    X^(l+1)    = Xh^(l+1) + (1/L) MLP^(l+1)(Xh^(l+1))              MLP          (F)
    Z          = (1/(sigma_{L+1} sqrt(D))) f_x^(L+1)(X^(L))        decoder      (D)

with the minimal (linear-module) instantiations, in the paper's own convention
— all weights `N(0,1)` with explicit `1/sqrt(fan_in)` prefactors, scale carried
by the prefactor not the init:

    (E)  X^(0)  = (1/(sigma_0 sqrt(n0))) X W^(0),          W^(0)_ij ~ N(0, sigma_0^2)
    (M)  MPNN   = (1/(gamma_P sqrt(D))) P X W~,            W~   ~ N(0,1)
    (A)  MHSA   = (1/(gamma_A sqrt(D))) [ S(X) V ] W_O,    W_O  ~ N(0,1)
           V_{,h}   = (1/sqrt(D)) X W_{V,h}                W_V  ~ N(0,1)
           q_{u,h}  = (1/(sigma_QK sqrt(D))) W_{Q,h} x_u   W_Q  ~ N(0, sigma_QK^2)
           k_{v,h}  = (1/(sigma_QK sqrt(D))) W_{K,h} x_v   W_K  ~ N(0, sigma_QK^2)
           A_{uv,h} = D_h^{-alpha_A} q_{u,h} . k_{v,h},  masked to v in N(u) u {u}
           S^h      = softmax_v(A_{uv,h})                  (row-stochastic on the graph)
    (F)  MLP    = (1/sqrt(4D)) ((1/sqrt(D)) X W_1) W_2,    W_1,W_2 ~ N(0,1)
    (D)  z      = (1/(sigma_{L+1} D)) (1/N) 1^T X^(L) W^(L+1),  W^(L+1) ~ N(0, sigma_{L+1}^2)

| symbol | meaning |
|---|---|
| `D` | width of the residual stream |
| `L` | depth (number of blocks) |
| `H` | number of attention heads; `D_h = D/H` the head dimension |
| `alpha_A` | attention logit exponent, `A = D_h^{-alpha_A} q.k` |
| `N`, `d_bar` | nodes per graph, mean degree **including the self-loop** |
| `n0` | input node-feature dimension, `||x_u||^2 = n0` |
| `gamma_P`, `gamma_A` | scale-shift factors of the two message-passing operators (their Eqn 14/19) |
| `sigma_0`, `sigma_QK`, `sigma_{L+1}` | init-scale rescalers used to hit a **global** learning rate |

Two conventions must be fixed before any counting, because both are load-bearing:

- **Heads partition the width**, `D = H D_h`. `W_V, W_O in R^{D x D}` regardless of
  `H`; only the *slicing* of their columns/rows changes. This is why `H` will
  turn out to be absent from the V/O sector.
- **The self-loop is mandatory.** `softmax` over an empty neighbourhood is
  undefined; a node with `d_u = 0` and no self-loop produces `NaN` (or, worse,
  a silent uniform-zero row under a masked-fill implementation). Asserted in
  `skills/dmft-graph/scripts/gt.py`.

### 0a. Desiderata (theirs, §3, transcribed)

1. **Feature stability**: `x^(l)_{u;i} = Theta_{D,L}(1)`.
2. **Update stability**: `Delta x^(0)`, `Delta x^(L+1)` are `Theta(1)`; each
   *residual branch* moves the stream by `Theta(1/L)` so the `L` blocks sum to
   `Theta(1)`.
3. **Complete learning**: updates do not converge to their linearisation; in
   practice `1/L` (not `1/sqrt(L)`) branch scaling, i.e. CompleteP `alpha = 1`.

Adding a third branch per block changes desideratum 2 by a factor of 3 and
nothing else: `3 x Theta(1/L) x L = Theta(1)`. The forward recursion of their
Prop. 1 becomes `(1 + 1/L^2)^3` per block instead of `(1 + 1/L^2)^2`, so
`E||X^(l)||_F^2 = ND(1 + O(L^{-1}))` still. **Depth counting is untouched by
adding attention.** This is the first thing to check and the least interesting.

### 0b. The backward signal, once, for the whole file

Everything below contracts against the same object. From (D),

    g_{u,i} := dz / dx^(L)_{u,i} = (1/(sigma_{L+1} D N)) (R^(l) W^(L+1))_i = Theta(1/(D N))

using their `R^(l) = Theta(1)` (their Eqn 30, unchanged by a third branch). Note
`g` is **node-independent** for a graph-level readout — the `1/N` pooling sends
the *same* covector to every node. That fact is used twice below and is the
origin of the node-alignment assumption in §6.

---

## 1. The MPNN, MLP and encoder/decoder sectors — re-derived, all confirmed

These are the paper's own results. I re-derive them by direct coherence counting
rather than their Frobenius recursion, so agreement is two independent routes
(the F14 bar).

**Residual matrices, SGD.** With `W~` the MPNN matrix,

    grad_{W~} z |_{ji} = (1/(L gamma_P sqrt(D))) sum_u (P x^(l)_{alpha,u})_j g_i
                       = Theta( 1/(L D^{3/2}) )                       [g = Theta(1/(DN)); sum_u gives N]
    Delta W~_{ji}      = -eta_res Delta_alpha grad  = Theta( eta_res/(L D^{3/2}) )
    Delta x_beta       = (1/(L gamma_P sqrt(D))) (P x_beta)_j Delta W~_{ji}
                       = (1/(L sqrt(D))) * Theta(D) * Theta(eta_res/(L D^{3/2}))
                       = Theta( eta_res / (L^2 D) )

The `Theta(D)` is the **coherent** contraction `(P x_beta) . (P x_alpha) = Theta(D)`
— `Delta W~` is a rank-one outer product built from the `alpha` features, and it is
read out against the `beta` features. Demanding `Theta(1/L)`:

    eta_res^SGD = eta_0 D L                            ✓ their Eqn 34 / Table 1

**Residual matrices, Adam (sign-GD proxy).** `Delta W~_{ji} = -eta sign(g_{ji})`,
and `sign(g_{ji}) = sign(Delta_alpha) sign((Px_alpha)_j) sign(g_i)`, so the sum over
`j` is `||P x_beta||_1 = Theta(D)` — coherent by construction:

    Delta x_beta = (1/(L sqrt(D))) eta Theta(D) = Theta(eta sqrt(D)/L)
    => eta_res^Adam = eta_0 / sqrt(D)                  ✓ their Eqn 39 / Table 1

Note this route needs **no** alignment assumption, unlike their route through
the RMS-gradient surrogate. See §6.

**MLP.** `W_1` has fan-in `D`, `W_2` has fan-in `4D`; the `1/sqrt(4D)` prefactor
and the `4D` hidden units cancel exactly, as the paper says. Same rates. ✓

**Encoder.** `Delta X^(0) = (1/(sigma_0 sqrt(n0))) X_beta Delta W^(0)` and
`grad_{W^(0)} z` carries `1/(sigma_0 sigma_{L+1} D N sqrt(n0))`. Two factors of
`1/sigma_0`, so the *needed* learning rate grows as `sigma_0^2`:

    eta_0-layer^SGD  = eta_0 D sigma_0^2 x C_{alpha beta},   C = n0 sqrt(N_beta)/M_{alpha beta}
    eta_0-layer^Adam = eta_0 sigma_0                          ✓ their Eqns 24, 38

with `M_{alpha beta} = || (1/N_alpha) 1^T X_alpha X_beta^T ||_2`. Under sign-GD one
factor of `1/sigma_0` is removed by the sign, hence `sigma_0` not `sigma_0^2`. ✓

**Decoder.** Symmetric: `eta^SGD = eta_0 D sigma_{L+1}^2`, `eta^Adam = eta_0 sigma_{L+1}`. ✓

**Global-rate choices.** Setting the needed rate equal to the global rate:

| optimiser | global rate | `sigma_0` | `sigma_{L+1}` |
|---|---|---|---|
| SGD | `eta_0 D L` | `sqrt(L)` | `sqrt(L)` |
| Adam | `eta_0 / sqrt(D)` | `1/sqrt(D)` | `1/sqrt(D)` |

### 1a. An internal inconsistency in the paper

§2.4 (p. 8) states: *"so that to use a global learning rate we choose
`sigma_0 = 1/sqrt(D)` and `sigma_{L+1} = 1`."*

That cannot be right on the paper's own terms. Prop. 3 gives
`eta^{(L+1)}_Adam = eta_0 sigma_{L+1}`, and the global rate is `eta_0/sqrt(D)`;
`sigma_{L+1} = 1` would demand `eta_0 = eta_0/sqrt(D)`. Both **Table 1** (Adam
decoder row, `sigma_ell = 1/sqrt(D)`) and the closing sentence of **Prop. 3**
(*"one may use a global effective learning rate ... with the choices
`sigma_0 = sigma_{L+1} = 1/sqrt(D)`"*) say `1/sqrt(D)`. My derivation agrees with
Table 1 and Prop. 3. **The §2.4 sentence is a typo.** This is a transcription-level
disagreement, not a physics one, and it is filed here because a reader
implementing from §2.4 alone gets a decoder rate too large by `sqrt(D)` — which
looks exactly like a failure of width transfer.

---

## 2. The attention sector I: the value/output path

    MHSA(X) = (1/(gamma_A sqrt(D))) [ S(X) ((1/sqrt(D)) X W_V) ] W_O

Compare the MLP branch, `(1/sqrt(4D))((1/sqrt(D)) X W_1) W_2`. **These are the
same object** — two chained fan-in-`D` matrices — with one difference: an
`N x N` linear operator `S` inserted on the *node* index between them, and a
normaliser `gamma_A` instead of the constant `1/2`.

The node index is not the width index. `S` acts on `u`, the matrices act on `i`.
The two commute, so every step of the MLP counting goes through verbatim with
`X -> S X`, and the only question is what `S` does to the *scale* — which is
precisely what `gamma_A` is defined to absorb (§4). Therefore:

    eta_V = eta_O = eta_res  =  eta_0 D L  (SGD)  =  eta_0/sqrt(D)  (Adam)

**`H` is absent.** `W_V, W_O in R^{D x D}` have fan-in `D` whatever `H` is; the
head split only re-labels which columns feed which slice of `S`. Every sum in the
counting runs over `D` or over the node index, never over `H` or `D_h`. So the
V/O sector is head-blind.

*Domain of that claim (F19).* It is fixed by counting the **fan-in of `W_V` and
`W_O`, which is `D`**. It applies to exactly those two matrices. It does **not**
apply to `W_Q, W_K`, whose relevant sum is over `D_h` (§3), and it does not
survive the convention `D_h` fixed with `D = H D_h` *growing* — that is the same
statement, just with `H` as the width dial rather than a free one.

---

## 3. The attention sector II: the key/query path

This is where the graph transformer is not a GNN. Every step is labelled
coherent/incoherent, and **each label carries the training time at which it
holds** (F18 — the failure this program registered on exactly this contraction).

### 3a. Logit scale at init

`q_{u,h}, k_{v,h} in R^{D_h}` have `Theta(1)` entries (incoherent sum of `D` terms
of size `Theta(1)`, divided by `sqrt(D)`). `q` and `k` come from *different*
matrices, so at init they are uncorrelated and `q.k` is an **incoherent** sum of
`D_h` terms:

    q_{u,h} . k_{v,h} = Theta(sqrt(D_h))
    A_{uv,h}          = Theta( D_h^{1/2 - alpha_A} )

- `alpha_A = 1/2`: logits `Theta(1)` at init — attention is selective from step 0.
- `alpha_A = 1`: logits `Theta(D_h^{-1/2}) -> 0` — **attention at init is exactly
  uniform mean-aggregation over each graph neighbourhood.** For a *graph*
  transformer this has a concrete reading the sequence case does not have: at
  `alpha_A = 1` the MHSA branch initialises as a second, degree-uniform MPNN
  branch, and every bit of graph selectivity is learned.
- `alpha_A < 1/2`: logits diverge with width. §4 shows this is fatal, for a
  reason specific to graphs.

### 3b. The chain, contraction by contraction

Write `c_{uv,h} := dz/dA_{uv,h}`. Four contractions, four labels:

| # | contraction | sum over | label at `t=1` | label at `t >> 1` |
|---|---|---|---|---|
| C1 | `g_u^T W_{O,h}` | `D` | incoherent (`sqrt(D)`) | incoherent |
| C2 | `(g^T W_{O,h}) . (v_v - o_u)` | `D_h` | **incoherent** (`sqrt(D_h)`) | **coherent** (`D_h`) |
| C3 | `Delta q_u . k_v`, the `v'=v` term | `D_h` | coherent (`D_h`) | coherent |
| C4 | `sum_j Delta W_{Q;ij} x_{u,j}`, the `u'=u` term | `D` | coherent (`D`) | coherent |

**C2 is the one that flips, and it is the same contraction that broke
`derivations/03-attention.md` §D2b.** At init `W_O` is independent of the
backward signal `g`, so `(g^T W_O)` is a random `Theta(1/(N sqrt(D)))` covector
and its dot with `(v_v - o_u)` is a `sqrt(D_h)` incoherent sum. After one step
`W_O` acquires a rank-one learned part `Delta W_O ∝ g (x) o`, so
`(g^T W_O) ⊃ |g|^2 o_u^T`, and `o_u . (v_v - o_u) ⊃ -|o_u|^2 = Theta(D_h)` —
coherent. **The label flips between `t=1` and `t=2`.**

Consequently:

    c_{uv,h}  =  Theta( s_uv sqrt(D_h) / (L gamma_A N D) )     at t = 1
              =  Theta( s_uv D_h       / (L gamma_A N D) )     at t >> 1

### 3c. The learning rate, SGD

    grad_{W_Q,h; ij} z = sum_{u,v} c_{uv,h} D_h^{-alpha_A} k_{v,i} x_{u,j} / (sigma_QK sqrt(D))
    Delta q_{u,h,i}    = (1/(sigma_QK sqrt(D))) sum_j Delta W_{Q;ij} x_{u,j}     [C4: Theta(D)]
    Delta A_{uv,h}     = D_h^{-alpha_A} Delta q_{u,h} . k_{v,h}                  [C3: Theta(D_h)]

Composing, with `eta_Q` the learning rate actually applied to `W_Q`:

    Delta A = eta_Q * c-factor * D_h^{1-2 alpha_A} / sigma_QK^2
            = Theta( eta_Q D_h^{2 - 2 alpha_A} / (L gamma_A D sigma_QK^2) )      at t >> 1

**What should `Delta A` be?** The attention-pattern channel moves the stream by
`Delta x = (1/(L gamma_A sqrt(D))) W_O (sum_v Delta s_uv v_v) = Theta(Delta A / L)`
(incoherent over `D_h` per head, `H` heads, times `1/sqrt(D)` — the factors
cancel). Desideratum 2 wants `Theta(1/L)`, so **`Delta A = Theta(1)`**. Then

    eta_Q^SGD = eta_0 L D sigma_QK^2 D_h^{2 alpha_A - 2}                         (*)

### 3d. The learning rate, Adam

Under sign-GD, `Delta W_{Q;ij} = -eta_Q sign(g_{ij})` and, factorising the sign as
the paper does in its own Prop. 3 (`sign(g^{(0)}_{ij}) = sign(Delta_alpha)
sign(Xbar_{alpha,i}) sign((R W^{(L+1)})_j)`),

    Delta q_{u,h,i} = -(eta_Q/(sigma_QK sqrt(D))) s^k_i ||x_u||_1 = Theta(eta_Q sqrt(D)/sigma_QK)
    Delta A_{uv,h}  = D_h^{-alpha_A} Theta(D_h) Theta(eta_Q sqrt(D)/sigma_QK)
                    = Theta( eta_Q D_h^{1-alpha_A} sqrt(D)/sigma_QK )
    => eta_Q^Adam   = eta_0 D^{-1/2} sigma_QK D_h^{alpha_A - 1}                   (**)

### 3e. One `sigma_QK` serves both optimisers

Setting (*) equal to the global SGD rate `eta_0 D L`, and (**) equal to the global
Adam rate `eta_0/sqrt(D)`, gives the *same* answer:

    sigma_QK = D_h^{1 - alpha_A} = (D/H)^{1 - alpha_A}                            (***)

That coincidence is not an accident: `sigma_QK` enters SGD twice (once forward,
once in the gradient) and Adam once (the sign kills the gradient factor), and
`alpha_A` enters `Delta A` with exactly the compensating power. It is the sharpest
prediction in this file because it is an *equality between two different
optimisers' correction factors*, and nothing else in the paper's Table 1 has that
property (`sigma_0` is `sqrt(L)` for SGD and `1/sqrt(D)` for Adam).

**At `alpha_A = 1`, `sigma_QK = 1`: `W_Q` and `W_K` take the global learning rate
with no correction whatsoever, under both optimisers.**

### 3f. ⚠ The `sigma_QK` control is an IDENTITY at `alpha_A = 1`

`q = (1/(sigma_QK sqrt(D))) W_Q x` with `W_Q ~ N(0, sigma_QK^2)` is
**distributionally identical** to `(1/sqrt(D)) W'_Q x` with `W'_Q ~ N(0,1)`. The
forward pass does not know `sigma_QK`. Its only effect is on the *needed* learning
rate — `Delta x ∝ eta/sigma_QK^2` (SGD), `∝ eta/sigma_QK` (sign-GD). Therefore:

> "Global LR with `sigma_QK = D_h^{1-alpha_A}`" and "`sigma_QK = 1` with a
> per-group LR `eta_0 D L D_h^{2 alpha_A - 2}`" are **the same run**, exactly,
> not approximately.

And at `alpha_A = 1` both collapse onto "`sigma_QK = 1`, global LR" — so a
width-transfer sweep at `alpha_A = 1` **cannot** distinguish the derived rule from
the naive one. This program has mistaken an identity control for evidence five
times. The discriminating experiment must be run at `alpha_A = 1/2`, where the
two differ by `D_h`, and that is what the pre-registration commits to.

---

## 4. `gamma`, and why `alpha_A >= 1/2` is forced (the graph-specific result)

The paper introduces `gamma_l^2 := E||P X^(l)||_F^2 / E||X^(l)||_F^2` (their
Eqn 14/19) as a scale-shift to be divided out, and advocates in §2.5 treating it
as a scanned hyperparameter. Compute it for a general row-supported operator.
Normalise node features to `|x_v|^2 = D` and write `x_v . x_{v'} = D rho_{vv'}`:

    gamma^2 = (1/N) sum_u [ sum_v P_uv^2 + sum_{v != v'} P_uv P_uv' rho_{vv'} ]   (G)

Two limits, both exact:

- **decorrelated neighbours** (`rho = 0`): `gamma^2 = < sum_v P_uv^2 >`.
  For row-stochastic `P` this is `< 1/d_eff(u) >` with
  `d_eff(u) := (sum_v P_uv^2)^{-1}` the **participation ratio** (inverse Simpson
  index) of the row — the *effective* number of neighbours actually aggregated.
- **perfectly aligned neighbours** (`rho = 1`, the oversmoothed limit):
  `gamma^2 = < (sum_v P_uv)^2 > = 1` for any row-stochastic `P`.

Three consequences.

**(i) Softmax attention is self-normalising.** `S` is row-stochastic by
construction, so `gamma_A^2 in [<1/d_eff>, 1]` — bounded, and `-> 1` exactly in
the aligned limit. The unnormalised adjacency `P = A` has row sums `d_u`, giving
`gamma_P^2 -> d_bar^2` (aligned) or `d_bar` (decorrelated) — unbounded in the
graph. **The attention branch needs no `gamma` hyperparameter; the sum-aggregation
MPNN branch does.** This is the derivation of the paper's own §2.5 empirical
finding, extended to attention.

**(ii) `gamma` is a depth-dependent order parameter, not a constant.** `rho_{vv'}`
is the neighbour feature correlation, and message passing *increases* it with
depth — that is what oversmoothing is. So `gamma_l` genuinely drifts with `l`,
which is why the paper's Eqn 19 carries a layer index and why collapsing it to a
single scanned `gamma` (their Eqn 15) is an approximation. Formula (G) says how
big the approximation error is: `gamma_l^2` runs from `<1/d_eff>` to `1` as
`rho_l : 0 -> 1`, i.e. a factor `d_eff` over the depth of the network.

**(iii) `alpha_A >= 1/2` is forced.** `d_eff` is a function of the softmax
temperature, and the logits are `Theta(D_h^{1/2 - alpha_A})` (§3a):

| `alpha_A` | logits as `D -> inf` | `d_eff` | `gamma_A^2` |
|---|---|---|---|
| `> 1/2` | `-> 0` | `-> d_u` (uniform) | `D`-independent |
| `= 1/2` | `Theta(1)` | `Theta(1)` fraction of `d_u` | `D`-independent |
| `< 1/2` | `-> inf` | `-> 1` (hard argmax) | **drifts with `D`, `-> 1`** |

At `alpha_A < 1/2` the softmax saturates as the width grows, so the effective
aggregation collapses to a single neighbour and `gamma_A` drifts with `D`.
Feature scale then drifts with `D`, and learning-rate transfer in `D` must break.
**This is a width-transfer failure with no analogue in a dense transformer**: in
the sequence case a saturating softmax changes *which* token is attended, but the
aggregation is over one token either way at the relevant scale; on a graph the
`gamma` normalisation of the branch is explicitly part of the parameterisation,
so the saturation moves a factor the architecture depends on.

**F21 note.** `alpha_A < 1/2` turns attention into a hard `argmax`, i.e. a
discrete selection. Degenerate ties are not generic under random init, but two
graph-specific degeneracies are: a node whose neighbourhood is `{u}` alone
(self-loop only) has `softmax` over one element `= 1` identically for every
`alpha_A`, and a node all of whose neighbours carry identical features has exactly
tied logits and gets an index-order tie-break. Both are asserted against in the
model code.

---

## 5. The derived parameterisation

> **Read §9 before using this table.** Round 011 falsified the coherence label
> that fixes the `W_Q, W_K` row, found a second `Delta A` channel this section
> never counted, and reversed the `alpha_A = 1` recommendation. The rows the
> paper also states are unaffected.

`D_h = D/H`. `eta_0` is the single tuned base rate; `lambda_0` the base weight
decay. Rows the paper states are marked ✓; rows this file derives are marked
**NEW**.

| group | init | `sigma` | SGD needed `eta` | Adam needed `eta` | status |
|---|---|---|---|---|---|
| encoder `W^(0)` | `N(0,sigma_0^2)` | `sqrt(L)` / `1/sqrt(D)` | `eta_0 D sigma_0^2 C_{ab}` | `eta_0 sigma_0` | ✓ |
| MPNN `W~` | `N(0,1)` | 1 | `eta_0 D L` | `eta_0/sqrt(D)` | ✓ |
| attn `W_V` | `N(0,1)` | 1 | `eta_0 D L` | `eta_0/sqrt(D)` | **NEW** |
| attn `W_O` | `N(0,1)` | 1 | `eta_0 D L` | `eta_0/sqrt(D)` | **NEW** |
| attn `W_Q, W_K` | `N(0,sigma_QK^2)` | `sigma_QK = D_h^{1-alpha_A}` | `eta_0 D L` | `eta_0/sqrt(D)` | **NEW** |
| MLP `W_1, W_2` | `N(0,1)` | 1 | `eta_0 D L` | `eta_0/sqrt(D)` | ✓ |
| decoder `W^(L+1)` | `N(0,sigma_{L+1}^2)` | `sqrt(L)` / `1/sqrt(D)` | `eta_0 D sigma_{L+1}^2` | `eta_0 sigma_{L+1}` | ✓ |
| — | branch scale `1/L`, three branches | | | | **NEW** (count) |
| — | attention exponent | `alpha_A >= 1/2`; `alpha_A = 1` recommended | | | **NEW** |
| — | AdamW weight decay | `lambda = lambda_0 sqrt(D)` | | | ✓ |

**AdamW.** Their Eqn 12 requires `lambda eta` invariant. Every group's Adam rate is
`eta_0/sqrt(D)` under the `sigma` choices above — *including* `W_Q, W_K` — so
`lambda = lambda_0 sqrt(D)` covers the attention sector unchanged. No new entry.

**Why `alpha_A = 1` is the recommendation, in the paper's own language.** Their
desideratum 3 ("Complete Learning": updates should not converge to their
linearisation around the initialisation) applied to the *attention pattern*:

| | init `A` | trained `Delta A` | ratio |
|---|---|---|---|
| `alpha_A = 1/2` | `Theta(1)` | `Theta(1)` | `Theta(1)` — the init pattern survives the limit |
| `alpha_A = 1` | `Theta(D_h^{-1/2})` | `Theta(1)` | `D_h^{-1/2}` — the init pattern washes out |

This is structurally the *same* statement as the non-fan-in `sigma(W_down)` of
`derivations/06-moe.md` §1d ("trained coherent part beats init incoherent part"),
one level down, applied to the attention logits instead of the expert output. Two
independent reasons converge on `alpha_A = 1`: it is the complete-learning choice,
and it is the choice for which `sigma_QK = 1` and `W_Q, W_K` need no correction.

---

## 6. Two assumptions in the paper's counting that are about the DATA, not the model

Both are stated in the paper; neither is tested there. They are recorded because
they bound the domain of the whole Table 1 (F19).

**(a) Node alignment.** Their Eqn 36 assumes `M_l := (E||(1/N) 1^T X^(l)||^2)^{1/2}
= Theta(sqrt(D))`, i.e. the *node-mean* feature has `Theta(1)` entries. For
decorrelated nodes it would be `Theta(sqrt(D/N))`. Their Eqn 32 makes the
matching assumption `M^{(l)}_{ab} ~ D sqrt(N_beta)`. Re-deriving §1 with
decorrelated nodes instead gives `eta_res^SGD = eta_0 D L N_alpha` — **the `D` and
`L` exponents are unchanged; only an `N` factor moves.** So the alignment
assumption is a *graph-size* correction of the same species as their `C_{alpha
beta}`, not a width/depth risk. Under Adam it does not appear at all in the direct
sign-GD route (§1). Good news for their Table 1, and worth stating because their
own route (rescaling by an RMS-gradient surrogate) makes the `M`'s appear in
numerator and denominator and is therefore harder to audit.

**(b) The attention sector needs a *different* first-layer correction.** `C_{alpha
beta}` arises because the encoder gradient is proportional to the **pooled** input
`1^T X_alpha`, then read out against `X_beta`. The `W_Q/W_K` gradient is
proportional to the **un-pooled** node features `x_{alpha,u}`, because attention is
per-node. So the correct alignment factor for the attention sector is the
node-level Gram `X_beta X_alpha^T`, not the pooled `(1/N_alpha) 1^T X_alpha X_beta^T`.
For sparse bag-of-words features — exactly the regime where the paper measures
`C_{ab} = 13.5-22` (Cora/Citeseer/Pubmed, their Table 2) — pooling is what makes
`M_{ab}` small, so the attention sector's correction should be **much closer to 1**
than the encoder's. Prediction P7 below.

---

## 7. Where this derivation is most likely to be wrong

Ranked, most-likely first:

1. **The C2 label at `t >> 1`.** I assert `Delta W_O ∝ g (x) o` makes
   `(g^T W_O).(v-o)` coherent from step 2. `03-attention.md` §D2b is the record of
   this program getting the *same* contraction wrong in the other direction, and
   the correction there was found only by reading a figure caption. The
   consequence if C2 never becomes coherent: `sigma_QK = D_h^{3/2 - alpha_A ...}`,
   i.e. the whole `alpha_A = 1 => no correction` result moves by `sqrt(D_h)`.
   **Measured directly in P4.**
2. **The predicted first-step suppression is `Delta A|_{t=1} = Theta(D_h^{-1/2})`,
   but `03-attention.md` records Bordelon et al.'s Fig 12(b) as
   `(Delta A)^2 ~ N^{-2}` at `t=1`, i.e. `Delta A ~ D_h^{-1}`.** That is one power
   of `sqrt(D_h)` more suppression than I get. Either their measurement is of a
   different object (their D6 discusses across-head `Var(A) ~ N^{-2}`, which is
   *not* the same quantity), or one of C1–C4 has a second incoherent contraction I
   have not found. **I do not claim to have resolved this**, and the program's
   standing prior says the error is mine. P4 measures the exponent directly so the
   record contains a number rather than an argument.
3. **The sign factorisation in §3d** treats `sign(grad_{W_Q})` as a product of a
   key-sign and a feature-sign. With `N d_bar` node pairs contributing to each
   entry, the signs partially cancel and the factorisation degrades. The paper
   makes the identical approximation in its own Prop. 3; if it is wrong, it is
   wrong for their Table 1 too.
4. **`Delta s_uv = Theta(Delta A)`** assumes the softmax Jacobian is `Theta(1)`.
   It is `s_uv(delta - s_uv')`, which is `Theta(1/d_eff)` for spread attention —
   so on a high-degree graph the attention-pattern channel is suppressed by
   `d_eff` relative to what §3c assumes. This does not touch `D`, `L` or `H`, but
   it does mean the *graph* enters the Q/K rate through `d_eff`.
5. Everything here is one-step counting, so **by F18 it claims the `t -> large`
   labelling**, except where a row is explicitly marked `t = 1`.

---

## 8. Registered predictions (bars are in `rounds/011-graph-transformer/prereg.md`)

| # | prediction | control that must bite |
|---|---|---|
| P1 | feature RMS and per-branch `Delta x` flat in `D`, `H`; per-branch `Delta x ~ 1/L` | mis-scaled `eta_Q` must break `Delta A` |
| P2 | `gamma_A` flat in `D` at `alpha_A in {1/2, 1}`; drifts toward 1 at `alpha_A = 0` | `alpha_A = 0` is the control |
| P3 | optimal `eta_0` transfers across `D`, `L`, `H` under the §5 table | `alpha_A = 0`, and `sigma_QK = 1` at `alpha_A = 1/2`, must not |
| P4 | `Delta A\|_{t=1} ~ D_h^{-1/2}` under SGD; `~ D_h^0` under Adam | the two optimisers are each other's control |
| P5 | `A_init ~ D_h^{1/2 - alpha_A}` | — |
| P6 | `gamma_l` rises with `l` toward 1 (oversmoothing), per formula (G) | a decorrelating (random-rewire) graph must not |
| P7 | the attention alignment factor stays near 1 where `C_{alpha beta} >> 1` | dense features must give both near 1 |

---

## 9. CORRECTIONS FORCED BY ROUND 011

Kept as an appendix rather than folded into the text above, so the record shows
what was derived before measuring and what measurement changed
(`rounds/011-graph-transformer/results.md`).

### 9a. §3 counted one channel out of two — the derivation was incomplete

`Delta A` also moves because the block's **input** moves. The residual stream is
designed so `Delta xt = Theta(1)`; with `q = (1/(sigma_QK sqrt(D))) W_Q xt` and
`W_Q` random, `Delta q = Theta(1)` per coordinate, and `Delta(q.k)` is an
incoherent sum over `D_h`:

    **Delta A |_stream  =  Theta( D_h^{1/2 - alpha_A} )**                        (S)

— exactly the order of `A_init`, and carrying **no dependence on the Q/K
learning rate at all**. Measured on four configurations (E8): `−0.454`, `−0.482`
at `alpha_A = 1` and `+0.080`, `+0.069` at `alpha_A = 1/2`, against `(S)`'s
`−0.5` and `0`.

**Why this hid.** At `alpha_A = 1` channel (S) and the Q/K channel both give
`D_h^{-1/2}`, so the total agreed with a derivation that had counted only one of
them. P5's `A_init` slopes and P4's `Delta A` slopes are the *same number* at
that exponent. The failure only became visible at `alpha_A = 1/2`, where the two
separate. Registered as **F23**; the guard is to enumerate every *input* to an
observable, not only the parameters that "belong" to it.

### 9b. The C2 coherence flip is FALSIFIED at `t <= 256` under SGD

§3b asserted that `Delta W_O ∝ g (x) o` makes the backward-through-`W_O`
contraction coherent from step 2. It does not, at any horizon run. With the Q/K
group frozen alone (E8), the SGD slope is `−0.562` at `t=1` and `−0.544` at
`t=256` — a drift of `0.018` over 256 steps, against a coherent prediction of
`0`. The **incoherent** labelling is what the data support, and it fits at both
`alpha_A` under the `alpha_A`-compensating rate, so it is a two-point test.

Under **signGD** the same channel is nearly coherent already (`−0.151`) and does
drift (`−0.065` by `t=256`), which is what §3d's sign argument predicts and which
shows the probe is not blind to the effect.

Consequence for §3e. With the incoherent labelling the needed rate is
`eta_Q^SGD = eta_0 D L sigma_QK^2 D_h^{2 alpha_A - 3/2}`, hence

    sigma_QK = D_h^{3/4 - alpha_A}      (incoherent, supported at t <= 256)
    sigma_QK = D_h^{1 - alpha_A}        (coherent, §3e, NOT supported)

At `alpha_A = 1` these are `D_h^{-1/4}` and `1`. **Neither is established
empirically** — §3 of the results shows the transfer harness cannot resolve a
`D_h` mis-scaling of the Q/K logit sector (0.12 dec against a 0.62-dec
sensitivity for the same sector's `W_V/W_O` rate). The `sigma_QK` row of §5
should be read as *derived, contested, unmeasured*.

### 9c. The `alpha_A = 1` recommendation is reversed by measurement

§5 recommended `alpha_A = 1` on two grounds: complete learning of the attention
pattern, and `sigma_QK = 1`. Both are undercut.

With C2 incoherent, at `alpha_A = 1` **every** channel of `Delta A` is
`Theta(D_h^{-1/2})`, and so is `A_init`. The attention matrix therefore converges
to *uniform over each graph neighbourhood at all times*, not just at
initialisation — the MHSA branch's limit is a mean-aggregation MPNN branch with a
learned value/output projection. That is a stable, transferring parameterisation
of a model whose attention does nothing. The complete-learning table in §5
compared `A_init` against a `Delta A` that assumed coherence; with (S) in hand
the correct table is:

| | `A_init` | `Delta A` (all channels, measured) | attention in the limit |
|---|---|---|---|
| `alpha_A = 1/2` | `Theta(1)` | `Theta(1)` | non-degenerate |
| `alpha_A = 1` | `Theta(D_h^{-1/2})` | `Theta(D_h^{-1/2})` | **uniform aggregation** |

So for *practice* the corrected recommendation is **`alpha_A = 1/2`**. For
*theory* `11` §9 still prefers `alpha_A = 1`, because at `1/2` the DMFT closure
needs `<softmax(Theta(1) Gaussian field)>` and drops an unclosed Jensen gap. The
two routes disagree, and this file does not resolve it.

### 9d. §1a overstated the consequence of the paper's §2.4 typo

The inconsistency is real (Table 1 and Prop. 3 say `1/sqrt(D)`, §2.4 says `1`).
I claimed it "looks exactly like a failure of width transfer". Measured: drift
0.110 dec, TRANSFERS (E10). Both conventions leave the forward pass unchanged
(`z = Theta(D^{-1/2})` either way); only the nominal decoder scale moves by
`sqrt(D)`, without producing the preregistered transfer failure. The typo is
worth fixing; it is not a transfer failure.

### 9e. What survived unchanged

§1 (all of the paper's own rows), §2 (`W_V`, `W_O` take the residual rate and are
head-blind), §4 (formula (G), verified exactly on matched equal-norm inputs and
at its decorrelated/aligned endpoints, and `alpha_A >= 1/2` forced by
softmax saturation), §5's encoder/MPNN/MLP/decoder/AdamW rows, and §6a/6b (the
alignment assumptions). §7's ranked list of likely errors put the C2 label first
and the missing-channel problem nowhere — the ranking was right about *where* to
look and wrong about *what* was there.
