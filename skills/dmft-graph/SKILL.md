---
name: dmft-graph
description: Parameterisation and scaling limits for GNNs and GRAPH TRANSFORMERS (DeZoort-Hanin, arXiv 2607.05017, extended to attention) — width/depth/head scaling, the message-passing normaliser gamma and where it comes from, the attention exponent alpha_A, the first-layer data-dependent correction C_ab, and what the node index does to a DMFT. Use for GNN/GT hyperparameter transfer and graph scaling limits.
---

> **Status: DERIVED IN ROUND 011, PARTIALLY MEASURED.** Derivations:
> `derivations/10-graph-transformer.md` (heuristic one-step),
> `derivations/11-graph-transformer-dmft.md` (cavity). Measured:
> `rounds/011-graph-transformer/results.md`. Not a reconstruction — every claim
> below points at either a derivation section or a measured number, and the ones
> that are neither say so.

# Graph transformers (delta on `dmft-derivation` and `dmft-attention`)

Prerequisites: `dmft-derivation` Phases 0–1; `05-completep-dmft-sgd.md` for the
depth sector (the branch multiplier is `1/L`, i.e. CompleteP `alpha = 1`,
unchanged); `03-attention.md` for the key/query response pair.

## The single most important structural fact

**Width is the mean-field direction. The NODE index is a data index and is not
averaged over.**

Everything else follows. The message-passing operator `P` and the attention
matrix `S` are never *disorder* — they are fixed linear maps on the node index of
every kernel, sandwiched between them, never cavity-expanded. Nothing
self-averages in the node direction, so every statement of the form "the graph
contributes `Theta(1)` factors" (which the paper makes throughout its §3, and
which `10` inherits) is a **fixed-graph** statement, not a limit theorem. A
theory of graph *ensembles* needs a second average that has not been shown to
commute with the width limit.

## The parameterisation

`D` width, `L` depth, `H` heads, `D_h = D/H`. Weights `N(0,1)` with explicit
`1/sqrt(fan_in)` prefactors (the paper's convention), scale carried by the
prefactor.

| group | `sigma` | SGD needed `eta` | Adam needed `eta` |
|---|---|---|---|
| encoder `W^(0)` | `sqrt(L)` (SGD) / `1/sqrt(D)` (Adam) | `eta_0 D sigma_0^2 C_ab` | `eta_0 sigma_0` |
| MPNN `W~` | 1 | `eta_0 D L` | `eta_0/sqrt(D)` |
| attention `W_V`, `W_O` | 1 | `eta_0 D L` | `eta_0/sqrt(D)` |
| attention `W_Q`, `W_K` | `sigma_QK` (below) | `eta_0 D L` | `eta_0/sqrt(D)` |
| MLP `W_1`, `W_2` | 1 | `eta_0 D L` | `eta_0/sqrt(D)` |
| decoder `W^(L+1)` | `sqrt(L)` / `1/sqrt(D)` | `eta_0 D sigma_{L+1}^2` | `eta_0 sigma_{L+1}` |
| AdamW decay | — | — | `lambda = lambda_0 sqrt(D)`, all groups |

- **`W_V` and `W_O` are head-blind.** Both are `D x D` with fan-in `D` whatever
  `H` is; the head split only re-labels columns. Do not attach `H` to them.
  *Domain (F19): this is fixed by counting the fan-in of exactly those two
  matrices. It does not extend to `W_Q, W_K`, whose relevant sum is over `D_h`.*
- **`W_Q, W_K` need a `sigma_QK`, and its exponent is CONTESTED AND UNMEASURED.**
  Two candidates, differing by which label the backward-through-`W_O`
  contraction carries: `D_h^{1-alpha_A}` (coherent — asserted in `10` §3e and
  **falsified at `t <= 256`**) or `D_h^{3/4-alpha_A}` (incoherent — what round
  011 E8 supports). Whichever it is, the **same** `sigma_QK` serves SGD and Adam,
  which nothing else in the table does (`sigma_0` is `sqrt(L)` for SGD and
  `1/sqrt(D)` for Adam). Do not quote either exponent as established: the
  transfer harness moves `lr*` only 0.12 dec under a `D_h` Q/K mis-scaling,
  against 0.62 dec for a `sqrt(D)` mis-scaling of `W_V/W_O` in the same sector.
- **`sigma_QK` and the Q/K learning rate are THE SAME KNOB.** `q =
  (1/(sigma_QK sqrt(D))) W_Q x` with `W_Q ~ N(0,sigma_QK^2)` is distributionally
  identical to `sigma_QK = 1`; only the needed rate moves. So a "control" that
  compares `sigma_QK = D_h^{1-alpha_A}` against `sigma_QK = 1` **is an algebraic
  identity at `alpha_A = 1`** and proves nothing there. Run it at
  `alpha_A = 1/2`. `gt.py` has `qk_correction_is_identity()` for this.
- **The paper's §2.4 says `sigma_{L+1} = 1` for Adam. That is a typo** — its own
  Table 1 and the closing line of its Prop. 3 both say `1/sqrt(D)`, and
  `eta^{(L+1)} = eta_0 sigma_{L+1}` forces `1/sqrt(D)`. Worth fixing, but
  **measured consequence is mild**: implementing the typo gives drift 0.110 dec
  and still TRANSFERS. Both conventions leave the forward pass unchanged
  (`z = Theta(D^{-1/2})`); only the decoder's needed rate moves, so it
  under-trains by `sqrt(D)` without destabilising anything.

## `gamma`: what it is, and why it must be scanned

The paper defines `gamma_l^2 = E||P X^(l)||_F^2 / E||X^(l)||_F^2` and advocates
scanning it. Derivation `10` §4 computes it. With `|x_v|^2 = D` and
`x_v . x_{v'} = D rho_{vv'}`:

    gamma^2 = < sum_v P_uv^2 + sum_{v != v'} P_uv P_uv' rho_{vv'} >              (G)

**Verified to within 1%** for `P = A~`, `P = D^-1 A`, and softmax attention at
three `alpha_A` (round 011 E9); ~9% for the unnormalised `P = A`, where node-norm
heterogeneity (which (G) assumes away) is amplified. Consequences:

- **Row-stochastic operators are self-normalising**: `gamma^2` lies in
  `[<1/d_eff>, 1]` with `d_eff = (sum_v P_uv^2)^{-1}` the participation ratio of
  the row. Softmax attention is row-stochastic *by construction*, so **the
  attention branch needs no `gamma` hyperparameter**. Sum aggregation (`P = A`,
  GIN-style) does: `gamma -> d_bar`. Measured `gamma = 6.6` at mean degree 7.8,
  bracketing the paper's own `gamma = 7` (Pascal) and `17` (MNIST).
- **`gamma` is a depth-dependent order parameter, not a constant.** `rho`
  rises with depth — that is oversmoothing — so `gamma_l` rises toward the
  row-sum value. This is why the paper's Eqn 19 carries a layer index and why
  collapsing to one scanned `gamma` (their Eqn 15) is an approximation. (G) says
  the size of the approximation: a factor `d_eff` across the network.
- **Degree-normalisation does not give `gamma = 1`.** The paper takes `gamma = 1`
  with `P = A~`; (G) gives `gamma ~ <1/d_eff>^{1/2} ~ 0.42` at mean degree 7.8.
  That is a *constant* — `D`- and `L`-independent — so it only rescales `eta_0`
  and does not break transfer. But it is not 1.

## `alpha_A`: the graph-specific constraint

`A_uv = D_h^{-alpha_A} q_u . k_v`, so `A_init = Theta(D_h^{1/2 - alpha_A})`
(measured slopes `+0.50 / -0.01 / -0.51` at `alpha_A = 0, 1/2, 1`).

**`alpha_A >= 1/2` is forced, for a reason with no dense-transformer analogue.**
Below `1/2` the logits diverge with width, the softmax saturates, `d_eff -> 1`,
and `gamma_A` drifts with `D` — so the branch normalisation the architecture
depends on is width-dependent and learning-rate transfer in `D` must break.
Measured at `alpha_A = 0`: `d_eff` `2.33 -> 1.21` and `gamma_A` `0.807 -> 0.952`
over `D = 32 -> 512`, both still moving at the largest width.

**Round 011 measured the choice between `1/2` and `1`, and it does not come out
where `10` §5 predicted.** With the Q/K backward contraction incoherent (below),
at `alpha_A = 1` *every* channel of `Delta A` vanishes as `D_h^{-1/2}` and so
does `A_init`, so the attention matrix converges to uniform-over-neighbourhood
**at all times** — the MHSA branch's limit is a mean-aggregation MPNN branch with
a learned value/output projection. It is stable and it transfers; it just has no
learned attention. `alpha_A = 1/2` is the only tested exponent whose attention
limit is non-degenerate. But theory pulls the other way — see the last row — so
this is a **named open problem, not a recommendation**:

| | `alpha_A = 1/2` | `alpha_A = 1` |
|---|---|---|
| `A` in the limit | CLT sum: a **random Gaussian field** that never stops fluctuating | LLN average: **deterministic**, computable from the kernels |
| heads | head-indexed field survives | collapse (measured across-head sd slope `-0.52`) |
| attention at init | selective, `Theta(1)` logits | exactly uniform mean-aggregation over each neighbourhood |
| DMFT closure | needs `<softmax(Gaussian field)>` — an **unclosed Jensen gap** | closure is exact |
| `sigma_QK` | `sqrt(D_h)` | 1 |

`11` §9 item 1 is the reason to prefer `alpha_A = 1` for *theory*: at `alpha_A =
1/2` the naive closure drops a `Theta(1)` Jensen gap that nobody in this program
knows how to close. Round 011's E8 is the reason not to be confident about it for
*practice* — see below.

## The open problem this skill must not hide

**Does the Q/K channel ever reach `Theta(1)` attention-pattern updates at
`alpha_A = 1`?**

`Delta A` has two channels, and `10` §3 enumerated only one:

1. **Q/K channel** — the Q/K weights move. Its size depends on whether the
   backward path through `W_O` (contraction C2) is coherent. Incoherent gives
   `D_h^{-1/2}` under the global rate; coherent gives `Theta(1)`.
2. **Stream channel** — the block's *input* moves by `Theta(1)`, which moves the
   logits with no reference to the Q/K rate at all. Size
   `Theta(D_h^{1/2 - alpha_A})`, i.e. the same order as `A_init`.

The stream channel was **missing from the derivation** and was found only because
E3 disagreed with it. It ties or beats the Q/K channel at both `alpha_A` tested,
which is exactly why the total was consistent with the derivation at
`alpha_A = 1` and inconsistent at `alpha_A = 1/2`. See
`rounds/011-graph-transformer/results.md` for what E8's freeze-one-group
decomposition then showed and for how far the horizon was pushed. **If C2 is
permanently incoherent, then at `alpha_A = 1` the attention pattern stays uniform
in the width limit and the MHSA branch degenerates into a second mean-aggregation
MPNN branch** — the parameterisation would still transfer, but it would be
transferring a model with no learned attention. That is the failure mode to test
for, and "the sweep transferred" does not rule it out.

## Checks specific to this skill

- **Self-loops are mandatory.** `softmax` over an empty neighbourhood is `NaN`
  under `masked_fill(-inf)`. `gt.py` asserts it.
- **Never validate on a degree-regular graph.** There, `D^-1/2 A D^-1/2` and
  `A/d` are the *same matrix*, so "degree-normalised vs constant-gamma" is an
  identity control. `gt.py` asserts a nonzero degree spread.
- **Never test the Q/K rule at `D_h = 1`** — every `D_h^k` correction is 1.
  `gt.py` refuses to construct it.
- **Never test the Q/K rule at `alpha_A = 1`** — the comparison is an identity
  (above).
- **Decompose `Delta A` by freezing groups** before attributing it to the Q/K
  learning rate. The stream channel is not small.
- **Report `d_eff` and the attention entropy beside any `gamma` claim.** `gamma`
  and `d_eff` move together and `gamma` alone cannot say why.
- Degenerate reduction: `P = I`, no attention branch, `N = 1` must reproduce
  `05-completep-dmft-sgd.md` exactly.

## Not established here

- **No DMFT solver.** `11` derives the single-site system; nothing evaluates it.
  There is no theory-vs-simulation comparison of the *dynamics*, hence no MC
  floor for one.
- **No real datasets**, no edge features, no positional encodings, no LayerNorm,
  no AdamW leg, no real Adam in the transfer sweeps (signGD proxy only).
- The `C_ab` first-layer correction is verified only in *mechanism* on synthetic
  sparse features, not against the paper's Cora/Citeseer/PubMed numbers.
- Everything in `10` is one-step counting, so by F18 it claims the `t -> large`
  labelling — and E8 is precisely the measurement of whether `t -> large` is
  reached in any horizon that was run.
