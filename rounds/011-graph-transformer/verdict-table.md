# Verdict table: my derived parameterisation vs arXiv 2607.05017, entry by entry

Mine: `derivations/10-graph-transformer.md` §5, as corrected by §9.
Theirs: their Table 1, Eqns 9/12, Props 1–3, Remarks 4–5.
Measured: `results.md`. `D` width, `L` depth, `H` heads, `D_h = D/H`.

**Reading the status column.** *AGREE* = same exponent, derived by a different
route (their Frobenius-norm recursion vs my coherent/incoherent contraction
counting), which is the F14 two-independent-routes bar. *NEW* = the paper has no
such entry; its §4 names attention as an explicit limitation. *DISAGREE* = we
give different answers. *CONTESTED* = I give two different answers depending on a
labelling the measurement did not settle.

---

## A. Rows the paper states

| # | quantity | theirs | mine | status | measured |
|---|---|---|---|---|---|
| A1 | encoder LR, SGD | `eta_0 D sigma_0^2 x C_ab` | `eta_0 D sigma_0^2 x C_ab` | **AGREE** | not isolated |
| A2 | encoder LR, Adam | `eta_0 sigma_0` | `eta_0 sigma_0` | **AGREE** | not isolated |
| A3 | residual LR, SGD | `eta_0 D L` | `eta_0 D L` | **AGREE** | `D` exponent **+1.041** vs `+1` (control moves `lr*` 0.94 dec) |
| A4 | residual LR, Adam | `eta_0 / sqrt(D)` | `eta_0 / sqrt(D)` | **AGREE** | not resolved (§3 of results) |
| A5 | decoder LR, SGD | `eta_0 D sigma_{L+1}^2` | `eta_0 D sigma_{L+1}^2` | **AGREE** | not isolated |
| A6 | decoder LR, Adam | `eta_0 sigma_{L+1}` | `eta_0 sigma_{L+1}` | **AGREE** | not isolated |
| A7 | `sigma_0` for a global rate | `sqrt(L)` (SGD) / `1/sqrt(D)` (Adam) | same | **AGREE** | — |
| A8 | `sigma_{L+1)` for a global rate | Table 1 + Prop. 3: `sqrt(L)` / `1/sqrt(D)` | same | **AGREE with Table 1** | — |
| A9 | `sigma_{L+1}` as printed in **§2.4** | "`sigma_{L+1} = 1`" (Adam) | `1/sqrt(D)` | **DISAGREE — I believe §2.4 is a typo** | consequence measured: 0.110 dec, **TRANSFERS** — the typo does *not* present as a width-transfer failure |
| A10 | branch multiplier | `1/L` (CompleteP `alpha = 1`) | `1/L`, and a third branch changes nothing but a factor 3 | **AGREE** | branch RMS `~ L^{-1.06 … -1.11}` |
| A11 | feature stability | `x^(l) = Theta_{D,L}(1)` | same | **AGREE** | flat in `D` (`+0.004`) and `H` (`+0.001`) |
| A12 | output scale at init | `z = Theta(M_L/D)`, `M_L ~ sqrt(D)` | same | **AGREE** | — |
| A13 | AdamW weight decay | `lambda = lambda_0 sqrt(D)` | same, **and unchanged by the attention sector** (every group sits at `eta_0/sqrt(D)`) | **AGREE + extension** | not attempted |
| A14 | first-layer correction | `C_ab = n0 sqrt(N_b) / M_ab` | same | **AGREE** | mechanism reproduced (grows with sparsity), magnitude bar failed (2.06× vs ≥3×) |
| A15 | `gamma` normalisation | defined by Eqn 14; "scan it as a hyperparameter" | **derived**: `gamma^2 = <sum_v P_uv^2 + sum_{v≠v'} P_uv P_uv' rho_{vv'}>` | **AGREE + I supply the formula they leave empirical** | corrected E9 verifies the identity exactly on six operators; an independent scan verifies the decorrelated/aligned endpoints |
| A16 | `gamma = 1` for `P = A~` | assumed throughout | `gamma` runs from `<1/d_eff>^{1/2}` for decorrelated features to `1` for aligned features | **AGREE in the aligned regime; earlier disagreement withdrawn** | independent scan: `0.2869` at `rho=0`, `1.0000` at `rho=1` on a mean-degree-14 graph |
| A17 | normalised `P` is needed for transfer (§2.5) | empirical claim | (G) says `gamma_P` is `D`-independent, so `P = A` should **not** break *width* transfer, only stability | **DISAGREE with the transfer reading** | `P = A`: **TRANSFERS**, drift 0.144 dec — but every `lr >= 2.7e-2` diverges. Their claim is about stability/sharpness, and that part holds |

## B. Rows the paper does not have (the attention sector)

| # | quantity | mine | status | measured |
|---|---|---|---|---|
| B1 | `W_V` LR | `eta_0 D L` (SGD) / `eta_0/sqrt(D)` (Adam) | **NEW** | mis-scaling it by `sqrt(D)` moves `lr*` **0.62 dec** → the sector is resolvable and the exponent is right |
| B2 | `W_O` LR | same as `W_V` | **NEW** | same control |
| B3 | `H`-dependence of `W_V, W_O` | **none** — fan-in is `D` whatever `H` is | **NEW** | branch RMS flat in `H` (`+0.016`); `H` transfer TRANSFERS (SGD 0.035 dec) |
| B4 | `W_Q, W_K` init rescaler | `sigma_QK = D_h^{1-alpha_A}` (coherent C2) **or** `D_h^{3/4-alpha_A}` (incoherent C2) | **NEW, CONTESTED** | **not resolvable by this harness**: the Q/K mis-scaling moves `lr*` only 0.12 dec against a 0.62-dec sensitivity for the same sector's `W_V/W_O` |
| B5 | one `sigma_QK` serves SGD *and* Adam | yes, for either labelling | **NEW** | unmeasured |
| B6 | logit scale at init | `A_init = Theta(D_h^{1/2-alpha_A})` | **NEW** | **`+0.499 / −0.013 / −0.513`** at `alpha_A = 0, 1/2, 1` vs `+0.5 / 0 / −0.5` |
| B7 | `alpha_A >= 1/2` is forced | below it the softmax saturates with width, `d_eff -> 1`, `gamma_A` drifts with `D` | **NEW** | at `alpha_A = 0`: `d_eff` `2.33 → 1.21`, `gamma_A` `0.811 → 0.951`, both still moving at `D = 512` |
| B8 | attention needs no free `gamma` hyperparameter if (G) is evaluated | row-stochastic ⇒ `gamma_A ∈ [<1/d_eff>, 1]`, bounded; sum-aggregation is not | **NEW** | `gamma_A` flat in `D` (`−0.010`, `−0.019` after probe correction); `P = A` gives `gamma = 5.8` at mean degree 7.8 |
| B9 | second `Delta A` channel | `Delta A\|_stream = Theta(D_h^{1/2-alpha_A})`, independent of the Q/K rate | **NEW — and missing from my own §3 until the measurement found it (F23)** | `−0.454, −0.482, +0.080, +0.069` vs `−0.5, −0.5, 0, 0` |
| B10 | C2 (backward through `W_O`) becomes coherent after one step | **asserted in §3b, FALSIFIED at `t <= 256` under SGD** | **DISAGREE WITH MYSELF** | Q/K-only slope `−0.562 (t=1) → −0.544 (t=256)`, drift 0.018; coherent predicts 0 |
| B11 | recommended `alpha_A` | §5 said `1`; §9c says **`1/2`** for practice | **REVERSED BY MEASUREMENT** | at `alpha_A = 1` every channel of `Delta A` is `Theta(D_h^{-1/2})` ⇒ attention → uniform aggregation at all times |
| B12 | attention concentration | `alpha_A = 1` is an LLN (deterministic `S`), `1/2` is a CLT (Gaussian field) — from the DMFT, `11` §M4b | **NEW (cavity route only)** | across-seed sd `−0.531 / −0.031`; across-head sd `−0.522 / −0.022` vs `−0.5 / 0` |
| B13 | the attention sector's alignment factor | the **node-level** Gram, not the pooled `C_ab` — so it stays near 1 where `C_ab` blows up | **NEW** | node-level `2.80 → 1.12` while pooled `14.6 → 30.0` across sparsity 0 → 0.97 |
| B14 | `gamma_l` is a depth-dependent order parameter | rises with `l` as `rho_l` rises (oversmoothing), per (G) | **NEW** | `rho` `0.003 → 0.023`, `gamma_P` `0.377 → 0.400`, monotone — but tiny, and the intended control does not bite |

## C. Where I disagree with the paper, ranked by confidence

The former `gamma = 0.42` versus `1` item is removed: those are the decorrelated
and aligned endpoints of the same formula, not competing answers.

1. **A9 — the §2.4 sentence contradicts Table 1 and Prop. 3.** High confidence
   that it is a typo, because the paper's own two other statements agree with my
   derivation and with each other. Low consequence: measured 0.110 dec, transfers.
2. **A17 — "normalised message passing is important for robust transfer".** I
   agree with the *stability* half and disagree with the *width-transfer* half:
   (G) makes `gamma_P` `D`-independent, so an unnormalised operator cannot move
   the width scaling. Measured: `P = A` transfers (0.144 dec) but diverges above
   `lr = 2.7e-2`. Their figures show broadened/blurred optima, which is what
   losing stability margin looks like — so this may be a difference of reading
   rather than of substance.

**Where I disagree with myself is the more important list**, and it is B10/B11
above: the C2 coherence label, and the `alpha_A` recommendation that depended on
it. The paper is not implicated in either — that sector does not exist in it.
