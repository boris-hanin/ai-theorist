# CompleteP for Adam, derived from first principles — then checked against 2505.01618

Target: Dey, Zhang, Noci, Li, Bordelon, Bergsma, Pehlevan, Hanin, Hestness,
*Don't be lazy: CompleteP enables compute-efficient deep transformers*
(NeurIPS 2025, arXiv:2505.01618v4). Optimizer: **Adam, no weight decay**.

Conventions are the paper's: `N` is the residual width (`h^l in R^N`), `L` the
depth, `m_N = N/N_base`, `m_L = L/L_base`, and

    h^{l+1} = h^l + m_L^{-alpha} F_l(h^l),        alpha in [0.5, 1]

**Scoreboard: all six rules derived and matching Table 1.** The derivation also
pins down exactly what went wrong in round 004.

## The one fact that drives everything

Adam's update is

    Delta theta = -eta * mhat / (sqrt(vhat) + eps)

and `mhat/sqrt(vhat)` is `O(1)` **per entry, whatever the gradient's magnitude**.
So

    |Delta theta| = Theta(eta)   per entry, independent of gradient scale.

Every rule below follows from combining that with how many entries get summed
coherently downstream. This is the whole difference from SGD, where the update
inherits the gradient's scale and the counting runs the other way.

## D1. Hidden LR: `eta_base * m_N^{-1}` — CONFIRMED

Take a hidden linear layer `y = W x`, `W in R^{N x N}`, `x` with `Theta(1)`
entries. The gradient is `dL/dW_ij = g_i x_j`, so Adam's sign-like update is
`Delta W_ij ~ -eta * sign(g_i) sign(x_j)`. Then

    Delta y_i = sum_j Delta W_ij x_j = -eta sign(g_i) sum_j |x_j| = Theta(eta N)

— a **coherent** sum over all `N` inputs, because the update is aligned with `x`
by construction. Requiring `Delta y = Theta(1)`:

    **eta_hidden ∝ 1/N  =  eta_base * m_N^{-1}**

**Check:** Table 1, Hidden LR (AdamW) for muP and `alpha in {0.5,1}`:
`eta_base * m_N^{-1}`. **Match.**

## D2. Embedding, bias and LayerNorm LRs carry no `m_N` — CONFIRMED

An embedding row, a bias, and an LN gain are all **per-activation** parameters:
the entry `theta_i` feeds `h_i` directly with no sum over a fan-in. So
`Delta h_i = Theta(eta)` with no factor of `N`, and `Delta h = Theta(1)` needs

    **eta = eta_base**, width-independent.

**Check:** Table 1 — Emb. LR, Hidden Bias LR, Pre-LN LR and Final-LN LR all
carry no `m_N`. **Match.** The contrast with D1 is the whole content: it is not
that "some layers are special", it is whether the parameter's effect reaches the
activation through a coherent fan-in sum or not.

## D3. Hidden init variance `sigma^2_base * m_N^{-1}` — CONFIRMED

For `y = W x` with no forward multiplier, `y_i = sum_j W_ij x_j` is an
**incoherent** sum of `N` terms at init, so `y = Theta(sqrt(N) * sigma)`. For
`y = Theta(1)`: `sigma^2 ∝ 1/N`. **Check:** Table 1 Hidden Init Var
`sigma^2_base m_N^{-1}`. **Match.** Note this is the *incoherent* count while D1
is the *coherent* one — same layer, different question, and the two exponents
differ by a factor of `N` for exactly that reason.

## D4. Unembedding forward multiplier `m_N^{-1}` — CONFIRMED

With `f = m_N^{-1} W_u h`, at init `W_u h` is incoherent (`Theta(sqrt N)`), so
`f = Theta(N^{-1/2}) -> 0` — the muP property that the network starts at zero
output. After a step, `Delta W_u` is aligned with `h`, so `Delta W_u . h` is
coherent (`Theta(eta N)`), and `Delta f = m_N^{-1} Theta(eta N) = Theta(eta)`.
**Check:** Table 1 Unemb. Fwd. `X_L W_unemb^T m_N^{-1}`. **Match.**

## D5. The depth LR factor `m_L^{alpha - 1}` — CONFIRMED

With the width rules in place, one Adam step moves a block's output by
`Delta F_l = Theta(eta_base * d_L)`, where `d_L` is whatever extra depth factor
the LR carries. Each block enters the stream through `m_L^{-alpha}`, and all
`2L` blocks are driven by the same loss, so their contributions add
**coherently**:

    Delta h_total ~ L * m_L^{-alpha} * eta_base * d_L  ~  m_L^{1-alpha} d_L eta_base

Requiring `Delta h_total = Theta(1)`:

    **d_L = m_L^{alpha - 1}**

**Check:** Table 1 gives `eta_base m_N^{-1} m_L^{alpha-1}` for Hidden LR and
`eta_base m_L^{alpha-1}` for Pre-LN and Hidden Bias LR. **Match**, including the
fact that the depth factor attaches to *every* in-block parameter group while
the width factor attaches only to the fan-in ones. And the paper states the same
requirement directly in §6: "we need the weight update to satisfy
`Delta w_l^i = Theta(L^{alpha-1})`."

Note the embedding, final-LN and unembedding LRs get **no** depth factor in
Table 1 — consistent, since they sit outside the residual stack and the
`L`-block counting does not apply to them.

## D6. Why `alpha = 1` is the unique complete choice — CONFIRMED

D5 shows every `alpha in [0.5,1]` keeps the *total* stream update `Theta(1)`. So
stability alone does not pick `alpha`. What separates them is whether each block
still uses its own **nonlinearity** in its parameters.

Expand a block's response to its own weight update. For a two-matrix block
`F = W_(2) W_(1) h`,

    Delta_theta h^{l+1} = m_L^{-alpha} ( W_(2) h Delta W_(1) + W_(1) h Delta W_(2) )   <- linear in Delta W
                        + m_L^{-alpha} h Delta W_(1) Delta W_(2)                       <- second order

With `Delta W = Theta(m_L^{alpha-1})` from D5:

| term | order |
|---|---|
| linear | `m_L^{-alpha} * m_L^{alpha-1}` = **`m_L^{-1}`** |
| second order | `m_L^{-alpha} * m_L^{2(alpha-1)}` = **`m_L^{alpha-2}`** |

Their **ratio is `m_L^{alpha-1}`**. It is `Theta(1)` only at `alpha = 1`; for any
`alpha < 1` it vanishes as `L` grows, and the block converges to its own
linearization. At `alpha = 1/2` the ratio goes as `L^{-1/2}`.

**Check:** the paper's Eq. (6) gives exactly these two orders and says "the two
terms have the same order only when `alpha = 1`". Their Figure 6 measures the
relative distance from linearization and reports `alpha = 0.5` converging to the
linearization **at rate `1/sqrt(L)`** — which is `m_L^{alpha-1}` at
`alpha = 1/2`. **Match, including the exponent.**

So `alpha = 1` is not "better tuned"; it is the unique exponent at which the
second-order term survives the limit. Hence *Complete*P.

## What this says about round 004 — a precise diagnosis

Round 004 tried to measure exactly this ratio and got a flat slope at every
`alpha`, and I recorded the reason as: *numerator and denominator both carry
`m_L^{-alpha}`, so the ratio is scale-free in `L` by construction.*

**That diagnosis was wrong.** The `m_L^{-alpha}` does cancel — but the ratio is
still `m_L^{alpha-1}`, because the two terms differ in their **powers of
`Delta W`** (second order vs first), not in their prefactor. The estimator's
*structure* was right.

The actual bug: round 004 ran with `lr_depth_exp = 0.0`, i.e. **no depth factor
on the learning rate at all**, for both `alpha` values. But the entire counting
above requires `Delta W = Theta(m_L^{alpha-1})`, which is precisely what that
factor supplies. With it forced to zero, `Delta W` is identical for both
`alpha`, and the ratio cannot separate them — a flat slope is then the *correct*
output of a correctly-structured estimator fed the wrong parameterisation.

Two errors, one masking the other: a wrong parameterisation, and a wrong story
about why the result was uninformative. The registry guard from round 005
applies again — **enumerate every factor in the chain and label it** — and this
time the missing one was in the optimiser, not the forward pass.

## Summary

| # | Rule | Derived from | Table 1 |
|---|---|---|---|
| D1 | Hidden LR `∝ m_N^{-1}` | coherent fan-in sum of an aligned Adam update | match |
| D2 | Emb / bias / LN LR: no `m_N` | per-activation parameter, no fan-in sum | match |
| D3 | Hidden init var `∝ m_N^{-1}` | incoherent fan-in sum at init | match |
| D4 | Unemb forward `× m_N^{-1}` | coherent update vs incoherent init | match |
| D5 | Depth LR `× m_L^{alpha-1}` | `L` blocks each entering via `m_L^{-alpha}` | match |
| D6 | `alpha = 1` unique | linear `m_L^{-1}` vs 2nd-order `m_L^{alpha-2}` | match, incl. the `1/sqrt(L)` rate |

Not derived here: AdamW `eps` scaling (`eps_base m_N^{-1} m_L^{-alpha}` for
residual blocks) and weight decay (`lambda_base m_N`). Weight decay is out of
scope by instruction; `eps` is implemented per Table 1 and kept small enough
that the solve does not sit in the `eps`-dominated regime.


# Provenance: how every rule in the implementation was obtained

`skills/dmft-resnet-depth/scripts/completep.py` implements four
parameterisations. This table is the audit trail: for each rule, **where it came
from**, and whether it was derived here, transcribed from the paper, or chosen
by me. Anything in the third category is an implementation decision the paper
does not fix, and is flagged so a later disagreement can be traced to it.

## Derived here, then checked against Table 1

| Rule | Applies to | Derivation | Table 1 | Code |
|---|---|---|---|---|
| hidden LR `x m_N^{-1}` | muP, alpha | **D1** — coherent fan-in sum of an aligned Adam update | match | `groups()` group 1 |
| no `m_N` on emb / bias / LN LR | muP, alpha | **D2** — per-activation parameter, no fan-in sum | match | `groups()` groups 2,3 |
| hidden init var `x m_N^{-1}` | muP, alpha | **D3** — incoherent fan-in sum at init | match | `Model.__init__` |
| unemb forward `x m_N^{-1}` | muP, alpha | **D4** — incoherent init vs coherent update | match | `self.uscale` |
| in-block LR `x m_L^{alpha-1}` | alpha only | **D5** — `L` blocks each entering via `m_L^{-alpha}` | match | `groups()`, `dep` |
| residual `x m_L^{-alpha}` | alpha only | given by Eq. (1); `alpha=1` singled out by **D6** | match | `self.res` |

## Transcribed from the paper, NOT derived here

| Rule | Source | Why not derived |
|---|---|---|
| AdamW `eps x m_N^{-1} m_L^{-alpha}` (blocks), `m_N^{-1}` (outside) | Table 1 | Not attempted. Implemented as written, with `eps_base` set small enough that the run is not in the `eps`-dominated regime — so the rule is present but not exercised. |
| weight decay `lambda x m_N` | Table 1 | Out of scope by instruction (Adam, no weight decay). `weight_decay=0.0` throughout. |
| attention logits `Q^T K / N` for **all** parameterisations | paper §3 | Taken as given. Independently justified by `03-attention.md` D1/D2 and by 2405.15712's requirement that `alpha_A = 1` for the `N -> inf` limit to exist. |
| SP = no width rules at all | Table 1 | Definitional. |

## My implementation choices — NOT from the paper

These are free parameters the paper does not fix. If a result later disagrees
with the paper, **check these first**.

| Choice | Value here | Paper | Note |
|---|---|---|---|
| `N_base`, `L_base` | 64, 2 | 256, 2 | Smaller base to keep the sweep affordable. All parameterisations coincide at the base shape either way — verified as sanity check 1. |
| `sigma_base` | `N_base^{-1/2}` | free | Table 1 gives every layer variance `sigma^2_base` and only rescales *hidden* by `m_N^{-1}`, so `sigma_base` must independently make the **base** model well conditioned. At `sigma_base = 1` the init loss was 21.2 against `ln V = 4.16`; at `N_base^{-1/2}` it is 4.18. Recorded because it is a real degree of freedom, not a detail. |
| `eps_base` | `1e-12` | free | Deliberately far below the gradient scale so the `eps` rule is inert. |
| task | synthetic next-token, `V=64`, `S=16` | 300M tokens of real text | The biggest deviation. Named in advance as the first suspect for any disagreement. |
| heads, MLP ratio | 4 heads, `4N` | 4N stated | heads not specified at this scale. |
| Adam betas | (0.9, 0.95) | not extracted | conventional LLM values; not verified against the paper. |
| steps | 30-40 full batch | 300M tokens | far shorter horizon. |

## Two controls that are NOT informative, and why

1. **"CompleteP minus the depth-LR factor" is a no-op.** At `alpha = 1` that
   factor is `m_L^{alpha-1} = m_L^0 = 1`. Removing the identity changes nothing,
   and the first sweep duly returned byte-identical drift (0.042 both). Removed
   from the sweep with a comment rather than left in looking like a fifth
   result. *Same trap as round 005's `alpha_L = 1/2` row — a control is only
   evidence if the thing it removes is not already the identity.*

2. **The real depth control ladder is the parameterisation list itself.** muP is
   "no residual scaling"; `alpha = 0.5` is partial; `alpha = 1` is full. Those
   three rows are the ablation.

## Verification performed before any sweep

| Check | Result |
|---|---|
| All four parameterisations identical at the base shape (`m_N = m_L = 1`) | init loss identical to 4 d.p. (21.2407, then 4.1790 after the `sigma_base` fix); `res = uscale = 1` |
| Init loss near `ln V` | 4.179 vs 4.159 |
| Optimum interior in the swept LR grid at every dial value | yes, after regridding |
