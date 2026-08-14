# Constant `L M / D` and Jiang MoE hyperparameter transfer

> **Status before round 018 measurement: derived compatibility conditions and
> preregistered initialization check.** This combines Jiang et al. Table 2
> (arXiv:2601.20205v3) with the Chizat residual-scale identity. It is not a
> full MoE cavity solution; `derivations/07-moe-dmft.md` documents that gap.

## 1. Scope

Write residual width `D`, depth `L`, hidden width per expert `M`, expert count
`E`, active experts `A`, fixed sparsity `kappa=A/E`, and

```text
rho = L M / D.
```

The exact DMFT limit order would take the width variables to infinity first at
fixed dataset size and finite training horizon. Real language pretraining does
not obey that finite-data limit. Consequently the conclusion here is only
about source-parameterisation and structural mean-field compatibility. It does
not by itself predict a validation-loss curve.

## 2. The apparent extra rescaling is already in Jiang

Jiang's non-fan-in expert-down initialization is

```text
sigma_down = alpha_ffn^(-1) D^(-1/2)
           = (D/M) D^(-1/2)
           = sqrt(D)/M.
```

Their architecture applies `1/L` to every residual branch. Thus the effective
per-unit residual coefficient is

```text
(1/L) sigma_down = sqrt(D)/(L M).
```

This is the Chizat MLU residual scale. It is an identity between conventions,
not a new factor to apply to the raw down matrix. Applying it to the raw matrix
would produce `sqrt(D)/(L^2 M)` after the residual branch and suppress the
stream-init variance by an erroneous extra `L^-2`.

## 3. Substitute constant rho

On a constant-rho path,

```text
M = rho D / L,              alpha_ffn = M/D = rho/L.
```

Therefore

```text
raw down std                 = L/(rho sqrt(D)),
residualized down std        = 1/(rho sqrt(D)),
raw down Adam LR             proportional to 1/M = L/(rho D),
residualized down Adam LR    proportional to 1/(L M) = 1/(rho D).
```

Both effective quantities are independent of depth once expressed in the
normalized width coordinate. This is the core compatibility result.

Relative to a tuned reference `(L0,M0,D0)`, let `rX=X/X0`. Constant rho means
`rM=rD/rL`. Substitution into every source group gives:

| group | raw Adam LR ratio | Adam epsilon ratio |
|---|---:|---:|
| embeddings | `1` | `rD^-1` |
| norms | `1` | `1` |
| attention QKV/output | `rD^-1` | `rD^-1 rL^-1` |
| router | `rD^-1` | `rD^-1 rL^-1` |
| expert up | `rD^-1` | `rM^-1 rL^-1 = rD^-1` |
| expert down | `rM^-1 = rL/rD` | `rD rM^-2 rL^-1 = rL/rD` |
| ordinary biases | `1` | `rL^-1` |
| manual load bias | `1` | not in Adam |

The embedding initialization stays in its tuned reference coordinate. The
tied unembedding receives the source inverse-width multiplier `rD^-1`. The
attention, router, expert-up, and expert-down initializations retain all Jiang
source constants; only their scale ratios change. A single global LR or a
single global Adam epsilon is therefore incompatible with this path.

## 4. Which limiting process is preserved?

Jiang's structural order parameter is

```text
alpha_* = D/(M E L).
```

At constant rho,

```text
alpha_* = 1/(rho E),
Var(initial residual-stream contribution) proportional to D/(L A M)
                                                   = 1/(rho A)
                                                   = alpha_*/kappa.
```

Hence fixed `rho`, `E`, and `kappa` preserve the same positive `alpha_*`: the
same finite neural-SDE sector. They do **not** produce the universal
`alpha_*=0` neural ODE. Growing `E` and `A=kappa E` drives both quantities to
zero and recovers that ODE sector.

## 5. The depth caveat

Constant rho does not keep the FFN ratio fixed:

```text
alpha_ffn = rho/L.
```

The measured coherent/incoherent crossover in the expert-down alignment is
near `alpha_ffn=1.8`, and the clean asymptotic exponent was observed only for
`alpha_ffn>=16`. Thus a finite `rho=32` ladder through `L<=16` is structurally
defensible, though its deeper rungs remain crossover-sensitive. Sending
`L->infinity` at fixed rho forces `alpha_ffn->0`; that unbounded path must not
be advertised as an already validated corollary of Jiang transfer.

## 6. Decision

Constant `L M / D` is compatible with the Jiang MoE parameterisation provided:

1. `sqrt(D)/(L M)` is treated as the **effective residualized** down scale;
2. the raw matrix remains `sqrt(D)/M`;
3. all Jiang per-group Adam LRs and epsilons are applied after tuning at one
   reference shape;
4. `A/E` is fixed;
5. the claimed ODE/SDE sector is tracked through `alpha_*`; and
6. the ladder does not silently cross into an unvalidated `M/D` regime.

Round 018 tests the scale identities and the double-depth control before a
trained transfer campaign is allowed to run.
