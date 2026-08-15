# Round 018 — Jiang MoE compatibility of constant `rho = L M / D`

Committed before the new Monte Carlo measurement. This round is an algebraic
and initialization-scale audit, not a language-model loss experiment and not a
full solution of the MoE DMFT.

## Question

Is constant `L M / D`, together with the `sqrt(D)/(L M)` residual scale,
compatible with the source-faithful Jiang MoE parameterisation?

## Convention under test

The Jiang raw expert-down matrix has entry scale `sqrt(D)/M`. The architecture
then multiplies the MoE residual branch by `1/L`. Therefore

```text
raw down std x branch factor = (sqrt(D)/M) x (1/L) = sqrt(D)/(L M).
```

The negative control puts `sqrt(D)/(L M)` into the raw matrix and then applies
the architectural `1/L` again.

## Fixed path

`rho = 32`, `E = 4`, `A = 1`, with `(L,M,D)` equal to
`(2,256,16)`, `(4,256,32)`, `(8,256,64)`, `(16,256,128)`.
Thus `alpha_ffn=M/D` traverses `16,8,4,2` without crossing below the measured
coherent/incoherent crossover near `1.8`.

## Predictions and bars

1. Every exact constant-rho identity has relative error below `1e-12`.
2. The structural stream-init variance proxy `D/(L A M)` is constant, so its
   theoretical log slope in `L` is zero to `1e-12`.
3. The double-depth control has variance slope `-2` in `L` to `1e-12`.
4. In the paired reduced-model Monte Carlo, the correct variance slope has
   absolute value at most `0.20` and the double-depth slope is within `0.20` of
   `-2.0`.
5. `alpha_* = D/(M E L) = 1/(rho E) = 1/128` at every rung. This is a fixed
   finite neural-SDE sector, not Jiang's universal `alpha_*=0` neural ODE.

Failure of the Monte Carlo bars blocks a compatibility claim even if the
algebra passes. Passing does not certify trained loss-curve transfer; that is a
separate fixed-normalized-eta experiment.
