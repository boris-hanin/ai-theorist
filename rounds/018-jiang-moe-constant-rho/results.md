# Round 018 — results

Preregistration commit: `4ac7d43`.  Runner:
`skills/dmft-moe/scripts/constant_rho_compatibility.py`.

## Verdict

**PASS for parameterisation and structural-limit compatibility.** Constant
`rho=L M/D` is compatible with the Jiang MoE rules when `sqrt(D)/(L M)` is the
effective residualized expert-down scale. It is not an additional raw-matrix
factor.

This is not a trained loss-curve transfer verdict and not a solved MoE DMFT.

## Exact algebra

| check | measured | required | verdict |
|---|---:|---:|---|
| constant-rho relative error | `0` | `<1e-12` | PASS |
| constant-kappa relative error | `0` | `<1e-12` | PASS |
| residualized down-init identity error | `0` | `<1e-12` | PASS |
| residualized down-LR identity error | `0` | `<1e-12` | PASS |
| correct stream-variance slope in `L` | `0` | `0 +/- 1e-12` | PASS |
| double-depth variance slope in `L` | `-2.0000000000000004` | `-2 +/- 1e-12` | PASS |
| `alpha_*` spread | `0` | `0` | PASS |

Every rung has `rho=32`, `kappa=1/4`, and `alpha_*=1/128`. Thus it remains in
one finite neural-SDE sector. It is not the universal `alpha_*=0` neural ODE.

## Paired reduced-model measurement, 64 seeds

The preregistration did not state which of Jiang's two routing-initialization
conventions to use. The first measurement used Appendix E (zero router, random
biases). Rather than choose after seeing it, the final record includes both
that convention and the main-text convention (random `D^-1` router, zero
biases), and requires both to pass the same bars.

| routing convention | correct slope | double-depth slope | bars | verdict |
|---|---:|---:|---|---|
| main text | `-0.08358` | `-2.08437` | `abs(correct)<=0.20`, `abs(control+2)<=0.20` | PASS |
| Appendix E | `-0.00550` | `-2.00551` | same | PASS |

Main-text means (variance of `h^L-h^0` per stream coordinate):

| `L` | correct mean | double-depth mean |
|---:|---:|---:|
| 2 | `3.967e-3` | `9.933e-4` |
| 4 | `3.727e-3` | `2.329e-4` |
| 8 | `3.509e-3` | `5.455e-5` |
| 16 | `3.337e-3` | `1.305e-5` |

The correct model is approximately flat across an 8x depth range. The control
falls by about 75x and lands on the predicted extra `L^-2` law. This directly
rules out applying `sqrt(D)/(L M)` to the raw matrix and then applying the
architectural `1/L` again.

## Remaining scientific gate

The next experiment must test trained fixed-normalized-`eta` trajectories and
report the local stability edge separately. The finite `rho=32` ladder should
stop at `L=16` for the primary claim: `alpha_ffn=rho/L` is already `2` there,
near the measured crossover at `1.8`. `L>=32` is an explicit crossover probe,
not theory-certified transfer evidence.
