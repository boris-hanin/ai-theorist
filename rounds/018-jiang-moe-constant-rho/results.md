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

## Independent compute-host reproduction

The exact committed runner at `034b27b` was deployed to the 8x H100 host using
Python 3.10 and PyTorch 2.6.0+cu124. It reproduced slopes `-0.0835834380` and
`-2.0843739833` for the main-text convention, and `-0.0055024478` and
`-2.0055062915` for Appendix E. Both passed; the differences from the local
numbers are numerical roundoff. No training campaign was started.

## Trained fixed-eta transfer result

The next gate was run on SlimPajama with the pinned GPT-2 tokenizer. A first
eight-point reference grid stopped correctly because its optimum was at the
lower boundary. A preregistered adaptive lower bracket then selected the
interior normalized learning rate `eta=0.00390625`. That value was frozen and
applied without nonreference retuning to all Table 2 parameter groups.

| `L` | `D` | `M` | active non-embedding params | mean loss | mean fractional progress |
|---:|---:|---:|---:|---:|---:|
| 2 | 128 | 2048 | 1,187,336 | 6.21075 | 0.42668 |
| 4 | 256 | 2048 | 5,264,912 | 6.19194 | 0.42663 |
| 6 | 384 | 2048 | 13,019,160 | 6.19812 | 0.42554 |
| 8 | 512 | 2048 | 25,236,512 | 6.18824 | 0.42616 |
| 12 | 768 | 2048 | 66,206,256 | 6.19403 | 0.42546 |
| 16 | 1024 | 2048 | 134,465,600 | 6.18153 | 0.42632 |

Across a `113.25x` active-nonembedding parameter span, the log-progress slope
is `-0.0003816` against a preregistered absolute limit of `0.30`. Every
trajectory was finite, all eight optimizer groups were complete and disjoint,
routing was nondegenerate, and all three seeds were present at every scale.

The endpoint wrong-global-LR control reached mean loss `6.73747`, versus
`6.18153` for the theory-scaled endpoint: an `8.99%` degradation against the
preregistered minimum of `0.5%`. All gates passed. The compact immutable record
is `transfer-result-summary.json`; its hashes bind the locally preserved raw
preregistration, selection, runtime qualification, and full transfer result.

This establishes short-horizon fixed-normalized-eta transfer, not token-horizon
transfer or a loss scaling law. The finite `rho=32` primary family still stops
at `L=16`, where `alpha_ffn=2` is near the measured crossover at `1.8`.
`L>=32` remains an explicit crossover probe, not theory-certified transfer
evidence.
