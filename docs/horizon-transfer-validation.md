# Horizon-transfer validation record

Date: 2026-08-10

## Automated validation

- Python: 164 passed, 1 skipped because this sandbox forbids local socket binding.
- Web: TypeScript and ESLint passed.
- Production web build passed.
- Rendered product-shell tests: 7 passed.
- The horizon campaign also ran through the persistent campaign-job engine in
  the Python integration tests, producing a manifest, trial cache, and result.

## Empirical CPU campaign

Configuration:
`configs/autoscaler/horizon_transfer_cpu_smoke.json`.

This was a real 172-trial normalized-Transformer training campaign on the
deterministic synthetic Markov task. It used one fixed 5,872-parameter model,
2,048 unique tokens, a fixed 16-token batch, presented-token horizons from 256
to 2,048, two seeds, three schedule families, and automatic learning-rate-grid
expansion. The 2,048-token horizon was not used while fitting or freezing rules.

| Schedule | Fitted exponent | Bootstrap 95% interval | Fit R2 | Held-out result |
| --- | ---: | ---: | ---: | --- |
| Cosine to 10% | 0.643 | [0.629, 0.666] | 0.999 | No candidate met the 2% oracle-regret gate |
| 10% warmup, linear decay | 0.465 | [0.389, 0.466] | 0.949 | Fitted rule certified |
| Warmup-stable-decay | 0.535 | [0.485, 0.589] | 0.985 | Fitted and one-third rules certified |

The lowest-loss certified rule was linear warmup/decay with the fitted
`T^(-0.465)` peak-learning-rate factor. Its predicted held-out peak LR was
0.0202312 and its mean held-out validation loss was 1.30851. For WSD, the
preregistered `T^(-1/3)` rule remained certified with 0.77% measured-oracle
regret and passed mechanism discrimination; the unchanged-LR WSD control was
13.15% worse. This difference between schedule families is the main smoke-scale
finding.

The earlier five-rate screen was refused because short-horizon optima landed on
the upper grid boundary. The final result was obtained by expanding and
bracketing that grid, not by weakening the acceptance gate.

## Interpretation

This validates campaign mechanics and demonstrates that schedule family can
materially change horizon transfer. It does not establish a universal horizon
exponent: the corpus is synthetic, the model is tiny, there are only two seeds,
and the fit span is 4x. The checked-in A100 manifest expands to three seeds, an
8x fit span, a completely held-out 16x horizon, seven starting LR probes, and
bounded automatic grid expansion.
