# Joint horizon × batch validation record

Date: 2026-08-10

## Confirmation campaigns

Configurations:
`configs/autoscaler/joint_horizon_batch_cpu_smoke.json` and
`configs/autoscaler/joint_horizon_batch_cpu_wsd.json`.

Both confirmations used a fixed 5,872-parameter normalized Transformer, 2,048
unique tokens, three paired seeds, horizon-fit cells at
`T = 384, 768, 1536`, batch-fit cells at `B_examples = 2, 4, 8`, the unseen
composition corner `(1536, 8)`, and the doubly held-out corner `(3072, 16)`.
The linear warmup/decay campaign executed 177 unique neural-training records;
the WSD campaign executed 162. This geometry was selected after an earlier
110-run pilot exposed an over-broad software gate; neither confirmation corner
had been observed in that pilot.

The schedule-specific axis fits were:

| Schedule / axis | Exponent | Paired-bootstrap 95% interval | R2 | Gate |
| --- | ---: | ---: | ---: | --- |
| Linear, horizon | `beta = 0.610` | `[0.499, 0.698]` | 0.933 | pass |
| Linear, batch | `gamma = 0.123` | `[-0.026, 0.289]` | 0.469 | refuse |
| WSD, horizon | `beta = 0.469` | `[0.381, 0.569]` | 0.980 | pass |
| WSD, batch | `gamma = 0.521` | `[0.248, 1.086]` | 0.995 | pass |

At the composition cross-check, fitted-horizon + Adam-SDE had 1.15% regret for
linear decay and 1.96% for WSD. The one-third + Adam-SDE rule had 0% for both.
Every peak-only joint rule and both `q` rules missed the 2% cross-check gate.

At the doubly held-out corner, the fitted-horizon + full Adam-SDE transform
certified for both schedules:

| Schedule / rule | Held-out loss | Oracle regret | Mechanism |
| --- | ---: | ---: | --- |
| Linear, fitted horizon + Adam-SDE batch | 1.25736 | 0.00% | certified |
| Linear, one-third + Adam-SDE batch | 1.31608 | 4.67% | refused |
| Linear, best partial control | 1.40890 | 12.05% | control |
| WSD, fitted horizon + Adam-SDE batch | 1.24079 | 0.00% | certified |
| WSD, one-third + Adam-SDE batch | 1.25229 | 0.93% | certified |
| WSD, best partial control | 1.36887 | 10.32% | control |

The best partial control was materially worse than the scoring oracle in both
campaigns, so the joint mechanism was identifiable. Both winning transforms
predicted `beta1 = 0.2`, `beta2 = 0.6`, and `epsilon = 3.53553e-17`.

For linear decay, the fitted global peak coordinate was `0.0437684`; the νGPT
input, hidden, and rescaler groups used that value and the output group used
`0.0218842`. For WSD, those values were `0.0254303` and `0.0127152`.

This is evidence that the successful transfer came from the complete optimizer
transform. Applying only the fitted horizon LR, only the fitted batch LR, or
their peak-only product did not transfer. The linear result is also a direct
test of rule-specific gating: its empirical batch exponent was refused, but the
winning rule does not consume that exponent and was independently filtered by
the composition cross-check.

## Pilot and protocol correction

The first 110-run pilot used a different ladder and was withheld by the
then-current implementation because its empirical batch-power fit missed the global
`R2` gate. Review showed that the gate had been applied to every rule, including
fixed-theory rules that do not consume the fitted batch exponent. The protocol
was corrected to use rule-specific prerequisites before the confirmation
geometry was chosen. Acceptance thresholds were not changed.

## Interpretation

This settles the software path and establishes a clean positive example at
smoke scale. It is not a universal batch or horizon law: the model and corpus
are tiny and the synthetic corpus repeats.
These confirmations did not consume a qualified cross-estimator critical-batch
consensus, so their Adam-SDE-form winners are empirical certifications, not a
claim that the formal SDE regime was independently verified.
The A100 manifest raises the seed count to three, uses wider horizon coverage,
and preserves a separately held-out corner.
