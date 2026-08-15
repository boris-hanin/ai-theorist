# Joint token-horizon and batch transfer

## Question and scope

This campaign asks whether a schedule and a complete Adam optimizer transform
can be frozen from smaller token horizons and batches, then work at a larger
`(T, B)` corner. Model size `N`, the unique token stream `U`, architecture, and
schedule family stay fixed. This isolates horizon and batch transfer before
they are composed with model scaling or constant `T/N` ladders.

The campaign currently supports the normalized Transformer with Adam. AdamW,
SGD, and other architectures remain separate contracts.

## Geometry and execution order

The fit data are an L-shaped set of cells:

- tune peak LR over at least three `T` values at the smallest fit batch;
- tune peak LR over at least three `B` values at the smallest fit horizon;
- fit `eta*(T) proportional to T^(-beta)` and
  `eta*(B) proportional to B^gamma` independently;
- freeze every candidate rule;
- test frozen rules at the previously unseen opposite corner of the fit
  rectangle;
- reveal a full LR grid there and reject rules above the cross-check regret
  threshold;
- evaluate the surviving frozen rules at a larger horizon and batch;
- only then reveal the final LR oracle for scoring.

The cross-check prevents a separable law from reaching the final holdout merely
because both one-dimensional fits look plausible.

## Frozen candidates

Three partial controls deliberately omit at least one effect:

- no optimizer change;
- fitted horizon peak-LR factor only;
- fitted batch peak-LR factor only.

The joint candidates are:

- the product of the independently fitted peak-LR factors, with Adam moments
  fixed;
- the fitted horizon factor composed with the Adam-SDE batch transform;
- `T^(-1/3)` composed with the Adam-SDE batch transform;
- the Complete(d)P `q = m_B / m_T` transform;
- the exact-token-half-life variant of the same `q` transform.

The Adam-SDE composition applies all declared coordinates:

`eta' = eta * m_T^(-beta) * sqrt(m_B)`

`epsilon' = epsilon / sqrt(m_B)`

`1 - beta_i' = m_B * (1 - beta_i)`.

An invalid transformed beta is refused, never clipped. The model then expands
the transformed global peak coordinate into its architecture-specific νGPT
input, hidden, output, and rescaler parameter groups. Every result records the
group names, peak rates, formulas, and epsilon values.

## Rule-specific gates

All candidates require the declared seed count, fit spans, and an interior
source optimum. A candidate that uses a fitted horizon exponent additionally
requires interior horizon-axis optima, bootstrap uncertainty, a nonnegative
exponent, and the horizon `R2` gate. The analogous conditions apply only to
candidates that use the fitted batch exponent. Fixed-theory rules are not
incorrectly blocked by a fit they do not consume.

A joint rule is transfer-certified only if it passes its prerequisite gates,
the composition cross-check, an interior final LR oracle, and the default 2%
held-out regret threshold. Mechanism discrimination is stricter: the joint
effect must be distinguishable from the best of all three partial controls,
and the rule must recover at least 90% of the available joint improvement.

Empirical certification and theory-regime certification are separate. The two
Adam-SDE-form candidates are marked theory-qualified only when the campaign is
given a previously qualified critical-batch value and the target stays below
the configured fraction of it. A successful held-out prediction without that
input is labeled empirical rather than evidence that the SDE assumptions hold.

## Running it

CPU plumbing and empirical smoke campaign:

```bash
ai-theorist-autoscale joint-transfer \
  configs/autoscaler/joint_horizon_batch_cpu_smoke.json \
  --output runs/autoscaler/joint-horizon-batch-cpu/result.json
```

A100 protocol:

```bash
ai-theorist-autoscale joint-transfer \
  configs/autoscaler/joint_horizon_batch_a100.json \
  --device cuda \
  --output runs/autoscaler/joint-horizon-batch-a100/result.json
```

The same campaign is available through the persistent web job system as
`Joint horizon × batch holdout`. Completed neural cells are cached atomically.

## Limits

Passing this campaign settles only the fixed-model, fixed-corpus schedule and
optimizer transform on the tested regime. It does not yet prove transfer when
`N`, `U`, architecture, optimizer family, or data distribution changes. The
next composition stage inserts a qualified joint rule into a constant-TPP
model ladder and gives that three-coordinate prediction another holdout.
