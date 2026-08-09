# Autoscaler study builder

The study builder is the supported path from a plumbing check to a forecast-grade
scaling experiment. It makes the data distribution, training budget, model
ladder, and held-out test explicit before compute starts. Study JSON uses
`schema_version: 2`.

## 1. Run profiles

- **Smoke** is a small CPU plumbing check. The backend always refuses to issue a
  forecast from a smoke run, even if a curve happens to fit.
- **Pilot** widens the parameter range, data set, and update budget enough to
  diagnose learning-rate transfer and whether the loss has useful dynamic range.
- **A100 study** selects the larger CUDA preset. It still has to pass every
  transfer, readiness, negative-control, normalization/routing, and held-out
  calibration gate before a forecast appears.
- **Custom** exposes the current values without claiming a preset contract. Any
  manual edit to a profile-controlled value changes the UI label to Custom.

Profiles are presets, not evidence. The result records `run_profile`, and all
effective per-scale values are stored in the compiled plan and result.

## 2. Dataset and difficulty controls

The UI supports two deterministic task families:

- **Teacher regression** exposes input dimension, teacher width, teacher depth,
  label noise, training examples, and validation examples.
- **Markov language** exposes Markov order, state count, vocabulary, context
  length, random-token probability, training sequences, and validation
  sequences.

Easy, Scaling, and Stress are coherent task-complexity presets. Direct edits are
marked Custom. Difficulty increases learnable structure; it does not fake a
scaling curve by adding noise. Noise is controlled separately because it raises
the irreducible loss floor.

For a fixed dataset seed, a larger training set contains the smaller training set
as an exact prefix. Validation is generated from a separate stream and remains
bit-identical when training size changes. Regression target normalization uses a
third fixed calibration stream, so changing `n_train` cannot leak into validation
labels.

## 3. Automatic model ladders

The ladder builder generates 5–10 strictly increasing levels from:

- starting width and depth;
- geometric width and depth growth;
- width-only, depth-only, or joint scaling;
- fixed head dimension for normalized Transformers; and
- constant `L M / D` construction for sparse MoE studies.

Transformer widths are rounded to valid head-dimension multiples. The largest
level is visibly marked as the holdout and is excluded from the scaling-law fit.
Editing an individual row creates a manual override; the generated ladder can be
restored with one action.

## 4. Batch, microbatch, and horizon budgets

Batch size, microbatch size, update steps, and the derived sample/token budget
are first-class fields. Microbatching performs gradient accumulation and must
evenly divide the logical batch size. For a language study,

`token_horizon = steps × batch_size × context_length`.

For regression, the same field counts sample presentations with a context factor
of one. The UI lets the user edit either steps or the derived budget and previews
the effective budget at every scale. Automatic batch-size scaling is deliberately
not inferred yet; the declared logical batch remains fixed unless a later study
contract adds that policy.

## 5. Pilot readiness and recommendations

After training, the backend evaluates whether the observations support a useful
power law. It reports:

- parameter and compute span;
- loss dynamic range relative to seed uncertainty;
- the fraction of adjacent scale transitions with lower loss;
- concrete reasons for refusing a forecast;
- recommended changes to task complexity, budget, or ladder span; and
- a suggested next model shape and training-set size.

Readiness is a hard forecast gate. It is checked in addition to an interior
learning-rate optimum, largest-scale transfer, scaling-fit usability, held-out
prediction accuracy, negative-control rejection, and architecture-specific
routing or normalization invariants.

## 6. Joint model–data scaling

Fixed-data mode preserves the earlier one-axis experiment. Geometric mode grows
training examples at level `i` as

`N_i = round(N_0 × growth_factor^i)`.

It offers two explicit horizon policies:

- **Fixed updates** keeps optimizer steps constant, so only model and data size
  change.
- **Match data growth** scales steps in proportion to `N_i`, preserving the base
  number of epochs at fixed batch size.

The immutable validation stream never grows. Each trial receives a materialized
per-scale specification, and compute estimates use those same effective values.
The compiled plan identifies whether the declared scaling axes are model only,
model plus data, or model plus data plus training horizon.

## Operational workflow

1. Run Smoke to verify the complete path.
2. Select Pilot, then choose the task difficulty and model path you actually care
   about.
3. Inspect the generated parameter and per-level data/budget preview before
   launching.
4. Use the readiness report to change one identified deficiency at a time.
5. Move to the A100 profile only after transfer and dynamic range are credible.
6. Treat a forecast as valid only when the app displays it; a completed run with
   refusal reasons is a successful diagnostic, not a forecast.
