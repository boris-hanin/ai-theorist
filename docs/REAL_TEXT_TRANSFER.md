# Frozen public-text transfer campaigns

The Autoscaler can materialize an allow-listed public corpus directly from the
web campaign workspace.  The default is the `sample-10BT` configuration of
FineWeb-Edu.  OpenWebText is also available.  Arbitrary URLs are deliberately
not accepted by the compute service.

## Reproducibility contract

Corpus preparation records the dataset repository, configuration, split,
source revision, license, exact source-row ranges, content hashes, byte-token
counts, and a token-stream fingerprint.  Training and validation are sampled
from widely separated, non-overlapping source-row ranges and written to
separate JSONL files.  A completed snapshot is re-used only after both files
pass their recorded SHA-256 checks.

Transient dataset-server failures use bounded exponential backoff.  Each
completed 100-document page is fsynced with a checkpoint tied to the source
revision and materialization contract, so an interrupted download resumes from
the last verified source row.

The public data is currently tokenized with the deterministic `byte_v1`
tokenizer.  This makes the first real-data campaign self-contained and removes
tokenizer-version drift, but it is not yet the final modern tokenizer baseline.
The next tokenizer milestone should add a pinned OLMo 2 or GPT-2 tokenizer and
store its model/configuration alongside the token stream.

The command-line equivalent of the web action is:

```bash
ai-theorist-autoscale corpus-materialize \
  configs/autoscaler/fineweb_edu_a100.json \
  --output-root runs/autoscaler/public-corpora \
  --progress-jsonl
```

## Jiang-family experiments

Both executable transfer harnesses accept the frozen training and validation
paths through `--train-path` and `--validation-path`.  `--vocab-size 260` must
be used with `--tokenizer byte_v1`.  Each harness loads and freezes one paired
set of token windows before any model, learning-rate, seed, or negative-control
trial.  The same corpus fingerprint and sampled windows are therefore used
throughout the factorial campaign.

This data path applies to:

- the dense interleaved Jiang-attention + Chizat mean-field FFN experiment;
- the full sparse Jiang MoE experiment;
- fixed-model normalized-Transformer schedule and token-horizon transfer.

The horizon campaign freezes the same corpus and sampled-window contract, tunes
each schedule independently on at least three shorter horizons, freezes the
candidate `eta(T)` rules, tests them at the hidden longest horizon, and reveals
that horizon's LR oracle only for regret scoring. Its A100 manifest is
`configs/autoscaler/fineweb_horizon_transfer_a100.json`, and the resumable
launcher is `scripts/run_fineweb_horizon_transfer_a100.sh`.

It does **not** replace the exact Chizat equation-(22) nonlinear-regression
experiment.  That experiment remains a source-faithful synthetic control; text
would define a different model and loss.

## Interpretation boundary

Real web text fixes the most serious data-realism weakness, but a 64 MiB slice,
byte tokenizer, context 64, and a few hundred optimizer steps are still a
pilot-scale transfer assay.  A positive result establishes that the per-group
parameterization survives the move from a Markov generator to natural text at
the tested scales.  It is not yet evidence for frontier-scale loss prediction.

## 2026-08-10 A100 evidence

The A100 campaign froze FineWeb-Edu sample-10BT revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` into 67,156,577 training and
8,395,537 held-out byte tokens.  Source rows `0..14291` and
`5000000..5001948` are disjoint.  The token-stream fingerprint is
`666710b377c444e7c0354dc3496d4375adcb482a5d65d086db86a8cfa61315e1`.

Both campaigns used four joint scale points, seven normalized learning rates,
three paired seeds, 300 steps, batch size 16, and the same pre-sampled token
windows for every trial and control.

- Dense Jiang attention + Chizat mean-field FFN: all four scale-wise grid
  optima were `eta=0.03`; the base-selected fixed rate was exactly at each
  grid oracle, and the progress-vs-scale log slope was `0.02338`.  All seven
  group-only feature-velocity checks passed.  None of four negative controls
  was rejected.
- Jiang sparse MoE: all four scale-wise grid optima were again `eta=0.03`;
  the fixed rate was exactly at each grid oracle, and the progress-vs-scale log
  slope was `0.02184`.  None of three negative controls was rejected.  Maximum
  routing-load deviation reached `0.3333` at the largest shape.

The exact conclusion is therefore **HP transfer certified; mechanism
discrimination not certified** for both architectures.  The public web
evidence view preserves that distinction rather than turning a clean optimum
match into a stronger causal claim.
