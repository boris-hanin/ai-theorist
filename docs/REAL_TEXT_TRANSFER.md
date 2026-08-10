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
- the full sparse Jiang MoE experiment.

It does **not** replace the exact Chizat equation-(22) nonlinear-regression
experiment.  That experiment remains a source-faithful synthetic control; text
would define a different model and loss.

## Interpretation boundary

Real web text fixes the most serious data-realism weakness, but a 64 MiB slice,
byte tokenizer, context 128, and a few hundred optimizer steps are still a
pilot-scale transfer assay.  A positive result establishes that the per-group
parameterization survives the move from a Markov generator to natural text at
the tested scales.  It is not yet evidence for frontier-scale loss prediction.
