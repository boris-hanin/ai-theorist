# SlimPajama/GPT-2 paper-coordinate rerun

This campaign answers two separate questions without conflating them:

1. How does the rho=32 Jiang-Chizat ladder behave on the corpus, tokenizer,
   context, batch, optimizer core, and schedule used by the CompleteP paper?
2. Does this runtime obtain a calibrated raw loss for one literal CompleteP
   paper model before we interpret the Jiang raw losses?

The first is a matched training-coordinate comparison, not a CompleteP
architecture reproduction. Jiang deliberately retains tied token/readout
weights, learned absolute positions, GELU, `QK^T/d_head`, the mean-field FFN,
`1/L` residual branches, and all eight declared LR/epsilon parameter groups.
The second is an exact `N=256, L=2` CompleteP anchor with untied embeddings,
ALiBi, ReLU-squared, `QK^T/N`, `d_head=64`, `d_ff=4N`, and CompleteP's six
AdamW groups.

## Frozen paper coordinates

The original `cerebras/SlimPajama-627B` repository disappeared from Hugging
Face in August 2026. Both assays therefore bind the same surviving sample and
update budget:

- `DKYoon/SlimPajama-6B`, a 10% sample of the already-shuffled original
  SlimPajama `train/chunk1`, at source revision
  `b5f90f419b7489cdba26fdbc8c022fcb5562f968`;
- its released validation split, distinct from training, with the generated
  Parquet repository pinned to
  `c4f51dc260275e8e01aa0fbf46c64832dbee5369`;
- the pinned `gpt2_openai` tokenizer at revision
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`;
- document text tokenized without automatic special tokens, followed by one
  GPT-2 end-of-text token;
- context length 2,048 and global batch 128;
- 1,144 optimizer updates, or exactly 299,892,736 presented tokens;
- AdamW beta1=0.9, beta2=0.95, epsilon0=1e-16, and zero weight decay;
- 10% linear warmup followed by linear decay to zero;
- identical fixed validation windows and seed 11.

This surviving sample is public and does not require a Hugging Face token. Its
5.49 million training documents cover roughly 6B tokens, so the 300M-token
assay remains a low-repetition slice of SlimPajama. It is the closest available
public replacement, but it cannot recover the paper's unpublished example
order or make the run a byte-identical dataset reproduction.

## Acquisition and identity

The Parquet materializer resolves separate train and validation inventories,
rewrites every mutable conversion URL to the pinned conversion commit, and
downloads resumably. Every consumed file is recorded with its immutable URL,
byte count, and SHA-256 digest. Canonical JSONL retains source split and row
provenance. Train and validation come from the dataset's distinct released
splits.

Raw JSONL, tokenizer assets, and uint32 token shards are all verified before a
plan can compile. The preregistration records the source revision, source
inventory fingerprints, tokenizer fingerprint, combined token-stream
fingerprint, and every config/qualification digest. A 300M-token training run
is refused unless the unique train stream is at least as long as the presented
token budget.

## Execution

The queued controller is `scripts/run_slimpajama_gpt2_paper_rerun.sh`. It first
waits for the active 300M horizon run and all GPU processes to exit. It then
materializes the corpus, binds both plans, runs full parameter-group and CUDA
canaries, and writes the preregistration before any outcome is observed.

The 23-point, single-seed Jiang reference LR grid and one fixed CompleteP anchor
run share one dynamic eight-GPU pool (24 tasks, or three full pool waves).
CompleteP is restricted to the published base learning rate `2^-8`; it is a
calibration anchor, not an adaptive tuning study. After an interior Jiang
reference optimum is selected, every remaining Jiang rung uses the frozen
scalar coordinate through the complete per-group rules. The largest Jiang rung
remains hidden from fitting.

Final `result.json` reports the Jiang scaling fit and hidden-rung error, the
CompleteP anchor loss, every gate, and a precise claim scope. It must not label
the campaign an exact raw-loss reproduction: the accessible corpus is a
preserved sample, the paper's example order was not published, and the Jiang
architecture intentionally differs from CompleteP.
