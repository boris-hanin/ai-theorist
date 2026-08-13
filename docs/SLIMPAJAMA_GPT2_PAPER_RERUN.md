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

Both assays bind the same immutable artifacts and update budget:

- `cerebras/SlimPajama-627B`, direct train and validation shards from one full
  repository revision;
- the pinned `gpt2_openai` tokenizer at revision
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`;
- document text tokenized without automatic special tokens, followed by one
  GPT-2 end-of-text token;
- context length 2,048 and global batch 128;
- 1,144 optimizer updates, or exactly 299,892,736 presented tokens;
- AdamW beta1=0.9, beta2=0.95, epsilon0=1e-16, and zero weight decay;
- 10% linear warmup followed by linear decay to zero;
- identical fixed validation windows and seeds 11, 29, and 47.

SlimPajama is currently access-controlled by Hugging Face even though its
release is public. The controller never accepts a token in a JSON config or
command-line argument. The operator must accept the dataset terms and make a
credential available through `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or the
standard private Hugging Face token file. The controller waits without
allocating GPUs until that condition is satisfied.

## Acquisition and identity

The direct-shard materializer downloads deterministic compressed train and
validation paths at the resolved immutable commit. Downloads are resumable,
and every consumed compressed shard is recorded with repository path, byte
count, and SHA-256 digest. Decompression writes canonical JSONL with source
split, shard, and record provenance. Train and validation come from the
dataset's distinct released splits.

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

The Jiang reference LR grid and the three fixed CompleteP anchor replicates
share one dynamic eight-GPU pool. CompleteP is restricted to the published
base learning rate `2^-8`; it is a calibration anchor, not an adaptive tuning
study. After an interior Jiang reference optimum is selected, every remaining
Jiang rung uses the frozen scalar coordinate through the complete per-group
rules. The largest Jiang rung remains hidden from fitting.

Final `result.json` reports the Jiang scaling fit and hidden-rung error, the
three CompleteP anchor losses and their mean, every gate, and a precise claim
scope. It must not label Jiang's raw loss a numerical reproduction of
CompleteP because the architectures intentionally differ.
