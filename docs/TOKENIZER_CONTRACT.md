# Immutable tokenizer and token-stream contract

## Purpose

A token stream is reproducible only when both its bytes and the tokenizer that
created them are identifiable. A display name such as `OLMo tokenizer` or a
mutable repository branch is not sufficient. The Autoscaler therefore treats
the tokenizer, document-packing rule, and token shards as one verified dataset
identity.

The contract preserves the original deterministic `byte_v1` path and adds an
allow-listed production tokenizer. Raw `uint16_bin_v1` and `uint32_bin_v1`
files remain available for local compatibility, but they are explicitly
reported as unpinned and must not qualify a certified scaling-law result.

## Approved definitions

`GET /api/tokenizers` returns the backend registry. The initial remote preset
is:

- ID: `olmo2_1124`
- repository: `allenai/OLMo-2-1124-7B`
- immutable revision: `35a7ed2e8347efe11760bcaa1f758a3b2d978a90`
- vocabulary: 100,278 tokens
- implementation: `tokenizers==0.21.4`
- BOS, EOS, and UNK: `<|endoftext|>` at token ID 100,257
- PAD: `<|pad|>` at token ID 100,277
- document separator: `<|endoftext|>` at token ID 100,257

The registry contains the expected SHA-256 digest of `tokenizer.json`,
`tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, and
`merges.txt`. Resolution rejects a mutable revision, a missing file, any asset
digest mismatch, a vocabulary or special-token mismatch, a different runtime
version, or a failed fixed-string encoding canary.

The built-in `byte_v1` tokenizer has an equivalent generated manifest covering
its 260-token vocabulary, special-token IDs, byte encoding, and canaries. It
does not depend on remote assets.

## Tokenizer manifest

Every prepared corpus stores `tokenizer/manifest.json`. Its fingerprint covers:

- tokenizer ID and implementation;
- implementation package and exact version;
- repository and full immutable revision;
- allow-listed definition fingerprint;
- asset names, sizes, and SHA-256 digests;
- vocabulary size and every special-token ID;
- tokenizer normalizer, pre-tokenizer, post-processor, decoder, and model type;
- canonical test strings, token counts, and hashes of their token-ID sequences.

The manifest fingerprint is self-consistency protection. The in-code registry
is the trust anchor: loading a manifest also compares it with the allow-listed
definition, so rewriting a manifest and recomputing its fingerprint cannot
turn altered assets into an approved tokenizer.

## Packing and sharding

Pinned public text is encoded with the following
`document_eos_concatenation_v1` contract:

1. tokenize document text with `add_special_tokens=false`;
2. append exactly one declared document-separator token;
3. concatenate documents without padding;
4. permit cross-document attention, with the separator visible;
5. write little-endian unsigned 32-bit token IDs;
6. sample only complete windows contained in one shard.

Unsigned 32-bit storage is required because the approved OLMo-2 vocabulary is
larger than the 65,536 values representable by the legacy uint16 format.
Shards are written atomically and are approximately bounded by the configured
token limit; an individual document may make a shard exceed that limit rather
than being silently split.

After every completed shard, the materializer checkpoints the source JSONL
byte offset, source digest, tokenizer fingerprint, document count, token count,
and all completed shard hashes. Restart verifies that prefix and resumes at the
next document; an incomplete orphan shard is never admitted to the manifest.

`token-streams/manifest.json` records the packing contract, dtype, vocabulary,
tokenizer manifest reference, tokenizer fingerprint, per-split document and
token counts, and each shard's byte count, token bounds, and SHA-256 digest.
The content fingerprint covers ordered shard hashes and token counts. The
combined dataset fingerprint additionally covers the tokenizer and packing
contract.

## Identity propagation

The corpus materialization job ID includes the tokenizer definition
fingerprint. A registry change therefore cannot reuse a corpus prepared under
an older definition.

Before a training campaign receives a job ID, the backend verifies the
tokenizer manifest and every shard, then injects the resulting token-stream
identity into the job hash. Trial-cache identities use the same combined
dataset fingerprint. Results report all three distinct values:

- content fingerprint: the token shard content;
- tokenizer fingerprint: assets, implementation, special IDs, and canaries;
- dataset identity fingerprint: content plus tokenizer plus packing.

Model vocabulary must exactly equal the manifest vocabulary. The loader also
rejects an out-of-range token, an escaped manifest path, an altered shard,
inconsistent token counts, a changed packing separator, or disagreement
between tokenizer and stream manifests.

## Web workflow

For FineWeb-Edu or OpenWebText, select `OLMo 2 · immutable revision` before
preparing the snapshot. The preparation job downloads and verifies tokenizer
assets, materializes sharded streams, and returns the stream manifest path.
The campaign builder then uses that manifest instead of the raw JSONL paths and
sets the model vocabulary from the approved contract.

The interface displays the repository revision and definition fingerprint
before preparation. After preparation it displays content, tokenizer, and
combined identity fingerprints. A local pinned campaign accepts a verified
token-stream manifest path. Raw binary streams remain labeled as legacy and
unpinned.

The command-line smoke materialization is:

```bash
ai-theorist-autoscale corpus-materialize \
  configs/autoscaler/fineweb_edu_olmo2_cpu_smoke.json \
  --output-root runs/autoscaler/public-corpora \
  --progress-jsonl
```

## Validation obligations

Tests cover deterministic resolution, multi-shard generation, deterministic
window sampling, exact vocabulary enforcement, ambiguous-input refusal, shard
tampering, tokenizer-asset tampering, public-corpus integration, and verified
cache reuse. The real OLMo-2 assets and canary encodings must also be resolved
once on every new deployment image before expensive training is scheduled.
