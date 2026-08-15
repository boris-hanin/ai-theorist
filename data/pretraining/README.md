# Pretraining data fixtures

`sample_train.txt` and `sample_validation.txt` are original, repository-owned
natural-language fixtures for deterministic local integration tests. They
exercise the real tokenization, next-token training, validation separation,
content fingerprinting, caching, and web-job path. Their size is intentionally
too small for a scientific scaling claim.

For a research campaign, replace both paths with a versioned text/JSONL corpus
or a little-endian `uint16` token stream prepared with a production tokenizer.
Do not mix tokenizer versions, concatenate validation into training, or reuse a
corpus fingerprint after changing document filtering or split construction.
See `docs/PRETRAINING_RUNTIME.md` for the complete contract.
