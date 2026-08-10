# Learning-rate schedule and token-horizon transfer

## Scope

The first horizon-transfer campaign answers one question at a time: at fixed
normalized-Transformer architecture, unique training stream, and batch size,
does a peak-learning-rate rule and normalized schedule shape transfer to a
strictly held-out longer token horizon?

The campaign does **not** yet claim that the same rule transfers when model
size or batch size also changes. Those are later composition tests.

## Coordinates

Every run records five distinct quantities:

- `N`: trainable parameters;
- `U`: tokens in the frozen sampled training windows;
- `T`: presented training tokens, including repetition;
- `B`: tokens in one optimizer update;
- `S = T / B`: optimizer updates.

Tokens per parameter is `T / N`. Repetition is `T / U`. Neither is called
"dataset size" in the result artifact.

For real text, the result also records the full materialized corpus token
counts. Those corpus counts are provenance, not the denominator in `T/U`: the
training distribution for one campaign is the fixed set of sampled windows,
so `U = n_train_windows * context_length`.

## Frozen real-text mode

`dataset.task_type = tokenized_text` selects the production horizon path. The
runner loads the materialized training and validation streams once, samples
both window tensors once with a declared seed, and reuses those exact tensors
for every rate, seed, schedule, fit horizon, frozen-rule trial, and held-out
oracle trial. The content fingerprint and sampling contract are included in
every trial's cache identity and result metadata. A cache from another corpus
therefore cannot satisfy the run.

The web builder exposes the same FineWeb-Edu, OpenWebText, and local-file
snapshot controls as the real-text batch census. A non-local campaign cannot
start until snapshot preparation completes. The reference runner remains FP32
so that schedule/horizon results use the same normalized-Transformer
parameter-group implementation as the model-scale transfer checks.

## Schedules

The schedule is a normalized-time multiplier on the peak rates produced by the
architecture-specific parameter-group contract. It never replaces that
contract.

Implemented schedule families are:

- constant;
- cosine decay to a declared terminal fraction (the νGPT preset ends at 10%);
- linear warmup followed by linear decay;
- warmup-stable-decay with a cosine tail.

The artifact records the first, peak, final, mean, and integrated schedule
multiplier, plus the multiplier at every validation checkpoint.

## Candidate horizon rules

For source horizon `T0` and target horizon `T`, the one-point rules use

`eta(T) = eta(T0) * (T/T0)^(-beta)`.

The preregistered candidates are:

- `none`: `beta = 0`, a negative/control hypothesis;
- `nugpt_one_third`: `beta = 1/3`;
- `bjorck_032`: `beta = 0.32`;
- `fitted_power`: fit both coefficient and exponent on fit horizons only.

For every schedule family, at least three fit horizons are tuned independently.
The largest horizon is hidden while rules are fitted and frozen. Frozen rules
run first; only then is a full learning-rate grid run at the held-out horizon to
measure oracle regret.

## Qualification

By default, a campaign needs three seeds, an 8x fit-horizon span, interior
fit-scale and held-out optima, and bootstrap uncertainty. A frozen rule is
transfer-certified when its held-out loss is within 2% of the held-out oracle.

Mechanism discrimination is separate. When the no-transfer control is already
within 0.2% of the oracle, the artifact reports that the transfer mechanism is
not identifiable on the observed learning-rate plateau. Otherwise a rule must
recover at least 90% of the improvement from no-transfer to oracle to certify
mechanism discrimination.

## Product behavior

The web campaign builder exposes the corpus, horizon ladder, peak-LR grid,
fixed batch, and schedule families. The result view displays the corpus
fingerprint, `N/U/T/B/S`, fitted exponents, held-out oracle regret, and transfer
and refusal verdicts. Campaign jobs are persisted and trial caches are
resumable.

The local implementation smoke test is:

```bash
ai-theorist-autoscale horizon-transfer \
  configs/autoscaler/fineweb_horizon_transfer_cpu_smoke.json \
  --device cpu \
  --output runs/autoscaler/fineweb-horizon-cpu-smoke.json \
  --progress-jsonl
```

The complete A100 recipe first materializes a pinned FineWeb-Edu snapshot and
then resumes the preregistered trial cache:

```bash
scripts/run_fineweb_horizon_transfer_a100.sh \
  runs/autoscaler/fineweb-horizon-a100
```

## Joint composition

The next fixed-model stage is implemented in `JOINT_HORIZON_BATCH_TRANSFER.md`.
It fits horizon and batch axes separately, filters frozen rules at an unseen
fit-rectangle corner, and then evaluates a doubly held-out `(T, B)` corner.
Only after that qualification should a rule enter a constant-`T/N` model
ladder. AdamW/Power-Lines timescale transfer and SGD remain separate optimizer
contracts rather than being silently mixed into the νGPT/Adam campaign.
