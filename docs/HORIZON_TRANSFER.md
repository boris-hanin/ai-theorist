# Learning-rate schedule and token-horizon transfer

## Scope

The horizon-transfer campaign answers one question at a time: at fixed
architecture, frozen training sample, and batch size, does a peak-learning-rate
rule and normalized schedule shape transfer to a strictly held-out longer token
horizon? Two architecture contracts are implemented: νGPT's normalized
Transformer and the interleaved Jiang-MHSA + Chizat-FFN decoder.

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
start until snapshot preparation completes. The reference runners remain FP32
so that schedule/horizon results use the same architecture-specific
parameter-group implementations as the model-scale transfer checks.

## Schedules

The schedule is a normalized-time multiplier on the peak rates produced by the
architecture-specific parameter-group contract. It never replaces that
contract.

Implemented schedule families are:

- constant;
- cosine decay to a declared terminal fraction (the νGPT preset ends at 10%);
- linear warmup followed by linear decay;
- warmup-stable-decay with a cosine tail;
- linear warmup followed by a constant peak (the source-faithful Jiang preset
  uses a 50% warmup).

The artifact records the first, peak, final, mean, and integrated schedule
multiplier, plus the multiplier at every validation checkpoint.

## Candidate horizon rules

For source horizon `T0` and target horizon `T`, the one-point rules use

`eta(T) = eta(T0) * (T/T0)^(-beta)`.

The preregistered candidates are:

- `none`: `beta = 0`, a negative/control hypothesis;
- `nugpt_one_third`: `beta = 1/3`;
- `fitted_power`: fit both coefficient and exponent on fit horizons only.

`bjorck_032` remains accepted for reproducing older sensitivity runs, but is
not a default candidate: distinguishing `0.32` from `1/3` is false precision at
the seed counts and horizon spans used here. Reports group it with the
one-third theory family rather than treating it as a competing hypothesis.

For Jiang+Chizat, the CompleteP DMFT contract supplies the distinct model-scale
LR and epsilon formulas for embeddings, norms, attention QKV, attention output,
FFN up, FFN down, and other biases. At fixed `(L,M,D)` the horizon candidate
multiplies every already-parameterized group by the same `eta(T)` ratio. The
The `1/3` duration exponent is a cross-theory candidate, not a claim derived by
the Jiang CompleteP paper; the fitted rule and no-transfer control make that
uncertainty explicit.

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

The Jiang-MHSA + Chizat-FFN campaign reuses that verified corpus snapshot and
the group constants selected by the completed FineWeb reference calibration:

```bash
scripts/run_fineweb_jiang_chizat_horizon_a100.sh \
  runs/autoscaler/fineweb-jiang-chizat-horizon-a100 \
  runs/autoscaler/fineweb-horizon-a100/corpus
```

## 2026-08-10 A100 real-text horizon evidence

Both assays froze corpus fingerprint
`666710b377c444e7c0354dc3496d4375adcb482a5d65d086db86a8cfa61315e1`,
16,384 training windows, 1,024 validation windows, and seeds `11, 29, 47`.
Rules were fit on `65,536..524,288` presented tokens, frozen, and evaluated at
the hidden `1,048,576`-token horizon before revealing its LR grid oracle.

The 1.12M-parameter normalized Transformer completed 351 trials.  The
`T^(-1/3)` rule passed transfer and mechanism gates for all three schedules:
cosine loss `1.74412` with `0.400%` oracle regret; warmup/decay loss `1.73327`
with zero grid-oracle regret; and WSD loss `1.73637` with `0.202%` regret.  Flat
controls were `5.2%..9.5%` above their schedule oracles.  The old `0.32`
sensitivity point is retained only in that artifact; it is grouped with the
one-third family and no longer run by default.

The 306,688-parameter Jiang-MHSA + Chizat-FFN assay completed 114 trials while
preserving all seven CompleteP LR and epsilon groups.  It is a qualified
negative result for duration transfer.  The one-third rule produced loss
`2.13133`, or `2.927%` regret, and failed the `2%` gate.  A flat peak-LR rule
produced `2.07174`, only `0.049%` above the `2.07072` held-out oracle, so the
mechanism is non-identifiable and no horizon rule is certified.  The fitted
exponent was `-0.652` with `R^2=0.353`; the campaign correctly refused it
because fitted optimal LR increased with horizon.

## Joint composition

The next fixed-model stage is implemented in `JOINT_HORIZON_BATCH_TRANSFER.md`.
It fits horizon and batch axes separately, filters frozen rules at an unseen
fit-rectangle corner, and then evaluates a doubly held-out `(T, B)` corner.
Only after that qualification should a rule enter a constant-`T/N` model
ladder. AdamW/Power-Lines timescale transfer and SGD remain separate optimizer
contracts rather than being silently mixed into the νGPT/Adam campaign.
