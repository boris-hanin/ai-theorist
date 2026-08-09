# Autoscaler MVP validation report

Date: 2026-08-09

## Decision

The reduced-scope product, corrected normalized-learning-rate contract, joint
`L/M/D` studies, and sparse-MoE Adam path are implemented.  The sparse-MoE path
passes its two-worker A100 campaign, including fixed-eta transfer, routing
health, a wrong-global-rate control, and held-out validation-loss prediction.
The earlier standard-MLP A100 results remain historical because their original
verdict used proximity to a width-wise local LR optimum; they have not been
silently promoted under the corrected fixed-eta contract.

This is not a claim that every optimizer/architecture/horizon combination has
a usable scaling law.  A forecast is a measured privilege: the product emits a
number only after transfer, a negative control, scaling-law diagnostics, and a
completely held-out largest model all pass.  Several A100 experiments below
were correctly refused and were used to repair or narrow the method instead of
relaxing its gates.

## Validated product boundary

- Graph: `Embed -> repeat(pre-norm residual {MLP or top-k MoE}) -> Unembed`.
- Blocks: GELU or ReLU residual MLP; fixed-sparsity top-k MoE.
- Optimizers: actual PyTorch SGD and Adam for MLP; Adam for MoE.
- Scaling axes: MLP width/depth; independent MoE `L`, `M`, and `D`.
- Constant within a study: dataset, examples, batch size, and update horizon.
- Tuned value: one normalized learning-rate coordinate `eta`.
- Transfer: the current residual MLP converts to `lr_raw = eta` for Adam and
  `lr_raw = eta / sqrt(D)` for SGD.  Chizat mean-field SGD uses
  `lr_raw = L M eta / alpha^2`.  MoE Adam uses the Table-1 group rates
  `eta`, `eta/D`, and `eta/M` according to parameter role.
- Verdict: fixed-`eta` non-inferiority/trajectory convergence.  The local
  optimum and largest stable probe are separate diagnostics and cannot fail
  transfer merely because a larger model has more stability headroom.
- Target: final validation loss at the declared horizon.
- Shape policy: the default MoE ladder grows all three axes with `LM/D=4`.
- Prepared extension: bias-free Chizat particles with trained embed/unembed,
  semantic SGD/Adam rates, and Muon on U/W with auxiliary Adam on boundaries.
- Deferred: attention, arbitrary graphs, AdamW, Muon outside Chizat, data
  scaling, and horizon scaling.

## Round 016 Chizat optimizer-by-dataset result

The product now supports a strict Chizat-only nine-cell matrix: SGD, Adam, and
Muon crossed with linear, tanh-teacher, and sinusoid-plus-quadratic tasks.
Every trial trains embed and unembed and records the raw rates for embed, U, W,
and unembed.  Muon routing is semantic rather than rank-based, so trained
boundary matrices remain on auxiliary Adam.  Exact optimizer continuation,
dataset fingerprints, fixed-eta trajectories, and optimizer-specific negative
controls are part of the persisted result.

Round 013 remains strong exploratory evidence because its protocol was not
committed before its A100 runs.  Round 016 repairs provenance: its protocol,
implementation, and immutable manifest were committed before execution.

The replicated nine-cell A100 matrix completes 665 finite 600-step trials per
worker.  Final-loss non-inferiority passes in all nine cells.  Adam passes the
complete transfer package on all three tasks; Muon passes it on linear and
sinusoid-plus-quadratic, while Muon/tanh fails trajectory settlement.  SGD
does not pass a complete package because two controls do not bite and two
tasks select a boundary eta.  No cell passes every loss-law and held-out gate,
so the broad claim fails and the product issues no forecast.  See
`rounds/016-chizat-optimizer-dataset/results.md`.

## Automated and interactive validation

| Layer | Evidence | Result |
| --- | --- | --- |
| Python unit/e2e | 80 passed and 1 socket-restricted test skipped locally; includes MLP/MoE/Chizat schema, Muon numerics and routing, trained boundaries, three datasets, matrix expansion, fixed-`eta` gates, replay, checkpoints, scaling, and persistence | Pass |
| New Chizat LR contract | CPU smoke plus two-worker A100 campaigns through 1280 steps and M=2048; fixed-eta transfer passed and omitted-M control was rejected | Pass for Chizat subclaim |
| Joint `L/M/D` | Pure-axis and joint two-worker A100 campaigns; exact duplicates, wrong-rule controls, and a dedicated `LM/D=8` ladder | Pass; constant `LM/D` preferred |
| Sparse-MoE A100 | Six `LM/D=4` scales, six seeds, 84 trials, two independent A100 workers, held-out scale, wrong-global-rate control, and routing gate | Pass |
| Round-016 matrix | Nine optimizer/task cells, 665 trials per worker, 600 steps per trial, two independent A100 replicas | Broad claim fails; transfer subclaims retained |
| Real local API | 2/2 socket tests: health, strict compilation, disallowed-origin rejection, asynchronous run, polling, persisted result | Pass |
| Packaging | Wheel build, clean wheel install, installed CLI help, sample-spec generation, and plan compilation | Pass |
| Web static checks | ESLint, TypeScript, production Vinext build | Pass |
| Rendered web tests | Product shell and fixed-horizon contract | Pass |
| Desktop browser | MLP/MoE selection, fixed-node inspection, invalid-plan blocking, optimizer restriction, `LM/D` warning, immutable running plan, live result/refusal | Pass |
| Mobile browser | Narrow viewport with the MoE `D/L/M` editor visible and no horizontal overflow | Pass |
| Browser diagnostics | No warning- or error-level console entries | Pass |

The browser run deliberately used a tiny local study.  It refused a 14.2%
held-out miss and a negative control that could not be distinguished.  That is
the intended product behavior, not a UI success mask.

## Chizat fixed-eta method validation

Round 012 directly validates the corrected distinction on two A100s.  For
`M=64..2048`, `L=8`, `D=32`, `P=64`, and fixed normalized
`eta=79.43282347242814`, the learning-progress slope versus width was
`+2.30e-6` at 640 steps and `+1.08e-7` at 1280 steps.  The omitted-M mutation
gave `-0.387` and `-0.369` and was rejected.  All 108 duplicated full-grid
trials matched exactly across workers; the 640-step adjacent absolute-loss gap
settled from `5.67e-6` at the small-width end to `1.04e-9` at the large-width
end.  See `rounds/012-chizat-width-transfer/`.

This validates the result schema and verdict mechanism for the Chizat
parameterization.  It does not replace rerunning the separate standard-MLP
Adam/SGD product campaigns under schema v2.

## Joint `L`, `M`, and `D` transfer

Round 014 tests each pure axis and two simultaneous joint ladders with the
coherent Chizat group rates

```text
lr_U = L M eta / D
lr_W = L M D eta
```

Pure `L`, pure `M`, and coupled pure `D` pass at 80 steps.  A general joint
ladder also passes at 80 and 320 steps.  The user-proposed invariant path with
`LM/D=8` is the cleanest: its log-progress slope is `+0.00473`, versus
`-0.0503` on the changing-ratio joint ladder.  Removing `L`, removing `M`, and
using the incoherent square-root-`D` surrogate are rejected.  Every merged
campaign contains 20 or 30 exact duplicate trials across the two workers.

This is why the product's MoE defaults keep `LM/D` constant.  It is a preferred
shape path, not a substitute for the optimizer's separate group-rate rules.

## Hardware reproducibility

Both independent A100-SXM4-80GB workers reported PyTorch 2.6.0 with CUDA 12.4.
The same deterministic CUDA canary produced final loss
`0.5159386396408081` on each worker and on repeat execution.  The CPU value was
`0.5158814191818237`, a relative CPU/GPU drift of approximately 0.011%.  Peak
allocated GPU memory for the canary was 18,381,824 bytes.

## Sparse-MoE Adam calibration

The release-evidence campaign uses six scales
`(D,L,M) = (8,2,16), (18,3,24), (32,4,32), (50,5,40), (72,6,48),
(98,7,56)`.  Every scale has `LM/D=4`.  It holds 1,024 training examples,
256 validation examples, batch size 64, and 320 updates fixed while tuning one
normalized eta over six common seeds.

The final run selected `eta=0.1`, an interior and non-flat optimum.  Fixed eta
passes at the held-out S6 scale.  The deliberately wrong single global raw
rate, matched to the reference router/up rate, is rejected by a paired loss
increase.  The fit-only law has `R^2` above 0.98, and the completely held-out
largest scale is predicted within 6% of its observed final validation loss.
The full numerical values and the final adjacent-scale forecast are recorded
in `rounds/015-autoscaler-moe/results.md` and its canonical result artifact.

Routing is part of the verdict.  The accepted controller rate is 0.1.  Rates
0.01 and 0.03 allowed isolated collapse; 0.3 and 1.0 produced oscillation and
failed the routing gate.  The final gate requires, at every scale, mean
worst-expert deviation across seeds at most 0.25 and an individual-run hard
limit of 0.50.  These failed settings remain in the round record rather than
being discarded after the successful choice.

## Historical Adam calibration

The six-level Adam study used widths 96/144/216/324/486/729, repeats
2/3/4/6/8/12, 16,384 fixed training examples, 4,096 validation examples, 600
fixed updates, batch size 256, four common seeds, and 400 bootstrap fits.

- Selected reference LR: `0.001`, interior to the tested range.
- Largest-scale local probe: `0.001`; this is now an edge diagnostic, not the
  transfer gate.
- Wrong square-root-growth control: rejected.
- Fit-only scaling-law `R^2`: 0.987844.
- Held-out S6 prediction: 0.0059029.
- Held-out S6 observation: 0.0056535.
- Relative point error: 4.41%; accepted.
- Outcome: one adjacent-scale forecast issued.
- Limitation: the asymptotic floor was not identifiable, so the system did not
  issue an asymptotic scaling claim.
- Next-step prediction: 0.0056554 with bootstrap interval
  [0.0044735, 0.0061820] at 3.366 times the largest observed compute.

## Historical SGD calibration and falsification trail

The initial SGD ladder reused Adam's depth growth and constant LR.  It became
optimization-limited at larger levels; tripling the common horizon did not fix
that raw-LR rule.  A shallower ladder improved its fit, but the largest-scale
local probe significantly preferred a lower LR.  This showed that constant raw
LR was inferior in that experiment and motivated the inverse-square-root raw
conversion.  Under the corrected protocol, transfer itself must be judged at
fixed normalized `eta`; local-argmin movement is reported separately.

With that transfer rule at 600 steps on the shallower ladder:

- Predicted largest-scale LR: 0.0054433.
- Local largest-scale optimum: 0.0054433; accepted exactly.
- Constant-LR negative control: rejected with paired loss increase 0.0096260
  (SEM 0.0024917).
- Held-out point error: 3.08%.
- Outcome: forecast refused because `R^2 = 0.791628` and the observation missed
  the bootstrap interval.

A compact-ladder control also failed: its smallest model was already at the
fixed-horizon loss floor and loss did not decrease with compute.  The system
again refused the forecast.  These runs establish that optimizer transfer can
be valid while a capacity scaling law is not; the implementation keeps those
claims separate.

Two final 1,800-step campaigns tested whether a longer shared horizon repaired
the scaling signal.  It did not.  The compact ladder still had 19.0% held-out
error.  On the original ladder, the 600-step inverse-width rule no longer
transferred: its S6 LR was 0.01633, while the local probe preferred 0.03258 and
constant LR was better.  The held-out error was 61.3%.  This is important
negative evidence: the transfer parameterization is itself horizon-specific,
so the product must recalibrate it rather than extrapolate a 600-step rule into
a different training regime.

## Operational safety and provenance

- Runtime results are strict JSON with an immutable input manifest and study
  fingerprint.
- Invalid or out-of-scope specs fail before work starts.
- Plans are locked while running, and the API records failures for inspection.
- Local write endpoints reject untrusted browser origins.
- The SSH private key and worker addresses are not stored in the repository or
  result manifests.
- Raw scratch campaign directories remain runtime evidence.  Rounds 014 and
  015 deliberately promote merged or canonical JSON artifacts alongside the
  checked-in configurations and acceptance decisions.

## Release conclusion

The sparse-MoE MVP is ready for the next fixed-data, fixed-horizon research
phase in the tested regime.  Its A100 evidence validates the implementation,
the `LM/D` default path, Table-1 Adam group transfer, router-load refusal, and
one adjacent-scale validation-loss forecast.  It does not certify arbitrary
expert counts, sparsity fractions, tasks, horizons, optimizers, or full MoE
DMFT.  The historical MLP forecasts likewise remain historical until their
schema-v2 fixed-eta campaigns are rerun.  No horizon result may be silently
pooled into another horizon's law.

The Chizat optimizer-by-dataset extension is certified only for the retained
cell-level transfer subclaims above.  It is not certified for automatic loss
forecasting: zero of nine cells passes every committed transfer, control,
trajectory, scaling-law, and held-out gate.
