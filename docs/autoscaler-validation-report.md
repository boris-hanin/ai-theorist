# Autoscaler MVP validation report

Date: 2026-08-09

## Decision after the normalized-eta correction

The reduced-scope product and corrected learning-rate contract are implemented.
The local regression suite passes, but the previous A100 transfer verdicts used
proximity to a width-wise local LR optimum.  That quantity mixes transfer with
the moving edge of discrete-time stability, so those verdicts are retained
below as historical evidence and are **not** a certification of the corrected
contract.  The checked-in A100 campaigns must be rerun before release.

This is not a claim that every optimizer/architecture/horizon combination has
a usable scaling law.  A forecast is a measured privilege: the product emits a
number only after transfer, a negative control, scaling-law diagnostics, and a
completely held-out largest model all pass.  Several A100 experiments below
were correctly refused and were used to repair or narrow the method instead of
relaxing its gates.

## Validated product boundary

- Graph: `Embed -> repeat(pre-norm residual MLP) -> Unembed`.
- Blocks: GELU or ReLU residual MLP.
- Optimizers: actual PyTorch SGD and Adam.
- Scaling axes: width and repeat count only.
- Constant within a study: dataset, examples, batch size, and update horizon.
- Tuned value: one normalized learning-rate coordinate `eta`.
- Transfer: the current residual MLP converts to `lr_raw = eta` for Adam and
  `lr_raw = eta / sqrt(D)` for SGD.  Chizat mean-field SGD uses
  `lr_raw = L M eta / alpha^2`.
- Verdict: fixed-`eta` non-inferiority/trajectory convergence.  The local
  optimum and largest stable probe are separate diagnostics and cannot fail
  transfer merely because a larger model has more stability headroom.
- Target: final validation loss at the declared horizon.
- Deferred: attention, arbitrary graphs, AdamW, data scaling, and horizon
  scaling.

## Parallel Chizat-Muon product preparation

Muon is now integrated locally for the bias-free Chizat architecture only.
The product trains embed and unembed in every trial, routes U/W through Muon
and the boundary maps through auxiliary Adam, records all four raw rates, and
persists exact optimizer state for continuation.  Three versioned deterministic
task families and a strict nine-cell optimizer-by-dataset matrix are present.

This is implementation readiness, not a new release claim.  Round 013 provides
strong exploratory CPU and duplicated-A100 transfer evidence, but its
preregistration was not committed before execution.  The round-016 matrix
protocol must be committed before its fresh A100 cells begin.

## Automated and interactive validation

| Layer | Evidence | Result |
| --- | --- | --- |
| Python unit/e2e | 80 passed and 1 socket-restricted test skipped locally; includes Muon numerics/routing, trained boundaries, three datasets, matrix expansion, normalized/raw conversion, fixed-`eta` settling, replay, checkpoints, scaling, and persistence | Pass |
| New Chizat LR contract | CPU smoke plus two-worker A100 campaigns through 1280 steps and M=2048; fixed-eta transfer passed and omitted-M control was rejected | Pass for Chizat subclaim |
| A100 code regression | The prior isolated suite passed 26/26 on both workers, before the transfer-verdict correction | Historical; rerun required |
| Real local API | 2/2 socket tests: health, strict compilation, disallowed-origin rejection, asynchronous run, polling, persisted result | Pass |
| Packaging | Wheel build, clean wheel install, installed CLI help, sample-spec generation, and plan compilation | Pass |
| Web static checks | ESLint, TypeScript, production Vinext build | Pass |
| Rendered web tests | Product shell and fixed-horizon contract | Pass |
| Desktop browser | Component click/drag, fixed-node inspection, invalid-plan blocking, immutable running plan, live result/refusal | Pass |
| Mobile browser | 390 px viewport with no horizontal overflow | Pass |
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

## Hardware reproducibility

Both independent A100-SXM4-80GB workers reported PyTorch 2.6.0 with CUDA 12.4.
The same deterministic CUDA canary produced final loss
`0.5159386396408081` on each worker and on repeat execution.  The CPU value was
`0.5158814191818237`, a relative CPU/GPU drift of approximately 0.011%.  Peak
allocated GPU memory for the canary was 18,381,824 bytes.

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
- Raw A100 campaign directories remain runtime evidence; the checked-in
  configurations and this acceptance record are the durable reproducibility
  surface.

## Release conclusion

The implementation is ready for corrected A100 revalidation, not for a new
release claim.  No prior local-optimum result may be relabeled as fixed-`eta`
transfer.  A release requires common-seed fixed-`eta` trajectories, the
separate stability-edge report, optimizer-specific negative controls, and
held-out validation-loss calibration under the new result schema.  No horizon
result may be silently pooled into another horizon's law.
