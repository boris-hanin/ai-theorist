# Autoscaler MVP validation report

Date: 2026-08-09

## Decision

The reduced-scope product is implemented and validated for the declared typed
architecture, deterministic fixed-horizon training, Adam and SGD execution,
learning-rate tuning and transfer, held-out calibration, explicit forecast
refusal, and the local visual workbench.

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
- Tuned value: one global reference learning rate.
- Transfer: constant LR for Adam; `eta(D) = eta_ref sqrt(D_ref / D)` for SGD.
- Target: final validation loss at the declared horizon.
- Deferred: attention, arbitrary graphs, AdamW, Muon, data scaling, and horizon
  scaling.

## Automated and interactive validation

| Layer | Evidence | Result |
| --- | --- | --- |
| Python unit/e2e | 36 repository tests when socket access is enabled; schema, compiler, parameter counts, optimizer parity, deterministic replay, checkpoints, accumulation, tuning, transfer, scaling, persistence | Pass |
| A100 code regression | The isolated 26-test autoscaler suite on each independent worker | 26/26 on both |
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

## Hardware reproducibility

Both independent A100-SXM4-80GB workers reported PyTorch 2.6.0 with CUDA 12.4.
The same deterministic CUDA canary produced final loss
`0.5159386396408081` on each worker and on repeat execution.  The CPU value was
`0.5158814191818237`, a relative CPU/GPU drift of approximately 0.011%.  Peak
allocated GPU memory for the canary was 18,381,824 bytes.

## Adam calibration

The six-level Adam study used widths 96/144/216/324/486/729, repeats
2/3/4/6/8/12, 16,384 fixed training examples, 4,096 validation examples, 600
fixed updates, batch size 256, four common seeds, and 400 bootstrap fits.

- Selected reference LR: `0.001`, interior to the tested range.
- Largest-scale local probe: `0.001`; transferred LR accepted exactly.
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

## SGD calibration and falsification trail

The initial SGD ladder reused Adam's depth growth and constant LR.  It became
optimization-limited at larger levels; tripling the common horizon did not fix
that transfer rule.  A shallower ladder improved its fit, but the largest-scale
local probe significantly preferred a lower LR.  This falsified constant-LR
SGD transfer and motivated the inverse-square-root width parameterization.

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

The MVP is suitable for its next research phase: repeated A100 studies over
the supported architecture slice.  Adam has earned a calibrated adjacent-scale
forecast in the tested regime.  SGD training and optimizer-specific transfer
are validated, while the tested 600-step capacity regimes correctly remain
non-forecastable.  The tested 1,800-step regimes also remain non-forecastable
and show that SGD transfer must be retuned by horizon.  No horizon result may be
silently pooled into the 600-step law.
