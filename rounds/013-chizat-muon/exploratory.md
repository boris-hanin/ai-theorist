# Round 013 exploratory Chizat-Muon transfer screen

## Status

**Exploratory pass with two A100 replications, not certification.**  The CPU
screen was run before the round-013 preregistration was committed and therefore
cannot satisfy the program's provenance rule.  The subsequent A100 runs are
strong replication evidence, but they do not repair that provenance gap.

Artifact: `rounds/013-chizat-muon/artifacts/exploratory-cpu-h80.json`.

## Design

- Shapes `(L,M,D)`: `(2,64,8)`, `(4,128,16)`, `(8,256,32)`,
  `(12,512,48)`, `(16,1024,64)`.
- Reference shape: `(8,256,32)`.
- Eta grid: `0.03, 0.05, 0.08, 0.10, 0.14, 0.20, 0.30`.
- Five common seeds, 80 full-batch updates, `P=64`, `d0=8`.
- RMS-matched Muon for U/W; auxiliary Adam for trained embed/unembed.
- No weight decay.

## Result

The reference numerical optimum was interior at `eta=0.10`; the one-SEM rule
also selected `0.10`.  No tuning row diverged.

At fixed eta, mean final losses along the joint path were:

| Shape | Mean final loss |
|---|---:|
| J1 | 0.0011133 |
| J2 | 0.0004965 |
| J3 | 0.0002102 |
| J4 | 0.0002970 |
| J5 | 0.0002330 |

Every post-step-1 trajectory checkpoint passed the settling diagnostic.  The
final log learning-progress slope against the joint dial was `+0.00070`, well
inside the preregistered absolute `0.3` bar.

All three required strong controls were rejected:

| Control | Outcome |
|---|---|
| W rate `D eta` instead of `sqrt(D) eta` | rejected by trajectory gate |
| reuse plain-GD L/M/D multipliers | rejected by trajectory and progress gates |
| constant unembed rate instead of `eta/D` | rejected by trajectory gate |

The weaker constant-W control passed the asymptotic settling/progress gates but
had much worse final loss, increasing from `0.00395` at J1 to `0.01780` at J5.
It is retained as a negative result rather than promoted to a required control.

Freezing either boundary individually did not destroy transfer.  The ablations
did change losses, but the effect was not monotone: freezing the embed hurt J1
and improved several larger shapes, while freezing the unembed was generally
neutral-to-worse.  Thus this screen validates the declared rates as a coherent
end-to-end path; it does not show that both boundary updates are individually
necessary at this horizon.

## Important numerical diagnostic

The actual first-step Muon update RMS fell with scale even under the nominal
RMS-matched adjustment.  Relative to the full-rank nominal coordinate, U fell
from `0.876` at J1 to `0.312` at J5 and W from `0.523` to `0.100`.  The finite
dataset makes the matrix gradients rank-limited and poorly conditioned, so the
full-rank `0.2 * lr` argument is not quantitatively realized here.  Function-
space progress nevertheless transferred.  The A100 confirmation retained this
diagnostic; entrywise RMS invariance is not established.

## Independent A100 replications

The unchanged 80-step primary path and repaired required-control battery were
run independently on two NVIDIA A100-SXM4-80GB workers after each worker had
been idle for a 60-second safety window.  Both used the same five seeds and eta
grid as the CPU screen.  The local copies are:

- `rounds/013-chizat-muon/artifacts/a100-h80-worker1.json`
- `rounds/013-chizat-muon/artifacts/a100-h80-worker2.json`

Both workers selected `eta=0.10`, placed the numerical optimum at the same
interior grid point, passed every primary transfer gate, and rejected all three
required controls.  The two structured result payloads are identical after
removing host and timestamp metadata.  The A100 final progress slope was
`+0.000704` on both workers.

At the selected eta, the A100 shape-wise mean final losses were
`0.00111325, 0.000496532, 0.000210167, 0.000294893, 0.000233247`.  Relative to
the CPU values, the maximum absolute difference was `2.08e-6` and the maximum
relative difference was `0.70%`.  The CPU and A100 tuning curves also agreed
closely through the selected basin (`eta <= 0.10`).  Device sensitivity was
larger at the deliberately aggressive `eta=0.20` and `0.30` probes, but it did
not change either the selected eta or any acceptance decision.

On A100, `wrong_W_D` was rejected by the paired largest-shape loss gate,
`wrong_constant_unembed` by both trajectory and paired-loss gates, and
`wrong_sgd_LMD` by trajectory, progress, and paired-loss gates.  The update
audit reproduced the CPU warning: trained embed and unembed updates matched
their declared coordinates numerically, while the rank-limited U/W Muon update
RMS decreased with scale.  The evidence therefore supports function-space HP
transfer, not invariant entrywise Muon update RMS.

Guard logs are retained as `a100-h80-worker{1,2}.log` alongside the JSON
artifacts.  Worker 1 ran from `18:55:50Z` to `18:58:26Z`; worker 2 ran from
`18:55:57Z` to `18:58:33Z` on 2026-08-09.

## 160-step follow-up and battery repair

An initial 160-step follow-up used the truncated grid `0.08,0.10,0.14`; its
numerical optimum landed at the lower edge, so it has no valid eta-selection
verdict.  The primary fixed-eta trajectory and learning-progress gates still
passed.  The constant-unembed control reached a largest-shape mean loss of
`0.001039` versus `0.000110` for the primary rule, but the settling-only
control logic did not reject it.  This exposed a battery blind spot: settling
can show convergence without showing convergence to comparable performance.

Before A100 execution, the control battery was amended to add a common-seed
paired largest-shape final-loss test with tolerance
`max(2 SEM, 1% of primary loss)`.  The expanded 160-step eta bracket must be
rerun; the truncated result is retained at
`rounds/013-chizat-muon/artifacts/exploratory-cpu-h160.json` and is not a pass.

The repaired rerun is retained at
`rounds/013-chizat-muon/artifacts/exploratory-cpu-h160-v2.json`.  On the expanded grid, the
numerical optimum was interior at `eta=0.06`; the conservative one-SEM rule
selected `eta=0.03`.  Fixed-eta trajectory and progress gates passed, with a
final progress slope of `+0.00034`.  Mean primary losses from J1 through J5
were `0.000488, 0.000168, 0.0000753, 0.0000466, 0.0000508`.  All three required
controls were rejected by the repaired battery.  Thus the exploratory transfer
result reproduces at both 80 and 160 updates, with the expected horizon-
dependent eta selection.

## Pure-axis confound checks

The joint path could hide compensating L/M/D exponents, so the same 80-step
battery was repeated with two axes fixed.  All three independent ladders
passed fixed-eta trajectory, progress, interior-reference-optimum, and required
negative-control gates:

| Axis | Selected eta | Numerical eta | Final progress slope | Primary mean losses |
|---|---:|---:|---:|---|
| L: `2,4,8,16` | 0.10 | 0.10 | `+0.000028` | `0.000165, 0.000117, 0.000123, 0.000136` |
| M: `64..1024` | 0.14 | 0.14 | `+0.000138` | `0.000303, 0.000240, 0.000116, 0.000172, 0.000127` |
| D: `8..64` | 0.10 | 0.10 | `+0.000572` | `0.000895, 0.000497, 0.000223, 0.000207, 0.000198` |

Artifacts are `exploratory-cpu-pure-{L,M,D}-h80.json` under
`rounds/013-chizat-muon/artifacts/`.  These results rule out a simple cancellation that occurs
only when all three dimensions grow together.
