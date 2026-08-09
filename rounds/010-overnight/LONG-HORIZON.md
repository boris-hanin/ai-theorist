# Long-horizon transfer — the failure is NOT the MoE extension

Runs stopped by request; partial but the decisive comparison completed.

## Drift grows monotonically with training horizon (T1, box 1)

| dial | h=32 | h=128 | h=512 |
|---|---|---|---|
| depth `L` | 0.046 | 0.239 | **0.708** |
| active experts `a` (`kappa` fixed) | 0.037 | 0.282 | — |
| expert width `M` | 0.058 | 0.168 | — |
| embedding `D` | 0.024 | 0.122 | — |

Clean and systematic, not noise. **Every transfer claim this program has made was
measured at 8–24 steps**, where drift is 0.008–0.041. At 128 steps it is already
0.12–0.28; at 512 the depth dial is 0.708. The earlier passes were real but
**horizon-limited, and were reported without that qualifier.**

## The decisive control: dense Chizat fails too

Same code path, same horizon, same grid and seeds; only the expert machinery
changes. `E=1, a=1` makes routing inert, collapsing the block to Chizat's dense
2LP.

| arm | h=1024 drift |
|---|---|
| **DENSE `E=1, a=1` (Chizat)** | **0.425** |
| MoE `E=16, a=4` (from box 2) | 0.490 |

**The dense arm fails at essentially the same magnitude.** So the long-horizon
breakdown is **not** caused by the MoE extension — sparsity, routing and the
load-balancing rule are exonerated. The fault is upstream, in the Chizat-side
prescription as implemented here, or in the setup.

## What this does and does not license

**Does not** license "the Chizat + MoE prescription is wrong". Two reasons:

1. **No learning-rate decay.** Every run here holds `eta` fixed for up to 2048
   steps. Both source papers state transfer at a fixed token budget *with a
   schedule*. Once a model reaches interpolation the optimum can move for reasons
   that have nothing to do with the parameterisation. This is the single most
   likely confound and it is **untested**.
2. **The h=2048 pattern is not monotone** (depth worse at 1024 than 2048,
   embedding the reverse, 16 seeds), which is not what a clean breakdown looks
   like.

**Does** license: the transfer claims in this repository are established **only
for short horizons**, and the horizon dependence is real, systematic, and
architecture-independent within this setup.

## Next, in order

1. Repeat the dense-vs-MoE control **with a decaying `eta`**. If drift collapses,
   the failure is the fixed-rate setup and the prescriptions are intact.
2. Finish the h=1024 sparse arm (`E=4, a=1`) — only 2 of 5 points ran.
3. T2 zero-shot regret at long horizon, which converts drift in decades into the
   loss penalty a practitioner would actually pay. Never reached.
