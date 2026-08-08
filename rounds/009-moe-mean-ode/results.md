# Round 009 — residual MoE in the Neural Mean ODE scaling (NEW RESULT)

Derivation/prereg: `derivations/09-moe-mean-ode.md`, committed **before** any
measurement (`feac9b6`). Model: `skills/dmft-moe/scripts/moe_ode.py`.

Not a replication: merges the MoE parameterisation of **2601.20205** with the
joint `(L, M, D)` Mean-ODE analysis of **2509.10167** / **2603.18168**, which do
not treat sparse routing.

## The claim

Units are indexed by the triple `(i,j,l) in [E] x [M] x [L]`, but only **active**
experts contribute to a token's forward pass, so

    effective width  W_eff = L * a * M          (a = active experts, NOT E)
    residual scale   = sqrt(D)/(L M)
    rate             = O( 1/L + sqrt(D/(L a M)) + 1/sqrt(D) )

## Verdicts

| # | prediction | measured | verdict |
|---|---|---|---|
| N1 | init deviation slope `-1/2` in `L` | **-0.505** | **PASS** |
| N1 | slope `-1/2` in `a` | **-0.489** | **PASS** |
| N1 | slope `-1/2` in `M` | **-0.512** | **PASS** |
| N1/N3 | slope `0` in `E` at fixed `a` | **+0.030** | **PASS** |
| N1c | fan-in control: `M` drops out | **-0.013** (vs `-0.512` MLU) | **PASS** |
| **N2** | **invariant to the split at fixed `L a M`** | spread **1.10x** over 10 splits | **PASS** |
| N2c | fan-in not invariant | spread **3.90x** | **PASS** |
| N4 | optimal LR transfers | `L` 0.11, `a` 0.06, `M` 0.14, `D` 0.22, `E` 0.04 | **PASS** (asymptotically) |
| N5 | Euler term `1/L`, `a`/`M`-independent | — | **not tested** |

## N2 — the test that mattered

`09` §7 named the risky step in advance: §2 treats the `a` active experts as
adding **incoherently** at initialisation, but they share the router-conditioning
event `q >= q*(kappa)`, which correlates them. If that correlation were
`Theta(1)`, the `a`-averaging would be coherent and `W_eff = L M`, not `L a M`.

At fixed `L a M = 256`, across ten splits spanning `a` = 2→16 and `L` = 2→64:

| (L, a, M) | MLU | fan-in |
|---|---|---|
| (4, 2, 32) | 2.17e-2 | 3.07e-2 |
| (8, 2, 16) | 2.26e-2 | 2.26e-2 |
| (16, 4, 4) | 2.31e-2 | 1.16e-2 |
| (64, 2, 2) | 2.22e-2 | 7.87e-3 |
| (2, 16, 8) | 2.28e-2 | 1.61e-2 |
| … | | |
| **spread** | **1.10x** | **3.90x** |

**The `a`-averaging is incoherent** and `W_eff = L a M` stands. Depth, active
experts, and expert width are mutually interchangeable to 10%.

The `(8,2,16)` row has `M = D = 16`, where the two initialisations are literally
the same model — the identity control landing inside the sweep, and agreeing to
every printed digit.

## N4 — HP transfer, and a missing factor it caught

First pass: `D` drifted **0.65 decades** and pinned at the grid edge, while
`L`, `a`, `M` drifted 0.22–0.25. The `D` failure was a real gap: `09` states the
LR scaling for `L`, `M`, `a` but **never states its `D`-dependence**, and the
simulator used `lr = L M a eta`. Counting properly: the loss is per-coordinate
normalised so `b ~ 1/D`, hence `grad_w ~ 1/(L M a D)` and

    **lr = L * M * a * D * eta**

With that, `D` drift → **0.22, interior**. This is the second time in this
session a preregistered dial has caught a scaling factor the derivation left
implicit (cf. `05` §1's unstated `d_L` domain, F19).

The residual `L`/`a` drift is **finite-size**, not a broken rule — it shrinks
monotonically as the smallest configs are dropped, which is what an asymptotic
transfer claim predicts:

| dial | drift over all 5 | over the largest 3 |
|---|---|---|
| depth `L` (4→64) | 0.26 | **0.11** |
| active `a` (2→32) | 0.25 | **0.06** |

**`E` at fixed `a` is the cleanest row in the table: 0.04 decades.** That is the
new prediction — the expert count is absent from the rate — and it transfers
better than any other dial.

## The fan-in control does not bite for HP transfer, and that is expected

Fan-in drifts 0.27 / 0.31 / 0.13 / 0.21 / 0.02 — indistinguishable from MLU.
This is **not** a failed control: round 007 established that the down-projection
init cannot move the update scale (`Delta E = M * (base/M) * Theta(1)`, free of
`sigma_down`), and the optimal LR is set by the update. So HP transfer is
**not a discriminating test of the down-projection init**; N1/N2 are, and there
the control bites hard (`-0.013` vs `-0.512`; 3.90x vs 1.10x).

Stated explicitly because a non-biting control is normally a red flag (F17), and
here it is a derived consequence instead.

## What this buys — two budgets

    parameters  P = Theta(L E M D)      FLOPs  C = Theta(L a M D)

The rate depends on the FLOP budget. Minimising subject to `L a M D = C` with
`W = aM`:

    **L = C^{1/6},   a M = C^{1/2},   D = C^{1/3},   error = O(C^{-1/6})**

with the split of `aM` between expert count and expert width **free**. So at
fixed FLOPs a model can buy capacity by raising `E` at fixed `a` — costing
parameters but neither FLOPs nor convergence rate. That is a scaling-theoretic
statement of why MoE works, and it locates the benefit of 2601.20205's Finding
3.1 (more, smaller experts) in the **limit object**, not in the approximation
rate — the rate cannot see `E` at all, which N1/N3/N4 now confirm three separate
ways (`+0.030`, and 0.04 decades of LR drift).

## Not done

- **N5** (Euler term `1/L`, independent of `a` and `M`) — not measured.
- The DMFT of `09` §5–§7 is derived but has **no solver**, so the response-sector
  claim (`O(1/(LaM))`, hence a memoryless limit) is untested. Same status as
  MoE in round 006: parameterisation and limit structure validated, dynamics not.
- The `C^{1/6}` shape inherits round 008's caveat: the `1/sqrt(D)` term it
  depends on is only suggestively measured.
