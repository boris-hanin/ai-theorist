# Round 008 — the rates of 2509.10167 and 2603.18168

Derivation: `derivations/08-chizat-mlu.md` §1, §4, §5.
Model: `skills/dmft-resnet-depth/scripts/mean_ode.py` (`MeanODENet`) — a faithful
implementation of their §2.1: explicit `alpha`, plain GD with LR `LM eta/alpha^2`,
**no LayerNorm** (their model has none).

## The instrument that made this sharp

Their bound is `O(1/L + sqrt(D/(LM)))` and their own Figure 2 fits it as
`a/L + b/sqrt(ML)` with two hand-adjusted constants. That conflates the two
terms. But the two have **different statistical characters**:

- the **Euler** term is a *deterministic* discretisation bias — identical for
  every random seed,
- the **CLT** term is a *fluctuation* — independent across seeds.

So seed-**differencing** isolates the second and seed-**averaging** isolates the
first, and each exponent can be measured alone with no free constants.

## Verdicts

| # | quantity | measured | predicted | verdict |
|---|---|---|---|---|
| A1 | CLT term, slope in `L` | **-0.513** | `-1/2` | **PASS** |
| A1 | CLT term, slope in `M` | **-0.500** | `-1/2` | **PASS** |
| A2 | Euler term, slope in `L` | **-0.953** | `-1` | **PASS** |
| A2 | Euler term, slope in `M` | **+0.012** | **`0`** | **PASS** |
| B | lazy `alpha* ~ (ML)^{1/4}` | — | `+0.25` | **INCONCLUSIVE** |
| C | third term, slope in `D` | **-0.645** | `-1/2` | **suggestive, confounded** |

**A2's `M`-independence is the sharpest single number here.** The discretisation
error of the Mean ODE is a property of the layer grid and must not care about
block width at all; measured `+0.012` across `M` = 512 → 4096.

## A2 needed two instrument fixes, both of which were my error

1. First pass gave slope `-0.502` in `L` *and* `-0.539` in `M`, against
   predictions `-1` and `0`. Both readings were ~`1/sqrt(12)` of the A1
   magnitudes — I was measuring the **residual CLT noise in the seed-mean**, not
   the Euler bias, which sat below it. Fix: subtract the noise in quadrature
   using the A1-measured spread, `Euler^2 = bias^2 - spread^2/S`.
2. That still failed (raw bias `0.90x`–`1.26x` of the floor — no signal). The
   Euler *constant* is ~6x smaller than the CLT constant in this setting, so the
   term is only visible where CLT is suppressed. Fix: large `M` (2048–4096) and
   small `L` (2–8). Then the bias runs `2.4x`–`4.1x` above its floor and both
   exponents come out.

Neither failure was the theory's; both were reading a quantity whose noise I had
not accounted for. Recorded because "the slope is wrong" and "the slope is the
noise floor's slope" look identical in a log-log fit.

## B — inconclusive, and why

Theorem 2 bounds the distance to the **Neural Tangent ODE**, the `L, M -> inf`
limit of the *linearised* dynamics, as `O(1/alpha + 1/L + alpha/sqrt(ML))`,
minimised at `alpha* = (ML)^{1/4}`.

**First instrument (wrong by construction).** I compared each ResNet to *its own*
finite-size linearisation. Those share an initialisation, so the CLT fluctuation
**cancels exactly** and only the `1/alpha` term survives. The measurement
confirmed that: `alpha*` pinned at the grid edge for every config and the minimum
gap was ML-independent (`+0.038`). That is a clean confirmation of the `1/alpha`
term *in isolation*, and no test at all of the trade-off.

**Second instrument (better, still not good enough).** Comparing to a large
linearised reference (`L=64, M=512`) gave a real interior minimum at the smallest
config — `alpha* = 5.28` against a predicted `3.36` at `ML = 128` — but the other
three configs pinned at the grid edge again, so their `alpha*` is unmeasured and
the fitted slopes (`+0.540`, `-0.437`) are contaminated by three edge points and
mean nothing.

The reference also carries its **own** `alpha`-dependent error
(`1/L_ref + alpha/sqrt(M_ref L_ref)`), which at `alpha = 64` is not small and
which I did not control. **Reported as inconclusive.** Testing this properly
needs a reference whose error is uniformly negligible across the whole `alpha`
range, which is a bigger build than this round.

## C — the third term, and the shape prediction

`|loss - loss_ref|` vs `D` at fixed large `L, M` gives slope **-0.645** against
`-1/2`. Inside the ±0.15 bar, but **confounded**: the deviations are large
(63% of the reference loss at `D = 4`), and the regression problem itself changes
with `D` — with `P = 8` samples and `D` output coordinates, small-`D` models fit
relatively better. So this does not cleanly isolate the large-`D` limit of the
Mean ODE. Suggestive, not established.

**Consequence for the `P^{-1/6}` shape.** Balancing the three terms under
`P = Theta(LMD)` gives (`08` §5)

    L = P^{1/6},   M = P^{1/2},   D = P^{1/3},   error = O(P^{-1/6})

i.e. at fixed budget the optimum is **wide blocks, moderate embedding, shallow
depth**. Two of the three exponents this rests on are now measured cleanly
(A1, A2); the third is only suggestive (C). The shape claim is therefore
**derived and partially supported, not validated**. A direct fixed-`P` sweep is
also not possible with the present harness: at fixed `D` and fixed `LM` the CLT
term is constant, so the optimum degenerates to "all depth", and escaping that
degeneracy requires exactly the `D`-comparable metric that C shows is confounded.

## Status

| claim | status |
|---|---|
| CLT term `sqrt(D/(LM))`, at init | **validated** (round 007: `-0.502/-0.501/+0.510`) |
| CLT term after `k` GD steps | **validated** (A1) |
| Euler term `1/L`, and `M`-independent | **validated** (A2) |
| `L`/`M` interchangeability at fixed `LM` | **validated** (round 007 Q4: 1.03x over `(4,64) -> (256,1)`) |
| `1/alpha` nonlinearity term | validated in isolation (B, first instrument) |
| lazy optimum `alpha* = (ML)^{1/4}` | **inconclusive** |
| third term `1/sqrt(D)` | suggestive, confounded |
| `P^{-1/6}` optimal shape | derived; partially supported |
