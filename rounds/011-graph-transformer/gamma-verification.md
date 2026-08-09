# Verification of the `gamma = 0.42` disagreement — it is NOT a disagreement

Round 011 (subagent) reported, as its highest-confidence disagreement with
**2607.05017**, that degree-normalised message passing gives `gamma = 0.42`
rather than the paper's `gamma = 1` (its Eqn 14), verified to 1% on five
operators. Checked independently here.

## The formula is right; the framing is not

The derived expression

    gamma^2 = < sum_v P_uv^2  +  sum_{v != v'} P_uv P_uv' rho_vv' >

makes `gamma` a **function of the feature correlation `rho`**, and the two
values in dispute are its two endpoints:

- `rho = 1` (perfectly correlated features): the sum telescopes to
  `(sum_v P_uv)^2 = 1` for row-normalised `P`, giving **`gamma = 1` exactly**.
- `rho = 0` (independent features): it collapses to `<1/deg>`, giving
  `gamma = <1/deg>^{1/2}`.

Measured (`N = 200` geometric graph, mean degree 14, 400 draws per point):

| `rho` | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| `gamma` measured | **0.2869** | 0.5123 | 0.6544 | 0.7527 | **1.0000** |
| row-norm prediction | 0.2879 | 0.5587 | 0.7358 | 0.8779 | 1.0000 |

`<1/deg>^{1/2} = 0.2879` on this graph, matching the `rho = 0` limit to 0.3%.
Symmetric normalisation `D^{-1/2} A D^{-1/2}` behaves identically
(0.2769 at `rho=0`, 0.9915 at `rho=1`).

## Verdict

**The subagent's formula is correct, its number is correct for its own graph
ensemble at `rho = 0`** (`0.42^2 ~ 1/5.7`, consistent with its mean degree ~6),
**and its framing as a disagreement with the paper is wrong.** `gamma = 1` is
what the same formula gives for correlated features — the regime that obtains
once features align — and it is exactly 1, not approximately. What was missing
is a regime label, not a correction to Eqn 14. The entry is **withdrawn** from
the disagreement list in `verdict-table.md`.

## Caveat on this check

The intermediate `rho` values sit below the row-norm prediction (0.512 vs 0.559
at `rho = 0.25`) because the construction used here,
`h = sqrt(rho) z0 + sqrt(1-rho) z_v`, imposes *uniform* pairwise correlation
whereas the formula weights `rho_vv'` by the actual `P`-products. The two
endpoints — the ones in dispute — are exact, so the verdict stands; the middle
of that table tests the construction, not the formula.
