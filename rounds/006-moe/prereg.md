# Round 006 — MoE parameterisation and mean-field structure (arXiv 2601.20205)

**Committed before any measurement.** Derivation: `derivations/06-moe.md`.

## Scope

The **reduced** residual-MoE model of the paper's §4 (MoE blocks only, no MHSA,
scalar output), which is what their DMFT analyses. Not the full transformer of
§3.1 — no claim about the language-model experiments is made or tested here.
Optimiser: **SignGD** for the parameterisation predictions (their stated Adam
proxy, and what §1 of the derivation assumes); SGD noted where it differs.

## What would make this round a failure

Any of P1–P6 failing against a certified floor, **or** any control failing to
bite. Per the program's rule, a control that changes nothing is a red flag
(F17), and an identity control is not evidence (recorded four times already:
`05` §8c/§8d).

## Predictions and bars

Floors: seed-halving for simulation scatter, and where two runs share a seed the
**paired** floor (F20), not the individual one. Every gap is reported beside its
floor (F8).

| # | prediction | quantitative bar |
|---|---|---|
| **P1** | training-loss curves collapse across `alpha_ffn in {1,2,4,8}` under Table 1 | max curve spread `<= 2x` the seed floor |
| **P1c** | **control**: fan-in `sigma_down = m^{-1/2}` does **not** collapse | spread `>= 5x` the Table-1 spread |
| **P2** | loss curves collapse across `E in {8,16,32,64}` at fixed `kappa` | same bar as P1 |
| **P3** | `E_k(init) ~ alpha_ffn^{-1/2}`; `Delta E_k ~ alpha_ffn^{0}` | fitted slopes within `0.1` of `-0.5` and `0.0` |
| **P3c** | **control**: fan-in gives slopes `0.0` and `+0.5` | within `0.1` |
| **P4** | init stream variance `~ 1/(L a alpha_ffn)` | slope `-1 +/- 0.15` in each of `L`, `a`, `alpha_ffn` *separately* |
| **P5** | top-`a` selection threshold `-> q*(kappa)` deterministically | threshold s.d. across seeds `~ E^{-1/2}`, slope `-0.5 +/- 0.15` |
| **P6** | optimal LR transfers across `n`, `E`, `alpha_ffn` | drift `<= 0.25` decades across the sweep |
| **P6c** | **control**: fan-in drifts in `alpha_ffn` | `>= 0.5` decades |

## Declared in advance

- **P4 is the one I most expect to need care.** It is a *variance*, so it is
  rate-amplified (F12) and needs seed-averaging; and it mixes three dials, so a
  single joint fit could hide a compensating error. Fitting each dial
  **separately** is the guard, and is committed to here.
- **P1/P2 collapse is a null result**, so it is only meaningful with P1c biting.
  If P1c does not bite, P1 is uninformative and will be reported as such rather
  than as a confirmation.
- `gamma = 1` (their default) makes router logits `O(n^{-1/2})` at init while
  updates are `O(1)`. So at `t = 1` the router is in the anomalous regime of
  F18. Any router claim tested at one step is therefore **not** a test of the
  asymptotic labelling. P5 is measured at init (a pure geometry claim about
  order statistics) and is unaffected.
- I have **not** read Appendix E. The DMFT-proper check is a separate,
  genuinely blind comparison recorded in `07-moe-dmft.md`.

## Known-suspect prior artifact

`skills/dmft-moe/SKILL.md` is a RECONSTRUCTED file whose central claim
(`eta_up ∝ alpha`, `alpha` = sparsity) contradicts both Table 1 and
`derivations/06-moe.md` §1a. It is treated as a hint, not a source (F14), and
is to be rewritten from whatever this round validates.
