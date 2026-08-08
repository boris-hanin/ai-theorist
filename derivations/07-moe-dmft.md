# MoE DMFT — check of `06-moe.md` against the paper's Appendix E

> **Status: COMPARISON, not a full derivation.** `06-moe.md` derives the
> parameterisation and the *structural* part of the limit. This file records
> what happened when I then read Appendix E of 2601.20205, which I had
> deliberately held off reading. The **full cavity derivation for MoE (single-site
> processes, the response sector, a solver) is NOT done** — see §5.

## 1. The one that matters: `alpha_*`

Their Eq. (5), verbatim in structure:

    lim_{N->inf}  N / ( N_e(N) L(N) E(N) )  ==  alpha_*  <  inf

with `N = n_embd`, `N_e = n_hid`, `E = n_exp`, `L` depth. `06-moe.md` §4 derived,
from nothing but incoherent-sum counting,

    Var[ init contribution to the residual stream ]  =  (1/kappa) * N / (N_e E L)

**Identical up to the fixed constant `kappa`.** They introduce `alpha_*` as a
*scaling condition*; §4 shows what it physically *is*. That reframing is the
substantive content, and round 006's P4 confirms it as a measurement rather than
an interpretation: slopes `-1.005`, `-1.000`, `-0.988` in `L`, `E`, `alpha_ffn`.

**A non-trivial consistency check I did not design for.** The formula is
`N/(N_e E L)` with `N_e = alpha_ffn N`, so `N` **cancels**: the variance is
`1/(alpha_ffn E L)`, independent of width. That is exactly why P4's width row
came out flat (`-0.056`) — a row I had included only as a null control. The
control passing is a prediction of Eq. (5), not an accident.

## 2. Structural claims — all four match

| Appendix E | `06-moe.md` | verdict |
|---|---|---|
| order parameters are expert averages `(1/E) sum_k` (their `M^l_{sigma sigma C}`, Eqs. 22–24) | §3, mean-field over experts | **match** |
| a per-expert response sector (`Rbar^l_{phi xi}`, `Rbar^l_{g chi}`, Eqs. 25–26) plus a stream response `R^L_{h xi}` | §3's three levels, each needing its own reused-disorder pair | **match in structure** |
| the `alpha_* = 0` limit is "a universal neural ODE that depends on `kappa` but **not on the FFN ratio `N_e/N`**" | §1d: `alpha_ffn`-independence, because `sigma(W_down)` makes the trained part dominate the init by `alpha_ffn^{1/2}` | **match**, and §1d supplies the mechanism |
| `alpha_* > 0` gives a neural SDE | §4 | **match** |

Their per-expert scalar `A^l_k` (Eq. 17) couples the expert output into the
router path via `sigma-dot_k A_k r_k` in the backward pass (Eq. 10). `06-moe.md`
does not have this object at all — scale counting never needed it. **It is the
main thing the heuristic route misses**, and it is where the router's feature
learning actually lives.

## 3. A convention difference that changes one of my results

Their footnote 2: the router `r_k` is **initialised at zero**, and expert
diversity at initialisation comes from **random initial biases `b_k(0)`**.

`06-moe.md` §2 and round 006's P5 instead used `gamma = 1/2` (standard-normal
logits) with `b = 0`. Both give the *same structure* — selection becomes a
deterministic quantile threshold on `q = sigma(r) + b`, because top-`a`-of-`E`
on an exchangeable population is thresholding at the `(1-kappa)` quantile. But
the **closed form is convention-specific**:

    my setup   (gamma = 1/2, b = 0):     q*(kappa) = sigmoid( Phi^{-1}(1 - kappa) )
    their DMFT (r = 0, b_k(0) random):   q* = sigma(0) + (1-kappa)-quantile of the b law

So round 006's original parameter-free result — measured to 0.19 and 0.75 s.e. at
`kappa` = 1/8, 1/4 — verified the quantile *structure* but was a specialisation
of their setup, not a reproduction of it.

**Now implemented and checked in their convention** (`moe.py`, `gamma=None`,
`b_std>0`). The threshold is a bare Gaussian order statistic, so

    q*(kappa) = 1/2 + b_std * Phi^{-1}(1 - kappa)

Measured at `kappa` = 1/8, 1/4, 1/2, 3/4: deviations 0.92, 1.43, 0.75, 0.36 s.e.
**All four now agree**, where under my earlier `sigma(r)` convention the
`kappa >= 1/2` points were 4.1 and 2.8 s.e. off. That closes round 006's one
unexplained residual: it was the `sigma(.)` map applied to an order statistic,
not the quantile structure.

Worth noting separately: the *main text* initialises biases at zero ("we
conveniently initialize biases at zero so no one expert disproportionately
receives tokens at init"), while the *DMFT appendix* needs them non-zero to
generate init diversity. That is a difference between the empirical setup and
the theoretical one, not an inconsistency — but it means the DMFT's init
transient is not the experiments' init transient, and **the two conventions must
not be mixed**: taking `b = 0` from the main text and `r = 0` from App. E leaves
the gates exactly tied and routing degenerate, with nothing visible in the loss.
Registered as **F21**; `moe.py` now refuses that combination.

## 4. What I could NOT verify, and why I am not calling it a discrepancy

The block prefactor in their Eq. (8) extracts from the PDF as

    h^{l+1} = h^l + sqrt(N) / (L N_e E) sum_k sigma_k W^{l,2}_k(t) phi(h^{l,1}_k)
            = h^l + (1/L) sqrt( N / (N_e E) ) sum_k ...

My scale count of the second form does not come out `Theta(1)`: with unit-variance
entries, `[sum_k sigma_k W_k phi_k]_j` is an incoherent sum giving
`Theta(sqrt(kappa E N_e))`, so the block output would be
`(1/L) sqrt(kappa N)` per component rather than `(1/L) Theta(1)`. The two printed
forms are also not equal to each other.

**I am recording this as an extraction failure, not a paper error.** Nested
radicals over fractions are exactly what `pypdf` mangles, the two rendered forms
already disagree with each other in the extracted text, and F14 is the registered
failure mode for trusting a transcribed formula (a ResNet-round transcription was
once 3.3x off). Resolving it needs the LaTeX source or the rendered PDF page, not
more arguing from the text dump. **Open.**

## 5. What is still missing

`06-moe.md` + this file establish the parameterisation and the structure of the
limit. They do **not** constitute a DMFT derivation of MoE in the sense of
`00-method.md`. Specifically not done:

1. **The cavity derivation** (M1–M6) for the MoE block — in particular the
   per-expert response pairs and the Onsager terms, where `05`'s contribution-(c)
   error lived. The MoE case has *more* response pairs than any architecture
   handled so far (stream, per-expert, within-expert), so the coherent/incoherent
   labelling that was wrong twice in `05` §6 has three more places to go wrong.
2. **`A^l_k` and the router feature-learning channel** (§2 above), absent from
   the heuristic route entirely.
3. **A solver**, hence no theory-vs-simulation validation of the *dynamics* —
   only of the exponents.
4. Their `M^l_{sigma-dot sigma-dot AA}` term (Eq. 13), which has no counterpart
   in anything I derived.

Until those exist, the honest description of this program's MoE work is:
**parameterisation validated, limit structure validated, dynamics not**.
