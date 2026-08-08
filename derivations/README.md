# Two kinds of derivation, and which one each file is

This directory mixes **two methods that answer different questions**. They were
not distinguished when these files were written, and at least one file
(`03-attention.md`) claimed the wrong one in its header. This README is the
correction and the taxonomy.

| file | method | what it establishes |
|---|---|---|
| `00-method.md` | **DMFT** (cavity) | the machinery itself |
| `01-deep-mlp.md` | **DMFT** | full single-site system for a deep MLP |
| `02-deep-linear.md` | **DMFT** | algebraic closure, operators, responses |
| `03-attention.md` | **mixed** — D4 is DMFT, D1/D2/D3/D5/D6 are heuristic | the attention parameterisation, plus the structural form of the key/query process |
| `04-completep.md` | **heuristic only** — no DMFT anywhere | the CompleteP parameterisation, Adam |
| `05-completep-dmft-sgd.md` | **DMFT** (cavity) | the CompleteP residual limit under SGD — dynamics, not just exponents; solver-validated |
| `06-moe.md` | **heuristic + limit structure** — no cavity derivation | the MoE parameterisation (2601.20205 Table 1) and the three-level mean-field structure; measured in `rounds/006-moe` |
| `07-moe-dmft.md` | **comparison only** | what `06` got right/wrong against their Appendix E, and what a real MoE DMFT still needs |

## The two methods

### A. Heuristic scale analysis ("one-step")

Count how large each quantity is, propagate `Theta(.)` through the forward pass,
take **one optimiser step**, and require the induced change in the relevant
observable to be `Theta(1)`. Everything turns on labelling each contraction in
the chain **coherent** (the update is aligned with what it multiplies, so `n`
terms sum to `n`) or **incoherent** (random signs, so `n` terms sum to
`sqrt(n)`).

- **Answers:** what exponents make updates size-stable — i.e. *the
  parameterisation*. A table of powers.
- **Costs:** hours.
- **Sufficient for:** hyperparameter transfer, training stability, choosing
  `alpha`. This is what a muP/CompleteP-style derivation *is*.
- **Cannot answer:** what the loss curve is, how kernels evolve, whether heads
  collapse, what the limit object looks like. It has no dynamics in it.

### B. DMFT

Integrate the parameter dynamics exactly, split each field into frozen disorder
plus a memory integral, classify each disorder edge as single-use (Gaussian
source) or reused (response pair), cavity-expand the reused ones to get the
Onsager reaction, and close on population kernels. → `00-method.md`.

- **Answers:** the limiting *dynamics* — single-site processes, kernels,
  response functions, prediction trajectories.
- **Costs:** days, plus a solver to evaluate it.
- **Sufficient for:** everything above, plus loss curves, kernel evolution,
  head collapse, feature-learning structure.

## Why keeping both is worth it

Not redundancy — **cross-checking**. The two methods must agree on the
exponents, and where they disagree, one of them has a bug. This session
produced a concrete instance in each direction:

- **The heuristic track missed something real.** `03-attention.md` §D2b: my
  one-step argument wrote `delta k ∝ (backward) x q` and treated the backward
  coefficient as `Theta(1)`. It is not, at initialisation — the gradient
  reaching `k` travels through `W_O`, which is uncorrelated with `df/dht` at
  init, so that contraction is *incoherent* and suppressed. It becomes coherent
  only after the correlation builds over subsequent steps. **A one-step frame
  cannot see this, because the first step is precisely the anomalous one.**
- **The DMFT track caught it.** The correlation buildup is exactly what response
  functions track; the paper's own two-step argument (App E.1.2) and its Fig 12
  both live in that structure.

So the honest division of labour: **use the heuristic track to get the
parameterisation, and do not trust it for anything that depends on how
correlations develop over training.** If a claim involves more than one step,
it needs the DMFT track or an explicit measurement.

- **And the DMFT track made the same class of error, twice.** `05`'s
  contribution (c) labelled the per-block Onsager terms *incoherent* when they
  add coherently, and separately treated the response kernel `A^l` as `Theta(1)`
  when it carries a branch factor `L^{-alpha}`. At `alpha = 1/2` the two errors
  cancel **exactly**, so the formula looked right at the exponent everyone
  checks. Only `alpha = 1` exposed it, and only against a solver. So F18 is not
  "the heuristic track is the sloppy one" — coherent/incoherent labelling is the
  error-prone step in **both** methods. What the DMFT track adds is that its
  claims are *solvable*, hence falsifiable by measurement rather than by
  argument.

## Registered as F18

That failure mode is now in `registry/failure-modes.md` as **F18 — one-step
scale analysis is blind to correlation buildup**, with the D2b instance as its
evidence.

## What is missing

No DMFT derivation exists for **attention** (only the structural form of the
key/query process, D4) or for **MoE** (`06`/`07` give the parameterisation and
the limit structure, validated, but no cavity derivation, no response sector and
no solver — see `07-moe-dmft.md` §5). CompleteP now has one under **SGD**
(`05-completep-dmft-sgd.md`); the **Adam** case remains heuristic only.
Attention is still supported by heuristic analysis plus measurement. That is enough for
the HP-transfer claims made in rounds 004/005 and the CompleteP work, and it is
**not** enough to claim a replication of either paper's DMFT analysis — which is
why those rounds say so explicitly in their scope sections.
