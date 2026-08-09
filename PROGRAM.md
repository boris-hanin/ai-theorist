# The DMFT skill program: history and status

Four stages: identify skills → validate skills on known results → assess
solvability of new problems → solve. Each skill is crisp (one technique).

## Certification states

A skill's status is one of:

- **CERTIFIED** — its claims are backed by artifacts in this repo that can be
  re-run. Requires at least one exactly-solvable reduction and a stated
  Monte-Carlo floor.
- **PARTIAL** — certified on a proper sub-case, with the uncovered part named.
- **RECONSTRUCTED** — rewritten from the program record after the original was
  lost. Faithful in substance, not verbatim, and NOT re-validated. By the
  program's own F14 standard these files are hints, not sources: where one
  conflicts with a derivation or a measurement, the measurement wins.

"Validated in a prior session" is not a certification state. Every such claim
in this repo rests on the program record alone; those artifacts are gone.

## Skills

| Skill | Status | Files |
|---|---|---|
| dmft-derivation | **PARTIAL** — L=1 certified; deep-linear and general-depth nonlinear P=1 response solvers implemented (quick battery L=1–3); P>1 nonlinear coupling not certified | `skills/dmft-derivation/`, `rounds/001–003` |
| dmft-master | RECONSTRUCTED | `skills/dmft-master/` |
| dmft-resnet-depth | RECONSTRUCTED | `skills/dmft-resnet-depth/` |
| dmft-attention | RECONSTRUCTED — round 005 measured four transfer claims, but did not complete its own certification bar | `skills/dmft-attention/`, `rounds/005` |
| dmft-moe | **PARTIAL** — rewritten from round 006; sharp scaling subclaims pass, collapse/control bars are mixed; full DMFT absent | `skills/dmft-moe/`, `rounds/006–010` |
| dmft-graph | **FAILED VALIDATION** — round 011 failed its preregistered rule (P3, P4/P4b, P7, and controls P3c2/P3c3); static/scaling subclaims remain useful, but neither the full parameterisation nor a dynamics solver is certified | `skills/dmft-graph/`, `rounds/011` |

The four companion skills were originally delivered as downloadable archives
whose copies were lost (cloud-workspace recycling). Each carries a provenance
banner. The master skill's 9-step algorithm: scaling audit; edge classification
(single-use → Gaussian source / reused → response pair / readout carrier →
correlator / bilinear order parameter); populations & nesting; exact update
identities; disorder average; closure; simplification; solve; validate.

## Re-validation policy

**Re-validate lazily.** A RECONSTRUCTED skill is certified when it is next
actually used, not speculatively. Certifying all four up front is expensive and
the value does not materialise until one is needed.

What certification requires, when the time comes:

1. A pre-registration committed before results (`rounds/TEMPLATE/prereg.md`).
2. At least one exactly-solvable reduction reproduced.
3. A degenerate-case collapse onto an already-certified skill.
4. Finite-size simulation in the matched parameterisation, seed-averaged, with
   the Monte-Carlo floor reported beside the gap.
5. An ablation that bites, or a stated reason it cannot.
6. For any new check battery: a mutation test, before the battery is trusted.

Exception: `dmft-master`'s zero-shot claim — that a novel architecture is
handled by running Steps 0–8 without consulting the per-instance skills — is
the program's headline result and worth certifying on its own schedule rather
than lazily. It was originally validated on the hyperbolic rounds, which
postdated the synthesis; those artifacts are gone.

## Method invariants (learned the hard way; see registry/)

- Exact discrete-time predictions via the correlator rule with control
  variates — never Euler-marched theory curves (F4). F4 and F15 are the same
  channel: the readout correlator carries an explicit 1/γ₀, so its MC floor is
  O(1/(γ₀√S)) and GROWS as γ₀ shrinks.
- Response functions by exact forward-mode sensitivity — never finite
  differences in production (FD is for TESTING the sensitivity code; correct
  agreement is ε-independent and O(1/S)).
- Equal-time response diagonals can be nonzero and scale as `1/dt` (F1), but
  their endpoint weight in a causal sum comes from the exact update order.
  Round 003 falsified unit weight and favoured strict past in its L=2 solver.
- Seed-average before comparing (F10); check MC floors by sample-halving (F8)
  and report the floor beside every gap; ablations that change nothing are red
  flags, not passes (F17).
- **Before counting `Delta(observable)`, list every INPUT to that observable**,
  not only the parameters that "belong" to it (F23, round 011: the attention
  logits move because the block's input moves, with no reference to the Q/K
  learning rate — and the omission was invisible at the conventional exponent
  because two channels coincided there).
- **Read the whole loss-vs-`eta` row before trusting an `argmin`** (F24): under
  sign-like optimisers a "best loss during training" statistic creates a second
  basin at large `eta` and the optimum jumps between basins.
- **A check battery is not trustworthy until it has been mutation-tested.**
  Reintroduce the failure modes it claims to catch and confirm it catches them.
  The two-layer battery was initially blind to F4 (`rounds/001`).
- **This repository is the working directory of a round, not an export of one.**
  Four archive losses, all the same shape: the real work lived somewhere that
  was not a commit. See `rounds/README.md`.

## Where the program actually is

Stage 2 now has executable evidence beyond L=1 (deep-linear and nonlinear P=1,
plus several scaling rounds), but only the L=1 subcase meets the stated
certification definition cleanly.  Stages 3 and 4 still lack a general
solvability rubric.  Round 009 is a worked novel scaling example, not a general
methodology; Round 011 is a worked example of the method correctly refusing to
certify a derivation after failed bars.
