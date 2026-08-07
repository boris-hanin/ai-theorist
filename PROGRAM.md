# The DMFT skill program: history and status

Four stages: identify skills → validate skills on known results → assess
solvability of new problems → solve. Each skill is crisp (one technique).

## Skills

| Skill | Status | Files |
|---|---|---|
| dmft-derivation | **in repo, verbatim** | `skills/dmft-derivation/` |
| dmft-resnet-depth | validated; files in session archive (to restore) | — |
| dmft-attention | validated; files in session archive (to restore) | — |
| dmft-moe | validated; files in session archive (to restore) | — |
| dmft-master | validated; files in session archive (to restore) | — |

The four missing skill trees were delivered as downloadable archives in
earlier sessions ("math-assistant" trees and overlay archives) before
cloud-workspace recycling; re-attaching those archives to a session will
restore them here verbatim. The master skill's 9-step algorithm: scaling
audit; edge classification (single-use → Gaussian source / reused →
response pair / readout carrier → correlator / bilinear order parameter);
populations & nesting; exact update identities; disorder average; closure;
simplification; solve; validate.

## Method invariants (learned the hard way; see registry/)

- Exact discrete-time predictions via the correlator rule with control
  variates — never Euler-marched theory curves (F4).
- Response functions by exact forward-mode sensitivity — never finite
  differences in production (FD is for TESTING the sensitivity code).
- Equal-time response diagonals are generically nonzero (F1).
- Seed-average before comparing (F10); check MC floors by sample-halving
  (F8); ablations that change nothing are red flags, not passes (F17).
