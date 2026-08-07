# The DMFT skill program: history and status

Four stages: identify skills → validate skills on known results → assess
solvability of new problems → solve. Each skill is crisp (one technique).

## Skills

| Skill | Status | Files |
|---|---|---|
| dmft-derivation | **in repo, verbatim** | `skills/dmft-derivation/` |
| dmft-resnet-depth | RECONSTRUCTED (pending re-validation) | `skills/dmft-resnet-depth/` |
| dmft-attention | RECONSTRUCTED (pending re-validation) | `skills/dmft-attention/` |
| dmft-moe | RECONSTRUCTED (pending re-validation) | `skills/dmft-moe/` |
| dmft-master | RECONSTRUCTED (pending re-validation) | `skills/dmft-master/` |

The four companion skills were originally delivered as downloadable
archives whose copies were lost (cloud-workspace recycling; archives no
longer in chat). They have been REWRITTEN from the program record: each
carries a provenance banner, is faithful in substance but not verbatim,
and is pending re-validation before being treated as certified. The master skill's 9-step algorithm: scaling
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
