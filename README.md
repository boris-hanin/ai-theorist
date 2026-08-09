# ai-theorist

An executable research record for deriving and testing dynamical mean-field
theory (DMFT) and hyperparameter-scaling claims for neural networks.

The standard used here is **derivation + independent numerics + matched
finite-size simulations**, with explicit uncertainty and controls.  The record
also includes failed and inconclusive rounds.  Formal preregistration was adopted
during the project; it is not retroactively claimed for early rounds.

## What is in the repository

- `skills/dmft-derivation/`: the core derivation workflow and runnable solvers
  for two-layer, deep-linear, and general-depth nonlinear P=1 cases.
- `skills/dmft-master/`: a reconstructed nine-step synthesis.  It remains a
  specification, not a certified package.
- `skills/dmft-{resnet-depth,attention,moe,graph}/`: architecture-specific
  derivations, simulators, sweeps, and stated coverage limits.
- `derivations/`: long-form derivations and corrections.
- `rounds/`: eleven validation rounds with results and, where retained, raw
  data and preregistrations.
- `registry/failure-modes.md`: the canonical F1–F22 registry (F7, F9, and F13
  are intentionally marked lost and are never reused).
- `PROGRAM.md`: certification policy and current skill status.

## Current status

- The L=1 solver is certified against exact reductions (`rounds/001`).
- Deep-linear response dynamics and a general-depth nonlinear P=1 response
  solver are implemented; the quick battery checks L=1–3 (`rounds/002–003`).
  The equal-time response has a
  measured `1/dt` representative, but the previously claimed unit endpoint
  weight was falsified; the tested causal sum uses strict past.
- Attention, residual, and MoE scaling claims have executable measurements,
  with each round's failed or under-powered bars retained in its `results.md`.
- The Round 010 `C^-1/6` rate is supported by a large A100 artifact.  Its v2
  source had not been committed;
  `skills/dmft-moe/scripts/overnight_suite_v2.py` reconstructs the exact
  effective settings from the log, and the historical susceptibility table is
  explicitly labeled summary-only until rerun.
- Round 011's graph-transformer validation **failed** its preregistered rule.
  Several subclaims pass, but the full parameterisation is not certified.
- No claim here establishes production performance, broad task generality, or
  a graph-transformer DMFT dynamics solver.

## Reproduce the checks

Python 3.11 is the CI target.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
pytest
python skills/dmft-derivation/scripts/validate.py --quick
```

The deep-linear and nonlinear suites are more expensive:

```bash
python skills/dmft-derivation/scripts/validate_deep_linear.py --quick
python skills/dmft-derivation/scripts/validate_deep_nonlinear.py --quick
```

Large Round 010 jobs require a CUDA GPU.  Every new runner accepts a portable
output path; historical files with fixed remote paths are retained only when
needed for provenance and are labeled as such.

## Research and provenance policy

Raw results are never silently promoted after a failed bar.  Shared-seed
comparisons use paired uncertainty.  Expensive artifacts whose original source
was lost are described as reconstructions rather than original code.  See
`rounds/README.md` for the lifecycle and per-round preregistration status.

## License and citation

Copyright (c) 2026 Boris Hanin.  All rights reserved; no open-source license has
been granted.  Citation metadata is in `CITATION.cff`.
