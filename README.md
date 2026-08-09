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
- `rounds/`: fifteen numbered validation rounds, including explicitly pending
  or exploratory work, with results and, where retained, raw data and
  preregistrations.
- `registry/failure-modes.md`: the canonical failure registry (F7, F9, and F13
  are intentionally marked lost and are never reused).
- `PROGRAM.md`: certification policy and current skill status.
- `src/ai_theorist/autoscaler/`: the first product slice: strict residual-MLP,
  sparse-MoE, and νGPT normalized-Transformer schemas; real SGD and Adam
  training; one-dimensional normalized-eta tuning; transfer checks;
  fixed-horizon scaling fits; and refusal-aware held-out calibration.
- `apps/web/`: the drag-and-drop Autoscaler workbench.  Its canvas intentionally
  compiles only typed `Embed -> repeated {MLP, top-k MoE, or νGPT} block ->
  Unembed` graphs.

## Current status

- The L=1 solver is certified against exact reductions (`rounds/001`).
- Deep-linear response dynamics and a general-depth nonlinear P=1 response
  solver are implemented; the quick battery checks L=1–3 (`rounds/002–003`).
  The equal-time response has a
  measured `1/dt` representative, but the previously claimed unit endpoint
  weight was falsified; the tested causal sum uses strict past.
- Attention, residual, and MoE scaling claims have executable measurements,
  with each round's failed or under-powered bars retained in its `results.md`.
- Round 014 validates simultaneous Chizat transfer in `L`, `M`, and `D`; the
  theory-motivated `LM/D = constant` path is the flattest tested joint path.
- Round 015 validates the sparse-MoE product path on two A100s, including
  groupwise Adam transfer, routing-load guards, a wrong-global-rate control,
  and held-out validation-loss calibration.  This is product evidence, not a
  full MoE DMFT certification.
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

The Autoscaler has a separate end-to-end path:

```bash
ai-theorist-autoscale sample-spec /tmp/autoscaler.json --optimizer adam --quick
ai-theorist-autoscale sample-spec /tmp/moe.json --optimizer adam --architecture pre_norm_moe --quick
ai-theorist-autoscale sample-spec /tmp/nugpt.json --optimizer adam --architecture normalized_transformer --quick
ai-theorist-autoscale plan /tmp/autoscaler.json
ai-theorist-autoscale run /tmp/autoscaler.json --output runs/autoscaler/manual --summary
ai-theorist-autoscale-api
```

With the API running on port 8787, start `apps/web` with Node 22 and pnpm.  The
UI compiles a study, launches it asynchronously, monitors trial progress, and
shows a next-scale forecast only after every calibration gate passes.  See
`docs/autoscaler-validation.md` for the exact acceptance contract and A100
campaign configurations, and `docs/autoscaler-validation-report.md` for the
measured evidence, retained negative results, and current A100 revalidation
status after the fixed-eta transfer correction.
The normalized-Transformer contract is documented separately in
`docs/normalized-transformer-contract.md`; its A100 manifest is
`configs/autoscaler/a100_nugpt_adam.json`.

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
