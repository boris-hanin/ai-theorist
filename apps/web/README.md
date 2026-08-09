# AI Theorist Autoscaler web

A vinext product shell for composing the MVP residual architectures and launching
fixed-horizon tuning, transfer, and scaling-law studies through the local Python API.
The typed canvas includes the standard pre-norm MLP, Chizat particles, and the
sparse MoE.  Embed and unembed are trained; Muon is exposed only for the
validated Chizat semantic-routing contract.

Start the scientific service from the repository root with
`PYTHONPATH=src python -m ai_theorist.autoscaler.api`, then run this app. The UI
uses `http://127.0.0.1:8787` by default; override it with
`NEXT_PUBLIC_AUTOSCALER_API`.

## Prerequisites

- Node.js `>=22.13.0`

## Quick start

```bash
pnpm install
pnpm dev
```

The hosted shell is useful for reviewing the interface.  Training remains an
explicit local or A100-side service; without that service the UI says
`Compute offline` and does not simulate a run.

## Checks

```bash
pnpm lint
pnpm typecheck
pnpm test
```

`pnpm test` builds the production worker and verifies the rendered product
contract.  The Python and A100 evidence is recorded in
`../../docs/autoscaler-validation-report.md`.
