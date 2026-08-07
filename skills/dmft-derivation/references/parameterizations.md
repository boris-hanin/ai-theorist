# Parameterization bookkeeping (μP / mean-field, depth-μP, transformers)

## Base μP / mean-field (width only) — conventions of 2205.09653 / 2309.16620

Network: $f=\frac{\beta_L}{\gamma}\bm w^L\cdot\phi(\bm h^L)$,
$\bm h^{\ell+1}=\bm h^\ell+\beta_\ell\bm W^\ell\phi(\bm h^\ell)$ (residual) or
$\bm h^{\ell+1}=\beta_\ell\bm W^\ell\phi(\bm h^\ell)$ (MLP),
$\bm h^1=\beta_0\bm W^0\bm x$; all weight entries $\sim\mathcal N(0,1)$.

| Quantity | μP (width only) | depth-μP (1/√L residual) |
|---|---|---|
| Input branch β₀ | $D^{-1/2}$ | $D^{-1/2}$ |
| Hidden branch βₗ (0<ℓ<L) | $N^{-1/2}$ | $(LN)^{-1/2}$ |
| Readout β_L | $N^{-1/2}$ | $N^{-1/2}$ (no L factor) |
| Output divisor γ | $\gamma_0\sqrt N$ | $\gamma_0\sqrt N$ |
| Learning rate η(t) | $\eta_0(t)\gamma_0^2N$ | $\eta_0(t)\gamma_0^2N$ (independent of L) |
| Init variance σ²ₗ | 1 | 1 (not depth-dependent) |

Once 1/√L is inside residual branches, NO further depth-dependence of η, γ, σ
is needed. η₀, γ₀ = O(1) are the transferable hyperparameters.

## The richness dial γ₀

- Predictions move at O(1) for any γ₀; feature-kernel movement is O(γ₀).
- γ₀ → 0: lazy/kernel limit; kernels static; NTK regression dynamics.
- γ₀ = O(1): rich/feature-learning DMFT regime; γ₀ multiplies the memory
  (feedback) terms in the single-site processes.
- γ₀ is itself transferable across width and (with depth-μP) depth.

## Converting other parameterizations

- NTK parameterization (f = (1/√N)w·φ(...), LR O(1)): equals γ₀ ∝ 1/√N → lazy
  as N→∞. To study feature learning, move to μP first.
- Standard (PyTorch default): transfers HPs over neither width nor depth;
  convert weights/LR to the table above before applying the theory.

## What "hyperparameter transfer" means operationally (2309.16620)

In the joint N,L→∞ limit, predictions and kernels evolve at O(1) rates
independent of N and L; finite (N,L) networks approximate the same limiting
dynamics with O(N^{-1/2}) + O(L^{-1}) error and no systematic η₀-dependent
bias ⇒ optimal (η₀, γ₀, momentum, regularization) measured on a small proxy
transfer to large (N,L). μP alone transfers across width but NOT depth.

## Transformer axes (2405.15712, summary)

Per-head dim N, heads H, depth L; exponents α_A (attention-logit scaling),
α_L (residual scaling):
- Stable feature-learning N→∞ limit REQUIRES α_A = 1 (1/d_k logits, not
  1/√d_k). Heads then collapse to identical dynamics (variance O(N^{-2})).
- H→∞ at fixed N: heads become i.i.d. single sites (new head-averaged mean
  field); attention keeps learning; convergence O(H^{-1}).
- Depth: α_L = 1 keeps Θ(1) block-weight updates but kills init stochasticity
  (attention logits → 0 at init); α_L = 1/2 keeps init Brownian structure but
  freezes in-block weights at L→∞ (residual stream still updates Θ(1)).
- Bulk LR: η = η₀·N·H·L^{2α_L−1} (SGD).

Recipe generalization: for each architectural axis, (i) identify the
exchangeable population, (ii) define its averaged kernel order parameter,
(iii) audit the exponent of every action term to classify which survive the
limit.
