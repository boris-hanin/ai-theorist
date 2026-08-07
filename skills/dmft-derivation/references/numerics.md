# DMFT numerical solution algorithms

## A. General deep case: alternating Monte Carlo fixed point
(Algorithm 1 of arXiv:2205.09653, Appendix B)

Discretize time on a grid of $T$ points (Euler / discrete-GD form of all
integrals). Every kernel becomes a $PT\times PT$ matrix.

```
Input: K^x, targets y, sample count S, damping β ∈ (0,1]
Init:  Φ⁰ = K^x ⊗ 11ᵀ (constant in time), G^{L+1} = 11ᵀ
       guesses {Φˡ, Gˡ} (use lazy/NTK kernels), {Aˡ, Bˡ} = 0
repeat until kernels converge:
    # 1. deterministic error dynamics from current kernels
    K(t,t) = Σ_{ℓ=0..L} G^{ℓ+1}(t,t) ⊙ Φˡ(t,t)
    integrate df_μ/dt = Σ_α K_{μα}(t,t)Δ_α(t) on the grid → f, Δ
    # 2. layer-wise single-site sampling
    for ℓ = 1..L:
        draw S samples uˡ ~ GP(0, Φ^{ℓ-1})   # Cholesky of PT×PT covariance
        draw S samples rˡ ~ GP(0, G^{ℓ+1})
        solve the single-site integral equations forward in t for each sample
            → trajectories hˡ_n(t), zˡ_n(t), gˡ_n = φ̇(hˡ_n)zˡ_n
        # 3. kernel re-estimation
        Φ̃ˡ = (1/S)Σ_n φ(hˡ_n)φ(hˡ_n)ᵀ ;  G̃ˡ = (1/S)Σ_n gˡ_n gˡ_nᵀ
        # 4. response functions: sample-averaged Jacobians via AUTODIFF
        Ãˡ  = (1/S)Σ_n ∂φ(hˡ_n)/∂rˡ_n        (causal PT×PT)
        B̃^{ℓ-1} = (1/S)Σ_n ∂gˡ_n/∂uˡ_n
    # 5. damped update
    X ← (1-β)X + βX̃   for X ∈ {Φˡ, Gˡ, Aˡ, Bˡ}
```

Implementation notes:
- Response Jacobians: differentiate through the UNROLLED discrete single-site
  solve (autodiff), never finite differences.
- Damping β ≈ 0.3–0.7; smaller for larger γ₀. If diverging: anneal γ₀ upward
  from a small value, reusing converged kernels as init.
- Convergence metric: relative Frobenius change of all kernels between outer
  iterations < 1e-3 (and stable under β change).
- Cost: memory O(P²T²), time O(P³T³) per outer iteration. If PT ≳ N, direct
  network simulation may be cheaper than the theory solve.

## B. Two-layer (L=1): exact causal co-integration (no fixed point)

Key structural fact: single-site sample dynamics depend on the population only
through the deterministic Δ(t); Δ(t) depends on samples only through
equal-time kernels. So integrate everything forward jointly:

```
draw n=1..S:  u_n ~ N(0, K^x) ∈ R^P (static),  w_n(0) ~ N(0,1)
h_n(0) = u_n;  f = 0 ∈ R^P
for t on grid (step dt):
    Δ = -∂ℓ/∂f            # MSE: Δ = y - f
    Φ(t,t) = (1/S)Σ_n φ(h_n)φ(h_n)ᵀ
    G(t,t) = (1/S)Σ_n w_n² φ̇(h_n)φ̇(h_n)ᵀ
    f ← f + dt·[Φ(t,t) + G(t,t)⊙K^x]Δ
    for each n:
        h_n ← h_n + dt·γ₀·w_n·(K^x ⊙ [Δφ̇(h_n)ᵀ-broadcast]) …
              i.e. dh_{n,μ} = dt·γ₀·w_n·Σ_α K^x_{μα}Δ_α φ̇(h_{n,α})
        w_n ← w_n + dt·γ₀·Σ_α Δ_α φ(h_{n,α})
```
(Update f, h, w from the SAME time-t quantities — do not use already-updated
values within a step; or use RK4 on the joint system for higher accuracy.)

Statistical error: kernels are S-sample averages ⇒ O(1/√S) noise in Δ
trajectory; S ≈ 10⁴–10⁵ is cheap since each sample is a P-dim ODE.

## C. Deep linear: algebraic solve

Build causal PT×PT operators C, D from current kernels/responses and solve the
matrix fixed-point equations of `equations.md` §3 by iteration (still damped,
but no sampling noise). The L=1 whitened case reduces to the scalar ODE
∂ₜΔ = −2√(1+γ₀²(y−Δ)²)Δ — integrate with RK4, use as ground truth.

## D. Matching finite-width simulations (the sim side of validation)

To compare theory to experiment the simulation MUST use the same
parameterization and time discretization:
- f = (1/(γ₀N))·w²·φ(W⁰x/√D), entries of W⁰, w² ~ N(0,1).
- GD step θ ← θ + η·∇θ Σ_μ ℓ-gradient with η = γ₀²·N·dt  (matches
  dθ/dt = −γ²∇L with γ=γ₀√N, time t = (step)·dt).
- Same dt as the DMFT grid; same loss convention (ℓ = ½(y−f)² ⇒ Δ = y−f).
- Average several seeds; kernel observables self-average at large N but
  predictions have O(1/√N) fluctuations at finite width.
- Optional: subtract-init trick f(x) → f(x) − f₀(x) if γ₀√N is not large
  enough for f(0) ≈ 0.
