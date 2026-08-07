# Mandatory checks and reproduction targets

## Analytic sanity checks (run ALL that apply, every time)

1. **Lazy limit.** γ₀ → 0 (numerically: γ₀ ≈ 0.05): kernels must freeze at
   NNGP/NTK init values; MSE prediction dynamics must match
   f(t) = (I − e^{−K₀t})y with K₀ the init NTK.
2. **t = 0 kernels** must equal the NNGP/NTK recursions (analytic Gaussian
   integrals for tanh/ReLU/erf, or S-sample MC estimates at init).
3. **Exactly solvable cases:**
   - L=1 linear, whitened data, single output direction:
     ∂ₜΔ = −2√(1+γ₀²(y−Δ)²)Δ; final kernel H(∞) = I + [(√(1+γ₀²y²)−1)/y²]·yyᵀ.
   - Deep linear: algebraic matrix fixed-point solution.
4. **Perturbative regime:** small-γ₀ numerics must match Φ₀ + γ₀²Φ₂ expansion.
5. **Convergence audits:** dt → dt/2, S → 2S, damping β varied ⇒ results stable.

## Simulation match protocol

Train finite-width nets in the SAME parameterization/discretization (see
`numerics.md` §D). Compare:
- train-loss curves (theory deterministic vs sim seeds; agreement should
  improve with N),
- equal-time kernel trajectories Φ(t,t), G(t,t) (entrywise and Frobenius),
- final-kernel alignment A(Φ_DMFT, Φ_NN) = ⟨Φ₁,Φ₂⟩/(‖Φ₁‖‖Φ₂‖) as function of N
  (2205.09653 finds convergence by N ≈ 500–2500),
- the γ₀ trend: kernel movement ‖Φ(T,T) − Φ(0,0)‖ grows with γ₀ in BOTH.

## Published reproduction targets (easiest first)

| Target | Source | Settings |
|---|---|---|
| L=1 linear whitened ODE | 2205.09653 §4.1 | any P≤D, K^x=I; scalar RK4 |
| Deep linear kernels vs sims | 2205.09653 Fig. 2 | linear, N=1000, P=20, L≤5 |
| L=3 tanh kernels + loss | 2205.09653 Fig. 1 | P=10 CIFAR-10 pts, MSE, γ₀∈{0,1}, N≈2500, t≲100 |
| Weight decay fixed points | 2205.09653 Fig. 3 | ReLU, L=1, N=1000, λ=1, sweep γ₀ |
| Deep linear ResNet 1-step kernel | 2309.16620 Fig. 7 | linear residual, N=1000, L≤100, 1 example, closed form H₁(τ) |
| NTK depth-ODE convergence | 2309.16620 Fig. 6 | residual ReLU, 2 inputs angle θ, error O(L⁻²) |
| LR transfer sweeps | 2309.16620 Figs. 1,4 | conv ResNet CIFAR-10, 20 epochs, μP vs depth-μP grids |
| Lazy fluctuation law | 2304.03408 Prop. 3 | rank-one Σ^Δ(t,s)=κy²ts·e^{−λ(t+s)} vs 2-layer ensembles |
| Variance vs γ trend | 2304.03408 Fig. 3 | prediction variance ↓ with γ, kernel variance ↑ |

## Failure symptom → likely cause table

| Symptom | Likely cause |
|---|---|
| Theory loss decays 2× too fast/slow | loss convention (½ factor), or η vs γ² mismatch |
| Kernels don't move at γ₀=1 | γ₀ missing from memory term; or NTK parameterization used |
| Sim disagrees more as N grows | parameterization mismatch (standard vs μP) |
| Fixed point oscillates/diverges | damping too high, γ₀ too rich — anneal γ₀ |
| L≥2 theory off but L=1 fine | response functions A,B dropped or mis-signed |
| Predictions offset at t=0 | f(0)≠0 at finite γ₀√N — subtract init predictor in sims |
