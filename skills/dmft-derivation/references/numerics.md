# DMFT numerical solution algorithms

## 0. Two invariants that override the textbook algorithms

Both are registered failure modes. They apply to EVERY algorithm below.

**(i) Predictions come from the correlator rule, never from marching the
prediction ODE (F4).** Discretizing the *memory integrals* on a time grid is
fine and unavoidable. What is forbidden is obtaining $f$ by Euler-stepping
$df_\mu/dt=\sum_\alpha K_{\mu\alpha}(t,t)\Delta_\alpha(t)$ with the same step
as SGD: that accumulates $O(dt^2\lambda^2)$ per step and masquerades as
finite-width error. Instead evaluate $f$ directly from the sampled population
each step, which is what the finite-width network actually computes:

$$f_\mu(t)=\gamma_0^{-1}\big\langle w(t)\,\phi(h^L_\mu(t))\big\rangle
\;\approx\;\frac{1}{\gamma_0 S}\sum_{n=1}^{S}w_n(t)\,\phi(h^L_{n\mu}(t))$$

(from $f=\frac{1}{\gamma\sqrt N}\bm w^L\!\cdot\!\phi(\bm h^L)$ with
$\gamma=\gamma_0\sqrt N$). Advance $\{h,z,w\}$ by exact discrete update
identities from time-$t$ quantities, then READ $f$ off the updated population.
Use control variates on the sampled correlator.

Note the explicit $\gamma_0^{-1}$: this channel multiplies the sampled
correlator by $1/\gamma_0$, so its Monte-Carlo floor is $O(1/(\gamma_0\sqrt S))$
and GROWS as $\gamma_0\to0$ — F4 and F15 are the same channel. Antithetic
readout pairs (§E) make it exactly zero at $t=0$.

**(ii) Response functions come from exact forward-mode sensitivities, never
from finite differences in production.** Propagate state sensitivities
alongside the single-site solve and population-average them into
$\bar A,\bar B$. Finite differences are a TEST of the sensitivity code, not a
production path: a correct sensitivity implementation shows FD agreement that
is $\varepsilon$-independent and $O(1/S)$ (pure population feedback). See §E.

## A. General deep case: alternating Monte Carlo fixed point
(Algorithm 1 of arXiv:2205.09653, Appendix B)

Discretize the memory integrals on a grid of $T$ points (discrete-GD form).
Every kernel becomes a $PT\times PT$ matrix. Predictions still follow the
correlator rule of §0(i), not the marched ODE.

```
Input: K^x, targets y, sample count S, damping β ∈ (0,1]
Init:  Φ⁰ = K^x ⊗ 11ᵀ (constant in time), G^{L+1} = 11ᵀ
       guesses {Φˡ, Gˡ} (use lazy/NTK kernels), {Aˡ, Bˡ} = 0
repeat until kernels converge:
    # 1. error dynamics — correlator rule (F4), NOT a marched ODE
    #    march the single-site states with exact discrete updates, then
    #    read f off the population each step:
    #        f_μ(t) = (1/(γ₀S)) Σ_n w_n(t) φ(h^L_{nμ}(t))    [+ control variate]
    #        Δ(t)   = -∂ℓ/∂f |_{f(t)}
    #    K(t,t) = Σ_{ℓ=0..L} G^{ℓ+1}(t,t) ⊙ Φˡ(t,t) remains the diagnostic
    #    NTK, but it is not the integration path for f.
    # 2. layer-wise single-site sampling
    for ℓ = 1..L:
        draw S samples uˡ ~ GP(0, Φ^{ℓ-1})   # Cholesky of PT×PT covariance
        draw S samples rˡ ~ GP(0, G^{ℓ+1})
        solve the single-site integral equations forward in t for each sample
            → trajectories hˡ_n(t), zˡ_n(t), gˡ_n = φ̇(hˡ_n)zˡ_n
        # 3. kernel re-estimation
        Φ̃ˡ = (1/S)Σ_n φ(hˡ_n)φ(hˡ_n)ᵀ ;  G̃ˡ = (1/S)Σ_n gˡ_n gˡ_nᵀ
        # 4. responses: exact FORWARD-MODE sensitivities, sample-averaged
        Ãˡ  = (1/S)Σ_n ∂φ(hˡ_n)/∂rˡ_n        (causal PT×PT)
        B̃^{ℓ-1} = (1/S)Σ_n ∂gˡ_n/∂uˡ_n
        # write kernel rows BEFORE any same-step read of them (F17)
    # 5. damped update
    X ← (1-β)X + βX̃   for X ∈ {Φˡ, Gˡ, Aˡ, Bˡ}
```

Implementation notes:
- Responses: propagate sensitivities forward alongside the single-site solve
  (§0(ii)). Never finite differences in production; FD only to test the
  sensitivity code.
- Equal-time diagonals are generically NONZERO (F1) — do not mask $\bar A,\bar B$
  with a strict lower-triangular mask. The rule is set by computation order:
  a field read before the backward pass has $\bar A(t,t)=0$; a drive that sees
  the same step's forward pass has $\bar B(t,t)\neq0$. Linear cross-checks pass
  with this bug present (F1b), so they do not certify it.
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
              (antithetic: pair each (u_n, w_n) with (u_n, -w_n) — F15)
h_n(0) = u_n
f = (1/(γ₀S)) Σ_n w_n φ(h_n)        # correlator rule; = 0 exactly if antithetic
for t on grid (step dt):
    Δ = -∂ℓ/∂f            # MSE: Δ = y - f
    # advance ALL single-site states from the SAME time-t quantities
    for each n:
        h'_{n,μ} = h_{n,μ} + dt·γ₀·w_n·Σ_α K^x_{μα} Δ_α φ̇(h_{n,α})
        w'_n     = w_n     + dt·γ₀·Σ_α Δ_α φ(h_{n,α})
    h, w ← h', w'
    # THEN read the prediction off the updated population (F4) — do not
    # Euler-march f ← f + dt·[Φ + G⊙K^x]Δ
    f = (1/(γ₀S)) Σ_n w_n φ(h_n)
    # diagnostics only (not the integration path for f):
    Φ(t,t) = (1/S)Σ_n φ(h_n)φ(h_n)ᵀ ;  G(t,t) = (1/S)Σ_n w_n² φ̇(h_n)φ̇(h_n)ᵀ
```
This is the exact discrete-time trajectory of the corresponding finite-width
network under GD with η = γ₀²N·dt, with N neurons replaced by S i.i.d. sites —
so theory and simulation are compared under the SAME discretization, and the
only residual gap is Monte-Carlo error plus O(1/√N) on the sim side. RK4 on the
joint system is available when you want the gradient-FLOW limit instead, but
then the sim must be run at correspondingly small dt.

Statistical error: kernels and the readout correlator are S-sample averages ⇒
O(1/√S) noise, amplified to O(1/(γ₀√S)) in f (§0(i), F15). S ≈ 10⁴–10⁵ is cheap
since each sample is a P-dim ODE. Certify the floor by sample-halving (F8) and
report it next to any theory-vs-sim gap.

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
- Average several seeds BEFORE comparing (F10); kernel observables self-average
  at large N but predictions have O(1/√N) fluctuations at finite width.
- Optional: subtract-init trick f(x) → f(x) − f₀(x) if γ₀√N is not large
  enough for f(0) ≈ 0.

## E. Variance reduction, sensitivity testing, and guards

Registered fixes, each keyed to the failure mode it closes.

1. **Antithetic readout pairs (F15).** Pair every site with a twin sharing all
   Gaussian sources but with the readout weight negated: (u_n, w_n) and
   (u_n, −w_n). The γ₀⁻¹-amplified correlator ⟨wφ(h)⟩ is then exactly zero at
   t = 0, so f(0) = 0 and loss(0) are exact rather than O(1/(γ₀√S)).
   Diagnosis signature that you needed this: the theory-vs-sim gap's γ₀-trend
   flips sign once the fix is in.
2. **One joint QMC stream (F16).** If using quasi-Monte-Carlo, draw ONE
   scrambled-Sobol stream of dimension (families × dims) and slice it per
   source family. Independently seeded scrambles of the same Sobol sequence
   are NOT independent (measured same-dimension |corr| up to 0.96 with
   scipy.stats.qmc.Sobol). Signature of the bug: init-time cross-moments that
   should vanish don't; seed scatter far above 1/√S.
3. **Sample-halving (F8).** Run at S and S/2; the shift between them estimates
   the Monte-Carlo floor. Report the floor beside every theory-vs-sim gap — a
   gap below its own floor is not evidence of anything.
4. **Testing the sensitivity code.** Compare forward-mode responses against
   finite differences at several ε. A correct implementation gives agreement
   that is ε-independent and shrinks as O(1/S). ε-dependent disagreement means
   the sensitivity code is wrong; ε-independent but S-flat disagreement means
   a missing feedback channel, not noise.
5. **Write-order guard (F17).** Response rows Ā(t, s<t) are computable at time
   t and read by the same step's field assembly. Write before read, and assert
   the rows are nonzero at read time — writing after the read leaves the
   response sector silently OFF while spot-checks of Ā values still pass.
6. **Ablations must bite.** Turn the response sector off and confirm the answer
   changes. An ablation that changes nothing is a red flag, not a pass (F17).
   For L=1 this is trivially satisfied — responses genuinely vanish — which is
   exactly why L=1 cannot certify response-sector code.
