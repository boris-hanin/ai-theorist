# Solver library

> RECONSTRUCTED from the program record; pending re-validation. Numerical
> patterns validated across the program's solvers, keyed to Step 7. The
> original file also carried code-level templates (lost); the hyperbolic
> repo's solvers are working reference implementations of every pattern
> below (hyp_dmft.py, hyp_dmft_deep.py, hyp_dmft_res.py, theory_sim_res.py).

## Integration schemes

1. **Causal co-integration** (preferred whenever the closure permits):
   march t → t+dt integrating single-site samples, kernels, and the
   prediction correlator jointly — no fixed-point loop, no damping.
   Requires writing same-step kernel rows before same-step field reads
   (F17) and verifying equal-time (non)zeros of Ā/B̄ (F1).
2. **Alternating MC fixed point** (general deep case): sample S
   trajectories given kernels → re-estimate kernels + responses → damped
   update (β ≈ 0.3–0.7) → iterate. Response estimates need slow/extra
   damping (noise rectifies to positive kernel bias, F6); the Δ-residual
   map has operator norm ~ dt·λ·T and needs an inner damped loop (F5).
3. **Exact discrete-time predictions**: f each step from population
   correlators (the correlator rule) with control variates — never
   Euler-march the limit ODEs at SGD's step (F4).

## Sources and responses

4. **GP sources by growing Cholesky** of the time-block population
   covariance (append rows per step; leading minors frozen ⇒ earlier
   samples unchanged); diagonal jitter for near-degenerate time blocks.
5. **Responses by exact forward-mode sensitivities** per site (state
   sensitivities + explicit field-kernel feedback), population-averaged
   into Ā/B̄. NEVER finite differences in production; use FD only to TEST
   the sensitivity code (errors should be ε-independent and O(1/S) —
   pure population feedback).
6. **Theory-simulator control pair**: integrate the LIMIT maps with an
   explicit random matrix at large m (validates the reduction), and with
   an INDEPENDENT matrix for the backward transpose (physically realizes
   the no-response model). The pair brackets the solver: base closure
   must match the independent-backward control; the full solver must
   match the shared-matrix run. This is the decisive audit of the
   response sector (found F17).

## Variance reduction

7. Antithetic site pairs sharing all sources with opposite readout sign:
   kills the 1/γ₀-amplified channel exactly at init (w̄(0)=0, loss(0)
   exact) — F15.
8. Quasi-MC: ONE joint scrambled-Sobol stream of dimension
   (families × dims), sliced per family — independently-seeded scrambles
   of the same sequence are strongly cross-correlated (F16).
9. Stratify heavy scalar components (inverse-CDF grids for |readout|);
   quadrature-exact init populations when the site prior is
   low-dimensional (angle grids at d=2).
10. Heavy-tailed population averages (rate-amplified metrics, F12):
    S-halving to expose the MC floor (F8); median-of-seeds; robust
    metrics next to MSE.

## Guards and audits

- Seed-average before comparing; init-offset hygiene (F10).
- Ablations must change results or be reported under-powered.
- Stability edges can depend on the dial (width-creeping edges, radius-
  dependent hyperbolic edges): verify the operating point is inside the
  edge at ALL sweep scales before attributing gaps to theory.
- Memory/compute plan before building: sensitivity storage scales as
  (sites × P·T kicks × P·T history) per field family — size T, S, and
  precision (f32 histories, f64 states) to the machine first.
