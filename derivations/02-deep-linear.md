# Deep linear DMFT: the algebraic closure, with the operators actually defined

> **Method: DMFT (algebraic closure of the §1 system).**

`equations.md` §3 states the closure in terms of "causal operators **C**^ℓ (from
**A**^{ℓ-1}, **H**^{ℓ-1}, Δ) and **D**^ℓ (from **B**^ℓ, **G**^{ℓ+1}, Δ)" but
never defines them, and gives no formula for the responses at all. Both are
derived here.

This is the first rung with a live response sector, and it is algebraically
exact — **no Monte Carlo, no sampling floor**. Any response-sector bug shows up
unmasked. That is the whole reason to do it before the nonlinear case.

## 1. Specialisation

Set `phi(h) = h`, so `phi_dot = 1` and `phi_ddot = 0`. Then `g^l = z^l`,
`Phi^l = <h^l h^l> ≡ H^l`, and `G^l = <z^l z^l>`. From `01-deep-mlp.md` §6:

    h^l(t) = u^l(t) + gamma_0 int_0^t ds sum_a [A^{l-1}(t,s) + Delta_a(s) H^{l-1}(t,s)] z^l_a(s)
    z^l(t) = r^l(t) + gamma_0 int_0^t ds sum_a [B^l(t,s)     + Delta_a(s) G^{l+1}(t,s)] h^l_a(s)

with `u^l ~ GP(0, H^{l-1})`, `r^l ~ GP(0, G^{l+1})`, independent (they come from
different weight matrices: `W^{l-1}(0)` and `W^l(0)`).

**The Onsager delta of `01-deep-mlp.md` §7 is absent here**, because it carries
`phi_ddot`. This is exactly F1b, and it is why the checks below cannot certify
the equal-time diagonal — only the nonlinear `L=2` case can.

## 2. The operators (the missing definitions)

Vectorise over `(mu, t)` into `R^{PT}`. Define, as `PT x PT` matrices,

    C^l[(mu,t),(a,s)] = [ A^{l-1}_{mu a}(t,s) + Delta_a(s) H^{l-1}_{mu a}(t,s) ] * theta(t-s)
    D^l[(mu,t),(a,s)] = [ B^l_{mu a}(t,s)     + Delta_a(s) G^{l+1}_{mu a}(t,s) ] * theta(t-s)

`theta` is the causal mask; in discrete time the `dt` of the integral is folded
into the matrix. Both are lower-triangular in the time index, so all products
and all Neumann inverses below stay causal.

The two field equations become
`h^l = u^l + gamma_0 C^l z^l` and `z^l = r^l + gamma_0 D^l h^l`, hence

    (I - gamma_0^2 C^l D^l) h^l = u^l + gamma_0 C^l r^l
    (I - gamma_0^2 D^l C^l) z^l = r^l + gamma_0 D^l u^l

matching `equations.md` §3. Write `M_l = I - gamma_0^2 C^l D^l` and
`Nt_l = I - gamma_0^2 D^l C^l`.

## 3. Kernels

Since `u` and `r` are independent zero-mean Gaussians,

    H^l  = M_l^{-1}  [ H^{l-1}  + gamma_0^2 C^l G^{l+1} C^{l T} ] M_l^{-T}
    G^l  = Nt_l^{-1} [ G^{l+1}  + gamma_0^2 D^l H^{l-1} D^{l T} ] Nt_l^{-T}

Confirms `equations.md` lines 90–95.

## 4. Responses (not in `equations.md`)

For linear networks the response is **deterministic** — the same for every
sample — so no averaging is needed:

    A^l     = gamma_0^{-1} d h^l / d r^l  = gamma_0^{-1} * gamma_0 M_l^{-1} C^l  = M_l^{-1} C^l
    B^{l-1} = gamma_0^{-1} d z^l / d u^l  = gamma_0^{-1} * gamma_0 Nt_l^{-1} D^l = Nt_l^{-1} D^l

Both are causal, since `M^{-1} = sum_k (gamma_0^2 C D)^k` is a sum of products
of causal operators. Note the `gamma_0^{-1}` in the definition cancels exactly
against the `gamma_0` the source picks up through the memory — which is the
check that the convention in `equations.md` is the right one.

Sanity: at `gamma_0 -> 0`, `A^l -> C^l -> Delta * H^{l-1}` and the response
sector reduces to the trained-part memory alone, as it must.

## 5. Predictions by the correlator rule (F4)

`f_mu(t) = gamma_0^{-1} <w(t) h^L_mu(t)>` with `w = z^L`, so `f` is the
`(mu,t),(mu,t)` diagonal of the cross-correlation

    <z^L h^{L T}> = gamma_0 * Nt_L^{-1} [ G^{L+1} C^{L T} + D^L H^{L-1} ] M_L^{-T}

giving

    f_mu(t) = [ Nt_L^{-1} ( G^{L+1} C^{L T} + D^L H^{L-1} ) M_L^{-T} ]_{(mu,t),(mu,t)}

The `gamma_0^{-1}` and `gamma_0` cancel, so unlike the sampled case there is no
`1/gamma_0` amplification here — F15 has no analogue in the algebraic solve.
This is the exact prediction, not a marched ODE.

## 6. Coupling structure and why it is a fixed point

- `C^l` needs `A^{l-1}, H^{l-1}` — **from below**
- `D^l` needs `B^l, G^{l+1}` — **from above**
- `A^l` and `B^{l-1}` each need **both** `C^l` and `D^l`

So the upward recursion (`H^0 = K^x`, `A^0 = 0`) and the downward recursion
(`G^{L+1} = 1 1^T`, `B^L = 0`) cannot be run in sequence — each needs the
other's output. Hence a damped global fixed point over `{A, B, H, G, Delta}`,
with `Delta` itself set by `f` above. At `L = 1` the coupling is severed by the
boundary conditions (`A^0 = B^1 = 0` ⇒ `C^1 = Delta*K^x`, `D^1 = Delta`), which
is precisely why the two-layer case is causally integrable.

## 7. Checks this rung can and cannot certify

**Can:** the response machinery end to end — operator construction, causal
masking, the Neumann/`M^{-1}` structure, the fixed-point loop and its damping
(F5), and write-before-read ordering (F17), all with **zero sampling floor**.
Plus the exactly-solvable `L=1` whitened case as an anchor, and the lazy limit.

**Cannot:** the equal-time diagonal (F1) — `phi_ddot = 0`. Nor F6
(response-noise rectification) or F8/F15/F16, which are all Monte-Carlo
artifacts and simply do not exist in an algebraic solve.

So a green run here says the response *plumbing* is right. It says nothing
about the Onsager term. That gap is the entire content of Phase 4.

## 8. Reduction targets

1. `L = 1`: `C^1 = Delta*K^x`, `D^1 = Delta`, and the whole system must
   collapse onto the certified two-layer solver.
2. `L = 1`, whitened `K^x = I`, single output direction: the scalar ODE
   `dDelta/dt = -2 sqrt(1 + gamma_0^2 (y-Delta)^2) Delta`.
3. `gamma_0 -> 0`: `M, Nt -> I`, kernels freeze at `H^l = K^x`, `G^l = 1`, and
   `K^NTK = (L+1) K^x` for whitened data — lazy NTK dynamics in closed form.
4. Finite-width linear networks at matched parameterisation and `dt`.
