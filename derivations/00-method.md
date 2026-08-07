# How to derive a DMFT single-site process (the cavity route)

Worked derivations live alongside this file. This one is the method: the same
six moves, in the same order, for any architecture. It is the operational
content of `dmft-master` Steps 1–5.

MSRDJ (generating functional + Hubbard–Stratonovich) and the cavity method give
the same answer. Cavity is used here as the primary route because it makes
visible *where* the response functions come from — they arise from one specific
step, and skipping that step is the single most common way to get a wrong
answer that looks right (F1).

## The six moves

### M1. Integrate the parameter dynamics exactly

Under `theta_dot = -gamma^2 grad L`, every weight matrix splits as

    W(t) = W(0) + (learned part)

and the learned part is **rank-P per unit time**: it is an outer product summed
over the P data points. This is the whole reason the limit is tractable — the
learned part contributes through P-dimensional kernels, not through N^2 degrees
of freedom.

Keep `gamma_0` separate from `gamma = gamma_0 sqrt(N)` at every step. Confusing
them is a registered way to be off by `sqrt(N)`.

### M2. Substitute and split each field

For every field, substitute `W(t) = W(0) + learned` and collect:

    field = (frozen-disorder part) + (memory integral over kernels and Delta)

The memory integral is the easy half: it is an explicit `gamma_0`-weighted
convolution of order parameters against the error signal. Write it out.

### M3. Classify the frozen-disorder part

For each appearance of `W(0)`, ask: **is this matrix used anywhere else?**

- **Single-use.** The frozen part is a sum of N i.i.d. contributions with no
  feedback, so by CLT it is a Gaussian source with covariance equal to the
  population kernel of whatever it multiplies. Done — no response function.
- **Reused** (the same matrix appears in the forward pass and transposed in the
  backward pass, or twice in the forward pass). The frozen part is **not**
  Gaussian, because the thing it multiplies itself depends on `W(0)` through
  the other use. This is where responses come from. Go to M4.

### M4. Cavity expansion of a reused matrix — the Onsager term

Take the forward field `chi_i(t) = (1/sqrt(N)) sum_j W_ij(0) phi(h_j(t))`, and
suppose the same `W(0)` reappears transposed in the backward field
`xi_j(t) = (1/sqrt(N)) sum_i W_ij(0) g_i(t)`.

Remove row `i` from the system ("cavity"), so `h_j^{\i}` is independent of it.
Restoring the row perturbs `xi_j` by `(1/sqrt(N)) W_ij g_i`, so

    phi(h_j(t)) = phi(h_j^{\i}(t))
                  + sum_alpha int ds [ d phi(h_j(t)) / d xi_j(alpha,s) ]
                                     (1/sqrt(N)) W_ij g_i(alpha,s)

Substituting back, the first term is Gaussian (CLT, covariance = the population
kernel `Phi`), and the second gives

    (1/N) sum_j W_ij^2 <d phi / d xi> g_i   ->   <d phi / d xi> g_i

because `W_ij^2 -> 1` and the sum over j self-averages. So

    chi_i(t) = u_i(t) + sum_alpha int ds  <d phi(h(t))/d xi(s)>  g_i(s)

**The second term is the Onsager reaction.** It is not optional and it is not
small: it is the same order as the Gaussian source. Dropping it is the
"forgot the response functions" failure.

Define the response with an explicit `gamma_0^{-1}`,

    A(t,s) = gamma_0^{-1} <d phi(h(t)) / d r(s)>

so that the reaction reads `gamma_0 * A` and `A = O(1)`: the dependence of `h`
on the source `r` runs through the `gamma_0`-weighted memory, hence
`<d phi/d r> = O(gamma_0)`. The `gamma_0^{-1}` in the definition is bookkeeping,
not a divergence.

Run the same argument transposed to get the backward response `B`.

### M5. Close

Replace every population average by its single-site expectation:

    Phi(t,s) = <phi(h(t)) phi(h(s))>,   G(t,s) = <g(t) g(s)>

and read the responses off as `A`, `B` above. The unknowns are now `P*T x P*T`
matrices, not `N`-dimensional fields.

### M6. Fix the boundaries

At each end of the network one of the two uses of a matrix is missing, so the
corresponding response vanishes:

- the input side has no backward channel into the data: `A^0 = 0`;
- the output side has nothing above it: `B^L = 0`.

These two zeros are what make `L = 1` collapse to a response-free
McKean–Vlasov system. **They are the only reason L=1 is easy**, and they are
why L=1 can never certify response-sector code.

## The equal-time subtlety (F1, sharpened)

`h(t) = u(t) + gamma_0 * (memory over [0,t])`, so `u` enters `h` **directly**:

    d h(t) / d u(s)  contains  delta(t - s)

Therefore the response `B(t,s) = gamma_0^{-1} <d g(t) / d u(s)>` with
`g = phi_dot(h) z` contains an instantaneous piece

    B(t,s) ⊃ gamma_0^{-1} < phi_ddot(h(t)) z(t) > delta(t-s)

**In discrete time the delta becomes 1/dt**, so the diagonal entry of the
response kernel is larger than its off-diagonal neighbours by a factor `1/dt`.
It looks like an outlier. It is not — after the `gamma_0 * dt * B` weighting it
contributes at `O(1)` to the field:

    gamma_0 * dt * (gamma_0^{-1} <phi_ddot z> / dt) * phi(h(t)) = <phi_ddot(h(t)) z(t)> phi(h(t))

This is the precise content of F1, and it explains both halves of that entry:
masking the diagonal drops an `O(1)` term (hence 20–50% kernel error), and
"cleaning" an anomalously large diagonal is the natural wrong instinct.

It also explains **F1b**: the term carries `phi_ddot`, so it vanishes identically
for linear `phi`. A linear cross-check cannot see this bug. The minimal
architecture that *can* is a nonlinear network with `L = 2`.

*Status: derived here, not yet confirmed numerically. See
`03-deep-linear.md` and the L=2 experiment planned for the nonlinear solver —
until that measurement exists, this section is theory only.*

## Checklist before believing a derivation

1. Does it reduce correctly when a response is switched off by a boundary
   (`L=1`)?
2. Does it reduce to the linear case when `phi` is linear?
3. Does `gamma_0 -> 0` give the lazy/NTK answer in closed form?
4. Do two independent routes agree (cavity and MSRDJ; or derivation and a
   published equation)? If they disagree, **numerics decide** (F14).
5. Is every factor of `gamma_0`, `sqrt(N)`, `sqrt(D)` and `1/dt` accounted for?
