# Chizat mean-field learning-rate transfer in block width

> **Boundary limitation.** Round 012 trained only the Chizat residual-particle
> matrices and therefore certifies no embed/unembed rule.  Current end-to-end
> work uses trained fan-in embed and mean-field unembed groups; their SGD and
> Muon coordinates are derived separately in
> `derivations/12-chizat-trained-boundaries-muon.md`.

## Claim and scope

For the non-MoE mean-ODE residual network of Chizat et al., with fixed data,
fixed output dimension, fixed richness `alpha`, and plain GD/SGD, the
transferable hyperparameter is the normalized step `eta`.  The raw Euclidean
optimizer learning rate is

```text
lr_raw = L M eta / alpha^2.
```

No additional fitted power of `M` belongs in this rule.  A width-dependent
finite-horizon optimum near the stability boundary is a different observable.

## Model

For samples `a = 1,...,P`,

```text
h_a^0 = x_a
h_a^l = h_a^(l-1)
        + c sum_{j=1}^M w_lj sigma(u_lj . h_a^(l-1)),
c = alpha / (L M).
```

At initialization, `u_lj = O(D^-1/2)` per coordinate and
`w_lj = O(alpha^-1)` per coordinate.  The loss is the normalized squared
output error.

## Function-space counting

Let `A_al = d h_a^L / d h_a^l` be the downstream propagator.  The two
single-particle output Jacobians have the schematic form

```text
J_w,alj = c A_al sigma(u_lj . h_a^(l-1))
J_u,alj = c A_al w_lj sigma'(u_lj . h_a^(l-1)) h_a^(l-1)^T.
```

There are `L M` particles and every Jacobian contains one factor of `c`.
Consequently,

```text
sum_lj J_w J_w^T = Theta(alpha^2 / (L M)),
sum_lj J_u J_u^T = Theta(1 / (L M)).
```

Multiplying by Chizat's raw GD step gives

```text
(L M eta / alpha^2) sum_lj J_w J_w^T = Theta(eta),
(L M eta / alpha^2) sum_lj J_u J_u^T = Theta(eta / alpha^2).
```

Both function-space update channels are independent of `M`.  At `alpha = 1`
they have the same scale.  A group-natural optimizer can instead use
`lr_w = L M eta_w / alpha^2` and `lr_u = L M eta_u` when transfer in `alpha`
is also required; this distinction does not alter the width exponent.

The same cancellation is visible in particle space.  A per-particle gradient
is `Theta(1/M)` at fixed `L`, while the raw SGD rate is `Theta(M)`, so each
particle takes a `Theta(eta)` step independent of width.  This is the
Wasserstein/mean-field Euler coordinate.

## Finite-width correction and the stability edge

At fixed normalized `eta`, the empirical function-space kernel is

```text
K_M = (1/M) sum_j kappa(z_j)
    = K_infinity + M^-1/2 Xi + O_p(M^-1).
```

Therefore fixed-time trajectories should converge with ordinary
`M^-1/2` corrections.  This is the primary transfer test.

The largest stable Euler step is separate.  In a local quadratic regime,

```text
eta_critical(M) approximately 2 / lambda_max(H_M),
lambda_max(H_M) = lambda_infinity + q M^-1/2 + ...,
```

so `eta_critical(M)` can increase with width before saturating.  Over a short
width range this curve can resemble `M^p`; fitting `p` and inserting it into
the parameterization would confuse finite-width stability headroom with the
mean-field transfer law.

## Canonical validation protocol

1. Tune and report normalized `eta`; record the derived raw LR separately.
2. At common seeds, run the same `eta`, data, update horizon, and checkpoints
   across a pure `M` dial (and separately across a pure `L` dial).
3. Test absolute trajectory differences for settling compatible with
   `M^-1/2`; do not rely on relative/log loss near zero.
4. Sweep aggressive rates separately to locate the stability edge.  Its local
   optimum and largest finite rate are diagnostics, not the transfer verdict.
5. Require an omitted-`M` negative control for SGD.
6. Report train and validation trajectories separately.
7. Repeat at a second horizon and require every reported sweep optimum to be
   interior before interpreting an edge curve.

For the autoscaler MVP's standard fan-in residual MLP, the same normalized
coordinate discipline applies but the conversion is parameterization-specific:
`lr_raw = eta / sqrt(M)` for SGD and `lr_raw = eta` for Adam.  For a Chizat
mean-field component it is `L M eta / alpha^2` for SGD.  Adam must not inherit
the SGD multiplier: when its epsilon is negligible, `m/sqrt(v)` removes the
`1/M` gradient scale, giving a width-independent raw LR.  The epsilon-dominated
regime requires separate validation.

## A100 evidence motivating the corrected verdict

In the faithful `alpha = 1`, `L = 8` Chizat harness, at fixed normalized
`eta = 79.4328` and 80 updates, median final training MSE for
`M = 64,128,256,512` was

```text
0.0082817, 0.0078818, 0.0076848, 0.0077137.
```

That is only `0.0325` decades of spread.  Independently selecting the best
finite-horizon rate at every width instead produced a moving near-divergence
optimum.  The former is evidence about transfer; the latter is evidence about
the discretization's width-dependent stability margin.
