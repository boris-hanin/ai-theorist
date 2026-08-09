# Trained Chizat boundaries and a Muon transfer coordinate

## Scope and status

This note derives the candidate learning-rate coordinates implemented by
`chizat_lmd_transfer.py` and `chizat_muon_transfer.py`.  It is a scale-counting
derivation, not a certification.  Round 013 subsequently passed fixed-eta
trajectory, learning-progress, and negative-control gates on CPU and on two
independent A100 replicas, but remains exploratory because the preregistration
was not committed before execution.

The end-to-end scalar model is

```text
h^0 = x E
h^l = h^(l-1) + (1/(L M)) tanh(h^(l-1) U_l) W_l
f = h^L R
```

with `x in R^d0`, `E in R^(d0 x D)`, `U_l in R^(D x M)`,
`W_l in R^(M x D)`, and `R in R^(D x 1)`.  Boundary biases are absent so a
direct scalar bias cannot hide a failed representation-transfer rule.

## Initialization

```text
E_ad ~ N(0, 1/d0)
U_dj ~ N(0, 1/D)
W_jd ~ N(0, 1)
R_d  ~ N(0, 1/D^2)
```

The fan-in embed makes every coordinate of `h^0` order one.  The mean-field
unembed has norm `Theta(D^-1/2)`, so the initial scalar output vanishes as
`D^-1/2`.  This is intentional: it keeps the output carrier on the same
mean-field scale as the residual particles instead of injecting an order-one
random readout at initialization.

Initial conditions are coupled across shapes by drawing maximum-size arrays
once per seed and slicing them.  `E` is a literal shared slice.  `D R` is a
shared slice because the declared `1/D` scale must change with the shape.

## Plain-GD boundary rates

At initialization, the unembed tangent kernel is

```text
K_R(x,x') = h(x) . h(x') = Theta(D).
```

Thus `lr_R = eta/D` gives an order-eta function update.  Backpropagation to
one embed entry carries one factor `R_d = O(D^-1)`.  Summing its squared
Jacobian over the `D` output coordinates gives `K_E = Theta(D^-1)` at fixed
`d0`, hence

```text
lr_E = D eta,       lr_R = eta/D.
```

These boundary rates are independent of `L` and `M`.  The Chizat particle
groups retain their separately declared group-wise rules.  Boundary controls
freeze either map or replace both rates by a constant without changing the
particle rates.

## Muon plus auxiliary Adam

Muon is applied only to the mathematical residual-particle matrices `U_l` and
`W_l`.  The embed and unembed stay on auxiliary Adam even though both tensors
are matrices.  Tensor rank is not an optimizer-routing rule.

For a matrix of shape `(A,B)`, the polar-like Muon direction has RMS
`Theta(max(A,B)^-1/2)`.  The selected RMS adjustment multiplies it by

```text
0.2 sqrt(max(A,B)),
```

so its entrywise update RMS is approximately `0.2 * lr_raw`, independently of
aspect ratio.  Five quintic Newton--Schulz steps are performed in float32, with
momentum `0.95` and Nesterov enabled.

The mean-field unembed determines the group-wise coordinate.  For a `U`
update, `h . Delta U` contributes `sqrt(D) lr_U`, while contraction of the
existing `W` with `R` contributes `D^-1/2`; the product is `Theta(lr_U)`.  For
a `W` update, contraction with `R` contributes `D^-1/2`, so its raw rate needs
one `sqrt(D)` factor.  The candidate Muon rates are therefore

```text
embed, auxiliary Adam:  lr_E = eta
U, Muon:                 lr_U = eta
W, Muon:                 lr_W = sqrt(D) eta
unembed, auxiliary Adam: lr_R = eta/D.
```

No `L` or `M` power is added: Muon removes the gradient magnitude that made
plain-GD rates proportional to the population size, while the Chizat average
and gradient alignment sum the layer/particle contributions back to order
one.  That last cancellation is the main empirical claim of the new runner,
not an assumption allowed into the verdict.

## Falsification plan

The reference eta is selected at one preregistered shape by the lowest-eta
within-one-SEM rule.  It is then frozen.  Certification requires finite-size
trajectory settling, nontrivial scale-invariant learning progress, and
rejection of these controls:

1. `wrong_W_D`: use `D eta` rather than `sqrt(D) eta` for W.
2. `wrong_sgd_LMD`: reuse the plain-GD L/M/D multipliers after Muon has removed
   gradient magnitude.
3. `wrong_constant_unembed`: omit the `1/D` auxiliary-Adam readout factor.

A wrong rule is rejected when it fails trajectory settling, fails nontrivial
scale-invariant progress, or is resolvedly worse than the primary rule in a
common-seed paired final-loss comparison at the largest shape.  Settling alone
cannot distinguish convergence to the wrong loss.

Additional diagnostic controls use constant W rate, or freeze the embed or
unembed.  Full per-shape eta sweeps are stability-edge diagnostics only and
cannot alter the fixed-eta verdict.

## Numerical correctness contract

The optimizer is not trusted until tests cover square/tall/wide polar factors,
transpose equivariance, zero gradients, rectangular RMS adjustment, the exact
momentum/Nesterov update, semantic parameter routing, duplicate/missing-route
rejection, and exact checkpoint continuation.  Mutation checks must distinguish
the RMS adjustment from the original aspect-ratio rule and Nesterov from the
non-Nesterov update.
