# Joint depth, particle-width, and embedding transfer

## Scope

This note extends the fixed-`D` Chizat width rule in derivation 11 to the
fixed scalar task

```text
h_(l+1) = h_l + (1 / (L M)) tanh(h_l U_l) W_l,
f(x) = <h_L, w_out>,          w_out,j = O(1/D).
```

The input data, scalar target, embedding convention, horizon, normalized
coordinate `eta`, and common seeds are held fixed while `L`, `M`, and `D`
change.

## Group-natural Euclidean rates

With per-coordinate stream scale `O(1)`, the coherent mean-field counting gives

```text
lr_U = (L M / D) eta,
lr_W = (L M D) eta.
```

The two rates differ by `D^2`; one global Euclidean rate therefore cannot test
joint transfer in `D`.  At initialization the contractions are incoherent and
can display half-power transients,

```text
lr_U,transient ~ (L M / sqrt(D)) eta,
lr_W,transient ~ (L M sqrt(D)) eta,
```

but this is not the trained coherent law.  It is retained as a negative
control.

An isolated `U` update is also not a valid test of the coherent `U` rule: with
`W` frozen at a random initialization it cannot create the `W`/backward-field
alignment assumed by the coherent count.  The coupled network trajectory is
the primary observable; the isolated channel is reported as a diagnostic.

## Preferred joint shape path

The shape ratio

```text
rho = L M / D
```

is the natural joint invariant.  Holding `rho` fixed keeps the up-projection
rate fixed in normalized time and preserves the particle-to-stream aspect
ratio.  It should therefore give the cleanest joint transfer path.  This is a
shape constraint, distinct from the two optimizer-rate equations above.

The A100 test confirms the distinction:

- on a general joint ladder where `L M / D` grows from 16 to 256, the accepted
  progress slope is `-0.0503`;
- on a ladder with `L M / D = 8` exactly, it is `+0.00473`.

Both pass, but the invariant path is about ten times flatter.

## Acceptance protocol

Transfer requires all of the following at one fixed `eta`:

1. fixed-eta loss trajectories settle under the declared finite-size model;
2. every shape makes at least `0.1%` fractional progress;
3. log progress versus the declared scale dial has absolute slope at most
   `0.3`;
4. common-seed duplicate trials agree exactly across two A100 workers;
5. omitted-`L`, omitted-`M`, half-power-`D`, and single-rate mutations are
   reported separately and must bite on a path that identifies them.

The finite-horizon local optimum is still only a stability-edge diagnostic.
