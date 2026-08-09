# Graph transformer DMFT — the cavity derivation (M1–M6)

> **Method: DMFT (cavity).** The six moves of `00-method.md`, in order, for the
> base transfer graph transformer of `10-graph-transformer.md` §0.
> Target paper: **arXiv 2607.05017**. That paper contains **no DMFT** — its §3 is
> a Frobenius-norm scaling analysis. So there is nothing here to check against
> line by line; the checks available are internal (the four reductions of
> `00-method.md`'s checklist) and against `05-completep-dmft-sgd.md`, whose
> residual/CompleteP sector this must reproduce when the graph is a single node
> and the attention branch is removed.
>
> **Status: DERIVED, NOT SOLVER-VALIDATED.** There is no solver for these
> equations in this repo. What *is* measured (round 011) is the parameterisation
> that falls out of §7, plus the two structural predictions S1 and S2 of §8.
> Everything else in this file is theory only. §9 says where it is most likely
> wrong, and that section is not decoration — item 1 is a Jensen gap I do not
> know how to close.

---

## 0. What is different about a graph

One sentence, and it organises the whole file:

> **The width index `i = 1..D` is the mean-field direction. The node index
> `u = 1..N` is a data index and is NOT averaged over.**

So the limiting object is *not* a scalar single-site process. It is an
`N`-component process — one component per node of the graph — for a single width
coordinate, with the graph operators `P` and `S` acting as **fixed linear maps on
the node index of every kernel**. Nodes play exactly the role that sequence
positions play in a transformer DMFT and that data points play in an MLP DMFT:
the kernels are `(graph, node, time) x (graph, node, time)` matrices.

Two immediate consequences, both used later:

- `P` and `S` never appear as *disorder*. They are not random matrices being
  cavity-expanded; they are known operators sandwiched between kernels. M3's
  single-use/reused classification applies only to the weight matrices.
- Nothing self-averages in the node direction. Any claim of the form "the graph
  contributes `Theta(1)` factors" (which `10-graph-transformer.md` makes, and
  which the paper makes throughout its §3 — *"the message passing operator `P_a`,
  when properly normalized by `gamma_l` ... is treated as contributing order-one
  factors"*) is a statement about a **fixed graph**, not a limit theorem. Node
  fluctuations are `Theta(1)`, not `Theta(1/sqrt(N))`.

Index conventions for the rest of the file: `a = (alpha, u)` is a
(graph, node) pair; `t, s` are training times; `l` is the block index; `i` a
width coordinate; `h` a head. `Delta_alpha(t)` is the loss residual of graph
`alpha`. Set `sigma_QK = 1` and `alpha_A = 1` throughout (the recommendation of
`10` §5); other `alpha_A` is noted where it changes the structure, and it changes
it a lot (§8, S2).

---

## M1. Integrate the parameter dynamics exactly

Under SGD with the per-group rates of `10` §5, each matrix splits as
`W(t) = W(0) + (learned)`, with the learned part an integral of outer products.
For the MPNN matrix of block `l`:

    dW~^l/dt = -eta_res sum_alpha Delta_alpha(t) grad_{W~^l} z_alpha(t)
    grad_{W~^l} z_alpha = (1/(L gamma_P sqrt(D))) (P_alpha X^l_alpha)^T G^l_alpha

    => W~^l(t) = W~^l(0)
                 - (eta_res/(L gamma_P sqrt(D))) int_0^t ds sum_{alpha,u} Delta_alpha(s)
                       (P_alpha x^l_alpha,u(s)) (x) g^l_{alpha,u}(s)                    (M1.1)

**The learned part is rank-(number of nodes) per unit time**, not rank-(number of
graphs). That is the graph-specific version of "rank-`P` per unit time" in
`00-method.md` M1, and it is why the kernels below carry node indices. For a
graph-level task the *loss* has one residual per graph, but the *gradient* is an
outer product summed over that graph's nodes, so the node index survives.

Same structure for `W_V`, `W_O`, `W_1`, `W_2`. For `W_Q, W_K` the outer product
is between a key/query vector and a node feature, weighted by the attention
sensitivity `c_{uv,h} = dz/dA_{uv,h}`:

    W_Q^l(t) = W_Q^l(0)
               - (eta_Q D_h^{-alpha_A}/sqrt(D)) int ds sum_{alpha,u,v,h}
                     Delta_alpha(s) c^l_{uv,h}(s)  k^l_{alpha,v,h}(s) (x) x^l_{alpha,u}(s)  (M1.2)

Keep `eta_0` separate from `eta_res = eta_0 D L` at every step. This is the
`gamma_0` vs `gamma_0 sqrt(N)` discipline of `00-method.md`, and here it is worse
than usual because there are **three** distinct rate normalisations in play
(`eta_0 D L`, `eta_0 D sigma^2`, `eta_0 D L D_h^{2alpha_A - 2}`).

Rescale the backward field once and for all:

    ghat^l_{a,i}(t) := D N g^l_{a,i}(t) = Theta(1)                                   (M1.3)

---

## M2. Substitute and split each field

Substituting (M1.1) into the MPNN branch field
`b^l_{a,i} = (1/(L gamma_P sqrt(D))) (P X^l)_{a} . W~^l_{:,i}`:

    b^l_{a,i}(t) = u^{b,l}_{a,i}(t)
                 - (eta_0/(L gamma_P^2 N)) int_0^t ds sum_{b} Delta_{gr(b)}(s)
                       K^l_{ab}(t,s) ghat^l_{b,i}(s)                                  (M2.1)

with the **P-smoothed stream kernel**

    K^l_{ab}(t,s) := (1/D) (P x^l_a(t)) . (P x^l_b(s))  ->  [P H^l(t,s) P^T]_{ab}      (M2.2)

`H^l_{ab}(t,s) := (1/D) x^l_a(t) . x^l_b(s)` is the stream kernel. Note the
prefactor: `eta_res * D^{-1} * ... = eta_0 D L / (L^2 gamma_P^2 D) * D * (1/(DN))`
collapses to `eta_0/(L gamma_P^2 N)` — `Theta(1/L)` as desideratum 2 demands, and
**`D`-free**, which is the whole point of the parameterisation.

The MLP branch is identical with `P -> I` and the pre-activation nonlinearity
`phi` in the usual place (`01-deep-mlp.md`).

The attention branch splits into **two** memory channels, because the branch
depends on the weights through two different routes:

    (V/O channel)  same shape as (M2.1) with K -> S H^l S^T, gamma_P -> gamma_A
    (QK channel)   through the attention matrix S itself

and the second has no analogue anywhere else in this repo. It is §M4b.

---

## M3. Classify every frozen matrix

| matrix | forward use | backward use | class | consequence |
|---|---|---|---|---|
| `W^(0)` encoder | `X -> X^(0)` | transposed into `X` (**data**) | **single-use** | Gaussian source, covariance = input Gram `X X^T/n0`; `A^0 = 0` by M6 |
| `W~^l` MPNN | `P X -> b` | transposed into `X^l` | **reused** | response pair |
| `W_V^l` | `X -> V` | transposed into `X^l` | **reused** | response pair |
| `W_O^l` | `S V -> branch` | transposed into `o`, hence into `V` **and** `A` | **reused, twice over** | two response pairs |
| `W_Q^l, W_K^l` | `X -> q`, `X -> k` | transposed into `X^l` | **reused, and mutually contracted** | the class-(b) key/query pair of `03-attention.md` D4 |
| `W_1^l, W_2^l` MLP | standard | standard | **reused** | response pair |
| `W^(L+1)` decoder | `X^L -> z` | transposed into `X^L` | **reused** | Onsager term; `B^{L+1} = 0` by M6 |

Two entries need comment.

**`W_O` is reused twice over.** It carries the backward signal into `o`, and `o`
feeds *both* the value path (`v_v`) and the attention path (`A_{uv}`). So the
same frozen matrix generates two distinct Onsager reactions with two distinct
partner fields. In `05-completep-dmft-sgd.md` the analogous double-counting was
contribution (c), and it was **wrong twice** (labelled incoherent when it adds
coherently; and the response kernel treated as `Theta(1)` when it carried a
branch factor). The two errors cancelled exactly at `alpha = 1/2`. That is the
single most relevant precedent for this file.

**`W_Q, W_K` are a mutually-contracted pair.** `W_K` appears forward in `k` and
transposed in the backward pass into `Xt^l`; but the object it is contracted
*against*, `q`, is itself built from the same stream field. By M4 the frozen part
is not Gaussian and the cavity expansion yields a source-plus-reaction of the
form derived in `03-attention.md` D4.

---

## M4a. Cavity expansion — the ordinary reused edges

Take the MPNN edge. Remove width-coordinate row `i`, so that `x^{l,\i}` is
independent of it. Restoring the row perturbs the backward field
`xi^l_{a,j} = (1/sqrt(D)) sum_i W~_{ji} ghat_{a,i}` by `(1/sqrt(D)) W~_{ji} ghat_{a,i}`,
so

    (P x^l)_{a,j} = (P x^{l,\i})_{a,j}
                    + sum_b int ds [ d (P x^l)_{a,j}(t) / d xi^l_{b,j}(s) ]
                                   (1/sqrt(D)) W~_{ji} ghat_{b,i}(s)

Substituting back, `(1/D) sum_j W~_{ji}^2 -> 1` self-averages over the width
index, giving

    b^l_{a,i}(t) = u^{b,l}_{a,i}(t)
                   + eta_0 sum_b int ds  A^l_{ab}(t,s)  ghat^l_{b,i}(s)
                   - (memory term of M2.1)                                            (M4.1)

with the **forward response**, defined with the explicit `eta_0^{-1}` of
`00-method.md` M4 so that it is `O(1)`:

    A^l_{ab}(t,s) := eta_0^{-1} < d (P x^l)_a(t) / d r^l_b(s) >                        (M4.2)

and the transposed argument gives the backward response `B^l_{ab}(t,s)`. Same for
`W_V`, `W_1`, `W_2`, and for the V-path use of `W_O`.

**The response kernels are `N x N` in the node index, times `T x T` in time,
times `P x P` in graphs.** For a fixed graph of `N` nodes and a horizon of `T`
steps that is an `(NT)^2` object per block — the cost statement for any future
solver, and the reason none is attempted here.

---

## M4b. The attention channel — the piece with no precedent here

The branch output is `(1/(L gamma_A sqrt(D))) W_O (S V)`. Its dependence on the
Q/K weights runs entirely through `S = softmax(A)`, and `A` is a **bilinear order
parameter**:

    A^l_{uv,h}(t) = D_h^{-alpha_A} q^l_{u,h}(t) . k^l_{v,h}(t)
                  = D_h^{1 - alpha_A} * [ (1/D_h) sum_i q_{u,h,i} k_{v,h,i} ]          (M4.3)

The bracket is a **population average over the `D_h` width coordinates**. So its
behaviour in the limit is decided by `alpha_A`, and this is the sharpest
structural statement in the file:

- **`alpha_A = 1`.** `A^l_{uv,h}(t) = (1/D_h) sum_i q k -> < q_u k_v >`, a
  *deterministic* kernel by the LLN over width coordinates. **The attention
  matrix concentrates**: in the limit `S^l_h(t)` is a deterministic, graph-masked,
  row-stochastic `N x N` matrix, computable from the single-site `(q,k)` process.
  It carries no head index once the kernels collapse (below).
- **`alpha_A = 1/2`.** `A^l_{uv,h}(t) = D_h^{1/2}[(1/D_h) sum_i q k]` is a **CLT**
  sum, not an LLN sum: it converges to a *Gaussian field* with `Theta(1)`
  fluctuations that survive the width limit, indexed by `(u,v,h)`. The attention
  matrix stays random forever, and `S = softmax(A)` is a nonlinear function of a
  random variable.

Head collapse, following `03-attention.md` D6: at init the kernels
`Q_h = K_h = V_h = H^l` for every head, the responses are built from head-averaged
kernels only, and induction gives identical kernels at all later times. So **the
kernels collapse across heads at every `alpha_A`; whether the attention *matrix*
collapses depends on `alpha_A`** — deterministic and head-independent at
`alpha_A = 1`, a head-indexed Gaussian field at `alpha_A = 1/2`. `H` then enters
the limit only through `D_h = D/H` inside (M4.3), which is exactly why `H` is
absent from the V/O sector of `10` §2 and present in `sigma_QK` of `10` §3e.

The key/query single-site process, from M4 applied to the mutually-contracted
pair (this is `03-attention.md` D4, transplanted to a node index):

    k^l_{v,h}(t) = u^l_{K,v,h}(t) + sum_{v'} int ds C^{k,l}_{vv'}(t,s) q^l_{v',h}(s)
    q^l_{u,h}(t) = u^l_{Q,u,h}(t) + sum_{u'} int ds C^{q,l}_{uu'}(t,s) k^l_{u',h}(s)   (M4.4)

with `u_K, u_Q` Gaussian sources of covariance `H^l` (the stream kernel — the
population kernel of what `W_K`, `W_Q` multiply, per M3), and `C^k, C^q`
**carrying no head index**. The graph enters (M4.4) only through the node indices
of `C`, whose support is the attention sensitivity `c_{uv,h}` and hence the
graph mask.

---

## M5. Close

Unknowns, all now `(PN T) x (PN T)` matrices instead of `D`-dimensional fields:

| object | definition |
|---|---|
| `H^l_{ab}(t,s)` | stream kernel `(1/D) x^l_a(t).x^l_b(s)` |
| `K^l = P H^l P^T` | MPNN-smoothed stream kernel — **no new unknown** |
| `Kap^l = S^l H^l (S^l)^T` | attention-smoothed stream kernel — new *only* through `S` |
| `A^l_{uv}(t)`, `S^l = softmax_mask(A^l)` | attention kernel and matrix (deterministic at `alpha_A=1`) |
| `G^l_{ab}(t,s)` | backward kernel `<ghat ghat>` |
| `A^l, B^l` per reused edge | forward/backward responses (M4.2) |
| `C^{k,l}, C^{q,l}` | key/query response pair (M4.4) |
| `gamma_P^l, gamma_A^l` | **order parameters, not constants** — see below |

**`gamma_l` is an order parameter.** `10` §4 formula (G) reads, in kernel
language,

    (gamma_P^l)^2 = tr[ P H^l(t,t) P^T ] / tr[ H^l(t,t) ]                              (M5.1)
    (gamma_A^l)^2 = tr[ S^l H^l(t,t) (S^l)^T ] / tr[ H^l(t,t) ]

so it is a *functional of the solution*, at every layer and every time. The
paper's practical recipe — scan a single constant `gamma` (their Eqn 15) — is the
approximation `gamma_l(t) = gamma`. (M5.1) says exactly what is being approximated
and predicts the direction of the error: `H^l` off-diagonal entries in the node
index grow with `l` (oversmoothing), so `gamma_l` **rises with depth** toward the
row-sum value. Measured as P6 in round 011.

Closure: replace every population average by its single-site expectation, exactly
as `00-method.md` M5, with the node index carried along as a free index
throughout.

---

## M6. Boundaries

- **Input side.** `W^(0)` multiplies the data `X`, which has no backward channel
  (the features are not learned). So `A^0 = 0`. This is why the encoder sector is
  response-free and why the paper's Prop. 2 first-layer computation is a pure
  one-step algebra with no memory in it.
- **Output side.** `B^{L+1} = 0`: nothing sits above the decoder.
- **A third, graph-specific boundary.** On a graph with isolated nodes, the row
  of `P` (and of `S`) supported only on the self-loop makes that node's branch
  field depend on no other node. Those nodes decouple from the node-index
  structure entirely and reduce to the plain residual-MLP DMFT of
  `05-completep-dmft-sgd.md`. **That is the degenerate-case collapse this
  derivation must satisfy** (`00-method.md` checklist item 1), and it is the
  cheapest available check: set `P = I`, drop the attention branch, and the
  system must reproduce `05` verbatim.

---

## 7. What the DMFT says about the parameterisation

The exponents that make (M2.1) and (M4.1) `D`-free and `Theta(1/L)` are exactly
the table of `10` §5. That is the cross-check the two-method split exists for
(`derivations/README.md`): the heuristic route and the cavity route must agree on
the exponents, and here they do, **with one addition the heuristic route cannot
state**:

> The heuristic route says `sigma_QK = D_h^{1-alpha_A}` makes `Delta A = Theta(1)`.
> The cavity route says *what `A` is* in the limit: an LLN average at
> `alpha_A = 1` and a CLT field at `alpha_A = 1/2`. Only the second route
> distinguishes "the attention pattern is a deterministic function of the kernels"
> from "the attention pattern is a random field that never stops fluctuating".

Both are stable parameterisations. They are different *limits*.

---

## 8. Two structural predictions the heuristic route cannot make

**S1 — attention concentration.** At `alpha_A = 1`, the across-seed standard
deviation of a fixed attention entry `A_{uv,h}` at fixed data must fall as
`D_h^{-1/2}`. At `alpha_A = 1/2` it must be flat in `D_h`. This is the LLN-vs-CLT
statement of (M4.3), and it is the cleanest falsifiable content of this file.
*(Note: at `alpha_A=1` the mean is `Theta(D_h^{-1/2})` too, so the test must be
the standard deviation at fixed graph across weight seeds, not the magnitude.)*

**S2 — head collapse is `alpha_A`-conditional.** The across-head spread of the
attention *matrix* must vanish as `D_h -> inf` at `alpha_A = 1` (deterministic
limit, no head index) and must **not** vanish at `alpha_A = 1/2` (head-indexed
Gaussian field). This is a prediction where the two exponents give qualitatively
different answers, so it is a control that cannot fail to bite by being an
identity.

---

## 9. Where this derivation is most likely wrong

Ranked. Item 1 is the one I cannot close.

1. **`< softmax(A) > != softmax(<A>)`.** At `alpha_A = 1/2` the closure needs
   `E[S]` and `E[S H S^T]` for `A` a Gaussian field, and the naive closure
   (M5.1 with `S = softmax(<A>)`) drops a Jensen gap of order `Var(A)` — which is
   `Theta(1)`, not small. Every kernel downstream of `S` inherits the error. I do
   not have a closed form for the softmax of a correlated Gaussian field on a
   graph neighbourhood, and I am not going to pretend the naive closure is
   controlled. **At `alpha_A = 1` this problem disappears entirely** (the field is
   deterministic), which is a third, independent reason to prefer `alpha_A = 1`
   and is the reason §M4b is stated in that gauge.
2. **The double use of `W_O` (M3).** Two Onsager reactions from one frozen
   matrix, with a relative sign and a relative branch factor that I have written
   down but not checked against anything. `05-completep-dmft-sgd.md` got the
   structurally identical double-count wrong **twice**, and the two errors
   cancelled at the conventional exponent so that the formula "looked right at the
   exponent everyone checks". The conventional exponent here is `alpha_A = 1`.
   That is precisely the configuration in which a compensating pair of errors
   would be invisible.
3. **Equal-time response diagonals (F1).** I have not determined, from the update
   graph, which of `A^l`, `B^l`, `C^{k,l}`, `C^{q,l}` have nonzero `(t,t)` entries.
   The rule is set by *computation order*, not by nonlinearity: a field read
   before the backward pass has `A(t,t) = 0`; a drive that sees the same step's
   forward pass has `B(t,t) != 0`. The attention branch is the awkward case
   because `S` is computed in the forward pass and `c_{uv}` in the backward pass
   of the *same* step. Unresolved.
4. **`gamma_l` as an order parameter (M5.1) makes the system implicit** in a way
   the MLP/MoE cases are not: the branch normaliser depends on the kernel it
   normalises. Whether that fixed point is unique, or whether the F5-style
   stiffness of the `Delta`-loop is aggravated by it, is unknown.
5. **No node-direction self-averaging (§0).** Every `Theta(1)` I have attached to
   a graph factor is a fixed-graph statement. A theory of *ensembles* of graphs —
   which is what the paper's datasets actually are — would need a second average
   over the graph law, and the two averages do not obviously commute with the
   width limit.
6. **No solver.** Nothing above has been evaluated. By this program's own
   standard (`PROGRAM.md`), claims are certified by measurement, and the only
   measurements in round 011 are of the *exponents* and of S1/S2 — not of the
   dynamics. The honest one-line status is: **parameterisation cross-checked by
   two routes; limit structure derived; dynamics not validated.**
