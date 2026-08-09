# Round 011 — graph-transformer parameterisation and HP transfer (arXiv 2607.05017)

**Committed before any measurement.** Derivations:
`derivations/10-graph-transformer.md` (heuristic one-step),
`derivations/11-graph-transformer-dmft.md` (cavity).
Model: `skills/dmft-graph/scripts/gt.py`. Runner:
`skills/dmft-graph/scripts/experiments.py`.

## Scope, and every deviation from the paper

The paper is DeZoort & Hanin, *Hyperparameter Transfer in Graph Neural Networks*
(2607.05017, 6 Jul 2026), obtained as PDF and extracted with `pypdf` — never a
web-fetch summary (F14).

**The paper contains no graph transformer.** Its §2.2 permits an MHSA module in
the residual block and its §4 names attention as an explicit limitation
("Extending this analysis to these additional architectural features is an
important next step"). Every equation, proposition and experiment it runs uses
MPNN + MLP only. So this round tests:

- the paper's own rows of Table 1, re-derived by a different route (a
  confirmation, not a discovery); and
- **the attention sector, which is new** — `W_V, W_O, W_Q, W_K`, the exponent
  `alpha_A`, and the `gamma_A` normalisation of the attention branch.

### FIDELITY — what differs from the paper's setup

| paper | here | why |
|---|---|---|
| MNIST-Superpixels, PascalVOC-SP, QM9, Cora/Citeseer/PubMed | synthetic random-geometric graphs, `N = 24`, `B = 12`, teacher-generated scalar targets | CPU only, float64, and these are *scaling* tests, not performance tests |
| `D` up to 1024, `L` up to 8, 400–800 epochs | `D` 32–512, `L` 2–8, 12–24 full-batch steps | same reason |
| Adam with `eps = 1e-14` | **signGD** as the Adam proxy (this repo's convention, `derivations/06-moe.md` §1), with `adam` available | signGD is what the derivation assumes; Adam adds a moment transient that is not the object under test |
| real sparse bag-of-words features | a `sparsity` dial on synthetic features | E6 only needs the *mechanism* of `C_ab`, not their numbers |
| graph classification / node classification / regression | graph-level scalar regression | one task type; see "coverage" below |
| no `gamma` on the attention branch (it has none) | `gamma_A = 1` by default, and measured | the derived claim is that this is *correct*, §4 of derivation 10 |

**Coverage bound, stated up front:** one task type, one graph family, one degree
regime, `H <= 8`, `L <= 8`, `D <= 512`, no real datasets, no edge features, no
positional encodings, no LayerNorm. Nothing here tests the paper's Fig 2/3/6
performance claims, its citation-network `C_ab` experiment, or its AdamW
`tau_epoch` result. Those are **not attempted**.

## What would make this round a failure

Any of P1–P7 failing against its bar, **or any control failing to bite** (F17: a
control that changes nothing is a red flag, not a pass), **or** a control turning
out to be an identity that I did not flag in advance.

## The identity-control hazard, declared in advance

`derivations/10` §3f: `sigma_QK` and the Q/K learning-rate correction are the
**same knob**. `q = (1/(sigma_QK sqrt(D))) W_Q x` with `W_Q ~ N(0, sigma_QK^2)` is
distributionally identical to `sigma_QK = 1`, so `sigma_QK` only moves the needed
learning rate. Therefore:

> **At `alpha_A = 1` the "derived" and "qk-global" parameterisations are the
> SAME RUN, exactly.** A width sweep at `alpha_A = 1` cannot discriminate them
> and will not be reported as if it could.

The discriminating comparison is run at `alpha_A = 1/2`, where they differ by
`D_h`. `gt.GraphTransformer.qk_correction_is_identity()` is asserted against in
`skills/dmft-graph/scripts/experiments.py` before that leg runs. Two further
identity hazards are guarded in `skills/dmft-graph/scripts/gt.py`: a
degree-regular graph makes "degree-normalised" and
"constant-gamma" the same operator (`_assert_graph_ok` refuses), and `D_h = 1`
makes every `D_h^k` correction equal 1 (the constructor refuses).

## Floors

- Slopes: reported with the least-squares s.e. of the fit.
- Transfer verdicts: `transfer.py::verdict`, **imported unmodified**. Three bars
  (F22): statistical resolution from the across-seed scatter of `log10 lr*`, a
  0.3-decade practical bar, and `tail_drift <= 1.3 x head_drift` for the SHAPE.
  A sub-threshold drift that is *not settling* returns SUSPECT, never a pass.
- Dial sweeps share seeds and data, so differences between dial values are
  common-random-number estimates and the **paired** floor applies, not the
  seed-to-seed spread (F20). Where a paired floor is quoted it says so.

## Predictions and bars

| # | prediction | quantitative bar |
|---|---|---|
| **P1** | feature RMS and each branch's `Delta x` RMS are flat in `D` and `H` | slope within `0.10` of `0` |
| **P1b** | each branch's contribution scales as `1/L` | slope within `0.15` of `-1` |
| **P2** | `gamma_A` is flat in `D` at `alpha_A in {1/2, 1}` | `\|slope\| <= 0.05` |
| **P2c** | **control**: at `alpha_A = 0`, `d_eff -> 1` and `gamma_A` rises with `D` | `d_eff` slope `<= -0.15`; `gamma_A` slope `>= +0.05` |
| **P3** | optimal `eta_0` transfers across `D`, `L`, `H` for SGD and signGD at `alpha_A = 1` | `verdict` returns TRANSFERS on all six legs |
| **P3c1** | **control**: `alpha_A = 0` breaks width transfer | FAILS or SUSPECT |
| **P3c2** | **control**: dropping the Q/K correction at `alpha_A = 1/2` breaks width transfer, while keeping it does not | drift(qk-global) `>= 2x` drift(derived), and derived TRANSFERS |
| **P3c3** | **control**: unnormalised `P = A` breaks width transfer (the paper's own §2.5 finding) | FAILS or SUSPECT |
| **P3c4** | **control**: an SGD rate without the `D` factor breaks width transfer | FAILS |
| **P4** | `Delta A\|_{t=1} ~ D_h^{-1/2}` under SGD, `~ D_h^{0}` under signGD, at `alpha_A = 1` | SGD slope within `0.15` of `-0.5`; signGD within `0.15` of `0` |
| **P4b** | `Delta A\|_{t=8}` is closer to `D_h^0` than `Delta A\|_{t=1}` under SGD | `\|slope(t=8)\| < \|slope(t=1)\|` |
| **P5** | `A_init ~ D_h^{1/2 - alpha_A}` | slope within `0.10` of `-0.5` (`alpha_A=1`) and `0.0` (`alpha_A=1/2`) |
| **P6** | neighbour correlation `rho_l` rises with `l`, and `gamma_P^l` rises with it, per formula (G) of `10` §4 | `rho_8 > rho_1`; `gamma_P` monotone increasing over the last 4 layers |
| **P7** | the node-level (attention) alignment factor stays near 1 where the pooled `C_ab` blows up with feature sparsity | `C_ab(sparsity 0.97)/C_ab(0) >= 3` while `C_node(0.97)/C_node(0) <= 1.5` |
| **S1** | across-seed sd of a fixed attention logit `~ D_h^{-1/2}` at `alpha_A = 1`, flat at `alpha_A = 1/2` | slopes within `0.10` of `-0.5` and `0.0` |
| **S2** | across-head sd of the attention matrix vanishes with `D_h` at `alpha_A = 1`, not at `alpha_A = 1/2` | same bars as S1 |

## Declared in advance: which prediction I most expect to fail, and why

**P4 (SGD), and it is not close.** `derivations/10` §7 item 2 lays out the
problem: my counting gives `Delta A|_{t=1} = Theta(D_h^{-1/2})`, one factor of
`sqrt(D_h)` *less* suppression than the `(Delta A)^2 ~ N^{-2}` that
`derivations/03-attention.md` §D2b records from Bordelon et al.'s Fig 12(b). The
same contraction — the backward path through `W_O` — is the one this program
already got wrong once, in the *other* direction, and the correction was found
only by reading a figure caption. The program's standing prior is that the error
is mine. If P4 returns `-1` rather than `-0.5`, the first suspect is my C2
labelling, not the measurement, and the consequence propagates: it would move
`sigma_QK` and therefore the headline "at `alpha_A = 1` the Q/K matrices need no
correction at all".

**Second most likely: P6.** `gamma_l` is an order parameter (`11` §M5.1) and its
depth trend is a *dynamical* claim about oversmoothing, made from a static
formula. At `L = 8` with `1/L` branches the stream barely moves, so `rho_l` may
simply not have room to rise, and the prediction would be untestable at this
depth rather than false. That distinction will be reported explicitly.

**P3c2 is the load-bearing control of the whole round.** P3's collapse results
are null results: they mean nothing unless something fails to transfer. There are
four controls precisely because the program has been burned by controls that did
not bite. If *none* of P3c1–P3c4 bites, P3 will be reported as UNDER-POWERED, not
as a confirmation.

## Not attempted, and named as such

- No DMFT **solver**. `derivations/11` derives the single-site system; nothing
  evaluates it. No theory-vs-simulation comparison of the dynamics exists, and
  no MC floor is quoted for one, because there is nothing to floor.
- No test of the paper's `C_ab` numbers on real citation networks.
- No AdamW / weight-decay leg. `lambda = lambda_0 sqrt(D)` is derived to be
  unchanged by the attention sector (all groups sit at `eta_0/sqrt(D)`), and that
  is an argument, not a measurement.
- No real Adam (only signGD as its proxy) in the transfer legs.
- No `alpha_A` between `1/2` and `1`, and no `alpha_A > 1`.
