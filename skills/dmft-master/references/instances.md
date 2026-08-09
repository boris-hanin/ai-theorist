# Instance traces

> RECONSTRUCTED from the program record; pending re-validation. Each
> validated computation of the program, written as a run of the master
> algorithm (step-by-step classification). Quantitative details beyond
> those below lived in the lost per-round reports.

## 1. Deep MLP (Bordelon–Pehlevan, arXiv 2205.09653)

- Step 0: standard μP table; γ = γ₀√N; LR γ²-scaled.
- Step 1: input/hidden matrices class (b) per layer (forward χ, backward
  ξ, responses Aˡ, Bˡ).  The equal-time diagonal has measured `1/dt`
  structure, while round 003 falsified a unit endpoint weight in the tested
  L=2 causal sum; derive the endpoint from update order (F1/F1b). Readout is
  class (c).
- Steps 5–7: alternating MC fixed point with damping; response noise
  rectification fixed by slow response damping (F6); Δ-map stiffness
  needs inner damped loop (F5); exact discrete-time correlator
  predictions with control variates replaced Euler-marched curves (F4).
- Step 8: exact limiting equations reproduced; solver vs finite sims
  across widths; lazy anchor.

## 2. Depth-μP ResNet (arXiv 2309.16620)

- Step 0: branch scale 1/√L pinned; block LRs need no extra depth
  exponent; depth-audit protocol (loss-drop flatness, per-block movement
  slope −1/2).
- Step 1: one class-(b) pair per block (multi-block sandwich).
- Step 8: depthwise HP transfer + depth-collapse of loss curves;
  F14: a fetched Eq. 7 transcription was 3.3× off — resolved by
  independent derivation + sims (sims are ground truth).

## 3. Multi-head attention (arXiv 2405.15712)

- Step 0: PER-AXIS audit (model width, heads, d_k) — exponents pinned
  independently per axis.
- Step 1: q/k logits class (d) bilinear order parameters through the
  softmax; value/output paths as MLP-like edges.
- Step 8: pre-registered ΔA = Θ(N^{−1/2})-freezing claim FALSIFIED by
  measurement and corrected to concentration (F3 dichotomy: per-entry
  movement vs order-parameter concentration are different mechanisms —
  test the discriminating observable).

## 4. Mixture-of-Experts (Jiang–Bordelon–Pehlevan–Hanin, arXiv 2601.20205)

- Step 0: routing/sparsity audit exposed a reconstructed bookkeeping error:
  `W_up` is blind to `alpha_ffn`; the width factor belongs to `W_down`'s init
  and rate (F2/F14).
- Step 2: router-conditioned expert populations (quenched conditioning —
  match theory and sims, F11).
- Step 8: MoE Table 1 was re-derived and measured in round 006; several
  collapse/control bars remained inconclusive or failed and are not promoted
  to certification.

## 5. Hyperbolic Busemann networks (novel; rounds 1–3)

- Round 1 (L=1): Step 0 audit revealed the naive limit is DEGENERATE ⇒
  invented centered hyperbolic μP (analytic radial centering + √m
  amplification); Step 6 ladder ⇒ exact McKean–Vlasov, no responses;
  F15 (1/γ₀-amplified MC noise, antithetic fix) discovered.
- Round 2 (L=2): exact identity h = −√m log(1 − 2z/(√m s ĉ)) ⇒ deep
  layers asymptotically linear with hyperbolic gain R(q̄); class-(b)
  response sector validated (loss gaps ≤0.31%, order params ≤1.4% at
  m=8192); deep-matrix exponent pW=0 pinned by feature velocity.
- Round 3: INVENTED horospherical residual connection (stream in the
  pre-normalizer a-space where Busemann pairings linearize; α/√L);
  multi-block solver; depth+width transfer; F16 (correlated Sobol
  streams) and F17 (response-row write-order race; shared-vs-independent
  backward-matrix theory-sim as the diagnostic instrument) discovered.

Zero-shot validation of the master algorithm: the hyperbolic rounds were
executed AFTER the synthesis, from the algorithm alone.
