# Review: "Information Mechanics" stub (Halmos & Hanin, draft of 2026-08-20)

**Method: hand derivation + independent numerics** (`12-information-mechanics-checks.py`,
1d spectral-accuracy grid, generic non-Gaussian densities). This file records
(A) which displayed equations were verified correct, (B) the errors found with
corrections, and (C) condensed application notes. Equation numbers refer to the
draft PDF's numbering.

Conventions used below (ℏ = m = 1 unless shown):

- `I_std(µ‖ρ₀) = ∫ ‖∇log µ − ∇log ρ₀‖² dµ`  (no 1/8 prefactor)
- Bohm quantum potential `Q_B[ρ] = −(ℏ²/2m) Δ√ρ/√ρ`
- The draft's page-1 convention: `I ≡ (1/8) I_std`.

## A. Verified correct

1. **Page-1 conservation identity** (von Renesse form). With `V = E₀ + (ℏ²/2m)Δ√ρ₀/√ρ₀`
   (nodeless ground state), the expansion of `⟨ψ|Ĥ|ψ⟩ − E₀` into
   `(1/2m)∫‖∇ϕ‖² dµ + (ℏ²/8m) I_std(µ‖ρ₀)` is exact. Key lemma checked
   numerically to 3e-6 relative error:
   `(1/4) I_std(µ‖ρ₀) = ∫‖∇√µ‖² + ∫ µ Δ√ρ₀/√ρ₀`.
2. **Hamiltonian system (6)–(7) conserves (8).** Also true for non-gradient `u`
   in the convective form, since `u·((u·∇)u) = (u·∇)(½‖u‖²)` pointwise; the
   *canonical* (symplectic, (µ,ϕ)) interpretation additionally needs `u = ∇ϕ`,
   which Madelung initialization preserves.
3. **Energy–dissipation equation (10)** for the gradient-flow branch (9): standard,
   correct as displayed.
4. **Damped energy balance (11)**: correct (see B7 for the `γ` vs `γ_t` nit).
5. **Example II.3 (KL), all four displays (14)–(17).** `∇δKL/δµ = ∇log µ − ∇log ρ₀`
   (checked to 2e-11); the conservative branch is isothermal Euler with drift
   `∇log ρ₀`; the dissipative branch is exactly Fokker–Planck
   `Δµ − ∇·(µ∇log ρ₀)` (checked to 1e-5).
6. **Score expansion** `Δ√ρ/√ρ = (1/4)‖∇log ρ‖² + (1/2)Δlog ρ`: correct (5e-5,
   finite-difference floor).
7. **JKO scheme (19) and the surrogate-force direction in (20)–(21)**: the sign is
   right — `(T_{µ→µ⁺} − id)/η ≈ −∇ δF/δµ` — modulo the notation issues in B4.

## B. Errors found (with corrections)

1. **Eq (18) [and hence (21)]: sign + factor-2 error in the quantum potential.**
   The draft defines `Q_µ = Δ√ρ/√ρ` and asserts `∇δI/δµ = ∇Q_µ − ∇Q_{ρ₀}`.
   With the draft's own `I = (1/8) I_std` (ℏ = m = 1), the true first variation is

   `δI/δµ = −(1/2)( Δ√µ/√µ − Δ√ρ₀/√ρ₀ ) = Q_B[µ] − Q_B[ρ₀]`,

   i.e. the identity holds **only with the Bohm-sign potential**
   `Q_B = −(1/2)Δ√ρ/√ρ`. Against the eq-(18) convention the stated formula is
   off by exactly **−2** (numerics: pairing ratio −1.99999 against a direct
   perturbation of `(1/8)I_std`). Fix: define `Q_µ = −(ℏ²/2m)Δ√µ/√µ` in (18)
   (this also matches Bohm [6] and makes Example II.2's "quantum pressure gap"
   literally correct), or keep (18)'s convention and write
   `∇δI/δµ = −(1/2)(∇Q_µ − ∇Q_{ρ₀})`. Underlying scalar lemma:
   `δI_std/δµ = −4Δ√µ/√µ` for the non-relative part.
2. **Prefactor convention for `I` is inconsistent across the draft.** Page 1
   absorbs the 1/8 into the definition (`I(µ‖ρ₀) = (1/8)∫‖∇log µ − ∇log ρ₀‖²µ`),
   but page 3 writes the physical realization as `(ρ₀, ℏ²/8m · I)` and the image
   case as `(ρ_image, 1/8 · I)` — which double-counts the 1/8 under the page-1
   definition. Pick `I_std` (no prefactor) once and carry `ℏ²/8m` explicitly.
3. **Stray `1/m` after setting m = ℏ = 1.** Eq (2) and Example II.2's (12)–(13)
   keep a `1/m` although the Hamiltonian on page 1 already set m = ℏ = 1. With
   constants restored the velocity equation is
   `Du/Dt = −(1/m) ∇ δ[(ℏ²/8m) I_std]/δµ`; writing `−(1/m)∇δI/δµ` is consistent
   only if `I` is *defined* as `(ℏ²/8m) I_std`. State the convention once.
4. **Eq (20)–(21) notation.** `(T_{µ→µ⁺} − i)/η`: the `i` must be `id` (identity
   map). Also `δF(ν‖ρ₀)/δµ(µ)` mixes the dummy `ν` and the argument `µ`; and the
   exact JKO optimality condition evaluates the variation at `µ⁺`
   (`T_{µ⁺→µ} = id + η ∇δF/δµ(µ⁺)`), so the displayed form is the `O(η)`
   explicit-in-`µ` approximation — worth saying, since the implicit evaluation
   point is exactly where the scheme's unconditional stability comes from.
5. **"Potentially requires fourth-order derivatives" (Sec II.B).** The quantum
   *force* `∇Q` needs third derivatives of `log ρ` (second derivatives of the
   score); it is the dissipative *PDE* (13) that is fourth-order in space
   (relative-DLSS type, hence the k⁴ stiffness of §D.1). As written the sentence
   conflates the two, and contradicts "third-order quantum force" two sentences
   earlier.
6. **Cross-references are systematically stale** (label collision, likely all
   pointing at a defunct Madelung-system label): page 1 "the rewriting (12)" and
   "the identification 12" → should be (1)–(2); "recovers (14) in the
   underdamped limit … and (16) in the overdamped limit" → should be (6)–(7)
   and (9); Example II.2 "exactly yields real-time quantum mechanics (12)" →
   (1)–(2); Sec II.B "first order integrator of the gradient flow (16)" → (9).
7. **Overdamped limit statement.** `γ_t ↑ ∞` at fixed time freezes the motion; the
   gradient flow (9) is recovered only after the time rescaling `t → γt`
   (equivalently on timescale `τ = t/γ`). Also (11) should read `γ_t` inside the
   integral if the damping is time-dependent.
8. **Missing standing hypotheses.** (i) `ρ₀ > 0` a.e. and `√ρ₀ ∈ H¹_loc` so that
   `V − E₀ = (ℏ²/2m)Δ√ρ₀/√ρ₀` and `Q_{ρ₀}` are defined (nodeless ground state);
   (ii) `µ_t > 0` — the Madelung/Schrödinger equivalence degrades at vacuum
   (nodes, quantized vortices); (iii) for the canonical structure, `u_t = ∇ϕ_t`.
9. **`W₂` listed as an admissible divergence** (Sec II.A discussion): it is not an
   information divergence in the f-divergence/local sense, and its first
   variation (the Kantorovich potential) fails the stated `C¹` smoothness
   requirement at densities with non-unique optimal maps. Either weaken
   Definition II.1's smoothness clause or drop `W₂` from the list.
10. **Editorial debris**: the note ": this will be much more startling and
    unnerving to the reader … let's recap the math for the score hamiltonian
    here and just show it" is still in the Section II.A body text. Typos:
    "one a probability density" → "once a"; "in solved the associated PDEs" →
    "in solving"; "a first variational that is C1" → "first variation";
    Def. II.1 "Given such a tuple, one a probability" sentence fragment;
    ref [1] has a mangled umlaut ("Zeitschrift f˘”r Physik").

## C. Application notes (condensed; discussion in the session record)

- **f-divergence ↔ barotropic-Euler dictionary.** For `F(µ) = ∫ f(µ/ρ₀) ρ₀`,
  `δF/δµ = f'(µ/ρ₀)`, and the conservative branch is relative barotropic Euler
  with pressure law `P(r) = r f'(r) − f(r)`, `r = µ/ρ₀`: KL → isothermal
  (`P = r`), χ² → polytropic γ=2 (shallow-water). Consequence: generic
  f-divergence conservative mechanics forms shocks; **relative Fisher is
  distinguished** — dispersive (quantum pressure), linearizes via Madelung to
  Schrödinger, hence globally well-posed where the f-divergence flows are not.
  Theorem-shaped selling point for choosing Fisher.
- **§D.1 rates.** Linearization of the dissipative Fisher flow at `ρ₀` is
  governed by the *square* of the Witten/Schrödinger generator: rate λ² vs λ
  for Fokker–Planck (k⁴ vs k²). Fast for sharp/high-frequency error, slow for
  global mass transport when λ < 1; the damped (inertial) branch at critical
  damping recovers ~λ. Position against the DLSS literature
  (Jordan–Kinderlehrer–Otto; Gianazza–Savaré–Toscani, who identify DLSS as the
  Wasserstein gradient flow of Fisher information; Jüngel–Matthes) — the
  ρ₀-relative version appears to be the new object.
- **Positioning to add**: Chow–Li–Zhou Wasserstein Hamiltonian flows;
  Khesin–Misiołek–Modin Madelung geometry; Lafferty's density manifold;
  Ambrosio–Gangbo; critically-damped Langevin diffusion (Dockhorn et al.) as the
  particle-level analog of the damped KL branch; Boffi–Vanden-Eijnden
  score-based transport (online score maintenance = the practical bottleneck);
  action matching (Neklyudov et al.) as the natural (µ,ϕ) parameterization.
- **ℏ as a mode-hopping knob**: conservative Fisher mechanics at scale ℏ is
  Schrödinger dynamics in `V = E₀ + (ℏ²/2m)Δ√ρ₀/√ρ₀`; real-time tunneling gives
  metastable-mode transitions with different (non-Arrhenius) scaling than
  Langevin — candidate advantage for the "Copenhagen sampler", and the honest
  headline experiment.
- **First experiments** (cheap, decisive): 1d/2d Gaussian mixtures —
  (i) conservative Fisher flow vs split-step Schrödinger (exactness check);
  (ii) JKO-surrogate force error vs η and particle count; (iii) measured k² vs
  k⁴ relaxation; (iv) two-mode tunneling time vs Langevin escape vs ℏ. Then a
  low-d latent of a pretrained diffusion model for the "video from images"
  claim, scored against latent-interpolation baselines.

## Status

Reviewed the 5-page stub only; supplementary material was not present in the
PDF. The two Examples and the page-1 identity are sound once B1–B3 are fixed;
nothing found threatens the framework itself.
