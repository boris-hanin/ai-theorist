# The windtunnel: extending the program past pre-training

Status: **SKETCH — nothing below is derived, measured, or certified.** This
document is a design for Stages II and III of a full-lifecycle windtunnel. It
proposes dials, order parameters, exactly-solvable reductions, certification
bars, and rounds. Every claim in it is a *target*, written before any
derivation, in the same spirit as a preregistration: if the program later
falsifies a mechanism proposed here, this file does not get quietly edited —
the round's `results.md` records the correction.

## 0. What "windtunnel" means, generalized from what this repo already does

The pre-training program in this repo is, abstractly:

1. **Dials.** Identify the scale dials of the system (width `D`, depth `L`,
   heads `H`, experts, sparsity).
2. **Units.** Find the parameterisation — the exponents on init scales and
   learning rates per parameter group — in which the training dynamics have a
   well-defined limit as the dials grow (μP / CompleteP / DMFT closure).
3. **Certification.** Prove the limit is the right one by the standard bar:
   derivation + independent numerics + matched finite-size simulation, with
   preregistration, MC floors beside every gap (F8), paired seeds (F10/F20),
   and controls that bite (F17).
4. **Payoff.** In the right units, decisions made small transfer big: tune
   hyperparameters in the scaled-down system, deploy at scale, with a
   *measured drift* and a *stated horizon of validity* (round 010's lesson:
   every transfer claim is horizon-limited until measured otherwise).

Steps 1–3 are the windtunnel's calibration; step 4 is the windtunnel. The
generalization to post-training and inference is the same four moves with new
dials, new units, and — critically — new *interfaces* between stages (§3).

What pre-training already provides, honestly stated: L=1 certified; deep-linear
and nonlinear P=1 response solvers implemented; MoE and attention scaling
subclaims measured; the graph round FAILED its bar; long-horizon transfer
drift is systematic (round 010). The paradigm works but is not finished, and
the additions below inherit its open liabilities — especially horizon
dependence, which is *worse* in RL, not better.

## I. Stage II: RL-based post-training

### I.1 What is structurally new

The fixed-dataset DMFT here rests on a fixed data kernel `Kx` over P points.
RL post-training breaks exactly that: **the data distribution is the policy's
own output distribution and moves with training.** The disorder average
acquires a self-consistent data average. This is the single deepest new
derivation problem, and it should be named up front the way "the node index is
a data index" is named in `dmft-graph`:

> **The prompt index is a data index; the rollout index is a sample index; and
> the rollout *distribution* is an order parameter.**

Three new axes, none of which exist in pre-training:

| axis | symbol | role | pre-training analog |
|---|---|---|---|
| rollouts per prompt (group size) | `G` | sample axis: advantage-estimator noise ~ `1/sqrt(G)` | solver population `S` |
| generation horizon (tokens) | `T` | memory axis: credit assignment smears one scalar reward over `T` log-prob gradients | time/response kernels |
| off-policyness (PPO epochs, replay lag) | `k` | the updating policy differs from the sampling policy — an Onsager-type correction | response functions Ā/B̄ |

And three new scale factors that the Step 0 audit must pin before anything is
derived:

- **reward scale** and **advantage normalisation**. GRPO's group-standardised
  advantage silently divides the effective learning rate by the reward's
  within-group sd. This is F12 (rate-amplified metrics) wearing a new coat:
  any η-transfer claim that does not first pin the advantage's scale in the
  dials is confounded. Prediction to test: with group-standardisation ON,
  η-transfer across reward scales is trivially flat (the control must NOT
  bite); with it OFF, transfer breaks at a derivable exponent.
- **KL coefficient β.** Its natural units are per-token nats. A β tuned at
  horizon `T` and reused at `T' > T` changes the total KL budget; the
  parameterisation question is whether β or β·T is the transferable object.
- **softmax logit scale at the readout.** Round 011's central finding — the
  attention logits receive Θ(1) updates only at `alpha_A = 1/2`, and the
  update has a channel carried by the block's *input* moving (F23) — transfers
  verbatim: the policy's action distribution is a softmax over logits, and
  "which parameterisation gives the *sampling distribution* Θ(1) updates per
  step" is the same question asked of the output head. The F23 discipline
  (enumerate every input to the logit before counting `Delta logit`) is
  mandatory here, because in an LLM the logit moves through the embedding,
  every block, and the unembedding simultaneously.

### I.2 Units: the μP-for-policy-gradient conjecture

Target derivation (`derivations/12-policy-gradient-mup.md`): a policy-gradient
update is a weighted log-likelihood update — SGD with the per-sequence
advantage `A_i` playing the role of the error signal `Delta`. Conjecture:

> In μP units, with advantages normalised to Θ(1) and per-token logit updates
> Θ(1), the optimum of (η, β, clip ε, sampling temperature τ) is flat in model
> width, and the DMFT limit of RL fine-tuning exists with the *policy output
> distribution* entering as a self-consistent order parameter.

The width sector of this is cheap to test (the harness of round 011 E4/E10
generalises: paired seeds, drift in decades, power audit). The genuinely new
derivation content is (a) the moving data distribution and (b) the `T` axis.

### I.3 The ladder of exactly-solvable reductions

The certification policy requires exact reductions. Candidates, cheapest
first — each collapses onto something already certified when a dial
degenerates (re-validation requirement 3):

1. **Softmax bandit under policy-gradient flow.** No network, `T = 1`, K arms.
   Exact ODEs (replicator-type dynamics). Gives ground truth for: advantage
   baselines, group-size `G` floors (finite-`G` dynamics vs the `G → ∞`
   deterministic limit — the MC-floor story of F8/F15 replayed with `G` in
   place of `S`), entropy collapse rates, and the F24 bimodality under
   sign-like updates.
2. **Linear-softmax contextual bandit.** Adds the data kernel; the frozen
   part is the existing L=1 machinery, so this is the degenerate-case collapse
   onto `dmft-derivation`.
3. **KL-regularised reward maximisation with Gaussian rewards.** Exact tilted
   distribution `p*(x) ∝ p₀(x) exp(r(x)/β)`; connects RL and best-of-N through
   the shared KL budget (§II.2) and gives closed-form
   overoptimisation curves to certify the harness against.
4. **GRPO on a two-armed bandit.** Exact finite-`G` update distribution;
   mutation target: reintroduce a length-biased reward and confirm the
   harness catches it (the mutation-test requirement, re-validation
   requirement 6).
5. **Tiny transformer + verifiable synthetic task** (arithmetic; graph tasks —
   `skills/dmft-graph/scripts/gt.py` already exists and its static sector is
   the sharply-confirmed part of round 011). First system where `T > 1` and
   the moving-distribution effect is live.

### I.4 What the RL windtunnel certifies, and its controls

The transfer claim, stated as a round-011-style scoreboard:

- **P-RL1**: optimum `η` (in μP units) flat across width `D` — SGD-style and
  Adam-style legs separately (the Adam leg was the unresolved one in 011;
  don't inherit that debt silently).
- **P-RL2**: optimum `η` flat across group size `G` once advantages are
  normalised; control: normalisation OFF must break it at the derived
  exponent (a control that must bite).
- **P-RL3**: optimum `(η, β)` flat across horizon `T` in the derived units —
  this is the round-010 lesson made a first-class preregistered axis, with
  drift-vs-horizon reported as a curve, never a single short-horizon pass.
- **P-RL4**: reward-model overoptimisation onset (proxy reward up, gold reward
  down) occurs at a KL budget predictable from small-scale measurement of the
  reward model's error distribution. This is reward hacking recast as a
  windtunnel claim: *the failure mode itself must scale predictably, or the
  windtunnel is unsafe to use.* If P-RL4 fails, that failure is the headline
  result, not a footnote.

New failure modes to expect (register with the F-schema only when actually
bitten; listed here as watch-items, not entries): advantage-normalisation as a
hidden learning rate (F12 pattern); length-biased reward confounding every
transfer sweep (F23 pattern — enumerate the channels into the reward);
"best reward during training" statistics inheriting F24's bimodality;
entropy-collapse making late-horizon dynamics non-exchangeable across seeds
(F10 pattern, but now the *policy* distribution collapses, not just the seed
noise); KL-estimator bias (k1 vs k3 estimators differ at exactly the KL scales
the windtunnel cares about).

## II. Stage III: inference-time compute scaling

### II.1 The central order parameter: the difficulty kernel

For a task distribution and a fixed model, define ρ(p): the distribution over
problems of the per-sample success probability `p` (one draw of the policy at
temperature τ solves the problem with probability `p`). Almost every
inference-scaling observable is a functional of ρ:

- **pass@k** = E_ρ[1 − (1−p)^k] — exact, no model of the transformer needed.
- **majority vote at N** = E_ρ[P(Binom(N, p) majority correct)] — exact;
  its ceiling is E_ρ[1{p > 1/2}] plus tie terms: problems with `p < 1/2` are
  a *frozen sector* that no amount of voting solves.
- **best-of-N under a verifier** = a functional of ρ *and* the verifier's ROC
  (per-problem false-positive/false-negative rates on sampled answers).

This is the same move the repo makes everywhere: find the population whose
exchangeable average the observable is (Step 2), and the observable becomes a
low-dimensional functional of a measurable kernel. The windtunnel content:
**measure ρ at small N (it needs only per-problem sample means), predict the
entire scaling curve at large N, then verify.** The estimator of ρ from k
samples per problem is itself noisy (a Beta-binomial deconvolution); its MC
floor in (problems, samples) must be derived and sample-halved (F8) before any
predicted curve is trusted.

### II.2 Best-of-N and the KL bridge

BoN has sharp math waiting to be certified rather than invented:

- KL(BoN‖base) = log N − (N−1)/N, exactly, independent of the distribution.
- BoN performance vs N is extreme-value theory on the reward distribution's
  upper tail: the tail class (measurable at small N) determines the large-N
  curve and its saturation.
- Under an imperfect verifier, BoN *degrades* past a critical N: selecting the
  max over more samples selects for verifier false positives. Verifier hacking
  at inference and reward hacking in training (P-RL4) are the same phenomenon
  spent from the same budget — KL from the base policy — and the shared unit
  is what lets the two windtunnels talk to each other (published empirical
  forms for gold reward vs √KL under BoN and RL exist and become
  certification targets, not assumptions).

Bars: predicted-vs-measured BoN curve at N beyond the fit range; control that
bites: swap in a verifier with a known injected false-positive rate and
confirm the predicted degradation point moves as derived (this doubles as the
mutation test for the whole Stage III harness).

### II.3 CoT length and agentic horizon

The two sequential axes, in increasing order of ambition:

- **Thinking budget `T_think`.** Model the per-problem solve event as a hazard
  rate over thinking tokens; ρ generalises to a kernel over (problem, budget).
  Windtunnel claims: (a) the marginal value of thinking tokens is predictable
  from truncated small-budget runs; (b) optimal budget in scaled units
  transfers across model width; (c) sequential (long CoT) vs parallel (more
  samples) compute allocation has a predictable optimum given ρ and the
  hazard curves. (c) is the practically valuable one — it prices sequential
  against parallel compute inside one framework.
- **Agentic horizon `K` (steps/tool calls).** Null model: a Markov chain with
  per-step success/recovery/derail probabilities measured on short horizons;
  long-horizon success compounds them. The preregistered control that must
  bite: error *autocorrelation*. If failures were independent, success decays
  geometrically; measured deviation from geometric is the signature that the
  iid null is wrong (F17 discipline: an ablation that changes nothing is a red
  flag). The honest expectation is that the null FAILS — errors are
  correlated — and the round's value is measuring the correlation structure
  that a real theory must reproduce. Design the round so failure is
  informative (round 011 is the template: the failed bar plus E8-style
  follow-ups was worth more than the passes).

### II.4 Stage coupling: the interfaces

The full-lifecycle windtunnel is not three windtunnels side by side; it is
three stages passing typed interfaces:

```
pre-training ──(μP units; loss/data scaling)──▶ base model
post-training ──(Δρ(p): how RL moves the difficulty kernel; KL spent)──▶ policy
inference ──(ρ(p) + verifier ROC + budget split)──▶ deployed performance
```

The known tension to preregister early: RLVR-style post-training sharpens
pass@1 but can *shrink* pass@k diversity — post-training spends the same
distributional budget that inference-time sampling wants to consume. In
interface terms: RL moves mass of ρ(p) rightward for some problems while
collapsing the exploration that gives other problems their nonzero `p`.
A windtunnel that certifies each stage separately but not the interface will
recommend post-training recipes that destroy inference-time scaling. The
composed claim — tune the RL recipe small, predict the *inference scaling
curve* of the post-trained model big — is the program's Stage IV analog, and
it should be attempted only after the per-stage bars hold.

## III. Repository additions

Layout, mirroring the existing structure exactly:

- `skills/rl-post-training/` — SKILL.md + `scripts/`: bandit/GRPO exact
  solvers, tiny-LM RLVR harness, transfer sweep runner (generalising
  `experiments.py` E4/E10: paired seeds, drift decades, power audit).
- `skills/inference-scaling/` — SKILL.md + `scripts/`: ρ-kernel estimator with
  its floor, BoN/vote/EVT predictors, verifier-ROC harness, hazard fits.
- `derivations/12-policy-gradient-mup.md`, `13-grpo-mean-field.md`,
  `14-best-of-n-evt.md`, `15-vote-difficulty-kernel.md`,
  `16-agentic-hazard.md`.
- `rounds/012+` — one round per certification, prereg committed before
  measurement, per the existing lifecycle. No new registry: failure modes
  continue F25+ in `registry/failure-modes.md`.

Infrastructure that does not yet exist in this repo and is the real cost:

1. **A model ladder.** Everything so far runs on numpy/torch toy nets.
   Stages II–III need small LMs. Ladder: bandit (no net) → linear-softmax →
   tiny transformer on synthetic verifiable tasks (reuse the graph-task
   generators) → smallest pretrained LM that exhibits the phenomenon. The
   windtunnel philosophy applied to itself: certify each rung against the rung
   below before climbing.
2. **A task + verifier suite** with *known* ground truth and *injectable*
   verifier error — required for the mutation tests.
3. **Seed discipline extended to sampling.** Paired comparisons now need
   shared prompt sets, shared init, and common random numbers through the
   rollout sampler; F20's paired-floor rule applies to every BoN/vote/transfer
   comparison, and temperature is a new axis that every prereg must pin.
4. **Provenance under GPU economics.** Round 010 lost its v2 source once.
   RL runs are longer and more expensive; the "commit runner code before
   launching" rule needs a pre-launch CI check, not prose.

## IV. Sequencing

Recommended order, by (information gained)/(cost):

1. **Round 012 — vote/pass@k windtunnel** (`ρ`-kernel → predicted curves,
   tiny LM or even a fixed public model). Cheapest; pure measurement +
   closed-form functionals; exercises the entire new harness (task suite,
   sampling seeds, floors) with no RL and no new dynamics theory. The
   mutation test (injected verifier error) comes free.
2. **Round 013 — bandit/GRPO exact ladder** (§I.3 items 1–4). Certifies the
   RL solver machinery against exact reductions before any transformer is
   touched; measures the `G` floor and the F24 bimodality in a controlled
   setting.
3. **Round 014 — μP transfer for RL fine-tuning** on the tiny-transformer
   task (§I.4 P-RL1–P-RL3), inheriting the 011 sweep harness.
4. **Round 015 — overoptimisation onset** (P-RL4 + BoN degradation, the KL
   bridge), the first stage-coupling round.

Rounds 012–013 could realistically be attempted with the repo's current
compute posture; 014–015 need the model ladder's upper rungs.
