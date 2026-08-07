# DMFT closed equation systems

Complete systems from Bordelon–Pehlevan arXiv:2205.09653 (conventions of the
NeurIPS 2022 / v3 paper), plus the depth-μP delta from arXiv:2309.16620.
All formulas use LaTeX. Verify equation-by-equation against the source PDFs
before high-stakes use; this file was compiled from a structured extraction.

## 0. Setup and conventions

Depth-$L$ MLP, width $N$, inputs $\bm x_\mu \in \mathbb{R}^D$, $\mu = 1..P$:

$$f_\mu=\frac{1}{\gamma\sqrt N}\,\bm w^L\cdot\phi(\bm h^L_\mu),\qquad
\bm h^{\ell+1}_\mu=\frac{1}{\sqrt N}\bm W^\ell\phi(\bm h^\ell_\mu),\qquad
\bm h^1_\mu=\frac{1}{\sqrt D}\bm W^0\bm x_\mu$$

Init: all weight entries i.i.d. $\mathcal N(0,1)$. Richness: $\gamma=\gamma_0\sqrt N$;
feature-learning limit $N\to\infty$ with $\gamma_0=O(1)$; $\gamma_0\to0$ lazy.
Dynamics: $\dot{\bm\theta}=-\gamma^2\nabla_{\bm\theta}\mathcal L$,
$\mathcal L=\sum_{\mu=1}^P\ell(f_\mu,y_\mu)$.
Error signal: $\Delta_\mu(t)=-\partial\ell/\partial f_\mu$ ($=y_\mu-f_\mu$ for
$\ell=\frac12(y-f)^2$).
Backprop fields: $\bm g^\ell_\mu=\gamma\sqrt N\,\partial f_\mu/\partial\bm h^\ell_\mu
=\dot\phi(\bm h^\ell_\mu)\odot\bm z^\ell_\mu$, $\bm z^\ell_\mu=\frac1{\sqrt N}\bm W^{\ell\top}\bm g^{\ell+1}_\mu$.

Order parameters:
$$\Phi^\ell_{\mu\alpha}(t,s)=\tfrac1N\phi(\bm h^\ell_\mu(t))\cdot\phi(\bm h^\ell_\alpha(s)),\quad
G^\ell_{\mu\alpha}(t,s)=\tfrac1N\bm g^\ell_\mu(t)\cdot\bm g^\ell_\alpha(s)$$
$$K^{NTK}_{\mu\alpha}(t,s)=\sum_{\ell=0}^{L}G^{\ell+1}_{\mu\alpha}(t,s)\,\Phi^\ell_{\mu\alpha}(t,s),
\qquad \Phi^0=K^x=\tfrac1D XX^\top\ \text{(static)},\quad G^{L+1}\equiv1.$$

## 1. General deep case ($N\to\infty$, fixed $P,t$)

Single-site fields $h^\ell_\mu(t),z^\ell_\mu(t)$, Gaussian sources
$$\{u^\ell_\mu(t)\}\sim\mathcal{GP}(0,\bm\Phi^{\ell-1}),\qquad
\{r^\ell_\mu(t)\}\sim\mathcal{GP}(0,\bm G^{\ell+1}).$$

Forward / backward single-site processes:
$$h^\ell_\mu(t)=u^\ell_\mu(t)+\gamma_0\int_0^t ds\sum_{\alpha=1}^P
\big[A^{\ell-1}_{\mu\alpha}(t,s)+\Delta_\alpha(s)\Phi^{\ell-1}_{\mu\alpha}(t,s)\big]
\,z^\ell_\alpha(s)\,\dot\phi(h^\ell_\alpha(s))$$
$$z^\ell_\mu(t)=r^\ell_\mu(t)+\gamma_0\int_0^t ds\sum_{\alpha=1}^P
\big[B^\ell_{\mu\alpha}(t,s)+\Delta_\alpha(s)G^{\ell+1}_{\mu\alpha}(t,s)\big]
\,\phi(h^\ell_\alpha(s))$$
$$g^\ell_\mu(t)=\dot\phi(h^\ell_\mu(t))\,z^\ell_\mu(t)$$

Self-consistency (averages over realizations of $u,r$):
$$\Phi^\ell_{\mu\alpha}(t,s)=\langle\phi(h^\ell_\mu(t))\phi(h^\ell_\alpha(s))\rangle,\qquad
G^\ell_{\mu\alpha}(t,s)=\langle g^\ell_\mu(t)g^\ell_\alpha(s)\rangle$$

Response functions (causal; zero for $s>t$):
$$A^\ell_{\mu\alpha}(t,s)=\gamma_0^{-1}\Big\langle\frac{\delta\phi(h^\ell_\mu(t))}{\delta r^\ell_\alpha(s)}\Big\rangle,\qquad
B^\ell_{\mu\alpha}(t,s)=\gamma_0^{-1}\Big\langle\frac{\delta g^{\ell+1}_\mu(t)}{\delta u^{\ell+1}_\alpha(s)}\Big\rangle$$

Boundary conditions: $\Phi^0=K^x$, $G^{L+1}=\bm1\bm1^\top$, $A^0=0$, $B^L=0$.
Readout identity: $r^L_\mu(t)=w(0)\sim\mathcal N(0,1)$ and $z^L_\mu(t)=w(t)$ with
$w(t)=w(0)+\gamma_0\int_0^t ds\sum_\alpha\Delta_\alpha(s)\phi(h^L_\alpha(s))$.

Prediction dynamics (deterministic; $f_\mu(0)=0$ in the limit):
$$\frac{df_\mu}{dt}=\sum_{\alpha=1}^P K^{NTK}_{\mu\alpha}(t,t)\,\Delta_\alpha(t)$$

Unknowns after time discretization ($T$ grid points): $\{\Phi^\ell,G^\ell\}$,
$\{A^\ell,B^\ell\}_{\ell=1}^{L-1}$ as $PT\times PT$ matrices, plus $f_\mu(t)$.

## 2. Two-layer case ($L=1$): responses vanish

$A^0=0$ and $B^L=B^1=0$ ⇒ no response functions. $z_\mu(t)=w(t)$ scalar,
$u\sim\mathcal{GP}(0,K^x)$ static in time ($u_\mu(t)=u_\mu$). Closed system:

$$\dot h_\mu(t)=\gamma_0\,w(t)\sum_\alpha K^x_{\mu\alpha}\,\Delta_\alpha(t)\,\dot\phi(h_\alpha(t)),
\qquad h_\mu(0)=u_\mu,\ \ \bm u\sim\mathcal N(0,K^x)$$
$$\dot w(t)=\gamma_0\sum_\alpha\Delta_\alpha(t)\,\phi(h_\alpha(t)),\qquad w(0)\sim\mathcal N(0,1)$$
$$\Phi_{\mu\alpha}(t,s)=\langle\phi(h_\mu(t))\phi(h_\alpha(s))\rangle,\qquad
G_{\mu\alpha}(t,s)=\langle\dot\phi(h_\mu(t))w(t)\dot\phi(h_\alpha(s))w(s)\rangle$$
$$\frac{df_\mu}{dt}=\sum_\alpha\big[\Phi_{\mu\alpha}(t,t)+G_{\mu\alpha}(t,t)K^x_{\mu\alpha}\big]\Delta_\alpha(t),
\qquad f_\mu(0)=0.$$

Note the dynamics of each single-site sample $(h,w)$ depend on the population
ONLY through the deterministic $\Delta(t)$, and $\Delta(t)$ depends on samples
only through equal-time kernel averages ⇒ exact forward co-integration works
(no self-consistent iteration needed). Derivation shortcut: appendix D.2 of
2205.09653.

## 3. Deep linear networks: algebraic closure

$\phi(h)=h$. With $\bm h^\ell=\mathrm{Vec}\{h^\ell_\mu(t)\}\in\mathbb R^{PT}$,
causal operators $\bm C^\ell$ (from $\bm A^{\ell-1},\bm H^{\ell-1},\Delta$) and
$\bm D^\ell$ (from $\bm B^\ell,\bm G^{\ell+1},\Delta$):
$$(\mathbf I-\gamma_0^2\bm C^\ell\bm D^\ell)\bm h^\ell=\bm u^\ell+\gamma_0\bm C^\ell\bm r^\ell,\qquad
(\mathbf I-\gamma_0^2\bm D^\ell\bm C^\ell)\bm g^\ell=\bm r^\ell+\gamma_0\bm D^\ell\bm u^\ell$$
$$\bm H^\ell=(\mathbf I-\gamma_0^2\bm C^\ell\bm D^\ell)^{-1}
\big[\bm H^{\ell-1}+\gamma_0^2\bm C^\ell\bm G^{\ell+1}\bm C^{\ell\top}\big]
(\mathbf I-\gamma_0^2\bm C^\ell\bm D^\ell)^{-\top}$$
$$\bm G^\ell=(\mathbf I-\gamma_0^2\bm D^\ell\bm C^\ell)^{-1}
\big[\bm G^{\ell+1}+\gamma_0^2\bm D^\ell\bm H^{\ell-1}\bm D^{\ell\top}\big]
(\mathbf I-\gamma_0^2\bm D^\ell\bm C^\ell)^{-\top}$$
with $\bm H^\ell=\langle\bm h^\ell\bm h^{\ell\top}\rangle$ replacing $\Phi^\ell$.
No Monte Carlo needed.

### Exactly solvable special case
$L=1$, linear, whitened data $K^x=\mathbf I$, single output direction
($\bm y$), $\Delta(t)=\|\bm\Delta(t)\|$, $y=\|\bm y\|$:
$$\partial_t\Delta(t)=-2\sqrt{1+\gamma_0^2\,(y-\Delta(t))^2}\;\Delta(t)$$
$$\bm H(t\to\infty)=\mathbf I+\frac{\sqrt{1+\gamma_0^2y^2}-1}{y^2}\,\bm y\bm y^\top$$
$\gamma_0\to0$: exponential (NTK) convergence; large $\gamma_0$: sigmoidal
trajectories.

### Small-$\gamma_0$ perturbation theory
$\Phi=\Phi_0+\gamma_0^2\Phi_2+O(\gamma_0^4)$, $G=G_0+\gamma_0^2G_2+O(\gamma_0^4)$;
leading NTK correction closed-form for deep linear (Eq. 12 of the paper).

## 4. Depth-μP residual delta (arXiv:2309.16620)

Architecture: $\bm h^{\ell+1}=\bm h^\ell+\frac{1}{\sqrt{LN}}\bm W^\ell\phi(\bm h^\ell)$,
readout $\frac{1}{\gamma_0\sqrt N\cdot\sqrt N}$-scaled as in μP (no extra $L$
factor), $\eta=\eta_0\gamma_0^2N$ independent of $L$. Layer time
$\tau=\ell/L\in[0,1]$.

Joint $N,L\to\infty$ single-site process (their Prop. 1, schematic form):
$$h(\tau;\bm x;t)=h(0;\bm x;t)+\int_0^\tau du(\tau';\bm x;t)
+\eta_0\gamma_0\int_0^\tau d\tau'\int_0^t ds\int d\bm x'\,
C_h(\tau';\bm x,\bm x';t,s)\,g(\tau';\bm x';s)$$
$du$ Brownian-in-depth increments with covariance from $\Phi(\tau)$; analogous
equation for $g$. Limits commute (depth-then-width = width-then-depth).

Lazy limit ($\gamma_0\to0$) NTK depth-ODEs (their Prop. 2):
$$\partial_\tau H(\tau)=\Phi(\tau),\quad \Phi=\langle\phi(h)\phi(h')\rangle_{\mathcal N(0,H(\tau))},\quad
\partial_\tau G=-G\,\langle\dot\phi(h)\dot\phi(h')\rangle,\quad
K=\int_0^1 d\tau\,G\,\Phi$$
Finite-$L$ NTK → limit at rate $O(L^{-2})$ (squared error); rich dynamics
converge $O(L^{-1})$ at large $N$.

## 5. Finite-width delta (arXiv:2304.03408, summary)

DMFT measure $p(q)\propto e^{NS(q)}$ over order parameters $q$; expand around
saddle $q_\infty$: $\langle(q-q_\infty)(q-q_\infty)^\top\rangle=\frac1N\Sigma+O(N^{-2})$
with propagator $\Sigma=-[\nabla^2S(q_\infty)]^{-1}$. New objects: 4-point
"uncoupled variances" $\kappa$ and sensitivity blocks
$D(t,s)=\langle\partial_{\Delta(s)}(\phi(h(t))^2+g(t)^2)\rangle$. Lazy rank-one
closed form: $\Sigma^\Delta(t,s)=\kappa y^2\,ts\,e^{-\lambda(t+s)}$. Physics:
richer training ⇒ larger kernel variance but SMALLER prediction variance.
