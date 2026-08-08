"""Residual MoE in the Neural Mean ODE scaling of `derivations/09-moe-mean-ode.md`.

Merges the MoE parameterisation of 2601.20205 with the joint (L, M, D) analysis
of 2509.10167 / 2603.18168.

    h^{l+1} = h^l + c_L * (1/a) sum_{i in A(h^l)} g_i(h^l) E_i^l(h^l)
    E_i^l(h) = sum_{j=1}^{M} w^{ij} phi(<u^{ij}, h>)          2LP, M units
    g_i      = sigma(<r_i, h>),  A = top_a({g_i + b_i})

    c_L  = 1/(L M)          so the drift c_L * M is Theta(1/L)
    u    ~ N(0, I_D/D)      <u,h> = Theta(1) at ||h|| = Theta(sqrt D)
    w    ~ N(0, s_w^2 I_D),  s_w = 1                       -> MLU
                             s_w = sqrt(M/D) * M^{-1/2} = D^{-1/2}  -> fan-in control

Derived predictions: effective width W_eff = L*a*M, so the init deviation is
sqrt(D/(L a M)) and is invariant to how a fixed L*a*M is split among the three.
`E` should be absent from the rate (only active experts fire).

Router biases are initialised NONZERO (F21): with a zero router and zero biases
every expert has an identical gate, top-k breaks the tie by index, and routing
silently collapses to a fixed subnetwork while the loss still falls.
"""

import math

import torch


def phi(x):
    return torch.tanh(x)


class MoEMeanODE:
    def __init__(self, D, M, L, E, kappa=0.25, down="mlu", b_std=1.0,
                 seed=0, dtype=torch.float64):
        g = torch.Generator().manual_seed(seed)
        rn = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
        self.D, self.M, self.L, self.E = D, M, L, E
        self.a = max(1, int(round(kappa * E)))
        self.kappa = kappa
        # s_w = 1 is MLU (residual scale c_L*||w|| = sqrt(D)/(LM), derivation 09 §2).
        # The fan-in control is sqrt(M/D) LARGER, matching the ratio of
        # derivations/08 §3 -- and note what that does to the init deviation:
        #   MLU    : s_w / sqrt(L a M) = 1/sqrt(L a M)      <- M averages
        #   fan-in : sqrt(M/D)/sqrt(L a M) = 1/sqrt(L a D)  <- M CANCELS
        # so under fan-in the expert width buys no averaging at all, exactly as
        # in the dense case (round 007 Q3c).
        self.s_w = 1.0 if down == "mlu" else math.sqrt(M / D)
        self.down = down
        self.c_L = 1.0 / (L * M)
        self.U = [rn(E, D, M) * D ** -0.5 for _ in range(L)]     # u^{ij}
        self.W = [rn(E, M, D) * self.s_w for _ in range(L)]      # w^{ij}
        self.R = [rn(D, E) * D ** -0.5 for _ in range(L)]        # routers
        self.b = [rn(E) * b_std for _ in range(L)]               # nonzero (F21)

    def params(self):
        return self.U + self.W + self.R

    def block(self, h, l, stats=None):
        gate = torch.sigmoid(h @ self.R[l])                       # (P,E)
        with torch.no_grad():
            q = gate + self.b[l]
            idx = q.topk(self.a, dim=-1).indices
            mask = torch.zeros_like(q).scatter_(-1, idx, 1.0)
            if stats is not None:
                stats["spread"] = float(q.std(-1).mean())
                stats["load"] = mask.mean(0)
        z = torch.einsum("pd,edm->epm", h, self.U[l])             # (E,P,M)
        Eo = torch.einsum("epm,emd->epd", phi(z), self.W[l])      # (E,P,D)
        wgt = (gate * mask).T.unsqueeze(-1)                       # (E,P,1)
        return self.c_L * (wgt * Eo).sum(0) / self.a

    def forward(self, X, stats=None):
        h = X
        for l in range(self.L):
            h = h + self.block(h, l, stats if (stats is not None and l == 0) else None)
        return h

    def init_deviation(self, X):
        """Per-coordinate RMS of h^L - h^0. Predicted ~ sqrt(D/(L a M))/sqrt(D)
        per coordinate, i.e. 1/sqrt(L a M)."""
        with torch.no_grad():
            return float((self.forward(X) - X).pow(2).mean().sqrt())

    def loss(self, X, Y):
        return 0.5 * (self.forward(X) - Y).pow(2).sum(-1).mean() / self.D


def group_lrs(net, eta):
    """Per-group learning rates. A SINGLE global LR is WRONG in D.

    Measured with one global lr = L M a D eta, the induced change in each
    group's own observable scales as D^-0.18 (W), D^+1.28 (U), D^+1.37 (R)
    instead of D^0 -- so the router and up-projection updates blow up with the
    embedding dimension. Derivation, using the coherent (LLN-alignment)
    labelling that governs the trained regime:

      W:  block output change per coord = c_L * (a M) * dW / a ... = dW
          grad_w ~ (c_L/a)(1/D)          =>  lr_W = L M a D
      U:  dz = <du, h> = lr_U (c_L/a) <w,b> ||h||^2,  <w,b> coherent = Theta(1)
          =>  lr_U = L M a / D
      R:  d<r,h> = lr_R (c_L/a) <E,b> ||h||^2,  <E,b> coherent = sqrt(M)
          =>  lr_R = L a sqrt(M) / D

    Note the spread between U and W is D^2. At INITIALISATION <w,b> and <E,b>
    are incoherent and carry an extra D^-1/2, so the one-step exponents differ
    from these by 1/2 -- the F18 signature. These are the asymptotic values, and
    HP transfer is an asymptotic claim.
    """
    L, M, a, D = net.L, net.M, net.a, net.D
    return ([L * M * a / D * eta] * L +           # U
            [L * M * a * D * eta] * L +           # W
            [L * a * math.sqrt(M) / D * eta] * L) # R


def gd(net, X, Y, eta, steps):
    """LR scaled so the per-unit update is Theta(eta).

    grad_w = (c_L/a) * g * phi * b  with c_L = 1/(LM) and, since the loss is
    per-coordinate normalised, b ~ 1/D. So grad_w ~ 1/(L M a D) and the
    compensating factor is L*M*a*D.

    The D was MISSING in the first version, which made Delta w ~ eta/D and made
    the optimal LR drift 0.65 decades across D (round 009 N4). It is the only
    dial whose LR dependence derivation 09 did not state explicitly."""
    lrs = group_lrs(net, eta)
    ps = net.params()
    for p in ps:
        p.requires_grad_(True)
    hist = []
    for _ in range(steps):
        loss = net.loss(X, Y)
        if not torch.isfinite(loss):
            hist.append(float("inf"))
            break
        gs = torch.autograd.grad(loss, ps)
        with torch.no_grad():
            for p, gr, lr in zip(ps, gs, lrs):
                p -= lr * gr
        hist.append(float(loss))
    for p in ps:
        p.requires_grad_(False)
    return hist


def odedata(D, P, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(11000 + seed)
    X = torch.randn(P, D, generator=g, dtype=dtype)
    Y = torch.randn(P, D, generator=g, dtype=dtype) * 0.5
    return X, Y


# ---------------------------------------------------------------------------
# Fixed-task variant for the C^{-1/6} rate test (round 010).
#
# The rate test must compare models at DIFFERENT D, so it needs an observable
# whose meaning does not change with D. Solution: a fixed low-dimensional task
# with a scalar readout, so every shape approximates the SAME function
# R^d0 -> R and the data is literally identical across shapes.
#
#   h^0 = W_embed x,  W_embed in R^{d0 x D}, entries N(0,1/d0)  -> h per-coord O(1)
#   ... L MoE blocks as above ...
#   f   = <w, h^L>,   sigma_w = 1/D   (muP readout: f(init) -> 0, trained part O(1))
class MoEFixedTask(MoEMeanODE):
    def __init__(self, d0, **kw):
        super().__init__(**kw)
        g = torch.Generator().manual_seed(kw.get("seed", 0) + 90000)
        D = self.D
        self.d0 = d0
        self.Wemb = torch.randn(d0, D, generator=g, dtype=torch.float64) / math.sqrt(d0)
        self.wout = torch.randn(D, generator=g, dtype=torch.float64) / D

    def params(self):
        return super().params() + [self.wout]

    def forward_scalar(self, X):
        h = X @ self.Wemb
        for l in range(self.L):
            h = h + self.block(h, l)
        return h @ self.wout

    def loss(self, X, Y):
        return 0.5 * (self.forward_scalar(X) - Y).pow(2).mean()


def gd_fixed(net, X, Y, eta, steps):
    # Readout: f = <w, h^L> is a COHERENT sum over D, so df = D * dw and dw must
    # be Theta(1/D). grad_w = (f-y) h is Theta(1) per coordinate, hence lr = eta/D.
    # (An earlier version had eta*D -- inverted -- and every run diverged.)
    lrs = group_lrs(net, eta) + [eta / net.D]
    ps = net.params()
    for p in ps:
        p.requires_grad_(True)
    for _ in range(steps):
        loss = net.loss(X, Y)
        if not torch.isfinite(loss):
            break
        gs = torch.autograd.grad(loss, ps)
        with torch.no_grad():
            for p, gr, lr in zip(ps, gs, lrs):
                p -= lr * gr
    for p in ps:
        p.requires_grad_(False)
    return net


def shape_for(C, kappa=0.25, a=4):
    """The optimal-rate shape of derivations/09 §4: D ~ C^(1/3), L ~ C^(1/6)."""
    L = max(2, int(round(C ** (1 / 6.0))))
    D = max(4, int(round(C ** (1 / 3.0))))
    M = max(2, int(round(C / (L * D * a))))
    return dict(L=L, D=D, M=M, E=int(a / kappa), kappa=kappa)
