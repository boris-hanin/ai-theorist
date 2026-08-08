"""ResNet with 2LP blocks in the (L, M, D) parameterisation of
arXiv 2509.10167 (Chizat) / 2603.18168 (Chaintron-Chizat-Maass), written to test
the **down-projection initialisation** and its consequences.

    h^0 = x                                   ||h|| = Theta(sqrt D)
    h^l = h^{l-1} + (1/L) W_down^l phi( W_up^{l T} LN(h^{l-1}) )
    f   = w . LN(h^L)

    W_up   in R^{D x M},  sigma_up   = D^{-1/2}
    W_down in R^{M x D},  sigma_down = ?      <- the whole question
    w      in R^D,        sigma_w    = D^{-1}   (muP readout, not fan-in)

Two choices of `sigma_down`, in the convention `||h|| = Theta(sqrt D)`:

    down="mlu"    sigma_down = sqrt(D)/M     mean-field; MLU residual scale
    down="fanin"  sigma_down = M^{-1/2}      NTP / CompleteP

    ratio = sqrt(D/M)    ->  IDENTICAL when M = D, divergent off that slice.

`M = D` is therefore a built-in identity control: any measured difference between
the two arms at `M = D` is a bug in the harness, not an effect. (This program has
mistaken an identity control for evidence four times; here it is used the other
way round, as a null that must hold.)

Optimiser is SignGD, the Adam proxy used throughout this program, with the muP
learning rates implied by each init's fan-in count:
    eta_up = base/D,  eta_down = base/M,  eta_w = base/D.
"""

import math

import torch


def phi(x):
    return torch.tanh(x)


def layer_norm(h, eps=1e-12):
    c = h - h.mean(-1, keepdim=True)
    return c / (c.pow(2).mean(-1, keepdim=True).sqrt() + eps)


def sigma_down(D, M, down):
    """The one line this whole round is about."""
    if down == "mlu":
        return math.sqrt(D) / M          # mean-field: residual scale sqrt(D)/(LM)
    if down == "fanin":
        return M ** -0.5                 # NTP / CompleteP; equals mlu iff M == D
    raise ValueError(down)


class ResNet2LP:
    def __init__(self, D, M, L, down="mlu", seed=0, dtype=torch.float64):
        g = torch.Generator().manual_seed(seed)
        rn = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
        self.D, self.M, self.L, self.down = D, M, L, down
        self.s_dn = sigma_down(D, M, down)
        self.Wu = [rn(D, M) * D ** -0.5 for _ in range(L)]
        self.Wd = [rn(M, D) * self.s_dn for _ in range(L)]
        self.w = rn(D) * D ** -1.0

    def params(self):
        return self.Wu + self.Wd + [self.w]

    def lrs(self, base):
        D, M, L = self.D, self.M, self.L
        return [base / D] * L + [base / M] * L + [base / D]

    def forward(self, X, keep=False):
        """X: (P, D) with per-coordinate Theta(1)."""
        h = X
        stream = [h.clone()] if keep else None
        for l in range(self.L):
            h = h + (self.Wd[l].T @ phi(self.Wu[l].T @ layer_norm(h).T)).T / self.L
            if keep:
                stream.append(h.clone())
        f = layer_norm(h) @ self.w
        return (f, stream) if keep else f

    def init_deviation(self, X):
        """||h^L - h^0|| / sqrt(P D) -- the CLT fluctuation of derivations/08 §1.

        Predicted ~ sqrt(D/(L M)) in the MLU parameterisation, and it is the
        cleanest probe available because it needs no training at all.
        """
        with torch.no_grad():
            _, st = self.forward(X, keep=True)
            return float((st[-1] - st[0]).pow(2).mean().sqrt())


def train(net, X, y, base, steps):
    ps, lrs = net.params(), net.lrs(base)
    for p in ps:
        p.requires_grad_(True)
    hist = []
    for _ in range(steps):
        loss = 0.5 * ((y - net.forward(X)) ** 2).mean()
        if not torch.isfinite(loss):
            hist.append(float("inf"))
            break
        gs = torch.autograd.grad(loss, ps)
        with torch.no_grad():
            for p, gr, lr in zip(ps, gs, lrs):
                p -= lr * torch.sign(gr)
        hist.append(float(loss))
    for p in ps:
        p.requires_grad_(False)
    return hist


def data(D, P, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(7000 + seed)
    X = torch.randn(P, D, generator=g, dtype=dtype)
    y = torch.randn(P, generator=g, dtype=dtype)
    return X, y / y.std()


# ---------------------------------------------------------------------------
# Faithful implementation of the parameterisation of 2509.10167 §2.1, with an
# explicit richness dial `alpha` and plain GD (their Eq. 5), for the rate tests
# of round 008. No LayerNorm -- their model has none, and adding one would
# change the analysis.
#
#     h^0 = x                                        x per-coordinate Theta(1)
#     h^l = h^{l-1} + (alpha/(L M)) sum_j w^j tanh(<u^j, h^{l-1}>)
#     u^j ~ N(0, I/D)        so <u,h> = Theta(1)
#     w^j ~ N(0, (1/alpha)^2 I)
#     GD:  Z <- Z - (L M eta / alpha^2) grad_Z Lhat
#
# With s_w = 1/alpha the residual scale is (alpha/(LM)) * ||w|| = sqrt(D)/(LM),
# i.e. the MLU scale, for every alpha -- so `alpha` moves richness WITHOUT
# leaving the MLU parameterisation, which is what their Theorem 2 varies.
class MeanODENet:
    def __init__(self, D, M, L, alpha=1.0, seed=0, dtype=torch.float64):
        g = torch.Generator().manual_seed(seed)
        rn = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
        self.D, self.M, self.L, self.alpha = D, M, L, alpha
        self.U = [rn(D, M) * D ** -0.5 for _ in range(L)]      # u^j = column j
        self.W = [rn(M, D) / alpha for _ in range(L)]          # w^j = row j

    def params(self):
        return self.U + self.W

    def forward(self, X):
        h, c = X, self.alpha / (self.L * self.M)
        for l in range(self.L):
            h = h + c * (phi(h @ self.U[l]) @ self.W[l])
        return h

    def loss(self, X, Y):
        return 0.5 * (self.forward(X) - Y).pow(2).sum(-1).mean() / self.D


def gd(net, X, Y, eta, steps):
    """Their Eq. (5): Z <- Z - (L M eta / alpha^2) grad. Plain GD, not SignGD."""
    lr = net.L * net.M * eta / net.alpha ** 2
    ps = net.params()
    for p in ps:
        p.requires_grad_(True)
    for _ in range(steps):
        gs = torch.autograd.grad(net.loss(X, Y), ps)
        with torch.no_grad():
            for p, gr in zip(ps, gs):
                p -= lr * gr
    for p in ps:
        p.requires_grad_(False)
    return net


def linearized_gd(net, X, Y, eta, steps):
    """Train the model LINEARISED about its own init -- the Neural Tangent ODE
    analogue: f_lin(p0+dp) = f(p0) + J(p0) dp, trained by the same GD rule.
    Theorem 2 bounds how far the true ResNet is from this object, so the gap
    between `gd` and `linearized_gd` is the quantity that should be minimised at
    alpha* = (M L)^{1/4}."""
    lr = net.L * net.M * eta / net.alpha ** 2
    p0 = tuple(p.detach().clone() for p in net.params())
    dp = [torch.zeros_like(p) for p in p0]
    for _ in range(steps):
        v = tuple(d.detach().requires_grad_(True) for d in dp)
        f0, Jv = torch.autograd.functional.jvp(
            lambda *ws: _fwd(net, X, list(ws)), p0, v, create_graph=True)
        loss = 0.5 * (f0 + Jv - Y).pow(2).sum(-1).mean() / net.D
        gs = torch.autograd.grad(loss, v)
        dp = [d - lr * g for d, g in zip(dp, gs)]
    with torch.no_grad():
        f0, Jv = torch.autograd.functional.jvp(
            lambda *ws: _fwd(net, X, list(ws)), p0, tuple(dp))
    return f0 + Jv


def _fwd(net, X, ws):
    L, M = net.L, net.M
    U, W = ws[:L], ws[L:]
    h, c = X, net.alpha / (L * M)
    for l in range(L):
        h = h + c * (phi(h @ U[l]) @ W[l])
    return h


def odedata(D, P, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(9000 + seed)
    X = torch.randn(P, D, generator=g, dtype=dtype)
    Y = torch.randn(P, D, generator=g, dtype=dtype) * 0.5
    return X, Y
