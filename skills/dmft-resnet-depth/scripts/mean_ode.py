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
