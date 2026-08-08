"""Residual net in the DMFT convention of `derivations/05-completep-dmft-sgd.md`,
trained with SGD. Written to test that file's derived exponents.

    h^1     = (1/sqrt(D)) W0 x
    hbar^l  = LN(h^l)                       <- PRE-LN, always on
    z^l     = (1/sqrt(N)) W^{l,1} hbar^l ;  a^l = phi(z^l)
    F_l     = (1/sqrt(N)) W^{l,2} a^l
    h^{l+1} = h^l + (beta0 / L^alpha) F_l
    f       = (1/(gamma sqrt N)) w . LN(h^{L+1}),   gamma = gamma_0 sqrt(N)

Pre-LN is on by construction (both 2405.15712 and 2505.01618 are pre-LN
transformers, and its absence was the leading suspect for the alpha=1
discrepancy recorded in derivations/05 §8c). In the DMFT limit LN is NOT a new
stochastic element: with h having i.i.d.-like entries, its mean self-averages to
0 and its std to sqrt(H^l(t,t)), so

    LN(h^l) -> h^l / sqrt(H^l(t,t))

a DETERMINISTIC, layer- and time-dependent gain set by the stream kernel. That
is what makes it tractable in the solver, and it is why it fixes the block input
scale: (1/N)|hbar|^2 = 1 identically, whatever the stream does.

SGD with `theta <- theta - d_L gamma^2 dt grad L`, `d_L` free so the derived
rule `d_L = L^{2 alpha - 1}` can be imposed or withheld.

All weights i.i.d. N(0,1) -- the DMFT convention, with every scale factor
explicit in the forward pass rather than folded into the init. `block_k=2`
matches the derivation; `block_k=1` is available for the simpler case.
"""

import math

import torch


def phi(x):
    return torch.tanh(x)


def layer_norm(h, eps=1e-12):
    """LN over the feature axis. h: (N, P) -> (N, P)."""
    mu = h.mean(dim=0, keepdim=True)
    c = h - mu
    return c / (c.pow(2).mean(dim=0, keepdim=True).sqrt() + eps)


class ResNet:
    def __init__(self, D, N, L, alpha=1.0, gamma0=1.0, beta0=1.0, block_k=2,
                 seed=0, device="cpu", dtype=torch.float64):
        g = torch.Generator(device=device).manual_seed(seed)
        rn = lambda *s: torch.randn(*s, generator=g, device=device, dtype=dtype)
        self.D, self.N, self.L = D, N, L
        self.alpha, self.gamma0, self.beta0, self.k = alpha, gamma0, beta0, block_k
        self.s = beta0 * L ** (-alpha)                 # residual branch scale
        self.gamma = gamma0 * math.sqrt(N)
        self.W0 = rn(N, D)
        self.W1 = [rn(N, N) for _ in range(L)]         # inner matrix
        self.W2 = [rn(N, N) for _ in range(L)] if block_k == 2 else None
        self.w = rn(N)
        self.W1_0 = [W.clone() for W in self.W1]       # keep init for movement stats

    def block_params(self):
        """In-block parameters -- these and only these take the depth LR factor
        d_L. The L-block accumulation that fixes d_L counts these; W^0 and the
        readout appear ONCE, not L times, so applying d_L to them is wrong.
        CompleteP Table 1 says the same: Emb and Unemb LR carry no depth factor.
        Applying d_L to W^0 was the cause of the alpha=1 falsification recorded
        in derivations/05 §8c -- it made the stream input move L times faster,
        giving stream-movement slope +1.55 instead of 0."""
        return self.W1 + (self.W2 if self.W2 is not None else [])

    def outer_params(self):
        """Outside the residual stack: no depth factor."""
        return [self.W0, self.w]

    def params(self):
        return self.block_params() + self.outer_params()

    def forward(self, X, keep=False):
        """X: (D, P). Returns f (P,), and optionally the per-layer stream."""
        rn = math.sqrt(self.N)
        h = (self.W0 @ X) / math.sqrt(self.D)
        stream = [h.clone()] if keep else None
        for l in range(self.L):
            z = (self.W1[l] @ layer_norm(h)) / rn if self.k == 2 else None
            if self.k == 2:
                F = (self.W2[l] @ phi(z)) / rn
            else:
                # k=1: the block IS the single matrix applied to phi(LN(h))
                F = (self.W1[l] @ phi(layer_norm(h))) / rn
            h = h + self.s * F
            if keep:
                stream.append(h.clone())
        f = (self.w @ layer_norm(h)) / (self.gamma * rn)
        return (f, stream) if keep else f

    # -- diagnostics -----------------------------------------------------
    def stream_kernel(self, X):
        """H^l(0) = (1/N)|h^l|^2 averaged over data, per layer."""
        with torch.no_grad():
            _, st = self.forward(X, keep=True)
            return torch.tensor([(h ** 2).sum(0).mean() / self.N for h in st])

    def block_movement(self):
        """RMS |W1(t) - W1(0)| relative to the init RMS (which is 1 here)."""
        with torch.no_grad():
            d = torch.stack([((a - b) ** 2).mean() for a, b in zip(self.W1, self.W1_0)])
            return float(d.mean().sqrt())


def train(net, X, y, dt, steps, d_L=None, record=False):
    """SGD, theta <- theta - d_L gamma^2 dt grad L, L = sum_mu (1/2)(y-f)^2.

    `d_L = None` applies the derived rule L^{2 alpha - 1}; pass 1.0 to withhold
    it (the negative control).
    """
    if d_L is None:
        d_L = net.L ** (2 * net.alpha - 1.0)
    base = net.gamma ** 2 * dt
    blk, out = net.block_params(), net.outer_params()
    ps = blk + out
    lrs = [d_L * base] * len(blk) + [base] * len(out)   # d_L on blocks ONLY
    for p in ps:
        p.requires_grad_(True)
    hist = []
    for _ in range(steps):
        f = net.forward(X)
        loss = 0.5 * ((y - f) ** 2).sum()
        if not torch.isfinite(loss):
            return float("inf"), hist
        gs = torch.autograd.grad(loss, ps)
        with torch.no_grad():
            for p, gr, lr in zip(ps, gs, lrs):
                p -= lr * gr
        if record:
            hist.append(float(loss))
    with torch.no_grad():
        out = float(0.5 * ((y - net.forward(X)) ** 2).sum())
    for p in ps:
        p.requires_grad_(False)
    return (out if math.isfinite(out) else float("inf")), hist


def data(D, P, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(500 + seed)
    X = torch.randn(D, P, generator=g, dtype=dtype)
    X = X / X.norm(dim=0, keepdim=True) * math.sqrt(D)
    y = torch.randn(P, generator=g, dtype=dtype)
    return X, y / y.std()
