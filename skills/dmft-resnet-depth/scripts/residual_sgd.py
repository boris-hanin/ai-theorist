"""Residual net in the DMFT convention of `derivations/05-completep-dmft-sgd.md`,
trained with SGD. Written to test that file's derived exponents.

    h^1     = (1/sqrt(D)) W0 x
    z^l     = (1/sqrt(N)) W^{l,1} h^l ;  a^l = phi(z^l)
    F_l     = (1/sqrt(N)) W^{l,2} a^l
    h^{l+1} = h^l + (beta0 / L^alpha) F_l
    f       = (1/(gamma sqrt N)) w . h^{L+1},   gamma = gamma_0 sqrt(N)

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

    def params(self):
        p = [self.W0] + self.W1 + [self.w]
        if self.W2 is not None:
            p += self.W2
        return p

    def forward(self, X, keep=False):
        """X: (D, P). Returns f (P,), and optionally the per-layer stream."""
        rn = math.sqrt(self.N)
        h = (self.W0 @ X) / math.sqrt(self.D)
        stream = [h.clone()] if keep else None
        for l in range(self.L):
            z = (self.W1[l] @ h) / rn
            if self.k == 2:
                F = (self.W2[l] @ phi(z)) / rn
            else:
                F = phi(z)
            h = h + self.s * F
            if keep:
                stream.append(h.clone())
        f = (self.w @ h) / (self.gamma * rn)
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
    lr = d_L * net.gamma ** 2 * dt
    ps = net.params()
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
            for p, gr in zip(ps, gs):
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
