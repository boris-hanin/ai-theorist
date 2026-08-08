"""Reduced residual-MoE model of arXiv 2601.20205 §4, in the paper's own
parameterisation convention (scale in the init std and the LR, no explicit
forward prefactors) so Table 1 can be read off directly.

    h^0     = W_embed x
    h^{l+1} = h^l + (1/L) f_MoE^l(LN(h^l))
    f(x)    = w_unembd . phi(LN(h^L))

    f_MoE(h) = (1/a) sum_{i in A(h)} g_i(h) E_i(h)
    g_i      = sigma(r_i),  r_i = W_router[:, i] . h
    A(h)     = top_a({g_i + b_i})                       (no-grad, per the paper)
    E_i(h)   = W_down^i phi((W_up^i)^T h)

Table 1 (their §3.3), with n = n_embd, m = alpha_ffn * n:

    group        init std                     LR
    router       n^-gamma                     n^-1
    bias         0                            1
    W_up         n^-1/2                       n^-1
    W_down       alpha_ffn^-1 n^-1/2          alpha_ffn^-1 n^-1

`down_init="table1"` is that rule; `down_init="fanin"` is the negative control
sigma = m^-1/2 = alpha_ffn^-1/2 n^-1/2, i.e. alpha_ffn^{+1/2} larger. The
control is the whole point: P1/P2 are collapse (null) results and mean nothing
unless the control fails to collapse.

Optimiser is SignGD -- the paper's stated Adam proxy and what the derivation in
`derivations/06-moe.md` §1 assumes. Experts are batched over the expert axis and
masked, rather than gathered, so that the expert-parallel arithmetic is exact and
the routing stays a no-grad mask (their convention: routers get gradient only
through g_i, never through the top-k).
"""

import math

import torch


def phi(x):
    return torch.tanh(x)


def layer_norm(h, eps=1e-12):
    """LN over the feature axis. h: (P, n)."""
    c = h - h.mean(-1, keepdim=True)
    return c / (c.pow(2).mean(-1, keepdim=True).sqrt() + eps)


class MoENet:
    def __init__(self, n, L, E, kappa=0.25, alpha_ffn=1.0, D=1, gamma=1.0,
                 down_init="table1", b_std=1.0, seed=0, dtype=torch.float64):
        """`b_std` is the init std of the router biases, and it must be NONZERO.

        This follows the DMFT convention of 2601.20205 App. E footnote 2: the
        router is initialised at zero and *expert diversity at initialisation
        comes from the random biases* `b_k(0)`. The gating variable is
        `q_k = sigma(r_k) + b_k`, so with `r = 0` and `b = 0` every expert has
        the identical `q_k = sigma(0)`, `topk` breaks the exact tie by index
        order, and the SAME experts are selected for every token in every layer.
        That is not sparse routing, it is a fixed subnetwork -- and it is silent,
        because the loss still goes down.

        `b_std = 0` is retained only as the explicit degenerate control (it
        reproduces the tie above whenever the router is also zero). The paper's
        *main text* does initialise biases at zero, but there the router carries
        `n^-gamma` noise which supplies the diversity instead; the two settings
        must not be mixed and matched.
        """
        g = torch.Generator().manual_seed(seed)
        rn = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
        self.n, self.L, self.E, self.kappa = n, L, E, kappa
        self.alpha_ffn = alpha_ffn
        self.m = int(round(alpha_ffn * n))
        self.a = max(1, int(round(kappa * E)))
        self.down_init = down_init

        s_up = n ** -0.5
        s_dn = (alpha_ffn ** -1.0) * n ** -0.5 if down_init == "table1" \
            else self.m ** -0.5
        self.s_dn_used = s_dn

        self.b_std, self.gamma = b_std, gamma
        self.W_embed = rn(D, n) * D ** -0.5
        # per layer: routers (n,E), biases (E,), up (E,n,m), down (E,m,n)
        # gamma=None is the App. E convention: router identically zero at init,
        # so ALL init routing diversity comes from the biases.
        self.Wr = [(torch.zeros(n, E, dtype=dtype) if gamma is None
                    else rn(n, E) * n ** (-gamma)) for _ in range(L)]
        self.b = [rn(E) * b_std for _ in range(L)]
        if b_std == 0.0 and gamma is None:
            raise ValueError(
                "b_std=0 with a zero-init router gives every expert an identical "
                "gate: top-k then selects by index order and routing is a fixed "
                "subnetwork. Set b_std>0 (App. E) or gamma to a finite value.")
        self.Wu = [rn(E, n, self.m) * s_up for _ in range(L)]
        self.Wd = [rn(E, self.m, n) * s_dn for _ in range(L)]
        # Readout init is n^-1, NOT fan-in n^-1/2. This is the level-1 instance
        # of exactly the same mean-field condition as sigma(W_down) in §1c:
        #   f(init) = sigma_w sqrt(n)  (incoherent),  Delta f = n * eta_w (coherent)
        # eta_w = 1/n gives Delta f = Theta(1); sigma_w = n^-1 then sends
        # f(init) = n^-1/2 -> 0, so the trained part dominates. With fan-in
        # n^-1/2 the init function stays Theta(1) and a random Theta(1)
        # component survives the width limit -- which broke the width collapse
        # test (11.5 s.e.) before this was fixed.
        self.w = rn(n) * n ** -1.0
        self._init_snapshot()

    def _init_snapshot(self):
        self.Wd0 = [W.clone() for W in self.Wd]
        self.Wu0 = [W.clone() for W in self.Wu]

    # -- parameter groups and their Table 1 learning rates -----------------
    def groups(self, base_lr):
        n, af = self.n, self.alpha_ffn
        gs = []
        for l in range(self.L):
            gs.append((self.Wr[l], base_lr / n))
            gs.append((self.b[l], base_lr * 1.0))
            gs.append((self.Wu[l], base_lr / n))
            gs.append((self.Wd[l], base_lr / (af * n)))
        gs.append((self.w, base_lr / n))
        return gs

    def params(self):
        return [p for p, _ in self.groups(1.0)]

    # -- forward ------------------------------------------------------------
    def moe_layer(self, h, l, stats=None):
        """h: (P, n) -> (P, n). Batched over experts with a hard mask."""
        P = h.shape[0]
        r = h @ self.Wr[l]                       # (P, E)
        gate = torch.sigmoid(r)                  # (P, E)
        with torch.no_grad():                    # routing is no-grad (paper)
            q = gate + self.b[l]
            idx = q.topk(self.a, dim=-1).indices
            mask = torch.zeros_like(q).scatter_(-1, idx, 1.0)
            if stats is not None:
                # the selection threshold: lowest selected q per token
                stats["thresh"] = q.gather(-1, idx[:, -1:]).squeeze(-1)
                stats["load"] = mask.mean(0)
        hup = torch.einsum("pn,enm->epm", h, self.Wu[l])     # (E,P,m)
        Eout = torch.einsum("epm,emn->epn", phi(hup), self.Wd[l])  # (E,P,n)
        if stats is not None:
            stats["E_rms"] = Eout.pow(2).mean().sqrt()
        wgt = (gate * mask).T.unsqueeze(-1)                  # (E,P,1)
        return (wgt * Eout).sum(0) / self.a

    def forward(self, X, keep=False, stats=None):
        h = X @ self.W_embed
        stream = [h.clone()] if keep else None
        for l in range(self.L):
            # The 1/L residual multiplier is CompleteP alpha=1 (paper §3.1/§4).
            # Omitting it flips the L-exponent of the init stream variance from
            # -1 to +1 -- caught by P4, see rounds/006-moe/results.md.
            h = h + self.moe_layer(layer_norm(h), l,
                                   stats if (stats is not None and l == 0) else None) / self.L
            if keep:
                stream.append(h.clone())
        f = layer_norm(h) @ self.w
        return (f, stream) if keep else f

    # -- diagnostics --------------------------------------------------------
    def expert_out_rms(self, X, l=0, which="init"):
        """RMS of a single expert's output E_k, the level-3 mean-field object."""
        with torch.no_grad():
            h = layer_norm(X @ self.W_embed)
            Wd = (self.Wd0 if which == "init" else self.Wd)[l]
            Wu = (self.Wu0 if which == "init" else self.Wu)[l]
            hup = torch.einsum("pn,enm->epm", h, Wu)
            return float(torch.einsum("epm,emn->epn", phi(hup), Wd).pow(2).mean().sqrt())

    def stream_init_var(self, X):
        """Var of the init contribution to the stream: (1/n)|h^L - h^0|^2.

        This is the object that equals alpha_* up to kappa (derivations/06 §4).
        """
        with torch.no_grad():
            _, st = self.forward(X, keep=True)
            return float((st[-1] - st[0]).pow(2).mean())


def train(net, X, y, base_lr, steps, record=True):
    """SignGD, the paper's Adam proxy. Returns the loss history."""
    gs = net.groups(base_lr)
    ps = [p for p, _ in gs]
    lrs = [lr for _, lr in gs]
    for p in ps:
        p.requires_grad_(True)
    hist = []
    for _ in range(steps):
        loss = 0.5 * ((y - net.forward(X)) ** 2).mean()
        if not torch.isfinite(loss):
            hist.append(float("inf"))
            break
        grads = torch.autograd.grad(loss, ps, allow_unused=True)
        with torch.no_grad():
            for p, gr, lr in zip(ps, grads, lrs):
                if gr is not None:
                    p -= lr * torch.sign(gr)
        if record:
            hist.append(float(loss))
    for p in ps:
        p.requires_grad_(False)
    return hist


def data(D, P, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(1000 + seed)
    X = torch.randn(P, D, generator=g, dtype=dtype)
    X = X / X.norm(dim=-1, keepdim=True) * math.sqrt(D)
    y = torch.randn(P, generator=g, dtype=dtype)
    return X, y / y.std()
