"""Transformer in the parameterisation of arXiv:2405.15712 §2.1, with the
scaling exponents left FREE.

    h^1_s     = (1/sqrt(D)) W0 x_s                              in R^{NH}
    ht^l_s    = h^l_s  + beta0 L^{-aL} MHSA(h^l_s)
    h^{l+1}_s = ht^l_s + beta0 L^{-aL} MLP(ht^l_s)
    A^l_{h,ss'} = N^{-aA} k^l_{hs} . q^l_{hs'}
    f         = (gamma0 N H)^{-1} w . mean_s h^L_s

SGD learning rate (paper Table 1, and derived independently in
`derivations/03-attention.md` D3):

    eta = eta_0 * N * H * L^{2 aL - 1}

Storing the key/query weights
-----------------------------
The paper gives `W_K` entries of size `Theta(N^{1-aA})` with forward prefactor
`N^{aA-3/2} H^{-1/2}`, and applies the bulk learning rate to `W_K` directly.
Here `W_K = N^{1-aA} What_K` with `What_K ~ N(0,1)`, so the forward prefactor
collapses to `N^{-1/2} H^{-1/2}` — **independent of aA** — and the exponent
instead enters through the learning rate on `What_K`:

    lr(What_K) = eta / N^{2(1 - aA)}

This is algebraically identical to the paper's convention (chain rule through
`W_K = N^{1-aA} What_K` gives exactly that factor) and it makes explicit where
`aA` actually acts: not on the forward pass, where `k` is `Theta(1)` for every
`aA`, but on how fast the key/query weights move. That is the content of the
paper's Table 3, which this module is checked against before use.
"""

import math

import torch
import torch.nn.functional as F


def _mk(shape, gen, device):
    return torch.randn(*shape, generator=gen, device=device)


class Transformer(torch.nn.Module):
    def __init__(self, D, N, H, L, alpha_A=1.0, alpha_L=1.0, gamma0=1.0,
                 beta0=1.0, seed=0, device="cpu", first_last_rescale=True):
        super().__init__()
        self.D, self.N, self.H, self.L = D, N, H, L
        self.aA, self.aL = alpha_A, alpha_L
        self.gamma0, self.beta0 = gamma0, beta0
        self.dm = N * H
        g = torch.Generator(device=device).manual_seed(seed)

        # Table 1: first/last layer multiplier * L^{1/2 - aL}, init / same.
        r = (L ** (0.5 - alpha_L)) if first_last_rescale else 1.0
        self.r_first = r

        self.W0 = torch.nn.Parameter(_mk((self.dm, D), g, device) / r)
        self.WK, self.WQ, self.WV, self.WO, self.W1, self.W2 = (
            torch.nn.ParameterList() for _ in range(6))
        for _ in range(L):
            self.WK.append(torch.nn.Parameter(_mk((H, N, self.dm), g, device)))
            self.WQ.append(torch.nn.Parameter(_mk((H, N, self.dm), g, device)))
            self.WV.append(torch.nn.Parameter(_mk((H, N, self.dm), g, device)))
            self.WO.append(torch.nn.Parameter(_mk((H, self.dm, N), g, device)))
            self.W1.append(torch.nn.Parameter(_mk((self.dm, self.dm), g, device)))
            self.W2.append(torch.nn.Parameter(_mk((self.dm, self.dm), g, device)))
        self.w = torch.nn.Parameter(_mk((self.dm,), g, device) / r)

    # -- forward ----------------------------------------------------------
    def block(self, h, l, keep=None):
        """One residual block. `keep` collects A and k for diagnostics."""
        N, H, dm = self.N, self.H, self.dm
        s = self.beta0 * self.L ** (-self.aL)

        hb = F.layer_norm(h, (dm,))                       # (B,S,dm)
        # k, q: (B,S,H,N).  Prefactor N^{-1/2} H^{-1/2} -- see module docstring.
        ck = 1.0 / math.sqrt(N * H)
        k = ck * torch.einsum("bsd,hnd->bshn", hb, self.WK[l])
        q = ck * torch.einsum("bsd,hnd->bshn", hb, self.WQ[l])
        v = torch.einsum("bsd,hnd->bshn", hb, self.WV[l]) / math.sqrt(dm)

        A = torch.einsum("bshn,bthn->bhst", k, q) / (N ** self.aA)
        sig = torch.softmax(A, dim=-1)
        vs = torch.einsum("bhst,bthn->bshn", sig, v)
        mhsa = torch.einsum("bshn,hdn->bsd", vs, self.WO[l]) / math.sqrt(dm)
        if keep is not None:
            keep.append({"A": A.detach(), "k": k.detach(),
                         "hb": hb.detach()})
        ht = h + s * mhsa

        htb = F.layer_norm(ht, (dm,))
        m1 = htb @ self.W1[l].T / math.sqrt(dm)
        mlp = F.gelu(m1) @ self.W2[l].T / math.sqrt(dm)
        return ht + s * mlp

    def forward(self, X, keep=None):
        """X: (B, S, D) -> f: (B,)"""
        h = X @ self.W0.T / math.sqrt(self.D)
        h = h * self.r_first
        for l in range(self.L):
            h = self.block(h, l, keep)
        pooled = h.mean(dim=1)
        return (pooled @ self.w) * self.r_first / (self.gamma0 * self.dm)

    # -- per-group learning rates (Table 1 + the What_K chain rule) --------
    def param_groups(self, eta0, scale_N=True, scale_H=True, scale_L=True,
                     scale_A=True):
        """Bulk LR eta_0 * N * H * L^{2 aL - 1}; K/Q additionally / N^{2(1-aA)}.

        The `scale_*` flags exist so a sweep can drop exactly one predicted
        factor and check that transfer then FAILS. A control that still
        transfers means the sweep is not sensitive to that factor.
        """
        N, H, L = self.N, self.H, self.L
        eta = eta0
        if scale_N:
            eta *= N
        if scale_H:
            eta *= H
        if scale_L:
            eta *= L ** (2 * self.aL - 1)
        kq_div = (N ** (2 * (1 - self.aA))) if scale_A else 1.0
        kq = [p for pl in (self.WK, self.WQ) for p in pl]
        kq_ids = {id(p) for p in kq}
        rest = [p for p in self.parameters() if id(p) not in kq_ids]
        return [{"params": kq, "lr": eta / kq_div},
                {"params": rest, "lr": eta}]


# ---------------------------------------------------------------------------

def teacher_batch(B, S, D, seed=0, device="cpu"):
    """Fixed synthetic sequence task: the target mixes across positions, so a
    model that cannot attend is at a disadvantage."""
    g = torch.Generator(device=device).manual_seed(10_000 + seed)
    X = torch.randn(B, S, D, generator=g, device=device)
    U = torch.randn(D, 8, generator=g, device=device) / math.sqrt(D)
    sel = torch.softmax((X @ U).mean(-1), dim=1)            # (B,S) position weights
    z = torch.einsum("bs,bsd->bd", sel, X)                  # attention-like pooling
    a = torch.randn(D, generator=g, device=device) / math.sqrt(D)
    y = torch.tanh(z @ a) * 2.0
    return X, (y - y.mean()) / y.std()


def train(model, X, y, eta0, steps, **lr_flags):
    opt = torch.optim.SGD(model.param_groups(eta0, **lr_flags))
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = ((y - model(X)) ** 2).mean()
        if not torch.isfinite(loss):
            return float("inf")
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = ((y - model(X)) ** 2).mean().item()
    return final if math.isfinite(final) else float("inf")
