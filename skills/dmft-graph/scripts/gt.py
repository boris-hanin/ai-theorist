"""Base transfer GRAPH TRANSFORMER: the architecture of arXiv 2607.05017 §2.2
with the MHSA branch that the paper permits ("encoder, MPNN, MHSA, and decoder
modules can be chosen flexibly") but never writes down.

    X^(0)    = (1/(sigma_0 sqrt(n0))) X W^(0)                  W^(0) ~ N(0,sigma_0^2)
    Xt^(l+1) = X^(l)    + (1/L) (1/(gamma_P sqrt(D))) P X^(l) W~
    Xh^(l+1) = Xt^(l+1) + (1/L) (1/(gamma_A sqrt(D))) [S(Xt) V] W_O
    X^(l+1)  = Xh^(l+1) + (1/L) (1/sqrt(4D)) ((1/sqrt(D)) Xh W_1) W_2
    z        = (1/(sigma_{L+1} D)) (1/N) 1^T X^(L) W^(L+1)     W^(L+1) ~ N(0,sigma_{L+1}^2)

with, per head h (D_h = D/H, heads PARTITION the width):

    q_u = (1/(sigma_QK sqrt(D))) W_Q x_u ,   k_v likewise
    A_uv = D_h^{-alpha_A} q_u . k_v ,  masked to v in N(u) u {u}
    S    = softmax_v(A_uv)

Parameterisation implemented (derivations/10-graph-transformer.md §5):

    group            SGD needed eta                   Adam/signGD needed eta
    encoder          eta_0 D sigma_0^2 (x C_ab)       eta_0 sigma_0
    MPNN/V/O/MLP     eta_0 D L                        eta_0 / sqrt(D)
    Q,K              eta_0 D L sigma_QK^2 D_h^{2a-2}  eta_0 sigma_QK D_h^{a-1}/sqrt(D)
    decoder          eta_0 D sigma_{L+1}^2            eta_0 sigma_{L+1}

with a = alpha_A, and the global-rate choices sigma_0 = sigma_{L+1} = sqrt(L)
(SGD) or 1/sqrt(D) (Adam).

IDENTITY WARNING, and it is the whole reason `param="qk-global"` exists (see
`10-graph-transformer.md` §3f): rescaling sigma_QK is EXACTLY equivalent to
rescaling the Q/K learning rate, because the 1/sigma_QK in the forward pass
cancels the inflated init variance in distribution. So `param="derived"` and
`param="qk-global"` are THE SAME RUN at alpha_A = 1 (the correction factor
D_h^{2a-2} is 1 there). Any transfer test that means to discriminate them must be
run at alpha_A = 1/2. `groups()` raises if you ask for the discriminating
comparison at a value of alpha_A where it is an identity.
"""

import math

import torch


# --------------------------------------------------------------------------
# graphs
# --------------------------------------------------------------------------

def random_geometric(B, N, radius=0.34, seed=0, dtype=torch.float64):
    """Superpixel-like graphs: irregular degrees, local connectivity.

    Irregularity is load-bearing. On a REGULAR graph the symmetric
    degree-normalised operator D^-1/2 A D^-1/2 and the constant-gamma operator
    A/d are the same matrix, so "degree-normalised vs gamma-normalised" would be
    an identity control. Random geometric graphs have a genuine degree spread.
    """
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(B, N, 2, generator=g, dtype=dtype)
    d2 = (pos[:, :, None, :] - pos[:, None, :, :]).pow(2).sum(-1)
    adj = (d2 < radius ** 2).to(dtype)
    eye = torch.eye(N, dtype=dtype).expand(B, N, N)
    adj = torch.maximum(adj, eye)          # self-loops: MANDATORY (see below)
    return pos, adj


def operator(adj, mode="sym"):
    """Message-passing operator P from a dense adjacency WITH self-loops."""
    deg = adj.sum(-1)
    if mode == "sym":                       # \tilde A = D^-1/2 A D^-1/2
        dm = deg.pow(-0.5)
        return dm[:, :, None] * adj * dm[:, None, :]
    if mode == "row":                       # row-stochastic mean aggregation
        return adj / deg[:, :, None]
    if mode == "sum":                       # P = A, the unnormalised control
        return adj.clone()
    raise ValueError(mode)


def features(B, N, n0, seed=0, sparsity=0.0, dtype=torch.float64):
    """Node features, l2-normalised to ||x_u||^2 = n0 (the paper's Eqn 16).

    `sparsity` in [0,1) zeroes that fraction of entries BEFORE normalising, to
    imitate the sparse bag-of-words features of the citation networks where the
    paper measures C_ab = 13.5-22 (their Table 2).
    """
    g = torch.Generator().manual_seed(seed + 9161)
    X = torch.randn(B, N, n0, generator=g, dtype=dtype)
    if sparsity > 0:
        keep = (torch.rand(B, N, n0, generator=g, dtype=dtype) >= sparsity)
        keep[..., 0] = True                 # never produce an all-zero node
        X = X * keep
    X = X / X.pow(2).sum(-1, keepdim=True).sqrt() * math.sqrt(n0)
    return X


def teacher_targets(P, X, seed=0, width=16):
    """Graph-level scalar targets from a FIXED random 2-layer GCN teacher.

    Deterministic in `seed` and independent of the student's width/depth/heads,
    so a scaling sweep never changes the learning problem.
    """
    g = torch.Generator().manual_seed(seed + 4441)
    n0 = X.shape[-1]
    W1 = torch.randn(n0, width, generator=g, dtype=X.dtype) / math.sqrt(n0)
    W2 = torch.randn(width, width, generator=g, dtype=X.dtype) / math.sqrt(width)
    w = torch.randn(width, generator=g, dtype=X.dtype) / math.sqrt(width)
    h = torch.tanh(P @ (X @ W1))
    h = torch.tanh(P @ (h @ W2))
    y = (h.mean(1) @ w)
    return (y - y.mean()) / y.std()


def dataset(B=16, N=32, n0=8, radius=0.34, seed=0, op="sym", sparsity=0.0,
            dtype=torch.float64):
    pos, adj = random_geometric(B, N, radius, seed, dtype)
    P = operator(adj, op)
    X = features(B, N, n0, seed, sparsity, dtype)
    y = teacher_targets(operator(adj, "sym"), X, seed)
    mask = adj > 0
    _assert_graph_ok(adj, mask)
    return {"X": X, "P": P, "adj": adj, "mask": mask, "y": y}


def _assert_graph_ok(adj, mask):
    """F21-class guards, both graph-specific and both silent if violated."""
    N = adj.shape[-1]
    eye = torch.eye(N, dtype=torch.bool)
    if not bool(mask[:, eye].all()):
        raise ValueError("self-loops missing: softmax over an empty neighbourhood "
                         "is undefined and masked_fill(-inf) yields NaN rows")
    deg = mask.sum(-1).to(torch.float64)
    if float(deg.std()) < 1e-9:
        raise ValueError("degree-regular graph: degree-normalisation and a "
                         "constant gamma are then the SAME operator, so any "
                         "control comparing them is an identity, not evidence")


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class GraphTransformer:
    def __init__(self, D, L, H=4, n0=8, alpha_A=1.0, opt="sgd", seed=0,
                 sigma_QK=1.0, gamma_P=1.0, gamma_A=1.0, attn=True, mpnn=True,
                 param="derived", dtype=torch.float64):
        assert D % H == 0, "heads must partition the width"
        self.D, self.L, self.H, self.Dh = D, L, H, D // H
        self.n0, self.alpha_A, self.opt = n0, alpha_A, opt
        self.gamma_P, self.gamma_A = gamma_P, gamma_A
        self.attn, self.mpnn, self.param = attn, mpnn, param
        self.dtype = dtype
        if self.Dh == 1:
            raise ValueError("D_h = 1: every D_h^k correction collapses to 1 and "
                             "the Q/K comparison becomes an identity control")

        # global-rate init rescalers (10-graph-transformer.md §5)
        self.s0 = math.sqrt(L) if opt == "sgd" else D ** -0.5
        self.sL1 = math.sqrt(L) if opt == "sgd" else D ** -0.5
        self.sQK = sigma_QK

        g = torch.Generator().manual_seed(seed)
        rn = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
        self.W0 = rn(n0, D) * self.s0
        self.Wmp = [rn(D, D) for _ in range(L)]
        self.WQ = [rn(D, D) * self.sQK for _ in range(L)]
        self.WK = [rn(D, D) * self.sQK for _ in range(L)]
        self.WV = [rn(D, D) for _ in range(L)]
        self.WO = [rn(D, D) for _ in range(L)]
        self.W1 = [rn(D, 4 * D) for _ in range(L)]
        self.W2 = [rn(4 * D, D) for _ in range(L)]
        self.Wout = rn(D) * self.sL1
        for p in self.params():
            p.requires_grad_(True)

    def params(self):
        ps = [self.W0, self.Wout]
        for l in range(self.L):
            ps += [self.Wmp[l], self.WQ[l], self.WK[l], self.WV[l],
                   self.WO[l], self.W1[l], self.W2[l]]
        return ps

    # -- the parameterisation table -------------------------------------
    def groups(self, eta0, C_ab=1.0):
        """(param, lr) pairs implementing derivations/10 §5.

        `param="qk-global"` drops the D_h^{2a-2} / D_h^{a-1} Q/K correction, i.e.
        it is the naive extension of the paper's three-row Table 1 to attention.
        """
        D, L, Dh, a = self.D, self.L, self.Dh, self.alpha_A
        if self.opt == "sgd":
            res = eta0 * D * L
            enc = eta0 * D * self.s0 ** 2 * C_ab
            dec = eta0 * D * self.sL1 ** 2
            qk = eta0 * D * L * self.sQK ** 2 * (Dh ** (2 * a - 2))
            qk_naive = eta0 * D * L * self.sQK ** 2
        else:                                    # signgd / adam
            res = eta0 / math.sqrt(D)
            enc = eta0 * self.s0
            dec = eta0 * self.sL1
            qk = eta0 * self.sQK * (Dh ** (a - 1)) / math.sqrt(D)
            qk_naive = eta0 * self.sQK / math.sqrt(D)
        if self.param == "qk-global":
            qk = qk_naive
        elif self.param != "derived":
            raise ValueError(self.param)
        gs = [(self.W0, enc), (self.Wout, dec)]
        for l in range(self.L):
            gs += [(self.Wmp[l], res), (self.WV[l], res), (self.WO[l], res),
                   (self.W1[l], res), (self.W2[l], res),
                   (self.WQ[l], qk), (self.WK[l], qk)]
        return gs

    def qk_correction_is_identity(self):
        """True when `derived` and `qk-global` are algebraically the same run."""
        return abs(self.alpha_A - 1.0) < 1e-12

    # -- forward ---------------------------------------------------------
    def attention(self, Xt, l, mask):
        B, N, D = Xt.shape
        H, Dh = self.H, self.Dh
        q = (Xt @ self.WQ[l]).view(B, N, H, Dh) / (self.sQK * math.sqrt(D))
        k = (Xt @ self.WK[l]).view(B, N, H, Dh) / (self.sQK * math.sqrt(D))
        v = (Xt @ self.WV[l]).view(B, N, H, Dh) / math.sqrt(D)
        logits = torch.einsum("bnhd,bmhd->bhnm", q, k) * (Dh ** -self.alpha_A)
        logits = logits.masked_fill(~mask[:, None], float("-inf"))
        S = torch.softmax(logits, dim=-1)
        o = torch.einsum("bhnm,bmhd->bnhd", S, v).reshape(B, N, D)
        return S, logits, o

    def forward(self, batch, record=False):
        X, P, mask = batch["X"], batch["P"], batch["mask"]
        D, L = self.D, self.L
        rec = {"stream": [], "d_mp": [], "d_at": [], "d_mlp": [], "A": [], "S": []}
        Xl = (X @ self.W0) / (self.s0 * math.sqrt(self.n0))
        for l in range(L):
            if self.mpnn:
                dmp = (P @ Xl) @ self.Wmp[l] / (self.gamma_P * math.sqrt(D)) / L
            else:
                dmp = torch.zeros_like(Xl)
            Xt = Xl + dmp
            if self.attn:
                S, logits, o = self.attention(Xt, l, mask)
                dat = (o @ self.WO[l]) / (self.gamma_A * math.sqrt(D)) / L
            else:
                S = logits = None
                dat = torch.zeros_like(Xt)
            Xh = Xt + dat
            dmlp = (((Xh @ self.W1[l]) / math.sqrt(D)) @ self.W2[l]
                    / math.sqrt(4 * D) / L)
            Xl = Xh + dmlp
            if record:
                rec["stream"].append(Xl.detach())
                rec["d_mp"].append(dmp.detach())
                rec["d_at"].append(dat.detach())
                rec["d_mlp"].append(dmlp.detach())
                if S is not None:
                    rec["A"].append(logits.detach())
                    rec["S"].append(S.detach())
        z = (Xl.mean(1) @ self.Wout) / (self.sL1 * D)
        return (z, rec) if record else z

    def loss(self, batch):
        z = self.forward(batch)
        return 0.5 * (batch["y"] - z).pow(2).mean()


# --------------------------------------------------------------------------
# optimisation
# --------------------------------------------------------------------------

def step(net, batch, eta0, C_ab=1.0, state=None, t=1, eps=1e-14):
    """One optimiser step with the per-group rates of `net.groups`."""
    gs = net.groups(eta0, C_ab)
    for p, _ in gs:
        if p.grad is not None:
            p.grad = None
    net.loss(batch).backward()
    with torch.no_grad():
        for j, (p, lr) in enumerate(gs):
            gr = p.grad
            if gr is None:
                continue
            if net.opt == "sgd":
                p -= lr * gr
            elif net.opt == "signgd":
                p -= lr * torch.sign(gr)
            elif net.opt == "adam":
                m, v = state[j]
                m.mul_(0.9).add_(gr, alpha=0.1)
                v.mul_(0.999).addcmul_(gr, gr, value=0.001)
                mh = m / (1 - 0.9 ** t)
                vh = v / (1 - 0.999 ** t)
                p -= lr * mh / (vh.sqrt() + eps)
            else:
                raise ValueError(net.opt)
    return state


def adam_state(net):
    return [(torch.zeros_like(p), torch.zeros_like(p)) for p, _ in net.groups(1.0)]


def train(net, batch, eta0, steps=20, C_ab=1.0, val=None):
    st = adam_state(net) if net.opt == "adam" else None
    best = float("inf")
    for t in range(1, steps + 1):
        step(net, batch, eta0, C_ab, st, t)
        with torch.no_grad():
            L = float(net.loss(val if val is not None else batch))
        if not math.isfinite(L):
            return float("inf")
        best = min(best, L)
    return best


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------

def gamma_of(P, X):
    """gamma^2 = E||P X||_F^2 / E||X||_F^2 -- the paper's Eqn 14/19."""
    return float((P @ X).pow(2).sum() / X.pow(2).sum()).__pow__(0.5)


def attention_stats(net, batch):
    """gamma_A, effective span d_eff, logit spread, degree -- per derivations/10 §4."""
    with torch.no_grad():
        _, rec = net.forward(batch, record=True)
        out = []
        for l, S in enumerate(rec["S"]):
            # S: (B,H,N,N) row-stochastic on the mask
            d_eff = 1.0 / S.pow(2).sum(-1)                    # participation ratio
            A = rec["A"][l].masked_fill(~batch["mask"][:, None], float("nan"))
            Xl = rec["stream"][l]
            # gamma_A per head, averaged
            num = torch.einsum("bhnm,bmd->bhnd", S, Xl).pow(2).sum()
            gam = float((num / (net.H * Xl.pow(2).sum())).sqrt())
            out.append({"layer": l, "gamma_A": gam,
                        "d_eff": float(d_eff.mean()),
                        "A_std": float(A[~A.isnan()].std()),
                        "A_absmean": float(A[~A.isnan()].abs().mean())})
        return out


def branch_scales(net, batch):
    """RMS of the stream and of each residual branch's contribution."""
    with torch.no_grad():
        _, rec = net.forward(batch, record=True)
        rms = lambda ts: [float(t.pow(2).mean().sqrt()) for t in ts]
        return {"stream": rms(rec["stream"]), "d_mp": rms(rec["d_mp"]),
                "d_at": rms(rec["d_at"]), "d_mlp": rms(rec["d_mlp"])}


def delta_A(net, batch, eta0, steps=1, layer=0, C_ab=1.0):
    """RMS change in the attention LOGITS after `steps` optimiser steps.

    This is the F18 probe: the C2 contraction (backward through W_O, dotted into
    the value vectors) is INCOHERENT at t=1 and COHERENT thereafter, so the same
    quantity has two different exponents at t=1 and t>>1.
    """
    with torch.no_grad():
        _, r0 = net.forward(batch, record=True)
        A0 = r0["A"][layer].clone()
    st = adam_state(net) if net.opt == "adam" else None
    for t in range(1, steps + 1):
        step(net, batch, eta0, C_ab, st, t)
    with torch.no_grad():
        _, r1 = net.forward(batch, record=True)
        d = (r1["A"][layer] - A0)
        m = batch["mask"][:, None].expand_as(d)
        return float(d[m].pow(2).mean().sqrt()), float(A0[m].pow(2).mean().sqrt())
