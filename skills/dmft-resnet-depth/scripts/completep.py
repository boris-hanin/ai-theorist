"""Pre-LN transformer under SP / muP / alpha in {0.5, 1}, per Table 1 of
arXiv:2505.01618 (CompleteP). Optimiser: Adam, no weight decay.

Every rule below is derived in `derivations/04-completep.md` and matches the
paper's Table 1:

    hidden init var   sigma^2_base * m_N^{-1}          (SP: no m_N)
    hidden LR         eta_base * m_N^{-1} * m_L^{a-1}  (SP: eta_base)
    emb / LN / bias LR eta_base * m_L^{a-1}            (no m_N -- per-activation)
    emb / final-LN / unemb LR  eta_base                (outside the stack)
    unemb forward     x m_N^{-1}                       (SP: x 1)
    residual          h + m_L^{-a} F(LN(h))            (SP/muP: h + F(LN(h)))
    Adam eps          eps_base m_N^{-1} m_L^{-a}       (blocks); m_N^{-1} elsewhere

`m_N = N/N_base`, `m_L = L/L_base`. At the base shape all parameterisations
coincide, which is the sanity check the sweep starts from.

Attention uses Q^T K / N (i.e. alpha_A = 1) for every parameterisation, as the
paper does -- that is the muP attention scaling, and 2405.15712 shows it is
required for the N -> inf limit to exist at all.
"""

import math

import torch
import torch.nn.functional as F

N_BASE, L_BASE = 64, 2          # small base shape; paper uses 256 / 2
# sigma_base: Table 1 gives every layer variance sigma^2_base at the base shape,
# with m_N^{-1} applied to hidden only. Choose it so the BASE model is itself
# well conditioned -- fan-in scale at N_BASE -- otherwise init logits are
# Theta(sqrt(N_base)) and the run starts 5x above ln(V).
SIGMA_BASE = N_BASE ** -0.5


class Block(torch.nn.Module):
    def __init__(self, N, heads, g, dev):
        super().__init__()
        mk = lambda *s: torch.nn.Parameter(
            torch.randn(*s, generator=g, device=dev) * SIGMA_BASE)
        self.heads = heads
        self.ln1_g, self.ln1_b = mk(N), torch.nn.Parameter(torch.zeros(N, device=dev))
        self.ln2_g, self.ln2_b = mk(N), torch.nn.Parameter(torch.zeros(N, device=dev))
        self.Wq, self.Wk, self.Wv, self.Wo = mk(N, N), mk(N, N), mk(N, N), mk(N, N)
        self.W1, self.W2 = mk(4 * N, N), mk(N, 4 * N)
        self.b1 = torch.nn.Parameter(torch.zeros(4 * N, device=dev))
        self.b2 = torch.nn.Parameter(torch.zeros(N, device=dev))

    def forward(self, h, res):
        N = h.shape[-1]
        H, dh = self.heads, N // self.heads
        x = F.layer_norm(h, (N,)) * self.ln1_g + self.ln1_b
        B, S, _ = x.shape
        q = (x @ self.Wq.T).view(B, S, H, dh).transpose(1, 2)
        k = (x @ self.Wk.T).view(B, S, H, dh).transpose(1, 2)
        v = (x @ self.Wv.T).view(B, S, H, dh).transpose(1, 2)
        # Q^T K / N  (alpha_A = 1, muP attention scaling) -- per-head dim is dh
        a = torch.softmax((q @ k.transpose(-1, -2)) / dh, dim=-1)
        o = (a @ v).transpose(1, 2).reshape(B, S, N) @ self.Wo.T
        h = h + res * o
        y = F.layer_norm(h, (N,)) * self.ln2_g + self.ln2_b
        y = F.gelu(y @ self.W1.T + self.b1) @ self.W2.T + self.b2
        return h + res * y


class Model(torch.nn.Module):
    """`param` in {'sp','mup','alpha'}; `alpha` used only when param='alpha'."""

    def __init__(self, V, N, L, heads=4, param="alpha", alpha=1.0, seed=0,
                 device="cpu"):
        super().__init__()
        self.N, self.L, self.param, self.alpha = N, L, param, alpha
        self.mN, self.mL = N / N_BASE, L / L_BASE
        g = torch.Generator(device=device).manual_seed(seed)
        mk = lambda *s: torch.nn.Parameter(
            torch.randn(*s, generator=g, device=device) * SIGMA_BASE)

        self.emb = mk(V, N)
        self.pos = mk(64, N)
        self.blocks = torch.nn.ModuleList([Block(N, heads, g, device) for _ in range(L)])
        self.lnf_g = mk(N)
        self.lnf_b = torch.nn.Parameter(torch.zeros(N, device=device))
        self.unemb = mk(V, N)

        # Hidden init var * m_N^{-1}  (D3). SP leaves init alone.
        if param in ("mup", "alpha"):
            with torch.no_grad():
                for b in self.blocks:
                    for W in (b.Wq, b.Wk, b.Wv, b.Wo, b.W1, b.W2):
                        W.mul_(self.mN ** -0.5)
        # residual branch scale
        self.res = (self.mL ** -alpha) if param == "alpha" else 1.0
        # unembedding forward multiplier (D4)
        self.uscale = (self.mN ** -1.0) if param in ("mup", "alpha") else 1.0

    def forward(self, idx):
        h = self.emb[idx] + self.pos[: idx.shape[1]]
        for b in self.blocks:
            h = b(h, self.res)
        h = F.layer_norm(h, (self.N,)) * self.lnf_g + self.lnf_b
        return (h @ self.unemb.T) * self.uscale

    # -- Table 1 learning rates ------------------------------------------
    def groups(self, eta, eps_base=1e-12, scale_width=True, scale_depth=True):
        """Adam param groups. The scale_* flags drop a predicted factor so a
        sweep can check the corresponding control FAILS."""
        mN = self.mN if scale_width else 1.0
        dep = (self.mL ** (self.alpha - 1.0)) if (self.param == "alpha"
                                                  and scale_depth) else 1.0
        muP = self.param in ("mup", "alpha")
        e_out = eps_base * (mN ** -1 if muP else 1.0)
        e_blk = e_out * ((self.mL ** -self.alpha) if self.param == "alpha" else 1.0)

        fanin, peract = [], []
        for b in self.blocks:
            fanin += [b.Wq, b.Wk, b.Wv, b.Wo, b.W1, b.W2]
            peract += [b.ln1_g, b.ln1_b, b.ln2_g, b.ln2_b, b.b1, b.b2]
        outside = [self.emb, self.pos, self.lnf_g, self.lnf_b, self.unemb]
        gs = [
            # D1 + D5: fan-in weights take both factors
            {"params": fanin, "lr": eta * (mN ** -1 if muP else 1.0) * dep, "eps": e_blk},
            # D2 + D5: per-activation in-block params take only the depth factor
            {"params": peract, "lr": eta * dep, "eps": e_blk},
            # outside the residual stack: neither factor
            {"params": outside, "lr": eta, "eps": e_out},
        ]
        return gs


# ---------------------------------------------------------------------------

def task(B, S, V, seed=0, device="cpu"):
    """Fixed synthetic next-token task with a copy-like dependency, so the
    model has something attention can actually exploit."""
    g = torch.Generator(device=device).manual_seed(7000 + seed)
    x = torch.randint(0, V, (B, S + 1), generator=g, device=device)
    # target at position s is a deterministic mix of tokens s and s-2
    y = (x[:, :-1] * 3 + torch.roll(x[:, :-1], 2, dims=1)) % V
    return x[:, :-1], y


def train(model, x, y, eta, steps, **flags):
    opt = torch.optim.Adam(model.groups(eta, **flags), betas=(0.9, 0.95),
                           weight_decay=0.0)
    V = model.unemb.shape[0]
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        if not torch.isfinite(loss):
            return float("inf")
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item()
    return out if math.isfinite(out) else float("inf")
