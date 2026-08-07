"""Networks for the scaling-audit and HP-transfer harness.

One class covering MLPs and residual nets under either muP or the standard
(PyTorch-default) parameterisation, with the residual branch exponent `alpha`
left FREE rather than hardcoded. Keeping alpha free is the point: `algorithm.md`
Step 0 says never derive against unpinned exponents, and the repo's depth story
was previously built on alpha = 1/2 inherited from a paper without an audit.

Architectures
-------------
MLP        h^{l+1} = beta_h * W^l phi(h^l)
residual   h^{l+1} = h^l + L^{-alpha} * (block_k(h^l))
             k = 1:  beta_h * W phi(h)
             k = 2:  beta_h * W2 phi(beta_h * W1 phi(h))     (two-layer block)

Block depth k matters: CompleteP (arXiv 2505.01618) argues alpha = 1/2 is
right for k = 1 but asymptotically linearises blocks for k >= 2, where
alpha = 1 is needed for complete feature learning. This class can run both, so
the question is decided by measurement.

Parameterisations
-----------------
muP  raw weights ~ N(0,1); multipliers 1/sqrt(D), 1/sqrt(N), 1/(gamma_0 N) on
     the readout; per-group update scalings as derived in
     `derivations/01-deep-mlp.md` §3, so the base LR eta_0 is width- and
     depth-transferable (that is the claim under test).
sp   PyTorch default: W ~ N(0, 1/fan_in) with no forward multipliers and one
     global LR. The negative control -- it must FAIL to transfer, or the test
     is under-powered and says nothing.
"""

import numpy as np

import activations


class Net(object):
    def __init__(self, D, N, L, param="mup", gamma0=1.0, act="tanh", seed=0,
                 residual=False, alpha=0.5, block_k=1, lr_depth_exp=None):
        self.D, self.N, self.L = D, N, L
        self.param, self.gamma0, self.act = param, gamma0, act
        self.residual, self.alpha, self.block_k = residual, alpha, block_k
        # CompleteP's general rule is a per-layer LR factor L^{alpha-1};
        # depth-muP uses none. Default follows CompleteP unless overridden.
        self.lr_depth_exp = (alpha - 1.0) if lr_depth_exp is None else lr_depth_exp
        self.phi, self.phidot = activations.get(act)
        rng = np.random.default_rng(seed)

        if param == "mup":
            self.b_in, self.b_h = 1.0 / np.sqrt(D), 1.0 / np.sqrt(N)
            self.b_out = 1.0 / (gamma0 * N)
            self.W0 = rng.standard_normal((N, D))
            self.Wh = [[rng.standard_normal((N, N)) for _ in range(block_k)]
                       for _ in range(L - 1 if not residual else L)]
            self.w = rng.standard_normal(N)
        elif param == "sp":
            self.b_in = self.b_h = self.b_out = 1.0
            self.W0 = rng.standard_normal((N, D)) / np.sqrt(D)
            self.Wh = [[rng.standard_normal((N, N)) / np.sqrt(N)
                        for _ in range(block_k)]
                       for _ in range(L - 1 if not residual else L)]
            self.w = rng.standard_normal(N) / np.sqrt(N)
        else:
            raise ValueError("param must be 'mup' or 'sp'")
        self.n_blocks = len(self.Wh)

    # -- forward ------------------------------------------------------------
    def forward(self, X):
        """Returns f (P,), and the cache needed for backprop."""
        phi = self.phi
        h = self.b_in * (self.W0 @ X)                    # h^1, (N, P)
        hs, inner = [h], []
        scale = self.L ** (-self.alpha) if self.residual else 1.0
        for b in range(self.n_blocks):
            Ws = self.Wh[b]
            a = phi(h)
            mids = []
            for j, W in enumerate(Ws):
                a = self.b_h * (W @ a)
                mids.append(a)
                if j < len(Ws) - 1:
                    a = phi(a)
            inner.append(mids)
            h = h + scale * a if self.residual else a
            hs.append(h)
        f = self.b_out * (self.w @ phi(hs[-1]))
        return f, (hs, inner)

    # -- backward + update --------------------------------------------------
    def step(self, X, y, lr):
        """One GD step on L = sum_mu (1/2)(y-f)^2. Returns (f, per-group RMS)."""
        phi, phidot = self.phi, self.phidot
        f, (hs, inner) = self.forward(X)
        delta = y - f                                    # (P,)
        scale = self.L ** (-self.alpha) if self.residual else 1.0

        # dL/df_mu = -delta_mu ; propagate e^l = dF/dh^l (unnormalised)
        e = np.outer(self.w, delta) * phidot(hs[-1]) * self.b_out   # (N,P)
        gw = self.b_out * (phi(hs[-1]) @ delta)

        gWh = [[None] * len(Ws) for Ws in self.Wh]
        for b in range(self.n_blocks - 1, -1, -1):
            Ws, mids = self.Wh[b], inner[b]
            eb = e * scale if self.residual else e
            # walk back through the block
            cur = eb
            for j in range(len(Ws) - 1, -1, -1):
                a_in = phi(hs[b]) if j == 0 else phi(mids[j - 1])
                gWh[b][j] = self.b_h * (cur @ a_in.T)
                cur = self.b_h * (Ws[j].T @ cur)
                if j > 0:
                    cur = cur * phidot(mids[j - 1])
            e = (e + cur * phidot(hs[b])) if self.residual else (cur * phidot(hs[b]))
        gW0 = self.b_in * (e @ X.T)

        # muP: the derived per-group scalings; sp: one global LR.
        if self.param == "mup":
            s0 = lr * self.gamma0 ** 2 * self.N
            sh = s0 * (self.L ** self.lr_depth_exp if self.residual else 1.0)
            self.W0 += s0 * gW0
            for b in range(self.n_blocks):
                for j in range(len(self.Wh[b])):
                    self.Wh[b][j] += sh * gWh[b][j]
            self.w += s0 * gw
            groups = {"W0": s0 * gW0, "Wh": sh * gWh[0][0] if self.n_blocks else None,
                      "w": s0 * gw}
        else:
            self.W0 += lr * gW0
            for b in range(self.n_blocks):
                for j in range(len(self.Wh[b])):
                    self.Wh[b][j] += lr * gWh[b][j]
            self.w += lr * gw
            groups = {"W0": lr * gW0, "Wh": lr * gWh[0][0] if self.n_blocks else None,
                      "w": lr * gw}
        return f, groups

    def loss(self, X, y):
        f, _ = self.forward(X)
        return 0.5 * float(np.sum((y - f) ** 2)), f

    def features(self, X):
        _, (hs, _) = self.forward(X)
        return hs


def teacher_data(P, D, seed=0, D_out=1):
    """A fixed regression task: random inputs, random smooth teacher."""
    rng = np.random.default_rng(1000 + seed)
    X = rng.standard_normal((D, P))
    X = X / np.linalg.norm(X, axis=0, keepdims=True) * np.sqrt(D)
    Wt = rng.standard_normal((32, D)) / np.sqrt(D)
    vt = rng.standard_normal(32) / np.sqrt(32)
    y = vt @ np.tanh(Wt @ X)
    y = y / np.std(y)
    return X, y


def train(net, X, y, lr, steps):
    """Train and return the final loss (nan-safe: divergence -> inf)."""
    for _ in range(steps):
        f, _ = net.step(X, y, lr)
        if not np.all(np.isfinite(f)):
            return np.inf
    loss, f = net.loss(X, y)
    return loss if np.isfinite(loss) else np.inf
