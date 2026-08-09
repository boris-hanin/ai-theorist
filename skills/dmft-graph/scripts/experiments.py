"""Round 011 experiments: graph-transformer parameterisation and HP transfer.

Every verdict on a transfer sweep comes from
`skills/dmft-derivation/scripts/transfer.py::verdict` -- the three-bar rule
(statistical resolution, 0.3-decade practical bar, and the SHAPE of the drift,
F22). It is imported, not reimplemented.

    python experiments.py E1 E2 E3 E4 E5 E6 E7      (or `all`)
"""

import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "skills", "dmft-derivation", "scripts"))

import gt                                                        # noqa: E402
from transfer import verdict, render                             # noqa: E402

OUT = os.path.join(ROOT, "rounds", "011-graph-transformer")
torch.set_num_threads(max(1, os.cpu_count() // 2))

BASE = dict(B=12, N=24, n0=8, radius=0.36)


def fit(xs, ys):
    """log-log slope with a paired-halving style s.e. from the residuals."""
    lx, ly = np.log10(np.asarray(xs, float)), np.log10(np.asarray(ys, float))
    ok = np.isfinite(lx) & np.isfinite(ly)
    lx, ly = lx[ok], ly[ok]
    if lx.size < 2:
        return float("nan"), float("nan")
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, res, *_ = np.linalg.lstsq(A, ly, rcond=None)
    if lx.size > 2:
        r = ly - A @ coef
        se = float(np.sqrt((r ** 2).sum() / (lx.size - 2)
                           / ((lx - lx.mean()) ** 2).sum()))
    else:
        se = float("nan")
    return float(coef[0]), se


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)
    print("  -> wrote", name)


# ==========================================================================
# E1  step-0 / one-step scaling audit
# ==========================================================================

def E1():
    print("\n=== E1  scaling audit: features and per-branch updates ===")
    data = gt.dataset(seed=0, **BASE)
    res = {"D": {}, "L": {}, "H": {}}

    for tag, dials, build in [
        ("D", [32, 64, 128, 256],
         lambda d, s: gt.GraphTransformer(D=d, L=3, H=4, alpha_A=1.0, seed=s, **{})),
        ("L", [2, 3, 4, 6, 8],
         lambda l, s: gt.GraphTransformer(D=64, L=l, H=4, alpha_A=1.0, seed=s)),
        ("H", [1, 2, 4, 8],
         lambda h, s: gt.GraphTransformer(D=128, L=3, H=h, alpha_A=1.0, seed=s)),
    ]:
        rows = []
        for d in dials:
            acc = {k: [] for k in ("stream", "d_mp", "d_at", "d_mlp")}
            for s in range(4):
                bs = gt.branch_scales(build(d, s), data)
                for k in acc:
                    acc[k].append(float(np.mean(bs[k])))
            rows.append({"dial": d, **{k: float(np.mean(v)) for k, v in acc.items()},
                         **{k + "_sd": float(np.std(v)) for k, v in acc.items()}})
        sl = {k: fit([r["dial"] for r in rows], [r[k] for r in rows])
              for k in ("stream", "d_mp", "d_at", "d_mlp")}
        res[tag] = {"rows": rows, "slopes": sl}
        print(" ", tag, {k: "%.3f+-%.3f" % v for k, v in sl.items()})
    dump("E1-scaling-audit.json", res)
    return res


# ==========================================================================
# E2  gamma_A, effective attention span vs width, at three alpha_A
# ==========================================================================

def E2():
    print("\n=== E2  gamma_A and d_eff vs D, at alpha_A in {0, 1/2, 1} ===")
    data = gt.dataset(seed=0, **BASE)
    deg = float(data["mask"].sum(-1).to(torch.float64).mean())
    out = {"mean_degree": deg, "runs": {}}
    for a in (0.0, 0.5, 1.0):
        rows = []
        for D in (32, 64, 128, 256, 512):
            g_, de, asd = [], [], []
            for s in range(4):
                net = gt.GraphTransformer(D=D, L=3, H=4, alpha_A=a, seed=s)
                st = gt.attention_stats(net, data)
                g_.append(np.mean([r["gamma_A"] for r in st]))
                de.append(np.mean([r["d_eff"] for r in st]))
                asd.append(np.mean([r["A_std"] for r in st]))
            rows.append({"D": D, "gamma_A": float(np.mean(g_)),
                         "gamma_sd": float(np.std(g_)),
                         "d_eff": float(np.mean(de)), "A_std": float(np.mean(asd))})
        out["runs"]["alpha_A=%.2f" % a] = {
            "rows": rows,
            "slope_gamma": fit([r["D"] for r in rows], [r["gamma_A"] for r in rows]),
            "slope_deff": fit([r["D"] for r in rows], [r["d_eff"] for r in rows]),
            "slope_Astd": fit([r["D"] for r in rows], [r["A_std"] for r in rows]),
        }
        print("  alpha_A=%.1f  gamma slope %.3f  d_eff slope %.3f  A_std slope %.3f"
              % (a, out["runs"]["alpha_A=%.2f" % a]["slope_gamma"][0],
                 out["runs"]["alpha_A=%.2f" % a]["slope_deff"][0],
                 out["runs"]["alpha_A=%.2f" % a]["slope_Astd"][0]))
    dump("E2-gamma-vs-width.json", out)
    return out


# ==========================================================================
# E3  Delta A at t = 1 vs D_h  (the F18 probe), SGD vs signGD
# ==========================================================================

def E3():
    print("\n=== E3  Delta A at t=1 and t=8 vs D_h; SGD vs signGD ===")
    data = gt.dataset(seed=0, **BASE)
    out = {}
    for opt, eta0 in (("sgd", 0.05), ("signgd", 0.02)):
        for a in (1.0, 0.5):
            rows = []
            for D in (32, 64, 128, 256, 512):
                d1, d8, a0 = [], [], []
                for s in range(4):
                    n1 = gt.GraphTransformer(D=D, L=3, H=4, alpha_A=a, opt=opt, seed=s)
                    x, ai = gt.delta_A(n1, data, eta0, steps=1)
                    n8 = gt.GraphTransformer(D=D, L=3, H=4, alpha_A=a, opt=opt, seed=s)
                    y, _ = gt.delta_A(n8, data, eta0, steps=8)
                    d1.append(x); d8.append(y); a0.append(ai)
                rows.append({"D": D, "Dh": D // 4, "dA_t1": float(np.mean(d1)),
                             "dA_t8": float(np.mean(d8)), "A_init": float(np.mean(a0)),
                             "dA_t1_sd": float(np.std(d1))})
            key = "%s_alphaA=%.1f" % (opt, a)
            out[key] = {"rows": rows,
                        "slope_t1": fit([r["Dh"] for r in rows], [r["dA_t1"] for r in rows]),
                        "slope_t8": fit([r["Dh"] for r in rows], [r["dA_t8"] for r in rows]),
                        "slope_init": fit([r["Dh"] for r in rows], [r["A_init"] for r in rows])}
            print("  %-18s  dA(t=1) %.3f  dA(t=8) %.3f  A_init %.3f"
                  % (key, out[key]["slope_t1"][0], out[key]["slope_t8"][0],
                     out[key]["slope_init"][0]))
    dump("E3-deltaA.json", out)
    return out


# ==========================================================================
# E4  HP transfer sweeps  (verdict from transfer.py, unmodified)
# ==========================================================================

def _sweep(build, dials, lr_grid, steps, seeds, data):
    per = np.full((len(dials), len(lr_grid), len(seeds)), np.inf)
    for i, d in enumerate(dials):
        for j, lr in enumerate(lr_grid):
            for k, s in enumerate(seeds):
                per[i, j, k] = gt.train(build(d, s), data, lr, steps=steps)
    return np.median(per, axis=2), per


def _leg(name, build, dials, lr_grid, steps, seeds, data, out):
    losses, per = _sweep(build, dials, lr_grid, steps, seeds, data)
    v = verdict(losses, per, lr_grid, dials)
    print(render(name, dials, lr_grid, losses, v))
    out[name] = {k: (val.tolist() if isinstance(val, np.ndarray) else val)
                 for k, val in v.items()}
    out[name]["losses"] = losses.tolist()
    return v


def E4(quick=False):
    print("\n=== E4  learning-rate transfer across D, L, H ===")
    data = gt.dataset(seed=0, **BASE)
    seeds = (0, 1, 2, 3)
    steps = 12 if quick else 24
    out = {"steps": steps, "seeds": list(seeds)}

    # --- SGD, alpha_A = 1 (the recommended exponent) -------------------
    lrg = np.logspace(-3.0, -0.5, 8)
    _leg("SGD width D (alpha_A=1)",
         lambda D, s: gt.GraphTransformer(D=D, L=3, H=4, alpha_A=1.0, opt="sgd", seed=s),
         [32, 64, 128, 256], lrg, steps, seeds, data, out)
    _leg("SGD depth L (alpha_A=1)",
         lambda L, s: gt.GraphTransformer(D=64, L=L, H=4, alpha_A=1.0, opt="sgd", seed=s),
         [2, 3, 4, 6], lrg, steps, seeds, data, out)
    _leg("SGD heads H (alpha_A=1)",
         lambda H, s: gt.GraphTransformer(D=128, L=3, H=H, alpha_A=1.0, opt="sgd", seed=s),
         [1, 2, 4, 8], lrg, steps, seeds, data, out)

    # --- signGD (Adam proxy) -------------------------------------------
    lrs = np.logspace(-2.5, 0.0, 8)
    _leg("signGD width D (alpha_A=1)",
         lambda D, s: gt.GraphTransformer(D=D, L=3, H=4, alpha_A=1.0, opt="signgd", seed=s),
         [32, 64, 128, 256], lrs, steps, seeds, data, out)
    _leg("signGD depth L (alpha_A=1)",
         lambda L, s: gt.GraphTransformer(D=64, L=L, H=4, alpha_A=1.0, opt="signgd", seed=s),
         [2, 3, 4, 6], lrs, steps, seeds, data, out)
    _leg("signGD heads H (alpha_A=1)",
         lambda H, s: gt.GraphTransformer(D=128, L=3, H=H, alpha_A=1.0, opt="signgd", seed=s),
         [1, 2, 4, 8], lrs, steps, seeds, data, out)

    # --- CONTROL 1: alpha_A = 0 (unnormalised logits) ------------------
    # derivations/10 §4: gamma_A drifts with D, so width transfer must break.
    _leg("CONTROL alpha_A=0, width D (SGD)",
         lambda D, s: gt.GraphTransformer(D=D, L=3, H=4, alpha_A=0.0, opt="sgd", seed=s),
         [32, 64, 128, 256], lrg, steps, seeds, data, out)

    # --- CONTROL 2: Q/K correction dropped, at alpha_A = 1/2 -----------
    # At alpha_A = 1 this control is an ALGEBRAIC IDENTITY (10 §3f) and is not run.
    probe = gt.GraphTransformer(D=64, L=3, H=4, alpha_A=0.5, opt="sgd", seed=0)
    assert not probe.qk_correction_is_identity(), "control would be an identity"
    _leg("derived  alpha_A=0.5, width D (SGD)",
         lambda D, s: gt.GraphTransformer(D=D, L=3, H=4, alpha_A=0.5, opt="sgd",
                                          param="derived", seed=s),
         [32, 64, 128, 256], lrg, steps, seeds, data, out)
    _leg("CONTROL qk-global alpha_A=0.5, width D (SGD)",
         lambda D, s: gt.GraphTransformer(D=D, L=3, H=4, alpha_A=0.5, opt="sgd",
                                          param="qk-global", seed=s),
         [32, 64, 128, 256], lrg, steps, seeds, data, out)

    # --- CONTROL 3: unnormalised message passing P = A -----------------
    datA = gt.dataset(seed=0, op="sum", **BASE)
    _leg("CONTROL P=A unnormalised, width D (SGD)",
         lambda D, s: gt.GraphTransformer(D=D, L=3, H=4, alpha_A=1.0, opt="sgd", seed=s),
         [32, 64, 128, 256], lrg, steps, seeds, datA, out)

    # --- CONTROL 4: mis-scaled global LR (no D in the SGD rate) --------
    class _Bad(gt.GraphTransformer):
        def groups(self, eta0, C_ab=1.0):
            return [(p, lr / self.D) for p, lr in
                    gt.GraphTransformer.groups(self, eta0, C_ab)]
    _leg("CONTROL no-D SGD rate, width D",
         lambda D, s: _Bad(D=D, L=3, H=4, alpha_A=1.0, opt="sgd", seed=s),
         [32, 64, 128, 256], np.logspace(-1.0, 1.5, 8), steps, seeds, data, out)

    dump("E4-transfer.json", out)
    return out


# ==========================================================================
# E5  gamma_l vs layer (oversmoothing), with a decorrelating control
# ==========================================================================

def E5():
    print("\n=== E5  gamma_l vs depth: oversmoothing raises rho ===")
    out = {}
    for tag, kw in (("geometric", {}), ("rewired-dense", {"radius": 0.9})):
        b = dict(BASE); b.update(kw)
        data = gt.dataset(seed=0, **b)
        rows = []
        for s in range(4):
            net = gt.GraphTransformer(D=128, L=8, H=4, alpha_A=1.0, seed=s)
            st = gt.attention_stats(net, data)
            with torch.no_grad():
                _, rec = net.forward(data, record=True)
                gp = [gt.gamma_of(data["P"], X) for X in rec["stream"]]
                # neighbour feature correlation rho_l
                rho = []
                for X in rec["stream"]:
                    Xn = X / X.pow(2).sum(-1, keepdim=True).sqrt()
                    G = Xn @ Xn.transpose(1, 2)
                    m = data["mask"] & ~torch.eye(
                        X.shape[1], dtype=torch.bool).expand_as(data["mask"])
                    rho.append(float(G[m].mean()))
            rows.append({"gamma_A": [r["gamma_A"] for r in st],
                         "gamma_P": gp, "rho": rho})
        agg = {k: np.mean([r[k] for r in rows], axis=0).tolist()
               for k in ("gamma_A", "gamma_P", "rho")}
        out[tag] = agg
        print("  %-14s rho_l  %s" % (tag, " ".join("%.3f" % x for x in agg["rho"])))
        print("  %-14s gamP   %s" % ("", " ".join("%.3f" % x for x in agg["gamma_P"])))
        print("  %-14s gamA   %s" % ("", " ".join("%.3f" % x for x in agg["gamma_A"])))
    dump("E5-gamma-vs-depth.json", out)
    return out


# ==========================================================================
# E6  alignment factors: pooled (encoder, C_ab) vs node-level (attention)
# ==========================================================================

def E6():
    print("\n=== E6  C_ab (pooled) vs the node-level attention alignment ===")
    out = {}
    for sp in (0.0, 0.5, 0.9, 0.97):
        b = dict(BASE); b["B"] = 24
        d = gt.dataset(seed=0, sparsity=sp, **b)
        X, n0 = d["X"], b["n0"]
        B, N, _ = X.shape
        Cs, Ans = [], []
        for i in range(0, B, 2):
            Xa, Xb = X[i], X[i + 1]
            # pooled: M_ab = || (1/N_a) 1^T X_a X_b^T ||_2      (their Eqn 25)
            M = float((Xa.mean(0) @ Xb.T).pow(2).sum().sqrt())
            Cs.append(n0 * math.sqrt(N) / max(M, 1e-30))
            # node-level: the Gram that the Q/K gradient actually reads out
            MA = float((Xb @ Xa.T).pow(2).mean().sqrt() * math.sqrt(N))
            Ans.append(n0 * math.sqrt(N) / max(MA, 1e-30))
        out["sparsity=%.2f" % sp] = {
            "C_ab_pooled": float(np.mean(Cs)),
            "C_attn_node": float(np.mean(Ans)),
            "ratio": float(np.mean(Cs) / np.mean(Ans))}
        print("  sparsity %.2f   C_ab(pooled) %8.2f   C(node-level) %8.2f   ratio %6.2f"
              % (sp, np.mean(Cs), np.mean(Ans), np.mean(Cs) / np.mean(Ans)))
    dump("E6-alignment.json", out)
    return out


# ==========================================================================
# E7  S1/S2: attention concentration and head collapse vs alpha_A
# ==========================================================================

def E7():
    print("\n=== E7  attention concentration (S1) and head spread (S2) ===")
    data = gt.dataset(seed=0, **BASE)
    out = {}
    for a in (0.5, 1.0):
        rows = []
        for D in (32, 64, 128, 256, 512):
            per_seed = []
            for s in range(8):
                net = gt.GraphTransformer(D=D, L=2, H=4, alpha_A=a, seed=s)
                with torch.no_grad():
                    _, rec = net.forward(data, record=True)
                    A = rec["A"][0]                      # (B,H,N,N)
                    m = data["mask"][:, None].expand_as(A)
                    per_seed.append(A.masked_fill(~m, 0.0))
            stack = torch.stack(per_seed)                # (S,B,H,N,N)
            m = data["mask"][:, None].expand_as(stack[0])
            # S1: across-SEED sd of a fixed entry (LLN vs CLT in D_h)
            sd_seed = float(stack.std(0)[m].pow(2).mean().sqrt())
            # S2: across-HEAD sd at fixed seed and entry
            sd_head = float(stack.std(2).mean(0)[data["mask"]].pow(2).mean().sqrt())
            rows.append({"D": D, "Dh": D // 4, "sd_seed": sd_seed, "sd_head": sd_head})
        out["alpha_A=%.1f" % a] = {
            "rows": rows,
            "slope_seed": fit([r["Dh"] for r in rows], [r["sd_seed"] for r in rows]),
            "slope_head": fit([r["Dh"] for r in rows], [r["sd_head"] for r in rows])}
        print("  alpha_A=%.1f  sd_seed slope %.3f   sd_head slope %.3f"
              % (a, out["alpha_A=%.1f" % a]["slope_seed"][0],
                 out["alpha_A=%.1f" % a]["slope_head"][0]))
    dump("E7-concentration.json", out)
    return out


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    todo = ["E1", "E2", "E3", "E4", "E5", "E6", "E7"] if "all" in args else args
    for name in todo:
        globals()[name]()
