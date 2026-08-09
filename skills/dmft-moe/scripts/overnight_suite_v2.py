"""Reconstructed Round 010 v2 runner for the MoE Mean-ODE limit.

The original v2 source was not committed.  This reconstruction uses the exact
effective ladder, seed counts, horizons, and LR grid recorded in ``big.log``
and ``rounds/010-overnight/README.md``.  It is not claimed to be a byte-for-byte
copy of the lost source.  The default preset is the historical full A100 run;
``--smoke`` is a small portability check.
"""

import argparse
import json
import math
import os
import time
import traceback

import numpy as np
import torch


def shape(C, active=4):
    L = max(2, int(round(C ** (1 / 6))))
    D = max(4, int(round(C ** (1 / 3))))
    M = max(2, int(round(C / (L * D * active))))
    return L, D, M


class Suite:
    def __init__(self, device, output, smoke=False):
        self.device = device
        self.output = os.path.abspath(output)
        self.smoke = smoke
        self.results = {
            "schema_version": 2,
            "runner": "reconstructed from big.log; original v2 source was lost",
            "device": str(device),
        }
        g = torch.Generator(device=device).manual_seed(4242)
        self.X = torch.randn(16, 8, generator=g, device=device, dtype=torch.float64)
        self.Y = torch.randn(16, generator=g, device=device, dtype=torch.float64)

    def log(self, *args):
        print("[%s]" % time.strftime("%H:%M:%S"), *args, flush=True)

    def save(self):
        os.makedirs(os.path.dirname(self.output), exist_ok=True)
        tmp = self.output + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(self.results, handle, indent=1, allow_nan=False)
        os.replace(tmp, self.output)

    def empty_cache(self):
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

    def make(self, S, L, D, M, E, d0, generator):
        rn = lambda *dims: torch.randn(
            *dims, generator=generator, device=self.device, dtype=torch.float64)
        return {
            "U": [rn(S, E, D, M) * D ** -0.5 for _ in range(L)],
            "W": [rn(S, E, M, D) for _ in range(L)],
            "R": [rn(S, D, E) * D ** -0.5 for _ in range(L)],
            "b": [rn(S, E) for _ in range(L)],
            "We": rn(S, d0, D) / math.sqrt(d0),
            "wo": rn(S, D) / D,
        }

    def run_ensemble(self, C, seeds, steps=24, eta=0.3, E=16, active=4,
                     seed_offset=0, eta_bias=1.0, memory_elements=80_000_000,
                     dims=None):
        L, D, M = shape(C, active) if dims is None else dims
        cL, kappa, d0 = 1.0 / (L * M), active / E, self.X.shape[1]
        outputs, imbalance0, imbalance1 = [], [], []
        done = 0
        per = max(1, int(memory_elements / max(1, L * E * D * M)))
        while done < seeds:
            s = min(per, seeds - done)
            g = torch.Generator(device=self.device).manual_seed(9000 + seed_offset + done)
            p = self.make(s, L, D, M, E, d0, g)
            U, W, R, b, We, wo = (p[k] for k in ("U", "W", "R", "b", "We", "wo"))
            params = U + W + R + [wo]
            rates = ([L * M * active / D * eta] * L
                     + [L * M * active * D * eta] * L
                     + [L * active * math.sqrt(M) / D * eta] * L
                     + [eta / D])
            loads = [None] * L

            def fwd():
                h = torch.einsum("pk,skd->spd", self.X, We)
                for layer in range(L):
                    gate = torch.sigmoid(torch.einsum("spd,sde->spe", h, R[layer]))
                    with torch.no_grad():
                        q = gate + b[layer].unsqueeze(1)
                        mask = torch.zeros_like(q).scatter_(
                            -1, q.topk(active, -1).indices, 1.0)
                        loads[layer] = mask.mean(1)
                    z = torch.einsum("spd,sedm->sepm", h, U[layer])
                    expert = torch.einsum(
                        "sepm,semd->sepd", torch.tanh(z), W[layer])
                    h = h + cL * ((gate * mask).permute(0, 2, 1).unsqueeze(-1)
                                  * expert).sum(1) / active
                return h

            with torch.no_grad():
                fwd()
                imbalance0.append(float(max((load - kappa).abs().max()
                                            for load in loads)))
            for param in params:
                param.requires_grad_(True)
            for _ in range(steps):
                prediction = torch.einsum("spd,sd->sp", fwd(), wo)
                grads = torch.autograd.grad(
                    (0.5 * (prediction - self.Y).pow(2).mean(-1)).sum(), params)
                with torch.no_grad():
                    for param, grad, rate in zip(params, grads, rates):
                        param -= rate * grad
                    for layer in range(L):
                        b[layer] -= eta_bias * (loads[layer] - kappa)
            with torch.no_grad():
                h = fwd()
                outputs.append(torch.einsum("spd,sd->sp", h, wo).cpu())
                imbalance1.append(float(max((load - kappa).abs().max()
                                            for load in loads)))
            done += s
            del p, U, W, R, b, We, wo, params, loads
            self.empty_cache()
        return (torch.cat(outputs).numpy(), float(np.mean(imbalance0)),
                float(np.mean(imbalance1)))

    @staticmethod
    def slope(x, y):
        return float(np.polyfit(np.log(x), np.log(y), 1)[0])

    def E1(self):
        budgets = [1e3, 2e3, 4e3, 8e3, 1.6e4, 3.2e4, 6.4e4, 1.28e5,
                   2.56e5, 5.12e5, 1.024e6, 2.05e6, 4.096e6, 8.19e6,
                   1.6384e7, 3.28e7, 6.5536e7, 1.31e8, 2.621e8, 5.24e8, 1e9]
        seeds = [3072, 3072, 3072, 2048, 2048, 1536, 1536, 1024, 1024,
                 768, 768, 512, 512, 384, 256, 192, 128, 96, 64, 48, 32]
        if self.smoke:
            budgets, seeds = budgets[:3], [8, 8, 8]
        for eta_bias in (1.0, 0.0):
            key = "E1_rate_eta_bias_%g" % eta_bias
            row = {"C": [], "Ediff": [], "imb_before": [], "imb_after": [],
                   "steps": 24, "requested_budgets": budgets,
                   "requested_seeds": seeds}
            self.results[key] = row
            outputs = {}
            for C, n_seed in zip(budgets, seeds):
                try:
                    start = time.time()
                    f, i0, i1 = self.run_ensemble(
                        C, n_seed, steps=24, seed_offset=int(math.log10(C) * 1000),
                        eta_bias=eta_bias)
                    outputs[C] = f
                    row["imb_before"].append(i0)
                    row["imb_after"].append(i1)
                    L, D, M = shape(C)
                    self.log("E1 eb=%g C=%.3e L=%d D=%d M=%d S=%d %.3f->%.3f [%.0fs]"
                             % (eta_bias, C, L, D, M, n_seed, i0, i1,
                                time.time() - start))
                    self.save()
                except (RuntimeError, MemoryError) as exc:
                    row["stopped_at"] = C
                    row["error"] = repr(exc)
                    self.log("E1 stopped at C=%.3e: %r" % (C, exc))
                    self.empty_cache()
                    break
            keys = sorted(outputs)
            for A, B in zip(keys[:-1], keys[1:]):
                fa, fb = outputs[A], outputs[B]
                distance = math.sqrt(float(
                    ((fa.mean(0) - fb.mean(0)) ** 2).mean()
                    + fa.var(0, ddof=1).mean() + fb.var(0, ddof=1).mean()))
                row["C"].append(A)
                row["Ediff"].append(distance)
            if len(row["C"]) > 2:
                row["slope"] = self.slope(row["C"], row["Ediff"])
                row["slope_tail"] = self.slope(row["C"][-4:], row["Ediff"][-4:])
            self.save()

    def kernel_spread(self, D, seeds, aM, steps, memory_elements=80_000_000):
        M, L, E, active = max(2, aM // 4), 4, 16, 4
        cL, kappa, done, values = 1.0 / (L * M), active / E, 0, []
        per = max(1, int(memory_elements / max(1, L * E * D * M)))
        while done < seeds:
            s = min(per, seeds - done)
            g = torch.Generator(device=self.device).manual_seed(555 + done)
            p = self.make(s, L, D, M, E, 8, g)
            U, W, R, b, We, wo = (p[k] for k in ("U", "W", "R", "b", "We", "wo"))
            params = U + W + R + [wo]
            rates = ([L * M * active / D * 0.3] * L
                     + [L * M * active * D * 0.3] * L
                     + [L * active * math.sqrt(M) / D * 0.3] * L + [0.3 / D])
            loads = [None] * L

            def fwd():
                h = torch.einsum("pk,skd->spd", self.X, We)
                for layer in range(L):
                    gate = torch.sigmoid(torch.einsum("spd,sde->spe", h, R[layer]))
                    with torch.no_grad():
                        q = gate + b[layer].unsqueeze(1)
                        mask = torch.zeros_like(q).scatter_(
                            -1, q.topk(active, -1).indices, 1.0)
                        loads[layer] = mask.mean(1)
                    z = torch.einsum("spd,sedm->sepm", h, U[layer])
                    expert = torch.einsum("sepm,semd->sepd", torch.tanh(z), W[layer])
                    h = h + cL * ((gate * mask).permute(0, 2, 1).unsqueeze(-1)
                                  * expert).sum(1) / active
                return h

            for param in params:
                param.requires_grad_(True)
            for _ in range(steps):
                prediction = torch.einsum("spd,sd->sp", fwd(), wo)
                grads = torch.autograd.grad(
                    (0.5 * (prediction - self.Y).pow(2).mean(-1)).sum(), params)
                with torch.no_grad():
                    for param, grad, rate in zip(params, grads, rates):
                        param -= rate * grad
                    for layer in range(L):
                        b[layer] -= loads[layer] - kappa
            with torch.no_grad():
                values.append(fwd().pow(2).mean(dim=(1, 2)).cpu().numpy())
            done += s
            del p, U, W, R, b, We, wo, params, loads
            self.empty_cache()
        return float(np.concatenate(values).std(ddof=1))

    def E3(self):
        widths = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        seeds = 192
        arms = [("init", 512, 0), ("trained", 512, 24), ("init_wide", 4096, 0)]
        if self.smoke:
            widths, seeds, arms = widths[:2], 4, [("init", 32, 0), ("trained", 32, 1)]
        self.results["E3_sqrtD"] = {}
        for tag, aM, steps in arms:
            row = {"D": [], "seeds": seeds, "steps": steps, "aM": aM,
                   "spread": []}
            self.results["E3_sqrtD"][tag] = row
            for D in widths:
                value = self.kernel_spread(D, seeds, aM, steps)
                row["D"].append(D)
                row["spread"].append(value)
                self.log("E3 %s D=%-6d spread %.4e" % (tag, D, value))
                self.save()
            row["slope"] = self.slope(widths, row["spread"])
            row["slope_tail"] = (self.slope(widths[-5:], row["spread"][-5:])
                                 if len(widths) >= 5
                                 else self.slope(widths, row["spread"]))
            self.save()

    def optimum_lr(self, L, D, M, E, active, seeds=256, steps=24, grid=None):
        grid = np.logspace(-1.0, 2.4, 29) if grid is None else grid
        losses = []
        for eta in grid:
            C = L * D * M * active
            prediction, _, _ = self.run_ensemble(
                C, seeds, steps=steps, eta=float(eta), E=E, active=active,
                seed_offset=31 - 9000, eta_bias=1.0, dims=(L, D, M))
            per_seed_loss = 0.5 * (prediction - self.Y.cpu().numpy()) ** 2
            losses.append(float(per_seed_loss.mean()))
        values, lg = np.asarray(losses), np.log10(grid)
        i = int(np.argmin(values))
        if 0 < i < len(values) - 1:
            den = values[i - 1] - 2 * values[i] + values[i + 1]
            refined = (lg[i] - 0.5 * (values[i + 1] - values[i - 1]) / den
                       * (lg[i] - lg[i - 1])) if abs(den) > 1e-12 else lg[i]
        else:
            refined = lg[i]
        return float(refined), i in (0, len(values) - 1), losses

    def E2(self):
        base = {"L": 8, "D": 64, "M": 128, "E": 16, "active": 4}
        dials = [
            ("depth L", [4, 8, 16, 32, 64], lambda c, v: c.update(L=v)),
            ("active a", [8, 16, 32, 64, 128],
             lambda c, v: c.update(active=v, E=4 * v)),
            ("expert width M", [32, 64, 128, 256, 512], lambda c, v: c.update(M=v)),
            ("embedding D", [16, 32, 64, 128, 256], lambda c, v: c.update(D=v)),
            ("expert count E (a=4)", [16, 32, 64, 128, 256],
             lambda c, v: c.update(E=v, active=4)),
        ]
        seeds, steps, grid = 256, 24, np.logspace(-1.0, 2.4, 29)
        if self.smoke:
            dials, seeds, steps, grid = dials[:1], 4, 1, np.logspace(-1, 0, 3)
            dials[0] = (dials[0][0], dials[0][1][:2], dials[0][2])
        self.results["E2_transfer"] = {}
        for name, values, apply in dials:
            optima, edges, curves = [], [], []
            row = {"vals": [], "lrstar": optima, "edge": edges,
                   "losses": curves, "lr_grid": grid.tolist(), "seeds": seeds,
                   "steps": steps}
            self.results["E2_transfer"][name] = row
            for value in values:
                config = dict(base)
                apply(config, value)
                optimum, edge, loss = self.optimum_lr(
                    **config, seeds=seeds, steps=steps, grid=grid)
                optima.append(optimum)
                edges.append(edge)
                curves.append(loss)
                row["vals"].append(value)
                self.log("E2 %-22s %-6s lr* %+.3f %s" %
                         (name, value, optimum, "EDGE" if edge else ""))
                self.save()
            row["drift"] = max(optima) - min(optima)
            row["tail"] = max(optima[-3:]) - min(optima[-3:])
            self.save()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="*", choices=("E1", "E2", "E3"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "rounds", "010-overnight",
        "big_out_rerun.json"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    suite = Suite(args.device, args.output, args.smoke)
    for name in args.experiments or ("E1", "E3", "E2"):
        try:
            getattr(suite, name)()
        except Exception as exc:  # preserve partial work on spot preemption/failure
            suite.results[name + "_error"] = repr(exc)
            suite.save()
            traceback.print_exc()
    suite.results["done"] = True
    suite.save()


if __name__ == "__main__":
    main()
