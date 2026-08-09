"""Diagnose the post-training failure of the ``1/sqrt(D)`` kernel law.

The historical run tested three predictions:

* P1: training progress at a fixed step count depends on ``D``;
* P2: the exponent deviation grows with training horizon;
* P3: matching models by loss restores ``-1/2``.

P3 was originally implemented with a chunk-mean stopping rule.  Because the
CUDA memory chunk size shrinks with ``D``, that gave different stopping units
at different widths.  This version initialises every seed independently and
stops every seed independently, so results are invariant to memory chunking.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch


def log(*args):
    print("[%s]" % time.strftime("%H:%M:%S"), *args, flush=True)


def make_data(device, seed=4242):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(16, 8, generator=g, device=device, dtype=torch.float64)
    y = torch.randn(16, generator=g, device=device, dtype=torch.float64)
    return x, y


def _seeded_randn(generators, shape, device):
    """A batched tensor whose per-seed samples do not depend on chunk size."""
    return torch.stack([
        torch.randn(*shape, generator=g, device=device, dtype=torch.float64)
        for g in generators
    ])


def active_for_target(losses, active, target):
    """Monotone per-seed stopping mask, split out for mutation testing."""
    if target is None:
        return active
    return active & (losses.detach() > target)


def run(D, S, steps, eta=0.3, aM=512, L=4, E=16, a=4, target=None,
        device="cuda", memory_elements=80_000_000, X=None, Y=None,
        seed_offset=555):
    """Return kernel, loss, and steps-used arrays, one entry per seed.

    Seeds that reach ``target`` are frozen immediately; other seeds in the same
    memory chunk continue.  The returned step counts are per seed rather than
    per chunk, which removes the former width-dependent loss-matching bias.
    """
    if X is None or Y is None:
        X, Y = make_data(device)
    M = max(2, aM // 4)
    cL = 1.0 / (L * M)
    per = max(1, int(memory_elements / max(1, L * E * D * M)))
    kernels, losses, step_counts = [], [], []
    done = 0
    while done < S:
        s = min(per, S - done)
        generators = [torch.Generator(device=device).manual_seed(seed_offset + done + i)
                      for i in range(s)]
        rn = lambda *shape: _seeded_randn(generators, shape, device)
        U = [rn(E, D, M) * D ** -0.5 for _ in range(L)]
        W = [rn(E, M, D) for _ in range(L)]
        R = [rn(D, E) * D ** -0.5 for _ in range(L)]
        b = [rn(E) for _ in range(L)]
        We = rn(8, D) / math.sqrt(8)
        wo = rn(D) / D
        ps = U + W + R + [wo]
        lrs = ([L * M * a / D * eta] * L
               + [L * M * a * D * eta] * L
               + [L * a * math.sqrt(M) / D * eta] * L
               + [eta / D])

        def fwd():
            h = torch.einsum("pk,skd->spd", X, We)
            for layer in range(L):
                gate = torch.sigmoid(torch.einsum("spd,sde->spe", h, R[layer]))
                with torch.no_grad():
                    q = gate + b[layer].unsqueeze(1)
                    mask = torch.zeros_like(q).scatter_(-1, q.topk(a, -1).indices, 1.0)
                z = torch.einsum("spd,sedm->sepm", h, U[layer])
                expert = torch.einsum(
                    "sepm,semd->sepd", torch.tanh(z), W[layer])
                h = h + cL * ((gate * mask).permute(0, 2, 1).unsqueeze(-1)
                              * expert).sum(1) / a
            return h

        for p in ps:
            p.requires_grad_(True)
        active = torch.ones(s, dtype=torch.bool, device=device)
        used = torch.zeros(s, dtype=torch.int64, device=device)
        for _ in range(steps):
            f = torch.einsum("spd,sd->sp", fwd(), wo)
            per_seed_loss = 0.5 * (f - Y).pow(2).mean(-1)
            active = active_for_target(per_seed_loss, active, target)
            if not bool(active.any()):
                break
            grads = torch.autograd.grad(per_seed_loss[active].sum(), ps)
            with torch.no_grad():
                for p, grad, lr in zip(ps, grads, lrs):
                    p -= lr * grad
                used[active] += 1

        with torch.no_grad():
            h = fwd()
            f = torch.einsum("spd,sd->sp", h, wo)
            kernels.append(h.pow(2).mean(dim=(1, 2)).cpu().numpy())
            losses.append((0.5 * (f - Y).pow(2).mean(-1)).cpu().numpy())
            step_counts.append(used.cpu().numpy())
        for p in ps:
            p.requires_grad_(False)
        del U, W, R, b, We, wo, ps
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        done += s
    return (np.concatenate(kernels), np.concatenate(losses),
            np.concatenate(step_counts))


def slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def save(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=1, allow_nan=False)
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="diag_out.json")
    parser.add_argument("--seeds", type=int, default=512)
    parser.add_argument("--max-width", type=int, default=4096)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    torch.set_default_dtype(torch.float64)
    widths = [d for d in (32, 64, 128, 256, 512, 1024, 2048, 4096)
              if d <= args.max_width]
    if args.smoke:
        widths = widths[:2]
    seeds = min(args.seeds, 4) if args.smoke else args.seeds
    fixed_steps = (0, 2) if args.smoke else (0, 8, 24)
    horizons = (2,) if args.smoke else (2, 4, 8, 16, 32, 64)
    targets = (0.40,) if args.smoke else (0.40, 0.30, 0.20)
    matched_horizon = 20 if args.smoke else 200
    X, Y = make_data(args.device)
    out = {"schema_version": 2, "stopping": "per-seed",
           "seed_initialisation": "chunk-invariant", "smoke": args.smoke}

    log("P1: is training progress D-dependent at a fixed step count?")
    for steps in fixed_steps:
        los, spread = [], []
        for D in widths:
            k, l, _ = run(D, seeds, steps, device=args.device, X=X, Y=Y)
            los.append(float(l.mean()))
            spread.append(float(k.std(ddof=1)))
        out["P1_steps%d" % steps] = {
            "D": widths, "loss": los, "spread": spread,
            "slope": slope(widths, spread)}
        log(" steps=%-3d loss %s" % (steps, " ".join("%.4f" % v for v in los)))
        save(args.output, out)

    log("P2: does the deviation from -1/2 grow with the horizon?")
    for steps in horizons:
        spread = [run(D, seeds, steps, device=args.device, X=X, Y=Y)[0]
                  .std(ddof=1) for D in widths]
        out["P2_h%d" % steps] = {"steps": steps, "D": widths,
                                  "spread": spread,
                                  "slope": slope(widths, spread)}
        log(" horizon %-3d slope %+.4f" % (steps, out["P2_h%d" % steps]["slope"]))
        save(args.output, out)

    log("P3: match every seed by loss")
    for target in targets:
        spread, mean_steps, per_seed_steps = [], [], []
        for D in widths:
            k, _, used = run(D, seeds, matched_horizon, target=target,
                             device=args.device, X=X, Y=Y)
            spread.append(float(k.std(ddof=1)))
            mean_steps.append(float(used.mean()))
            per_seed_steps.append(used.tolist())
        out["P3_loss%.2f" % target] = {
            "D": widths, "spread": spread, "slope": slope(widths, spread),
            "mean_steps_used": mean_steps, "per_seed_steps_used": per_seed_steps}
        log(" target %.2f mean steps %s" %
            (target, " ".join("%.2f" % v for v in mean_steps)))
        save(args.output, out)
    log("DIAG DONE")


if __name__ == "__main__":
    main()
