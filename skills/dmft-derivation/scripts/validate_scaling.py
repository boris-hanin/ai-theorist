"""Battery for the Step-0 scaling audit and the HP-transfer harness (Leg C).

    python3 validate_scaling.py [--quick]

Leg C needs no DMFT solver: it tests the parameterisation table directly by
training networks and asking whether the optimal learning rate moves. That
makes it runnable for architectures whose limiting equations have not been
derived yet, and it is a cheap falsifier -- a wrong table shows up here long
before any solver exists.

The negative control (standard parameterisation) is not optional. If it also
"transfers", the optimum is flat and the whole sweep is under-powered.
"""

import argparse
import sys
import time

import numpy as np

import audit
import transfer


class Report(object):
    def __init__(self):
        self.rows = []

    def add(self, name, passed, detail):
        self.rows.append((name, passed, detail))
        print("  [%s] %-38s %s" % ("PASS" if passed else "FAIL", name, detail))

    @property
    def failed(self):
        return [r for r in self.rows if not r[1]]


def check_audit(rep, cfg):
    """S1-S3: Step-0 feature-velocity audit, measured in linear response."""
    ws = cfg["widths"]
    _, sl_mup = audit.width_audit("mup", ws, L=2, steps=1, lr=1e-4)
    _, sl_sp = audit.width_audit("sp", ws, L=2, steps=1, lr=1e-4)
    rep.add("S1 muP feature velocity flat in N", max(abs(s) for s in sl_mup) < 0.05,
            "per-layer slopes %s" % "  ".join("%+.3f" % s for s in sl_mup))
    rep.add("S2 SP feature velocity scales", max(abs(s) for s in sl_sp) > 0.2,
            "per-layer slopes %s  (negative control)"
            % "  ".join("%+.3f" % s for s in sl_sp))

    # The slope is only an exponent if it is measured in the linear-response
    # regime; at a large lr, nonlinear blowup contaminates it (measured: slopes
    # of -1.9 / +10.0 at lr = 0.05, which are not exponents of anything).
    _, sl_a = audit.width_audit("sp", ws, L=2, steps=1, lr=1e-3)
    _, sl_b = audit.width_audit("sp", ws, L=2, steps=1, lr=1e-5)
    drift = max(abs(a - b) for a, b in zip(sl_a, sl_b))
    rep.add("S3 slopes are lr-independent", drift < 0.02,
            "max slope change over 2 decades of lr = %.4f" % drift)


def check_width_transfer(rep, cfg):
    """S4-S6: does the optimal LR move with width?"""
    ws = cfg["widths"]
    seeds = tuple(range(cfg["seeds"]))
    kw = dict(L=1, steps=cfg["steps"], P=64, seeds=seeds)

    _, vm = transfer.width_transfer("mup", ws, np.logspace(-2, 0.4, cfg["ngrid"]), **kw)
    rep.add("S4 muP transfers across width", vm["status"].startswith("TRANSFERS"),
            "drift %.3f dec (sem %.3f)  lr*: %s"
            % (vm["drift_log10"], vm["sem_log10"],
               " ".join("%.2f" % x for x in vm["refined_log10_lr"])))

    _, vs = transfer.width_transfer("sp", ws, np.logspace(-5, -1, cfg["ngrid"]), **kw)
    rep.add("S5 SP FAILS to transfer (control)", vs["status"] == "FAILS",
            "drift %.3f dec (sem %.3f)  lr*: %s"
            % (vs["drift_log10"], vs["sem_log10"],
               " ".join("%.2f" % x for x in vs["refined_log10_lr"])))

    # The control should fail with a recognisable exponent, not just noisily.
    slope = float(np.polyfit(np.log10(np.array(ws, dtype=float)),
                             vs["refined_log10_lr"], 1)[0])
    rep.add("S6 SP lr* scales as a power of N", -1.3 < slope < -0.7,
            "d log10(lr*) / d log10(N) = %.2f  (expect ~ -1)" % slope)


NOT_COVERED = """
NOT COVERED:
  * Depth. This battery sweeps width only; the residual-branch exponent alpha
    is the subject of the depth sweep, not tested here.
  * Any claim about the infinite-width LIMIT. Leg C tests that a finite-width
    ladder shares an optimum; it says nothing about the limiting equations.
    The muP residual drift (0.04 decades, converging as -0.93/-0.90/-0.89/-0.89)
    is consistent with a vanishing finite-width correction, but this sweep
    cannot establish that -- only that it is small over the range tested.
  * gamma_0 transfer: only eta_0 is swept.
  * Real tasks. The target is a fixed random teacher on synthetic inputs.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = (dict(widths=[128, 512, 2048], seeds=3, steps=20, ngrid=9) if args.quick
           else dict(widths=[128, 512, 2048, 8192], seeds=5, steps=20, ngrid=13))
    np.seterr(all="ignore")

    print("Scaling-audit and HP-transfer battery  (%s)"
          % ("quick" if args.quick else "full"))
    print("widths=%s  seeds=%d\n" % (cfg["widths"], cfg["seeds"]))
    rep = Report()
    t0 = time.time()
    for fn in (check_audit, check_width_transfer):
        fn(rep, cfg)
    print("\n%d checks, %d failed, %.1fs"
          % (len(rep.rows), len(rep.failed), time.time() - t0))
    print(NOT_COVERED)
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
