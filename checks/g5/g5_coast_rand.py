
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
"""Random profiles inside the coast assumptions. Solver check of the strong lemma at r=0.

Assumptions at r=0, one storage, E=Pc=Pd=eta=1: one surplus hour equal to 1,
every other hour a deficit at least delta, and the first deficit at least gamma+delta.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import g5_core as C
from g5_coast import audit


def main():
    rng = np.random.default_rng(3)
    n_ok = n_neg = n_err = 0
    worst = 1e9
    for i in range(40):
        T = int(rng.integers(3, 9))
        delta = float(rng.choice([0.05, 0.1, 0.2, 0.4]))
        g = 1.0 - delta
        x = np.zeros(T)
        x[0] = 1.0
        x[1] = -(g + delta + float(rng.uniform(0.0, 3.0)))
        for t in range(2, T):
            x[t] = -(delta + float(rng.uniform(0.0, 2.5)))
        # occasionally a second late cover that is still >= delta
        u = [{"E": 1.0, "Pc": 1.0, "Pd": 1.0, "eta": 1.0, "delta": delta}]
        info = C.boundary_slack(x.tolist(), u, delta)
        if info.get("status") != "ok":
            n_err += 1
            print("err", info.get("status"), "T", T, "d", delta)
            continue
        n_ok += 1
        worst = min(worst, info["slack"])
        if info["slack"] < -1e-5:
            n_neg += 1
            print("VIOLATION", info, x)
        audit(x.tolist(), delta)
    print("coast-rand ok", n_ok, "neg", n_neg, "err", n_err, "worst_slack", worst)
    print("notes", len(C.STATUS_NOTE))


if __name__ == "__main__":
    main()
