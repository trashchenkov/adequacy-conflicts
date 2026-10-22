
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
"""Stress the T=3 gap: small Pc, so a carried tank can exceed e* at e0=0."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import g5_core as C
from g5_traj import solve_traj, best_cover


def show(x, u, r=0.0):
    ens, lol, st_e, st_h = C.separate_at_0(x, u) if r == 0 else (None, None, "skip", "skip")
    if r == 0:
        print("sep", ens, lol, st_e, st_h)
        A = solve_traj(x, u, 0.0)
        B = best_cover(x, u, 0.0, lol)
        print("A", None if not A.get("ok") else (round(A["ens"], 6), A["d"].round(4), A["e"].round(4), A["c"].round(4)))
        if B:
            print("B", round(B["ens"], 6), "cover", B["cover"], "d", B["d"].round(4), "e", B["e"].round(4), "c", B["c"].round(4))
        dE = u[0]["delta"] * u[0]["E"]
        info = C.boundary_slack(x, u, dE)
        print("slack", info)
        g = C.gamma_interval(x, u, ens + 1e-8, lol, tol=1e-3, hi=1.0)
        print("gamma", g)
    return


def main():
    cases = [
        ([0.2, -0.5, -0.3], {"E": 1, "Pc": 0.2, "Pd": 1, "eta": 1, "delta": 0.1}),
        ([0.3, -0.8, -0.4], {"E": 1, "Pc": 0.3, "Pd": 1, "eta": 1, "delta": 0.2}),
        ([0.2, -0.4, -0.55], {"E": 1, "Pc": 0.2, "Pd": 0.5, "eta": 0.5, "delta": 0.3}),
        ([1.0, -0.4, -0.9], {"E": 1, "Pc": 0.2, "Pd": 1, "eta": 1, "delta": 0.1}),
        ([0.5, -1.2, -0.2], {"E": 2, "Pc": 0.4, "Pd": 1, "eta": 1, "delta": 0.15}),
    ]
    for x, u in cases:
        print("---", x, u)
        show(x, [u])


if __name__ == "__main__":
    main()
