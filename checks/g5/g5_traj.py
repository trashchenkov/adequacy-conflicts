
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
"""Recover optimal trajectories and compare ENS-opt vs LOLH-constrained."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.optimize import linprog

import g5_core as C


def solve_traj(x, units, r, cover=None):
    """One storage. Returns dict with c,d,e,ens or None."""
    u = units[0]
    T = len(x)
    s = np.asarray(x, float) + float(r)
    sp = np.maximum(s, 0.0)
    sm = np.maximum(-s, 0.0)
    # c, d, e[0..T-1], e is end of hour t; cycle prev of 0 is e[T-1]
    n = 3 * T
    def cv(t): return t
    def dv(t): return T + t
    def ev(t): return 2 * T + t
    gamma = 1.0 - u["delta"]
    eta = u["eta"]
    bounds = []
    for t in range(T):
        bounds.append((0.0, u["Pc"] if sp[t] > 1e-12 else 0.0))
    for t in range(T):
        bounds.append((0.0, u["Pd"] if sm[t] > 1e-12 else 0.0))
    for t in range(T):
        bounds.append((0.0, u["E"]))
    A_eq = []
    b_eq = []
    for t in range(T):
        row = np.zeros(n)
        prev = T - 1 if t == 0 else t - 1
        row[ev(t)] = 1.0
        row[ev(prev)] = -gamma
        row[cv(t)] = -eta
        row[dv(t)] = 1.0
        A_eq.append(row)
        b_eq.append(0.0)
    A_ub = []
    b_ub = []
    for t in range(T):
        row = np.zeros(n)
        row[cv(t)] = 1.0
        A_ub.append(row)
        b_ub.append(sp[t])
        row = np.zeros(n)
        row[dv(t)] = 1.0
        A_ub.append(row)
        b_ub.append(sm[t])
        if cover is not None and t in cover and sm[t] > 1e-12:
            row = np.zeros(n)
            row[dv(t)] = -1.0
            A_ub.append(row)
            b_ub.append(-sm[t])
    cobj = np.zeros(n)
    for t in range(T):
        cobj[dv(t)] = -1.0
    res = linprog(cobj, A_ub=np.vstack(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.vstack(A_eq), b_eq=np.array(b_eq),
                  bounds=bounds, method="highs")
    if not res.success:
        return {"ok": False, "msg": res.message}
    z = res.x
    c = z[0:T]
    d = z[T:2 * T]
    e = z[2 * T:3 * T]
    ens = float(sm.sum() - d.sum())
    return {"ok": True, "c": c, "d": d, "e": e, "ens": ens, "sm": sm, "sp": sp,
            "sum_e": float(e.sum()), "sum_c": float(c.sum()), "sum_d": float(d.sum())}


def best_cover(x, units, r, lol):
    """Among cover sets with enough covers, min ENS trajectory."""
    s = np.asarray(x, float) + r
    sm = np.maximum(-s, 0.0)
    T = len(x)
    pd = units[0]["Pd"]
    deficit = [t for t in range(T) if sm[t] > 1e-9]
    forced = [t for t in deficit if sm[t] > pd + 1e-8]
    choice = [t for t in deficit if t not in forced]
    budget = int(lol) - len(forced)
    from itertools import combinations
    best = None
    if budget < 0:
        return None
    for k in range(budget + 1):
        for op in combinations(choice, k):
            cover = [t for t in choice if t not in op]
            tr = solve_traj(x, units, r, cover=cover)
            if not tr.get("ok"):
                continue
            tr["cover"] = cover
            tr["open"] = list(forced) + list(op)
            if best is None or tr["ens"] < best["ens"] - 1e-9:
                best = tr
    return best


def dump(tag, tr):
    if tr is None or not tr.get("ok", False):
        print(tag, "FAIL", tr)
        return
    np.set_printoptions(precision=4, suppress=True)
    print(f"--- {tag} ENS={tr['ens']:.6f} sumC={tr['sum_c']:.4f} sumD={tr['sum_d']:.4f} sumE={tr['sum_e']:.4f}")
    print("  c", tr["c"])
    print("  d", tr["d"])
    print("  e", tr["e"])
    print("  sm", tr["sm"])
    if "cover" in tr:
        print("  cover", tr["cover"], "open", tr["open"])


def main():
    # patient
    x = [1, -5, 0, 0, 0, -0.3, -0.3]
    u = [{"E": 1, "Pc": 1, "Pd": 1, "eta": 1.0, "delta": 0.05}]
    ens, lol, _, _ = C.separate_at_0(x, u)
    print("PATIENT ENS", ens, "LOL", lol)
    A = solve_traj(x, u, 0.0, cover=None)
    dump("A r=0", A)
    B = best_cover(x, u, 0.0, lol)
    dump("B r=0", B)
    print("G", B["ens"] - A["ens"], "deltaE", 0.05)
    # at r = delta E
    A2 = solve_traj(x, u, 0.05, cover=None)
    dump("A r=dE", A2)
    B2 = best_cover(x, u, 0.05, lol)
    dump("B r=dE", B2)

    # critical L=4 delta=0.1
    d = 0.1
    g = 0.9
    L = 4
    beta = g ** (L + 1)
    x = [1.0] + [-3.0] * L + [-beta]
    u = [{"E": 1, "Pc": 1, "Pd": 1, "eta": 1.0, "delta": d}]
    ens, lol, _, _ = C.separate_at_0(x, u)
    print("\nCRIT ENS", ens, "LOL", lol, "beta", beta)
    A = solve_traj(x, u, 0.0)
    B = best_cover(x, u, 0.0, lol)
    dump("A", A)
    dump("B", B)
    print("G", B["ens"] - A["ens"], "dSumE", B["sum_e"] - A["sum_e"], "delta*dSumE", d * (B["sum_e"] - A["sum_e"]))

    # T=4 alternating deficits
    x = [1.0, -0.5, 0.2, -0.4]
    u = [{"E": 1, "Pc": 1, "Pd": 1, "eta": 1.0, "delta": 0.2}]
    ens, lol, _, _ = C.separate_at_0(x, u)
    print("\nALT ENS", ens, "LOL", lol)
    A = solve_traj(x, u, 0.0)
    B = best_cover(x, u, 0.0, lol)
    dump("A", A)
    dump("B", B)
    if B:
        print("G", B["ens"] - A["ens"])


if __name__ == "__main__":
    main()
