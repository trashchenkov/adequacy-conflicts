import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
"""Second search: costs the first grid skipped, three arcs, and a two-unit fleet.

Uses the same one-storage evaluator. Fleet goes through ref_milp (HiGHS).
Stopping rule remains the one in G6_SEARCH_PLAN.md: halt on a confirmed
Gamma/S > 1/4, otherwise exhaust this budget.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g6_search import analyze, ratios


def one_storage(seconds=70):
    rng = np.random.default_rng(4)
    t0 = time.time()
    best = 0.0
    best_rec = None
    seen = 0
    over = []

    def consider(x, E, Pc, P, tag):
        nonlocal best, best_rec, seen
        nd = sum(1 for v in x if v < -1e-12)
        if nd > 12 or nd == 0:
            return
        r_lin, r_cyc = analyze(x, E, Pc, P)
        seen += 1
        for name, rho, where in ratios(r_lin, r_cyc, P):
            if rho > best:
                best = rho
                best_rec = {
                    "tag": tag, "run": name, "rho": float(rho), "where": where,
                    "x": [round(float(v), 5) for v in x],
                    "E": float(E), "Pc": float(Pc), "P": float(P),
                }
            if rho > 0.25 + 1e-3:
                over.append(best_rec)
                return True
        return False

    # Integer costs 1..4, two arcs, overlap-friendly surpluses.
    defs = [1.0, 2.0, 3.0, 4.0]
    from itertools import product
    shapes = [(2, 2), (3, 2), (3, 3), (4, 2)]
    for n1, n2 in shapes:
        tuples = list(product(defs, repeat=n1 + n2))
        if len(tuples) > 2500:
            pick = rng.choice(len(tuples), size=2500, replace=False)
            tuples = [tuples[i] for i in pick]
        for E in (3.0, 4.0, 5.0):
            for s2 in (0.0, 1.0, 2.0, E):
                if time.time() - t0 > seconds * 0.55:
                    break
                for vals in tuples:
                    if time.time() - t0 > seconds * 0.55:
                        break
                    x = [E] + [-float(v) for v in vals[:n1]] + [s2] + [-float(v) for v in vals[n1:]]
                    if consider(x, E, max(E, 4.0), 4.0, "grid4"):
                        return seen, best, best_rec, over, time.time() - t0
    # Three charge intervals, continuous deficits, small middle charge.
    while time.time() - t0 < seconds:
        E = float(rng.uniform(1.5, 6.0))
        P = float(rng.uniform(E * 0.5, E * 1.3))
        Pc = float(rng.choice([P, E, max(E, P) * 3]))
        lens = [int(rng.integers(2, 5)) for _ in range(3)]
        if sum(lens) > 11:
            continue
        x = []
        sur = [E, float(rng.uniform(0.0, E)), E]
        for L, s in zip(lens, sur):
            x.append(s)
            block = rng.uniform(0.3, P, size=L)
            # Plant one classic-like peak in the first block.
            if L >= 3 and s == sur[0]:
                block[L // 2] = P
            for v in block:
                x.append(-float(v))
        if consider(x, E, Pc, P, "tri"):
            return seen, best, best_rec, over, time.time() - t0
    return seen, best, best_rec, over, time.time() - t0


def fleet(seconds=80, seed=5):
    from ref_milp import solve
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best = 0.0
    best_rec = None
    seen = 0
    unexpected = []
    over = []

    def r_of(x, units, H, D, cyc):
        hi = max(0.0, float(-np.min(x)))
        def feas(r):
            try:
                return solve(x, units, float(r), H=H, D=D, duration_cyclic=cyc) is not None
            except RuntimeError as exc:
                unexpected.append(str(exc))
                return None
        f0 = feas(0.0)
        if f0 is None:
            return None
        if f0:
            return 0.0
        lo = 0.0
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            f = feas(mid)
            if f is None:
                return None
            if f:
                hi = mid
            else:
                lo = mid
        return hi  # conservative upper on the threshold

    def consider(x, units, tag):
        nonlocal best, best_rec, seen
        S = sum(u["Pd"] for u in units)
        T = len(x)
        # A small menu of (H, D) where a duration conflict is possible.
        menu = []
        for D in (1, 2):
            for H in (D, D + 1, D + 2, max(D, T // 2)):
                if 0 <= H <= T and 0 <= D <= T:
                    menu.append((H, D))
        local_best = 0.0
        local = None
        for cyc in (False, True):
            # r_H and r_D depend on one limit; compute the ones we need.
            cache_H = {}
            cache_D = {}
            for H, D in menu:
                if H not in cache_H:
                    cache_H[H] = r_of(x, units, H, None, cyc)
                if D not in cache_D:
                    cache_D[D] = r_of(x, units, None, D, cyc)
                rH, rD = cache_H[H], cache_D[D]
                rJ = r_of(x, units, H, D, cyc)
                if rH is None or rD is None or rJ is None:
                    continue
                # r_of returns an upper bound. Gap upper = rJ - 0 is not valid.
                # Use rJ_hi - max(rH_lo, rD_lo). Here r_of returns hi of the
                # bracket, and lo is hi - width. Width after 16 steps is hi/2^16
                # only if we track lo. The returned value is an UPPER bound on r,
                # which OVERESTIMATES Gamma and can false-alarm, never hide a hit
                # below the estimate. Confirm alarms with a tighter bracket.
                gap = rJ - max(rH, rD)
                rho = gap / S
                if rho > local_best:
                    local_best = rho
                    local = (H, D, cyc, rH, rD, rJ, rho)
        seen += 1
        if local and local_best > best:
            best = local_best
            best_rec = {
                "tag": tag, "rho": float(local_best), "detail": [float(v) if not isinstance(v, (bool, str)) else v for v in local],
                "x": [round(float(v), 5) for v in x],
                "units": units,
            }
        if local_best > 0.25 + 5e-3:
            over.append(best_rec)
        return local_best > 0.25 + 5e-3

    # Structured: classic pocket plus a second scarce charge, two different units.
    classics = [
        [2, -1, -1, -2, -2, -2],
        [3, -1, -1, -1, -3, -3, -3],
        [4, -2, -2, -4, -4, -4],
    ]
    while time.time() - t0 < seconds:
        kind = int(rng.integers(0, 4))
        if kind == 0:
            base = classics[int(rng.integers(0, len(classics)))]
            s2 = float(rng.uniform(0.0, 2.0))
            tail = [s2] + [float(-rng.uniform(0.3, 2.5)) for _ in range(int(rng.integers(2, 5)))]
            x = base + tail
            p1 = float(rng.uniform(0.4, 2.0))
            p2 = float(rng.uniform(0.4, 2.0))
            units = [
                {"E": float(rng.uniform(0.5, 3.0)), "Pc": float(rng.uniform(0.5, 4.0)), "Pd": p1, "eta": 1.0},
                {"E": float(rng.uniform(0.3, 3.0)), "Pc": float(rng.uniform(0.3, 4.0)), "Pd": p2, "eta": 1.0},
            ]
            tag = "struct"
        else:
            T = int(rng.integers(8, 13))
            n_ch = int(rng.integers(2, 4))
            x = np.zeros(T)
            idx = rng.choice(T, size=n_ch, replace=False)
            for t in range(T):
                if t in idx:
                    x[t] = float(rng.uniform(0.0, 4.0))
                else:
                    x[t] = -float(rng.uniform(0.2, 3.0))
            units = []
            for _ in range(2):
                units.append({
                    "E": float(rng.uniform(0.4, 4.0)),
                    "Pc": float(rng.uniform(0.4, 5.0)),
                    "Pd": float(rng.uniform(0.4, 2.5)),
                    "eta": 1.0,
                })
            tag = "rand"
            x = x.tolist()
        if consider(x, units, tag):
            break
        if seen % 10 == 0:
            print("fleet", seen, "best", round(best, 5), "sec", round(time.time() - t0, 1), "bad", len(unexpected), flush=True)
    return seen, best, best_rec, over, unexpected, time.time() - t0


if __name__ == "__main__":
    s_seen, s_best, s_rec, s_over, s_dt = one_storage(80)
    print("storage_seen", s_seen, "best", s_best, "sec", round(s_dt, 1), "over", len(s_over))
    print("storage_rec", s_rec)
    f_seen, f_best, f_rec, f_over, unexpected, f_dt = fleet(90)
    print("fleet_seen", f_seen, "best", round(f_best, 5), "sec", round(f_dt, 1), "over", len(f_over), "unexpected", len(unexpected))
    print("fleet_rec", f_rec)
    with open("g6_more.json", "w") as f:
        json.dump({
            "storage": {"seen": s_seen, "best": s_best, "rec": s_rec, "over": len(s_over)},
            "fleet": {"seen": f_seen, "best": f_best, "rec": f_rec, "over": len(f_over), "unexpected": unexpected[:8]},
        }, f, ensure_ascii=False, indent=2, default=str)
    print("wrote g6_more.json")
