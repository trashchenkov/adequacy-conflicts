
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
"""Independent numerical check of the three G4 claims.

Ground truth is storage LP (cyclic, delta=0), not the cut reduction.
Python: ../py g4_check.py
"""
import json
import math
import time
import numpy as np
from scipy.optimize import linprog

TOL_R = 2e-5
VIOL = 5e-4
POS = 1e-8


def run_length(pos, cyclic):
    T = len(pos)
    if not any(pos):
        return 0
    if all(pos):
        return T
    best = cur = 0
    limit = 2 * T if cyclic else T
    for i in range(limit):
        if pos[i % T]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return min(best, T) if cyclic else best


def M_lin(T, D):
    if D >= T:
        return T
    return T - (T // (D + 1))


def M_cyc(T, D):
    if D >= T:
        return T
    return T - math.ceil(T / (D + 1))


def brute_M(T, D, cyclic):
    best = 0
    for mask in range(1 << T):
        pos = [(mask >> t) & 1 for t in range(T)]
        if run_length(pos, cyclic) <= D:
            best = max(best, sum(pos))
    return best


class SolverBad(Exception):
    def __init__(self, status):
        self.status = status
        super().__init__(f"solver status {status}")


def _solve(cobj, A_ub, b_ub, A_eq, b_eq, bounds):
    res = linprog(
        cobj,
        A_ub=A_ub if len(A_ub) else None,
        b_ub=b_ub if len(b_ub) else None,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if res.status == 0 and res.success:
        return res
    if res.status == 2:
        return None
    raise SolverBad(int(res.status))


def schedule_lp(x, units, r, zero_hours, objective="ens"):
    """Min ENS (or pure feasibility) with ell forced to 0 on zero_hours.

    Returns (ens, pack) or None if infeasible. pack holds c, d, e, e0, ell.
    """
    x = np.asarray(x, float)
    T = len(x)
    m = len(units)
    s = x + r
    sp = np.maximum(s, 0.0)
    sm = np.maximum(-s, 0.0)
    zero = np.zeros(T, dtype=bool)
    for t in zero_hours:
        zero[t] = True
    nc = m * T

    def ic(i, t):
        return i * T + t

    def id_(i, t):
        return nc + i * T + t

    def ie(i, t):
        return 2 * nc + i * T + t

    def ie0(i):
        return 3 * nc + i

    def il(t):
        return 3 * nc + m + t

    n = 3 * nc + m + T
    cobj = np.zeros(n)
    if objective == "ens":
        cobj[3 * nc + m :] = 1.0
    bounds = [(0.0, 0.0)] * n
    for i, u in enumerate(units):
        for t in range(T):
            bounds[ic(i, t)] = (0.0, float(u["Pc"]) if sp[t] > POS else 0.0)
            bounds[id_(i, t)] = (0.0, float(u["Pd"]) if sm[t] > POS else 0.0)
            bounds[ie(i, t)] = (0.0, float(u["E"]))
        bounds[ie0(i)] = (0.0, float(u["E"]))
    for t in range(T):
        bounds[il(t)] = (0.0, 0.0 if zero[t] else float(sm[t]))

    A_eq = []
    b_eq = []
    A_ub = []
    b_ub = []
    for i, u in enumerate(units):
        eta = float(u["eta"])
        for t in range(T):
            row = np.zeros(n)
            row[ie(i, t)] = 1.0
            prev = ie(i, t - 1) if t else ie0(i)
            row[prev] = -1.0
            row[ic(i, t)] = -eta
            row[id_(i, t)] = 1.0
            A_eq.append(row)
            b_eq.append(0.0)
        row = np.zeros(n)
        row[ie(i, T - 1)] = 1.0
        row[ie0(i)] = -1.0
        A_eq.append(row)
        b_eq.append(0.0)
    for t in range(T):
        row = np.zeros(n)
        for i in range(m):
            row[ic(i, t)] = 1.0
        A_ub.append(row)
        b_ub.append(float(sp[t]))
        row = np.zeros(n)
        for i in range(m):
            row[id_(i, t)] = 1.0
        row[il(t)] = 1.0
        A_eq.append(row)
        b_eq.append(float(sm[t]))
    res = _solve(cobj, np.asarray(A_ub), np.asarray(b_ub), np.asarray(A_eq), np.asarray(b_eq), bounds)
    if res is None:
        return None
    z = res.x
    c = np.zeros((m, T))
    d = np.zeros((m, T))
    e = np.zeros((m, T))
    e0 = np.zeros(m)
    for i in range(m):
        for t in range(T):
            c[i, t] = z[ic(i, t)]
            d[i, t] = z[id_(i, t)]
            e[i, t] = z[ie(i, t)]
        e0[i] = z[ie0(i)]
    ell = z[3 * nc + m :]
    # residual check
    for i, u in enumerate(units):
        prev = e0[i]
        eta = float(u["eta"])
        for t in range(T):
            pred = prev + eta * c[i, t] - d[i, t]
            if abs(pred - e[i, t]) > 1e-6:
                raise SolverBad("residual")
            prev = e[i, t]
        if abs(e[i, T - 1] - e0[i]) > 1e-6:
            raise SolverBad("cyclic")
    return float(ell.sum()), {"c": c, "d": d, "e": e, "e0": e0, "ell": ell, "sm": sm, "sp": sp}


def feasible(x, units, r, H, D, cyclic, ens_limit=None):
    """True/False. Raises SolverBad."""
    x = np.asarray(x, float)
    T = len(x)
    sm = np.maximum(-(x + r), 0.0)
    sp = np.maximum(x + r, 0.0)
    N = [t for t in range(T) if sm[t] > POS]
    n = len(N)
    SPd = sum(float(u["Pd"]) for u in units)
    eta_max = max(float(u["eta"]) for u in units)
    surplus_cap = eta_max * float(sp.sum()) + 1e-6
    # enumerate the positive subset of N; the complement must be fully covered
    for mask in range(1 << n):
        pos = [False] * T
        cover = []
        npos = 0
        power_bad = False
        need = 0.0
        for k in range(n):
            t = N[k]
            if mask & (1 << k):
                pos[t] = True
                npos += 1
            else:
                if sm[t] > SPd + 1e-9:
                    power_bad = True
                    break
                cover.append(t)
                need += sm[t]
        if power_bad or npos > H or need > surplus_cap:
            continue
        if run_length(pos, cyclic) > D:
            continue
        out = schedule_lp(x, units, r, cover, objective="ens")
        if out is None:
            continue
        ens, _pack = out
        if ens_limit is None or ens <= ens_limit + 1e-7:
            return True
    return False


def first_mode(x, units, r, H, D, cyclic, ens_limit=None):
    x = np.asarray(x, float)
    T = len(x)
    sm = np.maximum(-(x + r), 0.0)
    sp = np.maximum(x + r, 0.0)
    N = [t for t in range(T) if sm[t] > POS]
    n = len(N)
    SPd = sum(float(u["Pd"]) for u in units)
    eta_max = max(float(u["eta"]) for u in units)
    surplus_cap = eta_max * float(sp.sum()) + 1e-6
    for mask in range(1 << n):
        pos = [False] * T
        cover = []
        npos = 0
        power_bad = False
        need = 0.0
        for k in range(n):
            t = N[k]
            if mask & (1 << k):
                pos[t] = True
                npos += 1
            else:
                if sm[t] > SPd + 1e-9:
                    power_bad = True
                    break
                cover.append(t)
                need += sm[t]
        if power_bad or npos > H or need > surplus_cap:
            continue
        if run_length(pos, cyclic) > D:
            continue
        out = schedule_lp(x, units, r, cover, objective="ens")
        if out is None:
            continue
        ens, pack = out
        if ens_limit is not None and ens > ens_limit + 1e-7:
            continue
        pack["cover"] = cover
        pack["pos"] = pos
        return pack
    return None


def bisect(check, hi):
    """Return (lo, hi, flag) with check(hi) true, check(lo) false unless lo=0 and feasible.
    flag is 'ok' or 'bad'."""
    try:
        if check(0.0):
            return 0.0, 0.0, "ok"
        if hi <= 0:
            return 0.0, 0.0, "ok"
        if not check(hi):
            # should be feasible once all deficits vanish; treat as bad data
            return None, None, "never"
        lo = 0.0
        a, b = 0.0, float(hi)
        for _ in range(40):
            if b - a <= TOL_R:
                break
            mid = 0.5 * (a + b)
            if check(mid):
                b = mid
            else:
                a = mid
        return a, b, "ok"
    except SolverBad:
        return None, None, "bad"


def positive_hours(x, r):
    s = np.asarray(x, float) + r
    return [i for i, v in enumerate(s) if v > POS]


def compute_B(units):
    m = len(units)
    beta = max(float(u["eta"]) for u in units)
    w = np.zeros(m)
    D = np.zeros(m)
    C = np.zeros(m)
    for i, u in enumerate(units):
        w[i] = 1.0 - float(u["eta"]) / beta
        D[i] = min(float(u["Pd"]), float(u["E"]))
        C[i] = min(float(u["Pc"]), float(u["E"]) / float(u["eta"]))
    # min t s.t. u·D <= t, (w-u)·C <= t, 0<=u<=w
    cobj = np.zeros(m + 1)
    cobj[-1] = 1.0
    A_ub = []
    b_ub = []
    row = np.zeros(m + 1)
    row[:m] = D
    row[-1] = -1.0
    A_ub.append(row)
    b_ub.append(0.0)
    row = np.zeros(m + 1)
    row[:m] = -C
    row[-1] = -1.0
    A_ub.append(row)
    b_ub.append(-float(np.dot(w, C)))
    bounds = [(0.0, float(w[i])) for i in range(m)] + [(0.0, None)]
    res = _solve(cobj, np.asarray(A_ub), np.asarray(b_ub), np.zeros((0, m + 1)), np.zeros(0), bounds)
    if res is None:
        raise SolverBad("B infeasible")
    return float(res.fun), res.x[:m]


def F_value(m_hours, devs, Q):
    return min(Q, sum(min(C, m_hours * p) for C, p in devs))


def cuts_ok(costs, devs, Q):
    if any(c < -1e-8 for c in costs):
        return False
    acc = 0.0
    for m, c in enumerate(sorted(costs, reverse=True), 1):
        acc += c
        if acc > F_value(m, devs, Q) + 1e-7:
            return False
    return True


def pure_cover_lp(x, units, r, hours):
    """Feasibility of discharging exactly the deficit on `hours` and 0 elsewhere."""
    x = np.asarray(x, float)
    sm = np.maximum(-(x + r), 0.0)
    # force ell=0 on cover hours AND on non-deficit; forbid discharge contribution
    # by zeroing ell everywhere except we set zero_hours = all hours, and only if
    # the required discharge fits. ell=0 everywhere means full cover of every deficit
    # hour — too strong. Instead zero ell on `hours` and require sm==0 outside hours
    # or d=0 and ell=sm outside, i.e. do not care about outside ENS but d can be 0.
    # schedule_lp with zero_hours=hours allows partial outside. For a pure cover we
    # need d=0 outside. Enforce by passing zero_hours = all deficit hours not in the
    # cover? That forces ell=0 there too, i.e. full cover of everything.
    # Build a dedicated check: zero_hours = hours, and add nothing if we then test
    # sum d on the complement is ~0 by re-solving with objective and bounds.
    # Practical approach: set Pd-available only on cover hours via a modified unit? 
    # Easiest: call schedule_lp with zero on cover, then check a second LP is unnecessary
    # if we tighten discharge UB to 0 off the cover by temporarily zeroing sm off cover.
    sm_full = np.maximum(-(x + r), 0.0)
    x_mod = x.copy()
    # Raising x on non-cover deficit hours to -r makes their deficit 0, so d must be 0
    # there, while cover hours keep their deficit. Charge hours unchanged.
    cover = set(hours)
    for t in range(len(x)):
        if sm_full[t] > POS and t not in cover:
            x_mod[t] = -r  # deficit vanishes; original discharge there is forbidden
    out = schedule_lp(x_mod, units, r, hours, objective="ens")
    return out is not None


def network_of(x, units, r):
    """Static network for exactly one positive hour and a common eta. Or None."""
    x = np.asarray(x, float)
    s = x + r
    pos = [t for t in range(len(x)) if s[t] > POS]
    if len(pos) != 1:
        return None
    etas = {round(float(u["eta"]), 10) for u in units}
    if len(etas) != 1:
        return None
    eta = float(units[0]["eta"])
    tc = pos[0]
    Q = eta * float(s[tc])
    devs = []
    for u in units:
        C = min(float(u["E"]), eta * float(u["Pc"]))
        p = float(u["Pd"])
        devs.append((C, p))
    P = sum(p for _C, p in devs)
    sm = np.maximum(-s, 0.0)
    return {"tc": tc, "Q": Q, "devs": devs, "P": P, "sm": sm, "eta": eta}


def proof_select(costs, A, B, P):
    """Proof construction. costs: id -> energy cost. Returns (F, storage_cost dict, boost)."""
    A, B = set(A), set(B)
    if len(B) >= len(A):
        new = {t: max(0.0, costs[t] - P / 4.0) for t in B}
        return set(B), new, P / 4.0
    O = A & B
    A2, B2 = A - O, B - O
    if len(B2) == 0:
        new = {t: costs[t] for t in A}
        return set(A), new, 0.0
    n_take = len(A2) - len(B2)
    if n_take < 0:
        raise RuntimeError("size invariant broken")
    cheapest = sorted(A2, key=lambda t: (costs[t], t))[:n_take]
    chosen = set(B2) | set(cheapest)
    new = {t: costs[t] for t in O}
    boost = 0.0
    for t in chosen:
        cut = min(costs[t], P / 4.0)
        boost = max(boost, cut)
        new[t] = costs[t] - cut
    return set(O) | set(chosen), new, boost


def avg_check(x, units, packs, r_base):
    """Average normalized packs, add q = max_j d - mean d on deficit hours.
    Return dict with max_q, S, feasible, ell_is_min, violations."""
    m = len(packs)
    S = sum(float(u["Pd"]) for u in units)
    T = len(x)
    c = sum(p["c"] for p in packs) / m
    d = sum(p["d"] for p in packs) / m
    e = sum(p["e"] for p in packs) / m
    e0 = sum(p["e0"] for p in packs) / m
    dtot = [d[:, t].sum() for t in range(T)]
    djs = [[p["d"][:, t].sum() for t in range(T)] for p in packs]
    sm0 = np.maximum(-(np.asarray(x, float) + r_base), 0.0)
    q = np.zeros(T)
    for t in range(T):
        if sm0[t] <= POS:
            q[t] = 0.0
            continue
        mx = max(dj[t] for dj in djs)
        q[t] = mx - dtot[t]
    r_new = r_base + float(q.max())
    s = np.asarray(x, float) + r_base + q
    sp = np.maximum(s, 0.0)
    sm = np.maximum(-s, 0.0)
    ell = sm - np.array(dtot)
    # reference min profile at the OLD deficit, per the identity
    ell_old = []
    for p in packs:
        ell_old.append(sm0 - p["d"].sum(axis=0))
    ell_min = np.minimum.reduce(ell_old)
    # identity: ell_new should equal ell_min
    ident = float(np.max(np.abs(ell - ell_min)))
    problems = []
    if float(q.max()) > (1.0 - 1.0 / m) * S + 1e-6:
        problems.append("q_bound")
    if np.any(q < -1e-7):
        problems.append("q_neg")
    if np.any(ell < -1e-6):
        problems.append("ell_neg")
    for i, u in enumerate(units):
        prev = float(e0[i])
        eta = float(u["eta"])
        for t in range(T):
            pred = prev + eta * c[i, t] - d[i, t]
            if abs(pred - e[i, t]) > 1e-5:
                problems.append("dyn")
                break
            if e[i, t] < -1e-6 or e[i, t] > float(u["E"]) + 1e-5:
                problems.append("box")
                break
            if c[i, t] < -1e-7 or c[i, t] > float(u["Pc"]) + 1e-5:
                problems.append("Pc")
                break
            if d[i, t] < -1e-7 or d[i, t] > float(u["Pd"]) + 1e-5:
                problems.append("Pd")
                break
            prev = float(e[i, t])
        if abs(float(e[i, T - 1]) - float(e0[i])) > 1e-5:
            problems.append("cyc")
    if np.any(c.sum(axis=0) > sp + 1e-5):
        problems.append("charge_sum")
    if np.any(d.sum(axis=0) > sm + 1e-5):
        problems.append("discharge_sum")
    # surplus hours must have q=0 and d=0
    for t in range(T):
        if sm0[t] <= POS and q[t] > 1e-8:
            problems.append("q_on_surplus")
    return {
        "max_q": float(q.max()),
        "S": S,
        "cap": (1.0 - 1.0 / m) * S,
        "ident": ident,
        "problems": sorted(set(problems)),
        "r_new": r_new,
        "ell": ell,
    }


def check_requirements_profile(ell, sm_mask_positive_allowed, H, D, cyclic, ens_limit, which):
    """which is a set of 'H','D','E' that ell must satisfy, positivity vs ell>0."""
    pos = [ell[t] > 1e-6 for t in range(len(ell))]
    h = sum(pos)
    dur = run_length(pos, cyclic)
    ens = float(np.maximum(ell, 0).sum())
    bad = []
    if "H" in which and h > H:
        bad.append("H")
    if "D" in which and dur > D:
        bad.append("D")
    if "E" in which and ens_limit is not None and ens > ens_limit + 1e-5:
        bad.append("E")
    return bad, h, dur, ens


def random_units(rng, m, equal_eta, eta=None):
    units = []
    if equal_eta:
        eta = 1.0 if eta is None else float(eta)
    for _ in range(m):
        e = float(rng.uniform(0.3, 2.5))
        units.append(
            {
                "E": e,
                "Pc": float(rng.uniform(0.3, 2.0)),
                "Pd": float(rng.uniform(0.3, 2.0)),
                "eta": eta if equal_eta else float(rng.uniform(0.25, 1.0)),
            }
        )
    return units


def random_x(rng, T, n_pos):
    x = -rng.uniform(0.4, 2.5, size=T)
    idx = rng.choice(T, size=n_pos, replace=False)
    x[idx] = rng.uniform(0.4, 3.0, size=n_pos)
    # sprinkle an exact zero sometimes
    if T >= 5 and rng.random() < 0.25:
        zc = int(rng.integers(0, T))
        if zc not in set(idx.tolist()):
            x[zc] = 0.0
    return x


def pick_HD(rng, T, cyclic):
    D = int(rng.integers(1, max(2, T - 1)))
    M = M_cyc(T, D) if cyclic else M_lin(T, D)
    if M >= D + 2:
        H = int(rng.integers(D + 1, M))
    elif M >= D + 1:
        H = D + 1
    else:
        H = int(rng.integers(0, T + 1))
        D = int(rng.integers(0, T + 1))
    return H, D


def hd_rates(x, units, H, D, cyclic):
    hi = max(0.0, float(-np.min(x))) + 1e-6
    def cH(r):
        return feasible(x, units, r, H, T_if(D, x), cyclic)

    # fix: D large means unconstrained duration
    def cH(r):
        return feasible(x, units, r, H, len(x), cyclic)

    def cD(r):
        return feasible(x, units, r, len(x), D, cyclic)

    def cJ(r):
        return feasible(x, units, r, H, D, cyclic)

    hlo, hhi, hf = bisect(cH, hi)
    dlo, dhi, df = bisect(cD, hi)
    jlo, jhi, jf = bisect(cJ, hi)
    if "bad" in (hf, df, jf) or "never" in (hf, df, jf):
        return {"flag": "bad", "detail": (hf, df, jf)}
    r0_lo = max(hlo, dlo)
    r0_hi = max(hhi, dhi)
    g_lo = jlo - r0_hi
    g_hi = jhi - r0_lo
    S = sum(u["Pd"] for u in units)
    return {
        "flag": "ok",
        "h": (hlo, hhi),
        "d": (dlo, dhi),
        "j": (jlo, jhi),
        "g_lo": g_lo,
        "g_hi": g_hi,
        "S": S,
        "r0_lo": r0_lo,
        "r0_hi": r0_hi,
        "pos_lo": positive_hours(x, r0_lo),
        "pos_hi": positive_hours(x, r0_hi),
    }


def T_if(D, x):
    return D


def hand_examples():
    out = {}
    # cyclic sharp
    x = np.array([2, -1, -1, -2, -2, -2], float)
    units = [{"E": 2.0, "Pc": 2.0, "Pd": 2.0, "eta": 1.0}]
    info = hd_rates(x, units, 3, 2, True)
    out["sharp_cyc"] = {k: info[k] for k in ("flag", "h", "d", "j", "g_lo", "g_hi", "S")}
    info_l = hd_rates(x, units, 3, 2, False)
    out["sharp_lin_on_cyc_profile"] = {k: info_l[k] for k in ("flag", "h", "d", "j", "g_lo", "g_hi")}
    # linear sharp T=5
    x5 = np.array([-1, 2, -1, -2, -2], float)
    info5 = hd_rates(x5, units, 2, 1, False)
    out["sharp_lin"] = {k: info5[k] for k in ("flag", "h", "d", "j", "g_lo", "g_hi", "S")}
    # fleet split of the sharp example
    units2 = [
        {"E": 1.0, "Pc": 1.0, "Pd": 1.0, "eta": 1.0},
        {"E": 1.0, "Pc": 1.0, "Pd": 1.0, "eta": 1.0},
    ]
    infof = hd_rates(x, units2, 3, 2, True)
    out["sharp_fleet"] = {k: infof[k] for k in ("flag", "h", "d", "j", "g_lo", "g_hi", "S")}
    # proportional eta=0.5, double charge window energy: y=4 so useful charge eta*y=2, E=2
    xh = np.array([4.0, -1, -1, -2, -2, -2])
    unitsh = [{"E": 2.0, "Pc": 4.0, "Pd": 2.0, "eta": 0.5}]
    infoh = hd_rates(xh, unitsh, 3, 2, True)
    out["sharp_eta_half"] = {k: infoh[k] for k in ("flag", "h", "d", "j", "g_lo", "g_hi", "S", "pos_lo", "pos_hi")}
    # S/2 sharpness: two monotone single-hour requirements
    xs = np.array([1.0, -1.0, -1.0])
    us = [{"E": 1.0, "Pc": 1.0, "Pd": 1.0, "eta": 1.0}]
    def one_hour(r, hour):
        return schedule_lp(xs, us, r, [hour], objective="ens") is not None and True

    def cover_hour(r, hour):
        # deficit hour `hour` fully covered. Other deficit hours free.
        return schedule_lp(xs, us, r, [hour]) is not None

    def cover_both(r):
        return schedule_lp(xs, us, r, [1, 2]) is not None

    hi = 1.0
    a, b, f = bisect(lambda r: cover_hour(r, 1), hi)
    c, d, g = bisect(lambda r: cover_hour(r, 2), hi)
    e, h, k = bisect(cover_both, hi)
    out["half_sharp"] = {"r1": (a, b, f), "r2": (c, d, g), "rj": (e, h, k)}
    # explicit average of the two extreme schedules
    # mode1: c=(1,0,0), d=(0,1,0), e0=0
    # mode2: c=(1,0,0), d=(0,0,1), e0=0
    p1 = {
        "c": np.array([[1.0, 0.0, 0.0]]),
        "d": np.array([[0.0, 1.0, 0.0]]),
        "e": np.array([[1.0, 0.0, 0.0]]),
        "e0": np.array([0.0]),
    }
    p2 = {
        "c": np.array([[1.0, 0.0, 0.0]]),
        "d": np.array([[0.0, 0.0, 1.0]]),
        "e": np.array([[1.0, 1.0, 0.0]]),
        "e0": np.array([0.0]),
    }
    out["half_sharp_avg"] = avg_check(xs, us, [p1, p2], 0.0)
    # B unit check: eta 1 and 0.5, D=C=1, w=0.5, B=0.25
    bunits = [
        {"E": 1.0, "Pc": 1.0, "Pd": 1.0, "eta": 1.0},
        {"E": 1.0, "Pc": 1.0, "Pd": 1.0, "eta": 0.5},
    ]
    B, _u = compute_B(bunits)
    out["B_unit"] = B
    # M formulas vs brute
    bad_m = []
    for T in range(1, 11):
        for D in range(0, T + 1):
            if brute_M(T, D, False) != M_lin(T, D) or brute_M(T, D, True) != M_cyc(T, D):
                bad_m.append((T, D, brute_M(T, D, False), M_lin(T, D), brute_M(T, D, True), M_cyc(T, D)))
    out["M_mismatches"] = bad_m
    return out


def position_audit(rng, n_sys=40):
    """LP feasibility vs cuts, and whether position of equal deficits matters."""
    mismatches = []
    pos_dependent = []
    checked = 0
    for s in range(n_sys):
        T = int(rng.integers(5, 8))
        tc = int(rng.integers(0, T))
        x = -rng.uniform(0.3, 2.0, size=T)
        x[tc] = float(rng.uniform(0.8, 3.0))
        m = int(rng.integers(1, 4))
        eta = float(rng.choice([1.0, 0.8, 0.5, 0.35]))
        units = random_units(rng, m, True, eta=eta)
        # force at least the charge hour positive at r=0 and others negative
        net = network_of(x, units, 0.0)
        if net is None:
            continue
        N = [t for t in range(T) if t != tc and x[t] < -POS]
        if len(N) < 2:
            continue
        # compare LP and cuts on random subsets
        for _ in range(12):
            k = int(rng.integers(1, len(N) + 1))
            F = list(rng.choice(N, size=k, replace=False))
            costs = [float(-x[t]) for t in F]
            cut = cuts_ok(costs, net["devs"], net["Q"])
            try:
                lp = pure_cover_lp(x, units, 0.0, F)
            except SolverBad as e:
                mismatches.append({"kind": "solver", "status": e.status})
                continue
            checked += 1
            if cut != lp:
                mismatches.append(
                    {
                        "x": x.tolist(),
                        "units": units,
                        "F": [int(t) for t in F],
                        "cut": cut,
                        "lp": lp,
                        "costs": costs,
                    }
                )
                if len(mismatches) >= 5:
                    return {"checked": checked, "mismatches": mismatches, "pos_dependent": pos_dependent}
        # position: permute deficit values over N, test cover of a fixed size by largest costs
        base_costs = sorted([float(-x[t]) for t in N], reverse=True)
        for _ in range(6):
            rng.shuffle(base_costs)
            # actually reassign the multiset randomly
        multiset = [float(-x[t]) for t in N]
        # several placements
        placements = []
        for _ in range(8):
            rng.shuffle(multiset)
            placements.append(list(multiset))
        # For each placement, feasibility of covering the two largest positions' hours
        # Use the same SET OF VALUES (the two largest numbers), placed differently.
        values_desc = sorted([float(-x[t]) for t in N], reverse=True)
        if len(values_desc) < 2:
            continue
        target = values_desc[:2]
        answers = []
        for place in placements:
            xx = x.copy()
            for t, val in zip(N, place):
                xx[t] = -val
            # cover hours whose deficit equals the two target values — if duplicates, cover two hours
            hours = []
            need = list(target)
            for t in N:
                if need and abs(-xx[t] - need[0]) < 1e-9:
                    hours.append(t)
                    need = need[1:]
            if need:
                continue
            try:
                lp = pure_cover_lp(xx, units, 0.0, hours)
            except SolverBad:
                continue
            costs = [float(-xx[t]) for t in hours]
            cut = cuts_ok(costs, net["devs"], net["Q"])
            answers.append((lp, cut, xx.tolist(), hours))
        lps = {a[0] for a in answers}
        if len(lps) > 1:
            pos_dependent.append(
                {
                    "units": units,
                    "answers": [
                        {"lp": a[0], "cut": a[1], "x": a[2], "hours": [int(h) for h in a[3]]} for a in answers
                    ],
                }
            )
        for a in answers:
            if a[0] != a[1]:
                mismatches.append({"kind": "perm", "lp": a[0], "cut": a[1], "x": a[2], "hours": a[3]})
    return {"checked": checked, "mismatches": mismatches, "pos_dependent": pos_dependent}


def combo_lemma(rng, trials=2000):
    """Direct test of the fleet set-selection construction against cuts."""
    fails = []
    for _ in range(trials):
        m = int(rng.integers(1, 4))
        devs = []
        for _i in range(m):
            p = float(rng.uniform(0.1, 2.0))
            C = float(rng.uniform(0.0, 4.0))
            devs.append((C, p))
        P = sum(p for _C, p in devs)
        Q = float(rng.uniform(0.0, P * 3))
        n = int(rng.integers(2, 8))
        # random costs in (0, P]
        costs = {t: float(rng.uniform(0.05, P)) for t in range(n)}
        # rejection-sample feasible A, B
        ids = list(range(n))

        def rand_set():
            k = int(rng.integers(0, n + 1))
            pick = list(rng.choice(ids, size=k, replace=False)) if k else []
            return pick

        A = B = None
        for _try in range(40):
            cand = rand_set()
            if cuts_ok([costs[t] for t in cand], devs, Q):
                if A is None:
                    A = cand
                elif B is None:
                    B = cand
                    break
        if A is None or B is None:
            # build feasible sets from a prefix of cheapest hours
            order = sorted(ids, key=lambda t: costs[t])
            feas = []
            for k in range(0, n + 1):
                if cuts_ok([costs[t] for t in order[:k]], devs, Q):
                    feas.append(order[:k])
            if len(feas) < 1:
                continue
            A = feas[int(rng.integers(0, len(feas)))]
            B = feas[int(rng.integers(0, len(feas)))]
        F, new, boost = proof_select(costs, A, B, P)
        if not set(B) <= F:
            fails.append({"why": "B not subset", "A": A, "B": B, "F": list(F)})
            continue
        if len(F) < len(A) and not (len(B) >= len(A)):
            fails.append({"why": "size", "A": A, "B": list(B), "F": list(F)})
            continue
        if boost > P / 4 + 1e-8:
            fails.append({"why": "boost", "boost": boost, "P": P})
            continue
        if not cuts_ok([new[t] for t in F], devs, Q):
            fails.append(
                {
                    "why": "cuts",
                    "devs": devs,
                    "Q": Q,
                    "P": P,
                    "costs": costs,
                    "A": list(A),
                    "B": list(B),
                    "F": list(F),
                    "new": {str(k): v for k, v in new.items()},
                }
            )
            if len(fails) >= 8:
                break
    return {"trials": trials, "fails": fails}


def sweep_half(rng, n_target, cyclic):
    rows = []
    fails_solver = 0
    tries = 0
    max_ratio_hi = 0.0
    worst = None
    while len(rows) < n_target and tries < n_target * 6:
        tries += 1
        T = int(rng.integers(5, 8))
        n_pos = int(rng.integers(1, min(4, T)))
        x = random_x(rng, T, n_pos)
        equal = bool(rng.random() < 0.5)
        units = random_units(rng, int(rng.integers(1, 4)), equal)
        H, D = pick_HD(rng, T, cyclic)
        try:
            info = hd_rates(x, units, H, D, cyclic)
        except SolverBad:
            fails_solver += 1
            continue
        if info["flag"] != "ok":
            fails_solver += 1
            continue
        S = info["S"]
        ratio = info["g_hi"] / S if S > 0 else 0.0
        if ratio > max_ratio_hi:
            max_ratio_hi = ratio
            worst = {
                "ratio_hi": ratio,
                "g_lo": info["g_lo"],
                "g_hi": info["g_hi"],
                "S": S,
                "H": H,
                "D": D,
                "x": x.tolist(),
                "units": units,
                "cyclic": cyclic,
                "h": info["h"],
                "d": info["d"],
                "j": info["j"],
            }
        viol = info["g_lo"] > 0.5 * S + VIOL
        rows.append(
            {
                "g_lo": info["g_lo"],
                "g_hi": info["g_hi"],
                "S": S,
                "viol": viol,
                "equal_eta": equal,
                "n_pos_nominal": n_pos,
                "H": H,
                "D": D,
                "T": T,
            }
        )
        if viol:
            rows[-1]["x"] = x.tolist()
            rows[-1]["units"] = units
            break
    return {
        "kept": len(rows),
        "tries": tries,
        "solver_bad": fails_solver,
        "violations": sum(1 for r in rows if r["viol"]),
        "max_g_hi_over_S": max_ratio_hi,
        "worst": worst,
        "n_pos_gamma": sum(1 for r in rows if r["g_lo"] > 1e-3),
        "max_g_lo_over_S": max((r["g_lo"] / r["S"]) if r["S"] else 0 for r in rows),
    }


def sweep_quarter(rng, n_target, cyclic):
    rows = []
    fails_solver = 0
    rejected_pos = 0
    construct_fail = []
    tries = 0
    max_ratio = 0.0
    worst = None
    while len(rows) < n_target and tries < n_target * 15:
        tries += 1
        T = int(rng.integers(5, 8))
        tc = int(rng.integers(0, T))
        x = -rng.uniform(0.3, 2.2, size=T)
        x[tc] = float(rng.uniform(0.6, 3.0))
        if rng.random() < 0.3:
            zc = int(rng.integers(0, T))
            if zc != tc:
                x[zc] = 0.0
        eta = float(rng.choice([1.0, 0.85, 0.6, 0.4]))
        units = random_units(rng, int(rng.integers(1, 4)), True, eta=eta)
        H, D = pick_HD(rng, T, cyclic)
        try:
            info = hd_rates(x, units, H, D, cyclic)
        except SolverBad:
            fails_solver += 1
            continue
        if info["flag"] != "ok":
            fails_solver += 1
            continue
        if info["pos_lo"] != info["pos_hi"] or len(info["pos_hi"]) != 1:
            rejected_pos += 1
            continue
        S = info["S"]
        ratio = info["g_hi"] / S if S else 0.0
        viol = info["g_lo"] > 0.25 * S + VIOL
        if ratio > max_ratio:
            max_ratio = ratio
            worst = {
                "ratio_hi": ratio,
                "g_lo": info["g_lo"],
                "g_hi": info["g_hi"],
                "S": S,
                "H": H,
                "D": D,
                "x": x.tolist(),
                "units": units,
                "h": info["h"],
                "d": info["d"],
                "j": info["j"],
                "pos": info["pos_hi"],
            }
        # construction at r0_hi
        r = info["r0_hi"]
        net = network_of(x, units, r)
        cfail = None
        if net is not None:
            try:
                modeH = first_mode(x, units, r, H, len(x), cyclic)
                modeD = first_mode(x, units, r, len(x), D, cyclic)
            except SolverBad:
                fails_solver += 1
                continue
            if modeH is None or modeD is None:
                cfail = "no mode"
            else:
                sm = net["sm"]
                # pure covered hours: ell ~ 0 and sm>0. first_mode may partially cover the
                # positive side; rebuild pure covers from the declared cover list.
                costs = {t: float(sm[t]) for t in range(len(x)) if sm[t] > POS}
                A = [t for t in modeH["cover"]]
                B = [t for t in modeD["cover"]]
                # A, B must be cuts-feasible. If the LP mode exists they are.
                try:
                    F, new, boost = proof_select(costs, A, B, net["P"])
                    ok_cuts = cuts_ok([new[t] for t in F], net["devs"], net["Q"])
                    if (not set(B) <= set(F)) or boost > net["P"] / 4 + 1e-6 or not ok_cuts:
                        cfail = {
                            "A": A,
                            "B": B,
                            "F": list(F),
                            "boost": boost,
                            "P": net["P"],
                            "ok_cuts": ok_cuts,
                            "x": x.tolist(),
                            "units": units,
                            "r": r,
                            "H": H,
                            "D": D,
                        }
                except RuntimeError as e:
                    cfail = str(e)
        if cfail:
            construct_fail.append(cfail)
        rows.append({"g_lo": info["g_lo"], "g_hi": info["g_hi"], "S": S, "viol": viol, "T": T, "H": H, "D": D})
        if viol:
            rows[-1]["x"] = x.tolist()
            rows[-1]["units"] = units
            break
    return {
        "kept": len(rows),
        "tries": tries,
        "rejected_not_one_positive": rejected_pos,
        "solver_bad": fails_solver,
        "violations": sum(1 for r in rows if r["viol"]),
        "construct_fails": construct_fail[:6],
        "n_construct_fails": len(construct_fail),
        "max_g_hi_over_S": max_ratio,
        "max_g_lo_over_S": max(((r["g_lo"] / r["S"]) if r["S"] else 0) for r in rows) if rows else None,
        "n_pos_gamma": sum(1 for r in rows if r["g_lo"] > 1e-3),
        "worst": worst,
    }


def sweep_triple(rng, n_target, cyclic):
    rows = []
    fails_solver = 0
    tries = 0
    worst = None
    max_over_23 = -1e9
    while len(rows) < n_target and tries < n_target * 8:
        tries += 1
        T = int(rng.integers(5, 8))
        n_pos = int(rng.integers(1, min(4, T)))
        x = random_x(rng, T, n_pos)
        # unequal eta on purpose for most, a few equal
        equal = tries % 7 == 0
        units = random_units(rng, int(rng.integers(2, 4)), equal)
        H, D = pick_HD(rng, T, cyclic)
        total_def = float(np.maximum(-x, 0).sum())
        ens_lim = float(rng.uniform(0.15, 0.8) * total_def)
        hi = max(0.0, float(-np.min(x))) + 1e-6
        try:
            B, _u = compute_B(units)
            e_lo, e_hi, ef = bisect(lambda r: feasible(x, units, r, len(x), len(x), cyclic, ens_lim), hi)
            h_lo, h_hi, hf = bisect(lambda r: feasible(x, units, r, H, len(x), cyclic, None), hi)
            d_lo, d_hi, df = bisect(lambda r: feasible(x, units, r, len(x), D, cyclic, None), hi)
            j_lo, j_hi, jf = bisect(lambda r: feasible(x, units, r, H, D, cyclic, ens_lim), hi)
        except SolverBad:
            fails_solver += 1
            continue
        if "bad" in (ef, hf, df, jf) or "never" in (ef, hf, df, jf) or e_lo is None:
            fails_solver += 1
            continue
        S = sum(u["Pd"] for u in units)
        r0_lo = max(e_lo, h_lo, d_lo)
        r0_hi = max(e_hi, h_hi, d_hi)
        g_lo = j_lo - r0_hi
        g_hi = j_hi - r0_lo
        bound_B = B + 0.5 * S
        bound_23 = (2.0 / 3.0) * S
        bound = min(bound_B, bound_23)
        viol = g_lo > bound + VIOL
        over = g_hi - bound_23
        if over > max_over_23:
            max_over_23 = over
            worst = {
                "g_lo": g_lo,
                "g_hi": g_hi,
                "S": S,
                "B": B,
                "bound": bound,
                "x": x.tolist(),
                "units": units,
                "H": H,
                "D": D,
                "ens": ens_lim,
                "e": (e_lo, e_hi),
                "h": (h_lo, h_hi),
                "d": (d_lo, d_hi),
                "j": (j_lo, j_hi),
            }
        # construction of 2/3 at r0_hi: three modes
        r = r0_hi
        try:
            pE = first_mode(x, units, r, len(x), len(x), cyclic, ens_lim)
            pH = first_mode(x, units, r, H, len(x), cyclic, None)
            pD = first_mode(x, units, r, len(x), D, cyclic, None)
        except SolverBad:
            fails_solver += 1
            continue
        avg_prob = None
        if pE and pH and pD:
            # rebuild pure-ish discharges already in packs
            rep = avg_check(x, units, [pE, pH, pD], r)
            if rep["problems"] or rep["ident"] > 1e-4 or rep["max_q"] > bound_23 + 1e-5:
                avg_prob = rep["problems"] + [f"ident={rep['ident']}", f"q={rep['max_q']}"]
            # profiles: ell_new should meet all three because it equals the pointwise min
            # of three profiles. Check H/D/ENS on rep['ell'] directly.
            bad, _h, _d, _e = check_requirements_profile(rep["ell"], None, H, D, cyclic, ens_lim, {"H", "D", "E"})
            if bad:
                avg_prob = (avg_prob or []) + bad
        else:
            avg_prob = ["missing mode"]
        rows.append(
            {
                "g_lo": g_lo,
                "g_hi": g_hi,
                "S": S,
                "B": B,
                "bound": bound,
                "viol": viol,
                "equal": equal,
                "avg_prob": avg_prob,
            }
        )
        if viol:
            rows[-1]["x"] = x.tolist()
            rows[-1]["units"] = units
            rows[-1]["H"] = H
            rows[-1]["D"] = D
            rows[-1]["ens"] = ens_lim
            break
    n_avg = sum(1 for r in rows if r["avg_prob"])
    return {
        "kept": len(rows),
        "tries": tries,
        "solver_bad": fails_solver,
        "violations": sum(1 for r in rows if r["viol"]),
        "avg_problems": n_avg,
        "avg_examples": [r["avg_prob"] for r in rows if r["avg_prob"]][:5],
        "max_g_hi": max((r["g_hi"] for r in rows), default=None),
        "max_g_lo": max((r["g_lo"] for r in rows), default=None),
        "max_g_hi_over_minbound": max(((r["g_hi"] / r["bound"]) if r["bound"] > 1e-9 else 0) for r in rows) if rows else None,
        "worst": worst,
        "n_pos_gamma": sum(1 for r in rows if r["g_lo"] > 1e-3),
    }


def sweep_avg_half(rng, n_target, cyclic):
    """At r0, average an H-mode and a D-mode and check the 1/2 construction."""
    bad = []
    ok = 0
    tries = 0
    solver_bad = 0
    max_q_over = 0.0
    while ok < n_target and tries < n_target * 8 and len(bad) < 5:
        tries += 1
        T = int(rng.integers(5, 8))
        x = random_x(rng, T, int(rng.integers(1, min(4, T))))
        equal = bool(rng.random() < 0.5)
        units = random_units(rng, int(rng.integers(1, 4)), equal)
        H, D = pick_HD(rng, T, cyclic)
        hi = max(0.0, float(-np.min(x))) + 1e-6
        try:
            hlo, hhi, hf = bisect(lambda r: feasible(x, units, r, H, len(x), cyclic), hi)
            dlo, dhi, df = bisect(lambda r: feasible(x, units, r, len(x), D, cyclic), hi)
            if hf != "ok" or df != "ok":
                solver_bad += 1
                continue
            r = max(hhi, dhi)
            pH = first_mode(x, units, r, H, len(x), cyclic)
            pD = first_mode(x, units, r, len(x), D, cyclic)
            if pH is None or pD is None:
                solver_bad += 1
                continue
            rep = avg_check(x, units, [pH, pD], r)
        except SolverBad:
            solver_bad += 1
            continue
        max_q_over = max(max_q_over, rep["max_q"] / rep["S"] if rep["S"] else 0)
        if rep["problems"] or rep["ident"] > 1e-4 or rep["max_q"] > 0.5 * rep["S"] + 1e-5:
            bad.append({"problems": rep["problems"], "ident": rep["ident"], "q": rep["max_q"], "S": rep["S"], "x": x.tolist(), "units": units})
            continue
        bad_req, *_ = check_requirements_profile(rep["ell"], None, H, D, cyclic, None, {"H", "D"})
        if bad_req:
            bad.append({"req": bad_req, "x": x.tolist(), "units": units, "ell": rep["ell"].tolist()})
            continue
        ok += 1
    return {"ok": ok, "tries": tries, "solver_bad": solver_bad, "bad": bad, "max_q_over_S": max_q_over}


def main():
    t0 = time.perf_counter()
    rng = np.random.default_rng(20261021)
    print("hand", flush=True)
    hand = hand_examples()
    print(json.dumps(hand, default=str)[:2000], flush=True)
    print("combo", flush=True)
    combo = combo_lemma(rng, 2000)
    print("combo fails", len(combo["fails"]), flush=True)
    print("position", flush=True)
    pos = position_audit(rng, 30)
    print("position checked", pos["checked"], "mis", len(pos["mismatches"]), "dep", len(pos["pos_dependent"]), flush=True)
    print("avg half", flush=True)
    avg = sweep_avg_half(rng, 80, True)
    print("avg", avg["ok"], "bad", len(avg["bad"]), "q", avg["max_q_over_S"], flush=True)
    print("half sweep", flush=True)
    half = sweep_half(rng, 200, True)
    print("half", half["kept"], "viol", half["violations"], "max", half["max_g_hi_over_S"], flush=True)
    print("half linear", flush=True)
    half_l = sweep_half(np.random.default_rng(20261022), 80, False)
    print("halfL", half_l["kept"], "viol", half_l["violations"], "max", half_l["max_g_hi_over_S"], flush=True)
    print("quarter", flush=True)
    quar = sweep_quarter(rng, 200, True)
    print("quarter", quar["kept"], "viol", quar["violations"], "cfail", quar["n_construct_fails"], "max", quar["max_g_hi_over_S"], flush=True)
    print("quarter linear", flush=True)
    quar_l = sweep_quarter(np.random.default_rng(20261023), 60, False)
    print("quarterL", quar_l["kept"], "viol", quar_l["violations"], "max", quar_l["max_g_hi_over_S"], flush=True)
    print("triple", flush=True)
    tri = sweep_triple(rng, 50, True)
    print("triple", tri["kept"], "viol", tri["violations"], "avgbad", tri["avg_problems"], flush=True)
    tri_l = sweep_triple(np.random.default_rng(7), 20, False)
    print("tripleL", tri_l["kept"], "viol", tri_l["violations"], flush=True)
    # multi-charge observed max already inside half sweep
    result = {
        "hand": hand,
        "combo_fails": combo["fails"],
        "combo_trials": combo["trials"],
        "position": pos,
        "avg_half": {k: avg[k] for k in ("ok", "tries", "solver_bad", "bad", "max_q_over_S")},
        "half": half,
        "half_linear": half_l,
        "quarter": quar,
        "quarter_linear": quar_l,
        "triple": tri,
        "triple_linear": tri_l,
        "seconds": time.perf_counter() - t0,
    }
    with open("g4_results.json", "w") as f:
        json.dump(result, f, default=str)
    print("DONE", result["seconds"], flush=True)


if __name__ == "__main__":
    main()
