
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
"""Core checks for G5: boundary slack, bisection, status handling.

Python is launched as `../py g5_*.py` (-I), so the script directory is added
to sys.path explicitly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

import ref_milp


TOL_FEAS = 1e-7
STATUS_NOTE = []


def reset_notes():
    STATUS_NOTE.clear()


def _solve_ref(x, units, r, objective="ens", H=None, retries=3):
    """ref_milp.solve with retries. Returns (value_or_None, status).

    status in {'optimal', 'infeasible', 'error'}.
    None value only for infeasible. Errors are recorded, not treated as zero.
    """
    gaps = [0.0, 1e-8, 1e-6]
    last = None
    for g in gaps[:retries]:
        try:
            # ref_milp hardcodes mip_rel_gap=0. Replicate call only on retry
            # by monkeypatching options through a local copy if needed.
            if g == 0.0:
                v = ref_milp.solve(x, units, r, objective=objective, H=H)
            else:
                v = _solve_gap(x, units, r, objective=objective, H=H, gap=g)
            return v, "optimal" if v is not None else "infeasible"
        except RuntimeError as e:
            last = str(e)
            STATUS_NOTE.append(f"retry gap={g} r={r} obj={objective}: {last}")
    STATUS_NOTE.append(f"SKIP solver error r={r} obj={objective}: {last}")
    return None, "error"


def _solve_gap(x, units, r, objective="ens", H=None, gap=1e-8, time_limit=60):
    """Same model as ref_milp.solve but with a nonzero MIP gap."""
    tau = 1.0
    T, m = len(x), len(units)
    s = np.asarray(x, float) + r
    sp, sm = np.maximum(s, 0.0), np.maximum(-s, 0.0)
    nc = m * T
    ic = lambda i, t: i * T + t
    idd = lambda i, t: nc + i * T + t
    ie = lambda i, t: 2 * nc + i * T + t
    ie0 = lambda i: 3 * nc + i
    il = lambda t: 3 * nc + m + t
    iy = lambda t: 3 * nc + m + T + t
    n = 3 * nc + m + 2 * T
    lb = np.zeros(n)
    ub = np.zeros(n)
    integ = np.zeros(n)
    for i, u in enumerate(units):
        for t in range(T):
            ub[ic(i, t)] = u["Pc"] if sp[t] > 0 else 0.0
            ub[idd(i, t)] = u["Pd"] if sm[t] > 0 else 0.0
            ub[ie(i, t)] = u["E"]
        ub[ie0(i)] = u["E"]
    for t in range(T):
        ub[il(t)] = sm[t]
        ub[iy(t)] = 1.0
        integ[iy(t)] = 1.0
    rows = m * T + m + 3 * T + 2
    A = lil_matrix((rows, n))
    lo, hi = [], []
    k = 0
    for i, u in enumerate(units):
        dl, eta = u.get("delta", 0.0), u["eta"]
        for t in range(T):
            A[k, ie(i, t)] = 1.0
            A[k, ie(i, t - 1) if t > 0 else ie0(i)] = -(1.0 - dl)
            A[k, ic(i, t)] = -tau * eta
            A[k, idd(i, t)] = tau
            lo.append(0.0)
            hi.append(0.0)
            k += 1
        A[k, ie(i, T - 1)] = 1.0
        A[k, ie0(i)] = -1.0
        lo.append(0.0)
        hi.append(0.0)
        k += 1
    for t in range(T):
        for i in range(m):
            A[k, ic(i, t)] = 1.0
        lo.append(0.0)
        hi.append(sp[t])
        k += 1
        for i in range(m):
            A[k, idd(i, t)] = 1.0
        A[k, il(t)] = 1.0
        lo.append(sm[t])
        hi.append(sm[t])
        k += 1
        A[k, il(t)] = 1.0
        A[k, iy(t)] = -sm[t]
        lo.append(-np.inf)
        hi.append(0.0)
        k += 1
    A[k, [il(t) for t in range(T)]] = tau
    lo.append(0.0)
    hi.append(np.inf)
    k += 1
    A[k, [iy(t) for t in range(T)]] = tau
    lo.append(0.0)
    hi.append(H if H is not None else np.inf)
    k += 1
    cost = np.zeros(n)
    if objective == "ens":
        cost[[il(t) for t in range(T)]] = tau
        if H is None:
            integ[:] = 0.0
    else:
        cost[[iy(t) for t in range(T)]] = tau
    res = milp(
        cost,
        constraints=LinearConstraint(A.tocsr(), lo, hi),
        bounds=Bounds(lb, ub),
        integrality=integ,
        options={"mip_rel_gap": gap, "time_limit": time_limit},
    )
    if res.status == 0:
        return float(res.fun)
    if res.status == 2:
        return None
    raise RuntimeError(res.message)


def separate_at_0(x, units):
    ens, st_e = _solve_ref(x, units, 0.0, objective="ens")
    lol, st_h = _solve_ref(x, units, 0.0, objective="lolh")
    return ens, lol, st_e, st_h


def joint_ens(x, units, r, lol_limit):
    """Min ENS subject to LOLH <= lol_limit. (value, status)."""
    return _solve_ref(x, units, r, objective="ens", H=float(lol_limit))


def boundary_slack(x, units, r):
    """ENS*(0) - ENS_H(r), H = LOLH*(0).

    Positive slack => joint meets the r=0 records at this r (Gamma <= r,
    since r_E = r_H = 0). Negative => violation of Gamma <= r.
    """
    ens, lol, st_e, st_h = separate_at_0(x, units)
    if st_e != "optimal" or st_h != "optimal":
        return {"status": "error", "ens": ens, "lol": lol, "st": (st_e, st_h)}
    v, st = joint_ens(x, units, r, lol)
    if st != "optimal":
        return {"status": st, "ENS": ens, "LOL": lol}
    return {
        "status": "ok",
        "ENS": float(ens),
        "LOL": float(lol),
        "joint": float(v),
        "slack": float(ens) - float(v),
    }


def _feasible_joint(x, units, r, ens_lim, lol_lim):
    v, st = joint_ens(x, units, r, lol_lim)
    if st == "error":
        return None
    if st == "infeasible":
        return False
    return v <= ens_lim + 1e-8


def _feasible_E(x, units, r, ens_lim):
    v, st = _solve_ref(x, units, r, objective="ens")
    if st == "error":
        return None
    if st == "infeasible":
        return False
    return v <= ens_lim + 1e-8


def _feasible_H(x, units, r, lol_lim):
    v, st = _solve_ref(x, units, r, objective="lolh")
    if st == "error":
        return None
    if st == "infeasible":
        return False
    return v <= lol_lim + 1e-8


def bisect_r(x, units, kind, ens_lim, lol_lim, hi=None, tol=1e-4):
    """Return (lo, hi, status) with status ok/error. Infeasible at hi => None bounds."""
    if hi is None:
        hi = max(0.0, -min(x)) + 1e-9
    if kind == "E":
        feas0 = _feasible_E(x, units, 0.0, ens_lim)
    elif kind == "H":
        feas0 = _feasible_H(x, units, 0.0, lol_lim)
    else:
        feas0 = _feasible_joint(x, units, 0.0, ens_lim, lol_lim)
    if feas0 is None:
        return None, None, "error"
    if feas0:
        return 0.0, 0.0, "ok"
    # ensure hi feasible; expand a few times
    for _ in range(8):
        if kind == "E":
            fh = _feasible_E(x, units, hi, ens_lim)
        elif kind == "H":
            fh = _feasible_H(x, units, hi, lol_lim)
        else:
            fh = _feasible_joint(x, units, hi, ens_lim, lol_lim)
        if fh is None:
            return None, None, "error"
        if fh:
            break
        hi = hi * 2 + 0.1
    else:
        return None, None, "unbounded"
    lo = 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if kind == "E":
            fm = _feasible_E(x, units, mid, ens_lim)
        elif kind == "H":
            fm = _feasible_H(x, units, mid, lol_lim)
        else:
            fm = _feasible_joint(x, units, mid, ens_lim, lol_lim)
        if fm is None:
            return None, None, "error"
        if fm:
            hi = mid
        else:
            lo = mid
    return lo, hi, "ok"


def gamma_interval(x, units, ens_lim, lol_lim, tol=1e-4, hi=None):
    rE = bisect_r(x, units, "E", ens_lim, lol_lim, hi=hi, tol=tol)
    rH = bisect_r(x, units, "H", ens_lim, lol_lim, hi=hi, tol=tol)
    rJ = bisect_r(x, units, "J", ens_lim, lol_lim, hi=hi, tol=tol)
    if "error" in (rE[2], rH[2], rJ[2]) or "unbounded" in (rE[2], rH[2], rJ[2]):
        return {"status": "error", "rE": rE, "rH": rH, "rJ": rJ}
    # conservative: Gamma in [rJ_lo - max(rE_hi, rH_hi), rJ_hi - max(rE_lo, rH_lo)]
    glo = rJ[0] - max(rE[1], rH[1])
    ghi = rJ[1] - max(rE[0], rH[0])
    return {
        "status": "ok",
        "rE": (rE[0], rE[1]),
        "rH": (rH[0], rH[1]),
        "rJ": (rJ[0], rJ[1]),
        "Gamma": (glo, ghi),
    }


def own_ens(x, units, r):
    """Min ENS by linprog, one storage or many, no LOLH binaries.

    Independent of ref_milp: scipy.linprog, highs-ds if it works, else highs.
    Returns (value, status).
    """
    return _own_lp(x, units, r, cover=None)


def own_joint_ens(x, units, r, cover_hours):
    """Min ENS forcing full cover on cover_hours (must be power-feasible)."""
    return _own_lp(x, units, r, cover=set(cover_hours))


def _own_lp(x, units, r, cover=None):
    T = len(x)
    m = len(units)
    s = np.asarray(x, float) + float(r)
    sp = np.maximum(s, 0.0)
    sm = np.maximum(-s, 0.0)
    # vars: for each unit, c_t, d_t, e_t (t=0..T-1) where e_t is end of hour t+1? 
    # Use e_t end of hour t, t=0..T-1, and e_before hour 0 equals e_{T-1} (cycle).
    # Layout per unit: c[0:T], d[T:2T], e[2T:3T]
    nu = 3 * T
    n = nu * m
    cvar = lambda i, t: i * nu + t
    dvar = lambda i, t: i * nu + T + t
    evar = lambda i, t: i * nu + 2 * T + t
    bounds = []
    for i, u in enumerate(units):
        for t in range(T):
            bounds.append((0.0, u["Pc"] if sp[t] > 1e-15 else 0.0))
        for t in range(T):
            ubd = u["Pd"] if sm[t] > 1e-15 else 0.0
            if cover is not None and t in cover:
                # full cover of the bus is a sum constraint; per-unit upper stays Pd
                pass
            bounds.append((0.0, ubd))
        for t in range(T):
            bounds.append((0.0, u["E"]))
    # constraints
    A_rows = []
    b_eq = []
    # dynamics
    for i, u in enumerate(units):
        gamma = 1.0 - u.get("delta", 0.0)
        eta = u["eta"]
        for t in range(T):
            row = np.zeros(n)
            prev = t - 1 if t > 0 else T - 1
            row[evar(i, t)] = 1.0
            row[evar(i, prev)] = -gamma
            row[cvar(i, t)] = -eta
            row[dvar(i, t)] = 1.0
            A_rows.append(row)
            b_eq.append(0.0)
    A_ub = []
    b_ub = []
    # bus charge <= surplus, discharge <= deficit (equality via two sides for balance of ell)
    # ell_t = sm_t - sum d >= 0 is sum d <= sm, and we minimise sum ell = sum sm - sum d
    # so maximise sum d, i.e. minimise -sum d.
    for t in range(T):
        row = np.zeros(n)
        for i in range(m):
            row[cvar(i, t)] = 1.0
        A_ub.append(row)
        b_ub.append(sp[t])
        row = np.zeros(n)
        for i in range(m):
            row[dvar(i, t)] = 1.0
        A_ub.append(row)
        b_ub.append(sm[t])
        if cover is not None and t in cover:
            if sm[t] <= 1e-15:
                # nothing to cover
                continue
            # sum d >= sm  (together with <= becomes equality)
            row = np.zeros(n)
            for i in range(m):
                row[dvar(i, t)] = -1.0
            A_ub.append(row)
            b_ub.append(-sm[t] + 1e-12)
    cobj = np.zeros(n)
    for i in range(m):
        for t in range(T):
            cobj[dvar(i, t)] = -1.0
    A_eq = np.vstack(A_rows) if A_rows else None
    A_ub_m = np.vstack(A_ub) if A_ub else None
    res = linprog(
        cobj,
        A_ub=A_ub_m,
        b_ub=np.array(b_ub) if A_ub else None,
        A_eq=A_eq,
        b_eq=np.array(b_eq) if A_rows else None,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        # highs-ds fallback
        res = linprog(
            cobj,
            A_ub=A_ub_m,
            b_ub=np.array(b_ub) if A_ub else None,
            A_eq=A_eq,
            b_eq=np.array(b_eq) if A_rows else None,
            bounds=bounds,
            method="highs-ds",
        )
    if not res.success:
        msg = res.message
        if res.status == 2:
            return None, "infeasible"
        STATUS_NOTE.append(f"own lp fail r={r}: {msg}")
        return None, "error"
    ens = float(np.sum(sm) + res.fun)  # res.fun = -sum d
    return ens, "optimal"


def enumerate_joint(x, units, r, lol_limit):
    """Second solver: enumerate which deficit hours stay uncovered.

    For one or several storages. Cover-set LP via own_joint_ens.
    LOLH counts strict deficits only. Hours with sm=0 are covered for free.
    """
    s = np.asarray(x, float) + float(r)
    sm = np.maximum(-s, 0.0)
    deficit = [t for t in range(len(x)) if sm[t] > 1e-10]
    # power feasibility: an hour is individually uncoverable if sm > sum Pd
    pd_sum = sum(u["Pd"] for u in units)
    forced_open = [t for t in deficit if sm[t] > pd_sum + 1e-9]
    choosable = [t for t in deficit if t not in forced_open]
    # must leave all forced_open uncovered; may leave extra up to lol_limit - len(forced)
    budget = int(lol_limit) - len(forced_open)
    if budget < 0:
        return None, "infeasible"
    # enumerate subsets of choosable to leave open, size <= budget,
    # but to MINIMISE ENS we want the best cover. Number of subsets:
    # we must cover a subset of choosable, leaving at most `budget` open.
    # Equivalent: open set size <= budget.
    from itertools import combinations

    best = None
    # Always allow opening up to `budget` choosable hours (the rest covered).
    # Also try opening fewer.
    n = len(choosable)
    # If 2^n is large, cap — caller should avoid that. T<=10, n<=10, 1024 ok
    # but each is an LP. For search we use MILP; this is the verifier.
    for k in range(budget + 1):
        for open_set in combinations(choosable, k):
            cover = [t for t in choosable if t not in open_set]
            val, st = own_joint_ens(x, units, r, cover)
            if st == "error":
                return None, "error"
            if st == "infeasible":
                continue
            if best is None or val < best:
                best = val
    if best is None:
        return None, "infeasible"
    return best, "optimal"
