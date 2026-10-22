"""Independent storage solver (LP + covered-hour enumeration, alternate MILP).

Dynamics match the task statement:
    e_t = (1-delta) e_{prev} + tau*(eta*c_t - d_t),  0 <= e <= E,  e_T = e_0.
Charge only from surplus, discharge only into deficit.
ENS = tau * sum ell_t,  LOLH = tau * #{t : ell_t > 0}.

This is not the reference formulation: binaries are avoided by enumerating the
set of fully covered deficit hours and solving an LP. For larger T an alternate
MILP uses coverage indicators z_t (d >= s- * z) rather than loss indicators.
"""
from __future__ import annotations

# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import itertools
import math

import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds


TOL = 1e-8
ENS_SLACK = 1e-9
LP_OPTS = {"presolve": True, "time_limit": 30.0}


def _pack(x, units, r):
    x = np.asarray(x, dtype=float)
    T = int(x.size)
    s = x + float(r)
    sp = np.maximum(s, 0.0)
    sm = np.maximum(-s, 0.0)
    return x, units, float(r), T, sp, sm


def _layout(T, m):
    """Per unit: e0, c[t], d[t], e[t]."""
    nu = 1 + 3 * T
    return nu, m * nu


def min_ens(x, units, r, tau=1.0, cyclic=True, cover=None, return_plan=False):
    """Minimum ENS. cover: hours forced to ell=0. None if infeasible / not optimal.

    If return_plan, result is (ens, plan) with per-unit c, d, e, e0.
    """
    x, units, r, T, sp, sm = _pack(x, units, r)
    m = len(units)
    nu, n = _layout(T, m)
    if cover is None:
        cover = ()
    cover = set(cover)

    # equalities: dynamics (m*T), optional cyclic (m), optional cover hours
    n_eq = m * T + (m if cyclic else 0) + len(cover)
    Aeq = np.zeros((n_eq, n))
    beq = np.zeros(n_eq)
    bounds = [(0.0, 0.0)] * n
    cost = np.zeros(n)  # minimize -sum d  <=> maximize discharged energy

    for i, u in enumerate(units):
        base = i * nu
        E = float(u["E"])
        Pc = float(u["Pc"])
        Pd = float(u["Pd"])
        delta = float(u.get("delta", 0.0))
        eta = float(u["eta"])
        alpha = 1.0 - delta
        bounds[base] = (0.0, E)  # e0
        for t in range(T):
            bounds[base + 1 + t] = (0.0, Pc)
            bounds[base + 1 + T + t] = (0.0, Pd)
            bounds[base + 1 + 2 * T + t] = (0.0, E)
            cost[base + 1 + T + t] = -tau

    # sum_i c <= sp, sum_i d <= sm
    Aub = np.zeros((2 * T, n))
    bub = np.zeros(2 * T)
    for t in range(T):
        for i in range(m):
            base = i * nu
            Aub[t, base + 1 + t] = 1.0
            Aub[T + t, base + 1 + T + t] = 1.0
        bub[t] = sp[t]
        bub[T + t] = sm[t]

    row = 0
    for i, u in enumerate(units):
        base = i * nu
        alpha = 1.0 - float(u.get("delta", 0.0))
        eta = float(u["eta"])
        for t in range(T):
            prev = base if t == 0 else base + 1 + 2 * T + (t - 1)
            Aeq[row, base + 1 + 2 * T + t] = 1.0
            Aeq[row, prev] = -alpha
            Aeq[row, base + 1 + t] = -tau * eta
            Aeq[row, base + 1 + T + t] = tau
            row += 1
        if cyclic:
            Aeq[row, base + 1 + 2 * T + (T - 1)] = 1.0
            Aeq[row, base] = -1.0
            row += 1

    for t in sorted(cover):
        if sm[t] <= TOL:
            # nothing to cover; discharge stays at 0 via sm bound
            # still add a trivial equality only if sm>0
            continue
        for i in range(m):
            base = i * nu
            Aeq[row, base + 1 + T + t] = 1.0
        beq[row] = sm[t]
        row += 1
    if row < n_eq:
        Aeq = Aeq[:row]
        beq = beq[:row]

    res = linprog(cost, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds,
                  method="highs", options=LP_OPTS)
    if not res.success:
        return (None, None) if return_plan else None
    discharged = -float(res.fun) / tau
    ens = tau * float(np.sum(sm) - discharged)
    if -1e-7 < ens < 0:
        ens = 0.0
    ens = float(ens)
    if not return_plan:
        return ens
    plan = []
    for i in range(m):
        base = i * nu
        v = res.x
        plan.append({
            "e0": float(v[base]),
            "c": [float(v[base + 1 + t]) for t in range(T)],
            "d": [float(v[base + 1 + T + t]) for t in range(T)],
            "e": [float(v[base + 1 + 2 * T + t]) for t in range(T)],
        })
    return ens, plan


def deficit_hours(x, r, tau=1.0):
    s = np.asarray(x, dtype=float) + float(r)
    sm = np.maximum(-s, 0.0)
    return [t for t in range(len(x)) if sm[t] > TOL]


def min_lolh(x, units, r, tau=1.0, cyclic=True):
    """Minimum LOLH. None if the LP does not return an optimum for a subset."""
    hours = deficit_hours(x, r)
    if not hours:
        return 0.0
    # Largest coverable subset. Hours outside `hours` already have ell=0.
    uncertain = False
    n = len(hours)
    for k in range(n, -1, -1):
        for comb in itertools.combinations(hours, k):
            ens = min_ens(x, units, r, tau=tau, cyclic=cyclic, cover=comb)
            if ens is None:
                # could be infeasible (expected) — cannot distinguish solver failure
                # from infeasibility here; a later successful smaller set is fine.
                # If EVERY subset including empty fails, report uncertainty.
                continue
            return tau * float(n - k)
    return None


def joint_ok(x, units, r, ens_lim, lolh_lim, tau=1.0, cyclic=True):
    """True/False if a mode meets both limits. None if no LP returned optimum."""
    hours = deficit_hours(x, r)
    max_bad = int(math.floor(float(lolh_lim) / tau + 1e-9))
    if len(hours) <= max_bad:
        ens = min_ens(x, units, r, tau=tau, cyclic=cyclic, cover=())
        if ens is None:
            return None
        return ens <= float(ens_lim) + ENS_SLACK
    need = len(hours) - max_bad
    saw_success = False
    # Prefer covering the smallest deficit hours first (cheaper for LOLH).
    sm = np.maximum(-(np.asarray(x, dtype=float) + float(r)), 0.0)
    ordered = sorted(hours, key=lambda t: (sm[t], t))
    for k in range(need, len(hours) + 1):
        for comb in itertools.combinations(ordered, k):
            ens = min_ens(x, units, r, tau=tau, cyclic=cyclic, cover=comb)
            if ens is None:
                continue
            saw_success = True
            if ens <= float(ens_lim) + ENS_SLACK:
                return True
        # combinations() follows `ordered`, so the first items are the smallest.
        # Still exhaustive.
    if not saw_success:
        return None
    return False


def _bisect(pred, hi, tol=1e-8):
    """Smallest r in [0, hi] with pred(r) true. Returns (value, status).

    status: 'optimal' or 'uncertain'. Assumes monotonicity, then audits the ends.
    """
    p0 = pred(0.0)
    if p0 is None:
        return None, "uncertain"
    if p0:
        return 0.0, "optimal"
    phi = pred(hi)
    if phi is None:
        return None, "uncertain"
    if not phi:
        return None, "infeasible_at_hi"
    lo = 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        pm = pred(mid)
        if pm is None:
            return None, "uncertain"
        if pm:
            hi = mid
        else:
            lo = mid
    return hi, "optimal"


def evaluate_case(case, tol=1e-8):
    """Return r_E, r_H, r_joint, Gamma and statuses at the competition data in `case`."""
    x = case["x"]
    units = case["units"]
    tau = float(case.get("tau", 1.0))
    cyclic = bool(case.get("cyclic", True))
    ens_lim = float(case["ens_limit"])
    lolh_lim = float(case["lolh_limit"])
    hi = max(0.0, -min(x))
    # firm r >= -min(x) wipes out every deficit
    hi = max(hi, 0.0)

    def pred_E(r):
        v = min_ens(x, units, r, tau=tau, cyclic=cyclic)
        if v is None:
            return None
        return v <= ens_lim + ENS_SLACK

    def pred_H(r):
        v = min_lolh(x, units, r, tau=tau, cyclic=cyclic)
        if v is None:
            return None
        return v <= lolh_lim + ENS_SLACK

    def pred_J(r):
        return joint_ok(x, units, r, ens_lim, lolh_lim, tau=tau, cyclic=cyclic)

    rE, sE = _bisect(pred_E, hi, tol=tol)
    rH, sH = _bisect(pred_H, hi, tol=tol)
    rJ, sJ = _bisect(pred_J, hi, tol=tol)
    out = {
        "rE": rE, "rH": rH, "rJ": rJ,
        "status": {"E": sE, "H": sH, "J": sJ},
    }
    if None not in (rE, rH, rJ) and "infeasible" not in (sE, sH, sJ):
        out["Gamma"] = rJ - max(rE, rH)
        out["Gamma_status"] = "optimal" if {sE, sH, sJ} == {"optimal"} else "uncertain"
    else:
        out["Gamma"] = None
        out["Gamma_status"] = "uncertain"
    return out


def competition_norms(x, units, tau=1.0, cyclic=True):
    """ENS_bar = min ENS at r=0, LOLH_bar = min LOLH at r=0.

    Returns (ens, lolh, ens_status, lolh_status).
    """
    ens = min_ens(x, units, 0.0, tau=tau, cyclic=cyclic)
    lolh = min_lolh(x, units, 0.0, tau=tau, cyclic=cyclic)
    return ens, lolh


def gamma_at_competition(x, units, tol=1e-8, tau=1.0, cyclic=True):
    ens, lolh = competition_norms(x, units, tau=tau, cyclic=cyclic)
    if ens is None or lolh is None:
        return {"Gamma": None, "Gamma_status": "uncertain", "ens_bar": ens, "lolh_bar": lolh}
    case = {
        "x": list(map(float, x)),
        "units": units,
        "ens_limit": ens,
        "lolh_limit": lolh,
        "tau": tau,
        "cyclic": cyclic,
    }
    ev = evaluate_case(case, tol=tol)
    ev["ens_bar"] = ens
    ev["lolh_bar"] = lolh
    return ev


# --- alternate MILP (coverage binaries), used for T large and as a cross-check ---

def _milp_model(x, units, r, tau, cyclic, ens_lim, lolh_lim, objective):
    """Different encoding from ref_milp: z_t = 1 means hour t is fully covered."""
    x = np.asarray(x, dtype=float)
    T, m = len(x), len(units)
    s = x + float(r)
    sp = np.maximum(s, 0.0)
    sm = np.maximum(-s, 0.0)
    nu = 1 + 3 * T  # e0, c, d, e
    # then ell[t], z[t]
    n = m * nu + 2 * T
    base_ell = m * nu
    base_z = m * nu + T

    lb = np.zeros(n)
    ub = np.zeros(n)
    integ = np.zeros(n)
    for i, u in enumerate(units):
        b = i * nu
        ub[b] = u["E"]
        for t in range(T):
            ub[b + 1 + t] = u["Pc"]
            ub[b + 1 + T + t] = u["Pd"]
            ub[b + 1 + 2 * T + t] = u["E"]
    for t in range(T):
        ub[base_ell + t] = sm[t]
        ub[base_z + t] = 1.0
        integ[base_z + t] = 1.0

    rows = []
    lo = []
    hi = []

    def add(coeffs, l, h):
        row = np.zeros(n)
        for idx, val in coeffs:
            row[idx] += val
        rows.append(row)
        lo.append(l)
        hi.append(h)

    for i, u in enumerate(units):
        b = i * nu
        alpha = 1.0 - float(u.get("delta", 0.0))
        eta = float(u["eta"])
        for t in range(T):
            prev = b if t == 0 else b + 1 + 2 * T + (t - 1)
            add([
                (b + 1 + 2 * T + t, 1.0),
                (prev, -alpha),
                (b + 1 + t, -tau * eta),
                (b + 1 + T + t, tau),
            ], 0.0, 0.0)
        if cyclic:
            add([(b + 1 + 2 * T + (T - 1), 1.0), (b, -1.0)], 0.0, 0.0)
        else:
            add([(b, 1.0)], 0.0, 0.0)

    for t in range(T):
        add([(i * nu + 1 + t, 1.0) for i in range(m)], 0.0, sp[t])
        # sum d + ell = sm
        add([(i * nu + 1 + T + t, 1.0) for i in range(m)] + [(base_ell + t, 1.0)], sm[t], sm[t])
        # coverage: sum d >= sm * z   <=>  -sum d + sm * z <= 0
        if sm[t] > TOL:
            add([(i * nu + 1 + T + t, -1.0) for i in range(m)] + [(base_z + t, sm[t])], -np.inf, 0.0)
        else:
            # no deficit: z fixed 1 so it does not count as a loss hour, ell already 0
            lb[base_z + t] = 1.0

    if ens_lim is not None:
        add([(base_ell + t, tau) for t in range(T)], 0.0, float(ens_lim))
    if lolh_lim is not None:
        # LOLH = tau * sum (1-z) over deficit hours, and z=1 on non-deficit
        # tau * sum_t (1 - z_t) <= lolh_lim, but non-deficit z=1 so they add 0.
        # Equivalent: -tau * sum z <= lolh_lim - tau * T
        add([(base_z + t, -tau) for t in range(T)], -np.inf, float(lolh_lim) - tau * T)

    cost = np.zeros(n)
    if objective == "ens":
        cost[base_ell:base_ell + T] = tau
        if lolh_lim is None:
            integ[:] = 0.0
    elif objective == "lolh":
        cost[base_z:base_z + T] = -tau
        # constant tau*T added after; minimizing -tau sum z = tau * uncovered - tau*T
        # wait LOLH = tau * sum(1-z) = tau*T - tau*sum z, but non-deficit are fixed z=1
        # and deficit z variable. Since non-deficit z fixed 1, cost on them doesn't matter.
        # fun = -tau * sum z, LOLH = tau * (T_deficit_free) ... 
        # We'll convert: every hour contributes, z=1 on non-deficit.
        # LOLH = tau * sum_t (1-z_t) = tau*T + fun, because fun=-tau*sum z.
        # YES including non-deficit where z=1: sum(1-z) equals number of uncovered deficit hours.
    else:
        raise ValueError(objective)

    A = np.vstack(rows) if rows else np.zeros((0, n))
    res = milp(
        cost,
        constraints=LinearConstraint(A, lo, hi),
        bounds=Bounds(lb, ub),
        integrality=integ,
        options={"mip_rel_gap": 0.0, "time_limit": 60.0},
    )
    return res, tau, T


def milp_value(x, units, r, tau=1.0, cyclic=True, ens_lim=None, lolh_lim=None, objective="ens"):
    """Return objective value or None if infeasible. Raises on non-terminal status."""
    res, tau, T = _milp_model(x, units, r, tau, cyclic, ens_lim, lolh_lim, objective)
    if res.status == 2:
        return None
    if res.status != 0:
        raise RuntimeError(f"milp status {res.status}: {res.message}")
    if objective == "lolh":
        return float(tau * T + res.fun)
    return float(res.fun)


def milp_feasible(case, r, which):
    x, U = case["x"], case["units"]
    tau = float(case.get("tau", 1.0))
    cyc = bool(case.get("cyclic", True))
    E, H = case["ens_limit"], case["lolh_limit"]
    try:
        if which == "E":
            v = milp_value(x, U, r, tau=tau, cyclic=cyc, objective="ens")
            return v is not None and v <= E + ENS_SLACK
        if which == "H":
            v = milp_value(x, U, r, tau=tau, cyclic=cyc, objective="lolh")
            return v is not None and v <= H + ENS_SLACK
        v = milp_value(x, U, r, tau=tau, cyclic=cyc, ens_lim=E, lolh_lim=H, objective="ens")
        return v is not None and v <= E + ENS_SLACK
    except RuntimeError:
        return None


def milp_evaluate(case, tol=1e-7):
    hi = max(0.0, -min(case["x"]))

    def wrap(which):
        def pred(r):
            return milp_feasible(case, r, which)
        return _bisect(pred, hi, tol=tol)

    rE, sE = wrap("E")
    rH, sH = wrap("H")
    rJ, sJ = wrap("J")
    out = {"rE": rE, "rH": rH, "rJ": rJ, "status": {"E": sE, "H": sH, "J": sJ}}
    if None not in (rE, rH, rJ) and all(s == "optimal" for s in (sE, sH, sJ)):
        out["Gamma"] = rJ - max(rE, rH)
        out["Gamma_status"] = "optimal"
    else:
        out["Gamma"] = None
        out["Gamma_status"] = "uncertain"
    return out
