"""Reference MILP for storage with self-discharge: r_E, r_H, r_joint. Normalized f_t = r.
Dynamics: e_t = (1-delta_i) e_{t-1} + tau*(eta_i c_t - d_t), cyclic e_T = e_0.
"""
import json, sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


def solve(x, units, r, tau=1.0, ens_lim=None, H=None, objective="ens", cyclic=True,
         D=None, duration_cyclic=True):
    T, m = len(x), len(units)
    s = np.asarray(x, float) + r
    sp, sm = np.maximum(s, 0), np.maximum(-s, 0)
    nc = m * T
    ic = lambda i, t: i * T + t
    idd = lambda i, t: nc + i * T + t
    ie = lambda i, t: 2 * nc + i * T + t
    ie0 = lambda i: 3 * nc + i
    il = lambda t: 3 * nc + m + t
    iy = lambda t: 3 * nc + m + T + t
    n = 3 * nc + m + 2 * T
    lb, ub, integ = np.zeros(n), np.zeros(n), np.zeros(n)
    for i, u in enumerate(units):
        for t in range(T):
            ub[ic(i, t)] = u["Pc"] if sp[t] > 0 else 0
            ub[idd(i, t)] = u["Pd"] if sm[t] > 0 else 0
            ub[ie(i, t)] = u["E"]
        ub[ie0(i)] = u["E"]
    for t in range(T):
        ub[il(t)] = sm[t]
        # A structurally non-positive hour cannot carry a positive-run flag.
        ub[iy(t)] = 1.0 if sm[t] > 1e-12 else 0.0
        integ[iy(t)] = 1
    n_run = 0
    if D is not None and D < T:
        n_run = T if duration_cyclic else T - int(D)
    A = lil_matrix((m * T + m + 3 * T + 2 + n_run, n)); lo, hi = [], []; k = 0
    for i, u in enumerate(units):
        dl, eta = u.get("delta", 0.0), u["eta"]
        for t in range(T):
            A[k, ie(i, t)] = 1
            A[k, ie(i, t - 1) if t > 0 else ie0(i)] = -(1 - dl)
            A[k, ic(i, t)] = -tau * eta
            A[k, idd(i, t)] = tau
            lo.append(0); hi.append(0); k += 1
        if cyclic:
            A[k, ie(i, T - 1)] = 1; A[k, ie0(i)] = -1; lo.append(0); hi.append(0)
        else:
            A[k, ie0(i)] = 1; lo.append(0); hi.append(0)
        k += 1
    for t in range(T):
        for i in range(m): A[k, ic(i, t)] = 1
        lo.append(0); hi.append(sp[t]); k += 1
        for i in range(m): A[k, idd(i, t)] = 1
        A[k, il(t)] = 1; lo.append(sm[t]); hi.append(sm[t]); k += 1
        A[k, il(t)] = 1; A[k, iy(t)] = -sm[t]; lo.append(-np.inf); hi.append(0); k += 1
    A[k, [il(t) for t in range(T)]] = tau; lo.append(0); hi.append(ens_lim if ens_lim is not None else np.inf); k += 1
    A[k, [iy(t) for t in range(T)]] = tau; lo.append(0); hi.append(H if H is not None else np.inf); k += 1
    # Longest positive run <= D: every window of D+1 hours contains a zero.
    if D is not None and D < T:
        width = int(D) + 1
        starts = range(T) if duration_cyclic else range(T - int(D))
        for s0 in starts:
            for j in range(width):
                A[k, iy((s0 + j) % T)] = tau
            lo.append(0); hi.append(float(D) * tau); k += 1
    cost = np.zeros(n)
    if objective == "ens":
        cost[[il(t) for t in range(T)]] = tau
        if H is None and D is None:
            integ[:] = 0
    else:
        cost[[iy(t) for t in range(T)]] = tau
    res = milp(cost, constraints=LinearConstraint(A.tocsr(), lo, hi), bounds=Bounds(lb, ub),
               integrality=integ, options={"mip_rel_gap": 0, "time_limit": 120})
    if res.status == 0: return res.fun
    if res.status == 2: return None
    raise RuntimeError(res.message)


def feasible(case, r, which):
    x, U, tau = case["x"], case["units"], case.get("tau", 1.0)
    E, H = case["ens_limit"], case["lolh_limit"]
    cyc = case.get("cyclic", True)
    if which == "E":
        v = solve(x, U, r, tau, cyclic=cyc); return v is not None and v <= E + 1e-9
    if which == "H":
        v = solve(x, U, r, tau, objective="lolh", cyclic=cyc); return v is not None and v <= H + 1e-9
    v = solve(x, U, r, tau, H=H, cyclic=cyc); return v is not None and v <= E + 1e-9


def feasible_hd(case, r, H, D, duration_cyclic=True):
    """Feasibility of H and D together. None from the solver is infeasible.
    Any other non-optimal status is raised by solve()."""
    x, U, tau = case["x"], case["units"], case.get("tau", 1.0)
    cyc = case.get("cyclic", True)
    v = solve(x, U, r, tau, H=H, D=D, cyclic=cyc, duration_cyclic=duration_cyclic)
    return v is not None


def bisect_hd(case, H, D, duration_cyclic=True, tol=1e-6):
    lo, hi = 0.0, max(0.0, -min(case["x"]))
    if feasible_hd(case, 0.0, H, D, duration_cyclic):
        return (0.0, 0.0)
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if feasible_hd(case, mid, H, D, duration_cyclic):
            hi = mid
        else:
            lo = mid
    return (lo, hi)


def bisect(case, which, tol=1e-6):
    lo, hi = 0.0, max(0.0, -min(case["x"]))
    if feasible(case, 0.0, which): return (0.0, 0.0)
    while hi - lo > tol:
        mid = (lo + hi) / 2
        (hi, lo) = (mid, lo) if feasible(case, mid, which) else (hi, mid)
    return (lo, hi)


if __name__ == "__main__":
    case = json.load(open(sys.argv[1]))
    rE, rH, rJ = bisect(case, "E"), bisect(case, "H"), bisect(case, "J")
    print(json.dumps({"rE": rE, "rH": rH, "rJ": rJ, "Gamma_upper": rJ[1] - max(rE[0], rH[0])}))
