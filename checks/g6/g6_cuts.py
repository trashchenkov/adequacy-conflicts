
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
"""One-storage discharge cuts between charge intervals.

A pure deficit arc j preceded by charge interval j can take total discharge
D_j. With cyclic storage, delta = 0 and a single device, (D_j) is feasible
iff every single arc is <= E, every proper consecutive block of arcs is <= E
plus the charge strictly inside the block, and the full cycle is <= sum U.
U_j = min(E, energy the preceding charge interval can inject).

This module checks that characterization against the level LP and exposes
the segmenter used by the search.
"""
import numpy as np
from scipy.optimize import linprog


TOL = 1e-8


def lp_feasible(D, U, E):
    """True/False, or raises on an unexpected HiGHS status."""
    D = [float(v) for v in D]
    U = [float(v) for v in U]
    k = len(D)
    if k == 0:
        return True
    if any(v < -TOL for v in D):
        return False
    n = 2 * k
    c = np.zeros(n)
    A_eq = np.zeros((k, n))
    b_eq = np.zeros(k)
    A_ub = np.zeros((k, n))
    b_ub = np.full(k, float(E))
    for j in range(k):
        A_eq[j, j] = 1.0
        A_eq[j, k + j] = 1.0
        A_eq[j, (j + 1) % k] = -1.0
        b_eq[j] = D[j]
        A_ub[j, j] = 1.0
        A_ub[j, k + j] = 1.0
    bounds = [(0.0, float(E))] * k + [(0.0, float(U[j])) for j in range(k)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if res.status == 0:
        return True
    if res.status == 2:
        return False
    raise RuntimeError("lp status %s" % res.status)


def cuts_feasible(D, U, E):
    D = [float(v) for v in D]
    U = [float(v) for v in U]
    k = len(D)
    if k == 0:
        return True
    if any(v < -TOL for v in D):
        return False
    if any(D[j] > E + TOL for j in range(k)):
        return False
    if sum(D) > sum(U) + TOL:
        return False
    for length in range(2, k):
        for s in range(k):
            cap = E + sum(U[(s + t) % k] for t in range(1, length))
            total = sum(D[(s + t) % k] for t in range(length))
            if total > cap + TOL:
                return False
    return True


def segment_detail(x, q, E, Pc):
    """Return (arcs, charges).

    charges[j] is the list of time indices in the charge interval immediately
    before arcs[j]. Zero-balance hours are kept in that interval (they add no
    energy and do not open a new arc).
    """
    x = [float(v) for v in x]
    T = len(x)
    y = [xi + q for xi in x]
    deficit = [y[t] < -TOL for t in range(T)]
    if not any(deficit):
        return [], []
    start = None
    for t in range(T):
        if deficit[t] and not deficit[(t - 1) % T]:
            start = t
            break
    if start is None:
        return [list(range(T))], [list()]

    s = (start - 1) % T
    while not deficit[(s - 1) % T]:
        s = (s - 1) % T
        if s == (start - 1) % T:
            break
    t = s
    arcs, charges = [], []
    seen = 0
    while seen < T:
        group = []
        while not deficit[t]:
            group.append(t)
            t = (t + 1) % T
            seen += 1
            if seen > T:
                break
        charges.append(group)
        arc = []
        while seen < T and deficit[t]:
            arc.append(t)
            t = (t + 1) % T
            seen += 1
        arcs.append(arc)
        if t == s or seen >= T:
            break
    return arcs, charges


def charge_U(x, q, groups, E, Pc):
    """Useful-energy caps, one per charge group. Each is min(E, sum min(Pc, y+))."""
    U = []
    for group in groups:
        u = 0.0
        for t in group:
            y = float(x[t]) + q
            if y > TOL:
                u += min(float(Pc), y)
        U.append(min(float(E), u))
    return U


def segment(x, q, E, Pc):
    """Return (arcs, U) at source level q.

    arcs[j] is the list of time indices in deficit arc j.
    U[j] is the useful charge available from the charge interval that
    immediately precedes arc j. A zero-balance hour breaks runs but does
    not split an energy arc and does not add charge.
    """
    arcs, charges = segment_detail(x, q, E, Pc)
    if len(arcs) == 1 and len(charges) == 1 and len(charges[0]) == 0:
        return arcs, [0.0]
    return arcs, charge_U(x, q, charges, E, Pc)


def check_random(trials, ks, seed=0):
    rng = np.random.default_rng(seed)
    bad = []
    statuses = []
    for _ in range(trials):
        k = int(rng.choice(ks))
        E = float(rng.uniform(0.2, 5.0))
        U = np.minimum(E, rng.uniform(0.0, 6.0, size=k))
        # Mix feasible-looking and arbitrary discharges.
        if rng.random() < 0.5:
            D = rng.uniform(0.0, E, size=k)
            D *= rng.uniform(0.0, 1.3)
        else:
            D = rng.uniform(0.0, E + 1.0, size=k)
        try:
            a = lp_feasible(D, U, E)
            b = cuts_feasible(D, U, E)
        except RuntimeError as exc:
            statuses.append(str(exc))
            continue
        if a != b:
            bad.append({
                "D": [round(float(v), 6) for v in D],
                "U": [round(float(v), 6) for v in U],
                "E": round(E, 6),
                "lp": a,
                "cuts": b,
            })
            if len(bad) >= 8:
                break
    return bad, statuses


if __name__ == "__main__":
    bad, statuses = check_random(4000, [1, 2, 3, 4, 5, 6], seed=1)
    print("mismatches", len(bad), "unexpected", statuses[:5])
    for row in bad[:8]:
        print(row)
    # Spot-check the segmenter on the tight one-window example.
    arcs, U = segment([2, -1, -1, -2, -2, -2], 0.0, 2.0, 2.0)
    print("classic", arcs, U)
    arcs, U = segment([4, -1, -1, 1, -3, -3], 0.0, 4.0, 10.0)
    print("two", arcs, U)
    arcs, U = segment([4, -1, -1, 1, -3, -3], 1.0, 4.0, 10.0)
    print("two_q1", arcs, U)
