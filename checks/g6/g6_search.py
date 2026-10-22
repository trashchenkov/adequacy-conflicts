import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
"""Targeted search for Gamma_HD / S > 1/4 with one storage.

Budget and stopping rule: G6_SEARCH_PLAN.md, written before this run.
Feasibility of a covered set is the cut characterization in g6_cuts.py
(checked there against the storage-level LP). Source level q raises every
balance x_t + q, so deficit hours shrink and may become new charge.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g6_cuts import segment_detail, cuts_feasible, lp_feasible

TOL = 1e-8
BIG = 10 ** 9

_BITS = {}


def bits_for(n):
    if n not in _BITS:
        if n <= 0:
            _BITS[n] = np.zeros((1, 0))
        else:
            masks = np.arange(1 << n)
            _BITS[n] = ((masks[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
    return _BITS[n]


def critical_points(x, E, Pc):
    dmax = 0.0
    for xi in x:
        if xi < 0:
            dmax = max(dmax, -xi)
    if dmax <= TOL:
        return [0.0]
    pts = {0.0, float(dmax)}
    for xi in x:
        if xi < 0:
            d = -xi
            pts.add(d)
            sat = d + Pc
            if 0.0 < sat < dmax:
                pts.add(sat)
        else:
            sat = Pc - xi
            if 0.0 < sat < dmax:
                pts.add(float(sat))
    return sorted(pts)


def linear_charge(x, hours, q_lo, q_hi, E, Pc):
    """Return (alpha, beta) with U(q) = min(E, alpha + beta q) on [q_lo, q_hi].

    The piece must not contain an individual Pc-saturation, so each hour is
    either saturated on the whole piece or linear on the whole piece.
    """
    alpha = 0.0
    beta = 0.0
    for t in hours:
        xi = float(x[t])
        if xi + q_lo >= Pc - 1e-9:
            alpha += Pc
        else:
            alpha += xi
            beta += 1.0
    return alpha, beta


def split_U(alpha, beta, q_lo, q_hi, E):
    """Split [q_lo, q_hi] so that min(E, alpha+beta q) is linear on each part."""
    if beta <= TOL or alpha >= E - 1e-9:
        return [(q_lo, q_hi, E if alpha >= E - 1e-9 else alpha, 0.0 if alpha >= E - 1e-9 else beta)]
    q_sat = (E - alpha) / beta
    if q_sat <= q_lo + 1e-12:
        return [(q_lo, q_hi, E, 0.0)]
    if q_sat >= q_hi - 1e-12:
        return [(q_lo, q_hi, alpha, beta)]
    return [(q_lo, q_sat, alpha, beta), (q_sat, q_hi, E, 0.0)]


def run_lengths(positions, T, n):
    """Linear and cyclic run length of the UNCOVERED deficit hours.

    Mask bit 1 means the deficit hour is storage-covered (not positive).
    """
    M = 1 << n
    masks = np.arange(M)
    pos = np.zeros((M, T), dtype=bool)
    for i, p in enumerate(positions):
        pos[:, p] = (masks & (1 << i)) == 0
    lin = _longest(pos)
    doubled = np.concatenate([pos, pos], axis=1)
    cyc = np.minimum(_longest(doubled), T)
    return lin.astype(np.int16), cyc.astype(np.int16)


def _longest(pos):
    M, T = pos.shape
    idx = np.arange(T)
    last = np.maximum.accumulate(np.where(~pos, idx, -1), axis=1)
    run = np.where(pos, idx - last, 0)
    return run.max(axis=1)


def _q_from_cuts(S0, cnt, cuts, n_masks):
    """cuts: list of (alpha, beta) aligned with rows of S0, cnt (n_cuts, M)."""
    q_need = np.zeros(n_masks)
    alive = np.ones(n_masks, dtype=bool)
    for i, (alpha, beta) in enumerate(cuts):
        numer = S0[i] - alpha
        denom = cnt[i] + beta
        bad = (denom <= TOL) & (numer > 1e-8)
        alive &= ~bad
        need = np.zeros(n_masks)
        pos = denom > TOL
        need[pos] = numer[pos] / denom[pos]
        q_need = np.maximum(q_need, need)
    q_need = np.where(alive, q_need, np.inf)
    return q_need


def offer(frontier, q_feas, unc, run, q_hi, n, T):
    good = np.isfinite(q_feas) & (q_feas < q_hi - 1e-12)
    if not np.any(good):
        return
    unc_g = unc[good].astype(np.int64)
    run_g = np.minimum(run[good].astype(np.int64), T)
    q_g = q_feas[good]
    flat = unc_g * (T + 1) + run_g
    order = np.argsort(q_g, kind="mergesort")
    flat_o = flat[order]
    _, first = np.unique(flat_o, return_index=True)
    q_exact = np.full((n + 1, T + 1), np.inf)
    q_exact.ravel()[flat_o[first]] = q_g[order][first]
    pref_run = np.minimum.accumulate(q_exact, axis=1)
    pref_u = np.minimum.accumulate(pref_run, axis=0)
    pref = np.full((T + 1, T + 1), np.inf)
    pref[: n + 1, :] = pref_u
    if n < T:
        pref[n + 1 :, :] = pref_u[n : n + 1, :]
    np.minimum(frontier, pref, out=frontier)


def analyze(x, E, Pc, P):
    """Return r_joint_lin, r_joint_cyc, each shape (T+1, T+1), indexed by (H, D).

    Entry is the minimal source level at which that pair is feasible.
    """
    x = [float(v) for v in x]
    T = len(x)
    E = float(E)
    Pc = float(Pc)
    P = float(P)
    r_lin = np.full((T + 1, T + 1), np.inf)
    r_cyc = np.full((T + 1, T + 1), np.inf)
    pts = critical_points(x, E, Pc)
    # Singletons at every critical level, plus open intervals between them.
    for i, q0 in enumerate(pts):
        _offer_at(x, q0, E, Pc, P, r_lin, r_cyc)
        if i + 1 == len(pts):
            break
        q1 = pts[i + 1]
        if q1 <= q0 + 1e-10:
            continue
        _offer_interval(x, q0, q1, E, Pc, P, r_lin, r_cyc)
    return r_lin, r_cyc


def _structure(x, q, E, Pc):
    arcs, charges = segment_detail(x, q, E, Pc)
    hours = [t for arc in arcs for t in arc]
    arc_id = np.array([j for j, arc in enumerate(arcs) for _ in arc], dtype=int)
    return arcs, charges, hours, arc_id


def _offer_at(x, q, E, Pc, P, r_lin, r_cyc):
    T = len(x)
    arcs, charges, hours, arc_id = _structure(x, q, E, Pc)
    n = len(hours)
    if n == 0:
        r_lin[:] = np.minimum(r_lin, q)
        r_cyc[:] = np.minimum(r_cyc, q)
        return
    if n > 14:
        return
    a = np.array([-x[t] - q for t in hours])
    bits = bits_for(n)
    M = bits.shape[0]
    ok = _energy_ok(a, arc_id, bits, x, q, charges, E, Pc, P)
    unc = (n - bits.sum(axis=1)).astype(np.int16)
    lin, cyc = run_lengths(hours, T, n)
    q_feas = np.where(ok, q, np.inf)
    offer(r_lin, q_feas, unc, lin, q + 1.0, n, T)
    offer(r_cyc, q_feas, unc, cyc, q + 1.0, n, T)


def _energy_ok(a, arc_id, bits, x, q, charges, E, Pc, P):
    k = int(arc_id.max()) + 1 if arc_id.size else 0
    M = bits.shape[0]
    if k == 0:
        return np.ones(M, dtype=bool)
    U = []
    for group in charges:
        u = 0.0
        for t in group:
            y = float(x[t]) + q
            if y > TOL:
                u += min(Pc, y)
        U.append(min(E, u))
    S = np.zeros((k, M))
    for j in range(k):
        cols = np.flatnonzero(arc_id == j)
        S[j] = bits[:, cols] @ a[cols]
    ok = np.ones(M, dtype=bool)
    for j in range(k):
        ok &= S[j] <= E + 1e-7
    ok &= S.sum(axis=0) <= sum(U) + 1e-7
    for length in range(2, k):
        for s in range(k):
            cap = E + sum(U[(s + t) % k] for t in range(1, length))
            total = np.zeros(M)
            for t in range(length):
                total += S[(s + t) % k]
            ok &= total <= cap + 1e-7
    for i, ai in enumerate(a):
        if ai > P + 1e-8:
            ok &= bits[:, i] <= 0
    return ok


def _offer_interval(x, q_lo, q_hi, E, Pc, P, r_lin, r_cyc):
    qm = 0.5 * (q_lo + q_hi)
    arcs, charges, hours, arc_id = _structure(x, qm, E, Pc)
    n = len(hours)
    T = len(x)
    if n == 0:
        r_lin[:] = np.minimum(r_lin, q_lo)
        r_cyc[:] = np.minimum(r_cyc, q_lo)
        return
    if n > 14:
        return
    # Split the interval so every charge cap is linear.
    coeffs = [linear_charge(x, g, q_lo, q_hi, E, Pc) for g in charges]
    pieces = [(q_lo, q_hi)]
    # Sequential split on each charge. Rebuild from the first charge outward.
    refined = [(q_lo, q_hi)]
    for alpha, beta in coeffs:
        nxt = []
        for lo, hi in refined:
            for lo2, hi2, _a, _b in split_U(alpha, beta, lo, hi, E):
                if hi2 > lo2 + 1e-12:
                    nxt.append((lo2, hi2))
        refined = nxt
    for lo, hi in refined:
        _offer_linear_piece(x, lo, hi, arcs, charges, hours, arc_id, E, Pc, P, r_lin, r_cyc)


def _offer_linear_piece(x, q_lo, q_hi, arcs, charges, hours, arc_id, E, Pc, P, r_lin, r_cyc):
    n = len(hours)
    T = len(x)
    k = len(arcs)
    bits = bits_for(n)
    M = bits.shape[0]
    a0 = np.array([-float(x[t]) for t in hours])
    S_arc = np.zeros((k, M))
    C_arc = np.zeros((k, M))
    for j in range(k):
        cols = np.flatnonzero(arc_id == j)
        S_arc[j] = bits[:, cols] @ a0[cols]
        C_arc[j] = bits[:, cols].sum(axis=1)
    # Power: q >= a0 - P for covered hours with a0 > P.
    q_pow = np.zeros(M)
    for i, ai in enumerate(a0):
        if ai > P:
            q_pow = np.maximum(q_pow, np.where(bits[:, i] > 0, ai - P, 0.0))
    alphas = []
    betas = []
    for group in charges:
        alpha, beta = linear_charge(x, group, q_lo, q_hi, E, Pc)
        # After the outer split, min(E, ·) is linear on this piece.
        if beta > TOL and alpha < E and alpha + beta * (0.5 * (q_lo + q_hi)) >= E - 1e-8:
            alphas.append(E)
            betas.append(0.0)
        elif alpha >= E - 1e-9:
            alphas.append(E)
            betas.append(0.0)
        else:
            alphas.append(alpha)
            betas.append(beta)
    cuts = []
    S_rows = []
    C_rows = []
    for j in range(k):
        cuts.append((E, 0.0))
        S_rows.append(S_arc[j])
        C_rows.append(C_arc[j])
    cuts.append((sum(alphas), sum(betas)))
    S_rows.append(S_arc.sum(axis=0))
    C_rows.append(C_arc.sum(axis=0))
    for length in range(2, k):
        for s in range(k):
            alpha = E + sum(alphas[(s + t) % k] for t in range(1, length))
            beta = sum(betas[(s + t) % k] for t in range(1, length))
            total_s = np.zeros(M)
            total_c = np.zeros(M)
            for t in range(length):
                total_s += S_arc[(s + t) % k]
                total_c += C_arc[(s + t) % k]
            cuts.append((alpha, beta))
            S_rows.append(total_s)
            C_rows.append(total_c)
    S0 = np.vstack(S_rows)
    cnt = np.vstack(C_rows)
    q_need = np.maximum(q_pow, _q_from_cuts(S0, cnt, cuts, M))
    q_feas = np.maximum(q_need, q_lo)
    unc = (n - bits.sum(axis=1)).astype(np.int16)
    lin, cyc = run_lengths(hours, T, n)
    offer(r_lin, q_feas, unc, lin, q_hi, n, T)
    offer(r_cyc, q_feas, unc, cyc, q_hi, n, T)


def ratios(r_lin, r_cyc, P):
    """Max of (r_HD - max(r_H, r_D)) / P over H, D and both run definitions."""
    out = []
    for name, r in (("lin", r_lin), ("cyc", r_cyc)):
        T = r.shape[0] - 1
        # r_H[h] = min over d, r_D[d] = min over h
        r_H = r.min(axis=1)
        r_D = r.min(axis=0)
        best = -1.0
        where = None
        for h in range(T + 1):
            for d in range(T + 1):
                base = max(r_H[h], r_D[d])
                gap = r[h, d] - base
                if np.isfinite(gap) and gap > best:
                    best = gap
                    where = (h, d, float(r[h, d]), float(base), float(r_H[h]), float(r_D[d]))
        rho = best / P if best >= 0 and P > 0 else 0.0
        out.append((name, rho, where))
    return out


def self_test():
    # Tight one-window example: Gamma / P = 1/4 at H=3, D=2.
    x = [2, -1, -1, -2, -2, -2]
    r_lin, r_cyc = analyze(x, 2, 2, 2)
    ok = True
    for r in (r_lin, r_cyc):
        rH = r.min(axis=1)
        rD = r.min(axis=0)
        if abs(rH[3]) > 1e-6 or abs(rD[2]) > 1e-6 or abs(r[3, 2] - 0.5) > 1e-6:
            ok = False
            print("classic mismatch", rH[3], rD[2], r[3, 2])
    # Two windows, independent full refills, same block repeated, should not exceed 1/4.
    x2 = [4, -1, -1, -2, -2, -2, 4, -1, -1, -2, -2, -2]
    r_lin, r_cyc = analyze(x2, 2, 10, 2)
    info = ratios(r_lin, r_cyc, 2)
    for name, rho, where in info:
        if rho > 0.25 + 1e-4:
            ok = False
            print("independent exceed", name, rho, where)
    print("self_test", "ok" if ok else "FAIL", "repeat_rho", [(n, round(r, 6)) for n, r, _ in info])
    return ok


def milp_compare(n_cases, seed=2):
    from ref_milp import solve
    rng = np.random.default_rng(seed)
    bad = []
    unexpected = []
    for case_i in range(n_cases):
        T = int(rng.integers(5, 9))
        n_ch = int(rng.integers(1, 4))
        E = float(rng.choice([1.0, 2.0, 3.0]))
        P = float(rng.choice([1.0, 2.0, 3.0]))
        Pc = float(rng.choice([P, 10.0]))
        x = np.zeros(T)
        charges = rng.choice(T, size=min(n_ch, T), replace=False)
        for t in range(T):
            if t in charges:
                x[t] = float(rng.choice([0.0, 1.0, E, E + 1]))
            else:
                x[t] = -float(rng.choice([1.0, 2.0, P]))
        r_lin, r_cyc = analyze(x.tolist(), E, Pc, P)
        units = [{"E": E, "Pc": Pc, "Pd": P, "eta": 1.0, "delta": 0.0}]
        # Compare a few (H, D) cells, both run definitions.
        pairs = [(1, 1), (2, 1), (min(3, T), min(2, T))]
        for cyclic_run, table in ((False, r_lin), (True, r_cyc)):
            for H, D in pairs:
                if H > T or D > T:
                    continue
                pred = float(table[H, D])
                lo, hi = _bisect_milp(x, units, H, D, cyclic_run, unexpected)
                if lo is None:
                    continue
                if pred < lo - 2e-3 or pred > hi + 2e-3:
                    bad.append({
                        "x": [round(float(v), 4) for v in x],
                        "E": E, "Pc": Pc, "P": P, "H": H, "D": D,
                        "cyc": cyclic_run, "pred": round(pred, 5),
                        "milp": [round(lo, 5), round(hi, 5)],
                    })
                    if len(bad) >= 6:
                        return bad, unexpected
    return bad, unexpected


def _bisect_milp(x, units, H, D, cyclic_run, unexpected):
    from ref_milp import solve
    hi = max(0.0, float(-np.min(x)))
    def feas(r):
        try:
            v = solve(x, units, r, H=H, D=D, duration_cyclic=cyclic_run)
        except RuntimeError as exc:
            unexpected.append(str(exc))
            return None
        return v is not None
    f0 = feas(0.0)
    if f0 is None:
        return None, None
    if f0:
        return 0.0, 0.0
    lo = 0.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        f = feas(mid)
        if f is None:
            return None, None
        if f:
            hi = mid
        else:
            lo = mid
    return lo, hi


def search(seed=3, seconds=90):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best = []
    seen = 0
    over = []

    def consider(x, E, Pc, P, tag):
        nonlocal seen
        if max(0, sum(1 for v in x if v < 0)) > 12:
            return
        r_lin, r_cyc = analyze(x, E, Pc, P)
        seen += 1
        if seen % 200 == 0:
            top = max(r["rho"] for r in best) if best else 0.0
            print("progress", seen, "best", round(top, 5), "sec", round(time.time() - t0, 1), flush=True)
        for name, rho, where in ratios(r_lin, r_cyc, P):
            rec = {
                "tag": tag, "run": name, "rho": rho, "where": where,
                "x": [round(float(v), 6) for v in x],
                "E": E, "Pc": Pc, "P": P,
            }
            best.append(rec)
            if rho > 0.25 + 1e-3:
                over.append(rec)

    # Exhaustive two-arc integer family. Deficits in {1,2,4}, P=4.
    defs = [1.0, 2.0, 4.0]
    shapes = [(3, 3), (4, 2), (5, 2), (4, 3), (3, 2), (5, 3)]
    surpluses = [0.0, 1.0, 4.0, 8.0]
    Es = [2.0, 4.0, 6.0]
    for n1, n2 in shapes:
        # Enumerate deficit tuples only when the product is small; otherwise sample.
        total = 3 ** (n1 + n2)
        grids = []
        if total <= 729:
            from itertools import product
            grids = list(product(defs, repeat=n1 + n2))
        else:
            grids = [tuple(float(v) for v in rng.choice(defs, size=n1 + n2)) for _ in range(80)]
        for E in Es:
            for s1 in surpluses:
                for s2 in (0.0, 1.0, E):
                    if time.time() - t0 > seconds:
                        return _finish(best, over, seen, t0)
                    for vals in grids:
                        if time.time() - t0 > seconds:
                            return _finish(best, over, seen, t0)
                        x = [s1] + [-v for v in vals[:n1]] + [s2] + [-v for v in vals[n1:]]
                        consider(x, E, max(E, 4.0), 4.0, "exh2")
    # Three arcs, sampled.
    while time.time() - t0 < seconds:
        lens = [int(rng.integers(2, 5)) for _ in range(int(rng.integers(2, 4)))]
        if sum(lens) > 12:
            continue
        E = float(rng.choice([2.0, 3.0, 4.0, 5.0, 8.0]))
        P = float(rng.choice([2.0, 4.0, E]))
        Pc = float(rng.choice([P, max(P, E), 20.0]))
        x = []
        for L in lens:
            x.append(float(rng.choice([0.0, 0.5, 1.0, E, E + 2])))
            for _ in range(L):
                x.append(-float(rng.uniform(0.2, P)))
        # Occasional structural zero inside the last block.
        if rng.random() < 0.3 and len(x) > 4:
            x[int(rng.integers(1, len(x)))] = 0.0
        consider(x, E, Pc, P, "rand")
    return _finish(best, over, seen, t0)


def _finish(best, over, seen, t0):
    best.sort(key=lambda r: -r["rho"])
    # Keep a diverse top: the overall best and the best in each tag/run.
    kept = best[:12]
    print("seen", seen, "seconds", round(time.time() - t0, 2), "over", len(over))
    for rec in kept[:8]:
        print(round(rec["rho"], 5), rec["tag"], rec["run"], "HDbase", rec["where"], "E", rec["E"])
    return {"seen": seen, "seconds": time.time() - t0, "over": over, "top": kept}


if __name__ == "__main__":
    t0 = time.time()
    if not self_test():
        raise SystemExit(1)
    bad, unexpected = milp_compare(10, seed=2)
    print("milp_bad", len(bad), "unexpected", unexpected[:4])
    for row in bad:
        print(row)
    if bad or unexpected:
        raise SystemExit(2)
    # Spend the remaining part of a fixed wall clock on the search.
    spent = time.time() - t0
    out = search(seed=3, seconds=max(60.0, 160.0 - spent))
    with open("g6_search.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote g6_search.json")
