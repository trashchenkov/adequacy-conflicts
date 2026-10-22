
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
"""Exact check of the reduced-cost cut lemma, then a physical overlap search.

Statement checked: at common caps, if H and D are feasible, some set F
containing D with |F| >= |H| is feasible after every cost is replaced by
(a - P/4)^+. For one storage the caps are the discharge cuts.

A hit is a real counterexample to the construction used for one charge
window. It is then evaluated on the physical profile by g6_search.analyze.
Solver statuses other than optimal/infeasible are recorded, never zeroed.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def bits_table(n):
    m = 1 << n
    bits = ((np.arange(m)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int64)
    return bits


def subset_or(good):
    """good[:, f] True => every submask of f becomes True. SOS, writes in place."""
    covered = good.copy()
    n = int(np.log2(good.shape[1]))
    for i in range(n):
        bit = 1 << i
        src = np.arange(1 << n)
        low = src[(src & bit) == 0]
        high = low | bit
        covered[:, low] = covered[:, low] | covered[:, high]
    return covered


def first_hit(bad):
    flat = np.flatnonzero(bad.ravel())
    if flat.size == 0:
        return None
    c, m = divmod(int(flat[0]), bad.shape[1])
    return c, m


def scan_caps(feas_builder, costs, pi, cap_iter, label):
    """feas_builder(caps) -> (feas, rfeas) boolean arrays shaped (C, M)."""
    c_count, n = costs.shape
    bits = bits_table(n)
    m = bits.shape[0]
    pop = bits.sum(axis=1)
    rc = np.maximum(costs - pi, 0).astype(np.int64)
    checked = 0
    for caps in cap_iter:
        feas, rfeas = feas_builder(caps, costs, rc, bits)
        mp = np.where(feas, pop, -1).max(axis=1)
        good = rfeas & (pop[None, :] >= mp[:, None])
        covered = subset_or(good)
        bad = feas & ~covered
        checked += 1
        hit = first_hit(bad)
        if hit is not None:
            ci, di = hit
            return {
                "label": label,
                "costs": costs[ci].tolist(),
                "caps": [int(v) for v in caps],
                "pi": int(pi),
                "D_mask": int(di),
                "M": int(mp[ci]),
                "checked_caps": checked,
                "instances": int(c_count),
            }
    return {"label": label, "hit": None, "checked_caps": checked, "instances": int(c_count)}


def grid(n, lo, hi):
    axes = [np.arange(lo, hi + 1, dtype=np.int64) for _ in range(n)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, n)


def k2_equal(n_a, lo, hi, pi):
    costs = grid(n_a + (n_a if False else 0), lo, hi)  # placeholder, replaced below
    return costs


def make_k2(n_a, n_b, costs, pi):
    n = n_a + n_b
    emax = int(costs.max() * max(n_a, n_b))

    def caps():
        for e in range(0, emax + 1):
            for g in range(e, 2 * e + 1):
                yield (e, g)

    def build(cap, costs, rc, bits):
        e, g = cap
        s_a = costs[:, :n_a] @ bits[:, :n_a].T
        s_b = costs[:, n_a:] @ bits[:, n_a:].T
        r_a = rc[:, :n_a] @ bits[:, :n_a].T
        r_b = rc[:, n_a:] @ bits[:, n_a:].T
        feas = (s_a <= e) & (s_b <= e) & (s_a + s_b <= g)
        rfeas = (r_a <= e) & (r_b <= e) & (r_a + r_b <= g)
        return feas, rfeas

    out = scan_caps(build, costs, pi, caps(), "k2-equal n=%d+%d pi=%d" % (n_a, n_b, pi))
    out["family"] = "equal"
    return out


def make_k2_unequal(n_a, n_b, costs, pi):
    """Residual caps: alpha, beta, gamma with alpha,beta <= gamma and alpha+beta >= gamma."""
    n = n_a + n_b
    smax_a = int(costs[:, :n_a].max() * n_a) if n_a else 0
    smax_b = int(costs[:, n_a:].max() * n_b) if n_b else 0
    # Costs differ by row; use a uniform upper bound.
    smax_a = int(costs.max() * n_a)
    smax_b = int(costs.max() * n_b)

    def caps():
        for a in range(0, smax_a + 1):
            for b in range(0, smax_b + 1):
                for g in range(max(a, b), a + b + 1):
                    yield (a, b, g)

    def build(cap, costs, rc, bits):
        a, b, g = cap
        s_a = costs[:, :n_a] @ bits[:, :n_a].T
        s_b = costs[:, n_a:] @ bits[:, n_a:].T
        r_a = rc[:, :n_a] @ bits[:, :n_a].T
        r_b = rc[:, n_a:] @ bits[:, n_a:].T
        feas = (s_a <= a) & (s_b <= b) & (s_a + s_b <= g)
        rfeas = (r_a <= a) & (r_b <= b) & (r_a + r_b <= g)
        return feas, rfeas

    out = scan_caps(build, costs, pi, caps(), "k2-unequal n=%d+%d pi=%d" % (n_a, n_b, pi))
    out["family"] = "unequal"
    return out


def make_k3(lengths, costs, pi):
    n = sum(lengths)
    bounds = []
    acc = 0
    for L in lengths:
        bounds.append((acc, acc + L))
        acc += L
    emax = int(costs.max() * max(lengths))

    def caps():
        for e in range(0, emax + 1):
            # U_j <= e. Skip the redundant all-U=e corner only inside the check,
            # still include it: it must not be a hit.
            for u0 in range(0, e + 1):
                for u1 in range(0, e + 1):
                    for u2 in range(0, e + 1):
                        yield (e, u0, u1, u2)

    def build(cap, costs, rc, bits):
        e, u0, u1, u2 = cap
        ss, rs = [], []
        for a, b in bounds:
            ss.append(costs[:, a:b] @ bits[:, a:b].T)
            rs.append(rc[:, a:b] @ bits[:, a:b].T)
        s0, s1, s2 = ss
        r0, r1, r2 = rs
        feas = (
            (s0 <= e) & (s1 <= e) & (s2 <= e)
            & (s0 + s1 <= e + u1) & (s1 + s2 <= e + u2) & (s2 + s0 <= e + u0)
            & (s0 + s1 + s2 <= u0 + u1 + u2)
        )
        rfeas = (
            (r0 <= e) & (r1 <= e) & (r2 <= e)
            & (r0 + r1 <= e + u1) & (r1 + r2 <= e + u2) & (r2 + r0 <= e + u0)
            & (r0 + r1 + r2 <= u0 + u1 + u2)
        )
        return feas, rfeas

    out = scan_caps(build, costs, pi, caps(), "k3 %s pi=%d" % (lengths, pi))
    out["family"] = "k3"
    return out


def random_real(trials, seed):
    """Real costs, equal arc cap, total between E and 2E. Also a k=3 sample."""
    rng = np.random.default_rng(seed)
    hits = []
    seen = 0
    for k in (2, 3):
        for _ in range(trials):
            lengths = [int(rng.integers(1, 5)) for _ in range(k)]
            if sum(lengths) > 8:
                continue
            p = float(rng.choice([1.0, 2.0, 4.0]))
            pi = p / 4.0
            costs = rng.uniform(pi / 2, p, size=sum(lengths))
            e = float(rng.uniform(0.3 * p, 2.5 * p))
            us = rng.uniform(0.0, e, size=k)
            seen += 1
            if violates_real(costs, lengths, e, us, pi):
                hits.append({
                    "costs": [round(float(v), 6) for v in costs],
                    "lengths": lengths,
                    "E": e,
                    "U": [round(float(v), 6) for v in us],
                    "P": p,
                })
                if len(hits) >= 5:
                    return {"seen": seen, "hits": hits}
    return {"seen": seen, "hits": hits}


def violates_real(costs, lengths, e, us, pi):
    n = len(costs)
    bits = bits_table(n).astype(np.float64)
    m = bits.shape[0]
    pop = bits.sum(axis=1)
    rc = np.maximum(costs - pi, 0.0)
    ss, rs = [], []
    acc = 0
    for L in lengths:
        ss.append(costs[acc:acc + L] @ bits[:, acc:acc + L].T)
        rs.append(rc[acc:acc + L] @ bits[:, acc:acc + L].T)
        acc += L
    k = len(lengths)
    feas = np.ones(m, dtype=bool)
    rfeas = np.ones(m, dtype=bool)
    for j in range(k):
        feas &= ss[j] <= e + 1e-9
        rfeas &= rs[j] <= e + 1e-9
    feas &= sum(ss) <= float(np.sum(us)) + 1e-9
    rfeas &= sum(rs) <= float(np.sum(us)) + 1e-9
    for length in range(2, k):
        for s in range(k):
            cap = e + sum(float(us[(s + t) % k]) for t in range(1, length))
            total = sum(ss[(s + t) % k] for t in range(length))
            rtotal = sum(rs[(s + t) % k] for t in range(length))
            feas &= total <= cap + 1e-9
            rfeas &= rtotal <= cap + 1e-9
    mp = np.where(feas, pop, -1).max()
    good = rfeas & (pop >= mp)
    covered = subset_or(good[None, :])[0]
    bad = feas & ~covered
    return bool(bad.any())


def physical(seconds, seed):
    """Overlap profiles, one storage, including T = 11..16 with at most 12 deficits."""
    from g6_search import analyze, ratios
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best = 0.0
    best_rec = None
    seen = 0
    over = []

    def consider(x, e, pc, p, tag):
        nonlocal best, best_rec, seen
        nd = sum(1 for v in x if v < -1e-12)
        if nd == 0 or nd > 12:
            return False
        r_lin, r_cyc = analyze(x, e, pc, p)
        seen += 1
        for name, rho, where in ratios(r_lin, r_cyc, p):
            if rho > best:
                best = float(rho)
                best_rec = {
                    "tag": tag, "run": name, "rho": float(rho),
                    "where": where, "x": [round(float(v), 5) for v in x],
                    "E": float(e), "Pc": float(pc), "P": float(p),
                }
            if rho > 0.25 + 1e-3:
                over.append(best_rec)
                return True
        return False

    # Control: the known tight example, and an abundant two-window copy.
    x = [2, -1, -1, -2, -2, -2]
    r_lin, r_cyc = analyze(x, 2, 2, 2)
    info = ratios(r_lin, r_cyc, 2)
    if any(abs(rho - 0.25) > 1e-6 for _, rho, _ in info):
        raise RuntimeError("classic control failed %s" % (info,))
    while time.time() - t0 < seconds:
        k = int(rng.integers(2, 4))
        lens = [int(rng.integers(2, 6)) for _ in range(k)]
        if sum(lens) > 12:
            continue
        e = float(rng.uniform(1.0, 8.0))
        p = float(rng.uniform(0.5 * e, 1.2 * e))
        pc = float(rng.choice([p, e, max(e, p) * 3.0]))
        # First charge fills the tank; the others are scarce. This is the overlap.
        sur = [e]
        for _ in range(k - 1):
            sur.append(float(rng.uniform(0.0, e)))
        xs = []
        for L, s in zip(lens, sur):
            xs.append(s)
            block = rng.uniform(0.15 * p, p, size=L)
            if L >= 3:
                block[0] = min(p, block[0])
                block[-1] = p
            xs.extend(-float(v) for v in block)
        # A zero hour inside a deficit run: breaks D, does not split energy.
        if rng.random() < 0.25 and len(xs) > 6:
            xs[int(rng.integers(2, len(xs) - 1))] = 0.0
        if consider(xs, e, pc, p, "overlap"):
            break
    return {
        "seen": seen,
        "seconds": round(time.time() - t0, 2),
        "best": best,
        "rec": best_rec,
        "over": len(over),
    }


def differential(n_templates, seed):
    """Continuous magnitudes on fixed two- and three-arc shapes. Plan: DE, at most 40."""
    from scipy.optimize import differential_evolution
    from g6_search import analyze, ratios
    rng = np.random.default_rng(seed)
    shapes = [(3, 3), (4, 2), (5, 2), (4, 3), (3, 2), (5, 3), (3, 3, 2), (4, 2, 2)]
    hits = []
    best = 0.0
    best_rec = None
    evals = 0
    t0 = time.time()
    for i in range(n_templates):
        shape = shapes[i % len(shapes)]
        # variables: E, scarce-fraction per extra charge, P, each deficit
        n_def = sum(shape)
        n_scarce = len(shape) - 1

        def objective(v, shape=shape, n_scarce=n_scarce, n_def=n_def):
            nonlocal evals, best, best_rec
            evals += 1
            e = float(v[0])
            p = float(v[1])
            fr = v[2:2 + n_scarce]
            defs = np.maximum(v[2 + n_scarce:], 0.05)
            xs = []
            cursor = 0
            for j, L in enumerate(shape):
                if j == 0:
                    xs.append(e)
                else:
                    xs.append(float(fr[j - 1]) * e)
                for _ in range(L):
                    xs.append(-float(min(defs[cursor], p)))
                    cursor += 1
            if sum(1 for z in xs if z < 0) > 12:
                return 0.0
            r_lin, r_cyc = analyze(xs, e, max(e, p), p)
            rho = 0.0
            where = None
            name = "lin"
            for nm, r, w in ratios(r_lin, r_cyc, p):
                if r > rho:
                    rho, where, name = float(r), w, nm
            if rho > best:
                best = rho
                best_rec = {
                    "tag": "de", "run": name, "rho": rho, "where": where,
                    "x": [round(float(z), 5) for z in xs],
                    "E": e, "P": p, "shape": list(shape),
                }
            return -rho

        bounds = [(1.0, 8.0), (0.8, 8.0)]
        bounds += [(0.0, 1.0)] * n_scarce
        bounds += [(0.1, 8.0)] * n_def
        differential_evolution(
            objective, bounds, seed=int(rng.integers(0, 10_000)),
            maxiter=18, popsize=8, atol=0.0, tol=0.02, workers=1,
            mutation=0.6, recombination=0.7, polish=False,
        )
        if best > 0.25 + 1e-3:
            hits.append(best_rec)
            break
    return {
        "templates": n_templates,
        "evals": evals,
        "seconds": round(time.time() - t0, 2),
        "best": best,
        "rec": best_rec,
        "hits": hits,
    }


def main():
    t0 = time.time()
    reports = []
    # Integer grid. P = 4 pi, costs are multiples of pi, so this is every
    # multiple of P/4 up to P. A second grid uses P = 8 and costs 1..8.
    reports.append(make_k2(3, 3, grid(6, 1, 4), 1))
    print("done", reports[-1]["label"], reports[-1].get("hit", reports[-1].get("D_mask")), flush=True)
    reports.append(make_k2(3, 2, grid(5, 1, 4), 1))
    print("done", reports[-1]["label"], reports[-1].get("hit", reports[-1].get("D_mask")), flush=True)
    reports.append(make_k2(2, 2, grid(4, 1, 8), 2))
    print("done", reports[-1]["label"], reports[-1].get("hit", reports[-1].get("D_mask")), flush=True)
    # Unequal residual caps, smaller support: the proof's residual step.
    reports.append(make_k2_unequal(2, 2, grid(4, 1, 4), 1))
    print("done", reports[-1]["label"], reports[-1].get("hit", reports[-1].get("D_mask")), flush=True)
    reports.append(make_k3((2, 2, 2), grid(6, 1, 4), 1))
    print("done", reports[-1]["label"], reports[-1].get("hit", reports[-1].get("D_mask")), flush=True)
    real = random_real(4000, seed=7)
    print("real", real["seen"], "hits", len(real["hits"]), flush=True)
    phys = physical(55, seed=11)
    print("physical", phys["seen"], "best", round(phys["best"], 5), "over", phys["over"], flush=True)
    de = differential(8, seed=13)
    print("de", de["evals"], "best", round(de["best"], 5), "sec", de["seconds"], flush=True)
    out = {
        "seconds": round(time.time() - t0, 2),
        "scans": reports,
        "real": real,
        "physical": phys,
        "de": de,
    }
    with open("g6_abstract.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print("wrote g6_abstract.json", "seconds", out["seconds"])
    # A scan dict with D_mask is a hit. real hits and physical over are hits.
    failed = any("D_mask" in r for r in reports) or real["hits"] or phys["over"] or de["hits"]
    if failed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
