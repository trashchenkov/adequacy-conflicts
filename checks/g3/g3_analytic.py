
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
"""Algebraic checks for the exact families. No MILP. Launch: ../py g3_analytic.py"""
import numpy as np


def simulate(c, d, delta, eta, e0):
    gamma = 1.0 - delta
    e = []
    ecur = float(e0)
    for t in range(len(c)):
        ecur = gamma * ecur + eta * c[t] - d[t]
        e.append(ecur)
    return np.array(e)


def canon(delta, beta):
    gamma = 1.0 - delta
    r = beta * delta / (2.0 - delta)
    d2 = gamma - beta - r
    d3 = beta - r
    e2 = beta + r
    c = np.array([1.0, 0.0, 0.0])
    d = np.array([0.0, d2, d3])
    e = simulate(c, d, delta, 1.0, 0.0)
    ens = (1.0 - r - d2) + max(0.0, beta - r - d3)
    lol = int((1.0 - r - d2) > 1e-10) + int((beta - r - d3) > 1e-10)
    ok = (
        abs(e[-1] - 0.0) < 1e-9
        and e.min() >= -1e-9
        and e.max() <= 1 + 1e-9
        and d2 >= -1e-12
        and abs(ens - (beta + delta)) < 1e-9
        and lol == 1
        and beta <= gamma ** 2 + 1e-12
    )
    return ok, r, ens, lol, e


def crit(delta, L, D):
    gamma = 1.0 - delta
    beta = gamma ** (L + 1)
    r = gamma * (1.0 - gamma ** L) / (L + gamma ** (-L))
    d_first = r / gamma ** L
    c = np.zeros(L + 2)
    d = np.zeros(L + 2)
    c[0] = 1.0
    d[1] = d_first
    d[-1] = beta - r
    e = simulate(c, d, delta, 1.0, 0.0)
    deep_def = D - r
    ens = L * deep_def - d_first  # late is exactly covered
    ens_bar = L * D + beta - gamma
    late_covered = abs((beta - r) - d[-1]) < 1e-9
    deep_open = deep_def - d_first > 1e-9
    ok = (
        abs(e[-1]) < 1e-8
        and e.min() >= -1e-8
        and e.max() <= 1 + 1e-8
        and d_first <= gamma + 1e-10
        and r < min(1.0, beta, D - 1)
        and late_covered
        and deep_open
        and abs(ens - ens_bar) < 1e-8
    )
    ratio = r / delta
    return ok, r, ratio, ens, e[0], d_first


def pinned_cover(delta, L):
    """Late deficit 1, no discharge on the deep coast. r = 1 - gamma**(L+1)."""
    gamma = 1.0 - delta
    r = 1.0 - gamma ** (L + 1)
    T = L + 2
    c = np.zeros(T)
    d = np.zeros(T)
    c[0] = 1.0
    d[-1] = gamma ** (L + 1)
    e = simulate(c, d, delta, 1.0, 0.0)
    delivered = d[-1]
    ok = abs(e[-1]) < 1e-8 and e.max() <= 1 + 1e-8 and abs(delivered - (1.0 - r)) < 1e-10
    return ok, r, delivered


def main():
    print("== canonical construction ==")
    for delta, beta in [(0.1, 0.8), (0.1, 0.81), (0.05, 0.5), (0.2, 0.5), (0.01, 0.5)]:
        ok, r, ens, lol, e = canon(delta, beta)
        print(f"  d={delta} beta={beta} ok={ok} r={r:.8f} ens={ens:.6f} lol={lol} e={np.round(e, 6)}")
    # task example
    print("  4/95", 4 / 95, "match", abs(0.8 * 0.1 / 1.9 - 4 / 95) < 1e-15)

    print("== critical construction ==")
    worst = 0.0
    for delta, L, D in [
        (0.1, 1, 2.0),
        (0.1, 5, 2.0),
        (0.05, 10, 3.0),
        (0.02, 15, 3.0),
        (0.01, 20, 3.0),
        (0.05, 4, 2.0),
        (0.001, 50, 2.0),
        (0.001, 200, 2.0),
    ]:
        ok, r, ratio, ens, e1, d1 = crit(delta, L, D)
        worst = max(worst, ratio)
        print(f"  d={delta} L={L} ok={ok} r={r:.6f} ratio={ratio:.4f} d1={d1:.4f} ens={ens:.4f}")
    # limsup witness: lambda = delta*L -> 0
    print("  worst ratio in this list", round(worst, 4))
    for lam in (0.2, 0.05, 0.01):
        # (1-e^{-lam})/lam is the limit of the ratio as delta->0, L=lam/delta
        print(f"  limsup shape lambda={lam} (1-exp(-l))/l={(1-np.exp(-lam))/lam:.6f}")

    print("== pinned late hour, absolute reserve ==")
    for delta, L in [(0.3, 1), (0.1, 10), (0.05, 20), (0.02, 25)]:
        ok, r, delivered = pinned_cover(delta, L)
        print(f"  d={delta} L={L} ok={ok} r={r:.6f} dE={delta:.4f} r>dE={r > delta + 1e-12}")

    print("== abstract inequality counterexample ==")
    u = np.array([0.0, 1.0, 0.1])
    delta = 0.01
    gamma = 1 - delta
    prev = np.roll(u, 1)
    signed = u - gamma * prev
    rhs = np.minimum(np.maximum(signed, 0.0), delta * 1.0).sum()
    lhs = delta * u.sum()
    print(f"  lhs={lhs:.6f} rhs={rhs:.6f} lhs>rhs={lhs > rhs + 1e-15}")

    print("== fixed-norm identity spot, eta=1 ==")
    # Sigma d = c - delta * Sigma e on the canonical schedule
    delta, beta = 0.1, 0.8
    gamma = 1 - delta
    r = beta * delta / (2 - delta)
    c = np.array([1.0, 0.0, 0.0])
    d = np.array([0.0, gamma - beta - r, beta - r])
    e = simulate(c, d, delta, 1.0, 0.0)
    # e0=0 is not inside e (e is end-of-hour). Sum of the cyclic SOC states is e0+e[:-1] wait:
    # end-of-hour states are e[0], e[1], e[2]=e0=0, so sum is e.sum().
    gap = c.sum() - d.sum() - delta * e.sum()
    print(f"  balance gap {gap:.3e}")


if __name__ == "__main__":
    main()
