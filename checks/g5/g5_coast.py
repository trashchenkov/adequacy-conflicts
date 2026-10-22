
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
"""Numerically audit the coast construction: ENS_new <= ENS*(r)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import g5_core as C
from g5_traj import solve_traj, best_cover


def audit(x, delta, E=1.0, Pc=1.0, Pd=1.0, eta=1.0):
    u = [{"E": E, "Pc": Pc, "Pd": Pd, "eta": eta, "delta": delta}]
    g = 1 - delta
    ens, lol, st_e, st_h = C.separate_at_0(x, u)
    if st_e != "optimal" or st_h != "optimal":
        print("status", st_e, st_h)
        return
    A = solve_traj(x, u, 0.0)
    B = best_cover(x, u, 0.0, lol)
    sm = B["sm"]
    # covers
    K = [t for t in B["cover"]]
    # e1
    e1 = B["e"][0]
    # build dreq on hours 1..T-1 indexed 0.. 
    T = len(x)
    dreq = np.zeros(T)
    for t in K:
        dreq[t] = sm[t]
    # psi backward, psi[t] = min e[t]
    psi = np.zeros(T)
    for t in range(T - 1, 0, -1):
        psi[t - 1] = (psi[t] + dreq[t]) / g
    # new alpha
    dreq2 = np.array([max(dreq[t] - delta, 0.0) if t in K else 0.0 for t in range(T)])
    # if hour flips entirely, dreq2 already 0 when dreq<=delta
    psi2 = np.zeros(T)
    for t in range(T - 1, 0, -1):
        psi2[t - 1] = (psi2[t] + dreq2[t]) / g
    # discharges of construction at r+delta, e1 unchanged
    dC = np.zeros(T)
    eC = np.zeros(T)
    eC[0] = e1
    # dump excess at hour index 1
    if T > 1:
        dC[1] = g * e1 - psi2[1]
        eC[1] = psi2[1]
        for t in range(2, T):
            eC[t] = psi2[t]
            dC[t] = dreq2[t]
    # ENS at new reserve
    s = np.asarray(x, float) + delta
    sm2 = np.maximum(-s, 0.0)
    # feasibility flags
    feas = True
    if dC[1] < -1e-8 or eC[1] < -1e-8 or eC[1] > E + 1e-8:
        feas = False
    if T > 1 and dC[1] > min(Pd, sm2[1]) + 1e-6:
        feas = False
    ens_new = float(sm2.sum() - dC.sum())
    G = B["ens"] - A["ens"]
    print(
        f"x={np.round(x,3)} d={delta} ENS*={A['ens']:.4f} ENS_B={B['ens']:.4f} G={G:.4f} "
        f"ENS_new={ens_new:.4f} ok={ens_new <= A['ens']+1e-6} feas={feas} "
        f"dC={np.round(dC,3)} psi2_1={psi2[1] if T>1 else None:.3f}"
    )


def main():
    g = 0.9
    audit([1, -3, -3, -3, -3, -3, -(g ** 6)], 0.1)
    audit([1, -1, -0.8], 0.1)
    audit([1, -4, -2, -0.4, -0.5], 0.1)
    audit([1, -5, -1, -1, -0.3], 0.05)
    # eta < 1
    audit([2, -1, -1, -0.4], 0.1, eta=0.5, Pc=2, E=1, Pd=1)
    # second solver cross-check
    x = [1, -1, -0.8]
    u = [{"E": 1, "Pc": 1, "Pd": 1, "eta": 1, "delta": 0.1}]
    v_ref, st = C.joint_ens(x, u, 0.05, 1)
    v_own, st2 = C.enumerate_joint(x, u, 0.05, 1)
    print("cross ref", v_ref, st, "own", v_own, st2)


if __name__ == "__main__":
    main()
