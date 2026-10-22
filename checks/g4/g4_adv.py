
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
"""Adversarial search: can H/D exceed S/4 outside one charge window, or S/2 at all?"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
from g4_check import hd_rates, positive_hours, compute_B, feasible, bisect, SolverBad

rng = np.random.default_rng(11)
best = []
viol_half = []
viol_quarter = []
over_quarter_multi = []
solver_bad = 0
n_ok = 0

def consider(x, units, H, D, cyclic, tag):
    global solver_bad, n_ok
    try:
        info = hd_rates(np.asarray(x, float), units, H, D, cyclic)
    except SolverBad:
        solver_bad += 1
        return
    if info["flag"] != "ok":
        solver_bad += 1
        return
    n_ok += 1
    S = info["S"]
    if S <= 1e-9:
        return
    rec = {
        "tag": tag,
        "g_lo": info["g_lo"],
        "g_hi": info["g_hi"],
        "ratio_lo": info["g_lo"] / S,
        "S": S,
        "H": H,
        "D": D,
        "cyclic": cyclic,
        "x": list(map(float, x)),
        "units": units,
        "pos_lo": info["pos_lo"],
        "pos_hi": info["pos_hi"],
        "h": info["h"],
        "d": info["d"],
        "j": info["j"],
    }
    best.append(rec)
    if info["g_lo"] > 0.5 * S + 5e-4:
        viol_half.append(rec)
    one = info["pos_lo"] == info["pos_hi"] and len(info["pos_hi"]) == 1
    if one and info["g_lo"] > 0.25 * S + 5e-4:
        viol_quarter.append(rec)
    if (not one) and info["g_lo"] > 0.25 * S + 5e-4:
        over_quarter_multi.append(rec)

# bimodal profiles, 1 or 2 charge hours, scarce energy
for i in range(250):
    T = int(rng.integers(6, 9))
    cyclic = bool(rng.integers(0, 2))
    n_pos = 1 if rng.random() < 0.45 else 2
    x = np.zeros(T)
    pos = rng.choice(T, size=n_pos, replace=False)
    for t in range(T):
        if t in set(pos.tolist()):
            x[t] = float(rng.uniform(0.5, 2.5))
        elif rng.random() < 0.55:
            x[t] = -float(rng.uniform(0.4, 1.1))
        else:
            x[t] = -float(rng.uniform(1.2, 2.8))
    m = int(rng.integers(1, 3))
    units = []
    eta = float(rng.choice([1.0, 0.7, 0.5]))
    # scarce aggregate energy relative to one expensive hour
    for _k in range(m):
        Pd = float(rng.uniform(0.8, 2.2))
        units.append({
            "E": float(rng.uniform(0.4, 1.6)),
            "Pc": float(rng.uniform(0.4, 2.0)),
            "Pd": Pd,
            "eta": eta,
        })
    D = int(rng.integers(1, 3))
    H = D + int(rng.integers(1, 3))
    consider(x, units, H, D, cyclic, "bimodal")

# two-pocket pattern: charge, cluster, charge, cluster
for i in range(80):
    a = float(rng.uniform(0.6, 1.4))
    b = float(rng.uniform(1.2, 2.4))
    y = float(rng.uniform(0.8, 2.0))
    x = np.array([y, -a, -a, -b, y * float(rng.uniform(0.4, 1.2)), -a, -b, -b])
    E = float(rng.uniform(0.7, 2.2))
    Pd = float(rng.uniform(max(b, 0.5), b + 1.0))
    units = [{"E": E, "Pc": float(rng.uniform(0.6, 2.5)), "Pd": Pd, "eta": 1.0}]
    consider(x, units, 3, 1, True, "pockets")
    consider(x, units, 4, 2, False, "pockets")

best.sort(key=lambda r: -r["ratio_lo"])
print("ok", n_ok, "solver_bad", solver_bad)
print("viol_half", len(viol_half), "viol_quarter", len(viol_quarter), "multi_over_1/4", len(over_quarter_multi))
print("top ratios:")
for rec in best[:8]:
    print(round(rec["ratio_lo"], 4), rec["tag"], "pos", len(rec["pos_hi"]), "H", rec["H"], "D", rec["D"],
          "g", round(rec["g_lo"], 4), "S", round(rec["S"], 4))
