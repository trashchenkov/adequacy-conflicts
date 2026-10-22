"""Tasks G2.2–G2.4. Launch only as: ../py g2_analysis.py [phase]

Phases: check, task2, task3, task4, all
Writes g2_results.json (merged across phases).
"""
from __future__ import annotations

# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import importlib.util
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_solver as S

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "g2_results.json"


def load_ref():
    spec = importlib.util.spec_from_file_location("ref_milp", Path(__file__).resolve().parents[2] / "src/ref_milp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_out():
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {}


def save_out(obj):
    OUT.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def unit(E, Pc, Pd, eta, delta):
    return {"E": float(E), "Pc": float(Pc), "Pd": float(Pd), "eta": float(eta), "delta": float(delta)}


def ref_eval(ref, case, tol=1e-8):
    ev = {}
    for key, which in (("rE", "E"), ("rH", "H"), ("rJ", "J")):
        ev[key] = [float(v) for v in ref.bisect(case, which, tol=tol)]
    rE, rH, rJ = ev["rE"], ev["rH"], ev["rJ"]
    # r in (lo, hi], so Gamma in (rJlo - max(rEhi, rHhi), rJhi - max(rElo, rHlo)]
    ev["Gamma_lo"] = rJ[0] - max(rE[1], rH[1])
    ev["Gamma_hi"] = rJ[1] - max(rE[0], rH[0])
    return ev


def sandwich(x, units, ens, lolh, r, slack_r=1e-6):
    """Feasibility of the joint constraint just below and at/above r."""
    below = S.joint_ok(x, units, max(0.0, r - slack_r), ens, lolh)
    at = S.joint_ok(x, units, r, ens, lolh)
    above = S.joint_ok(x, units, r + slack_r, ens, lolh)
    return {"below": below, "at": at, "above": above}


# ---------- exact ex1 schedule (fractions via float cross-check + algebra) ----------

def exact_ex1():
    """Explicit cyclic schedule at r=4/95 and the energy obstruction below it."""
    r = 4 / 95
    alpha = 0.9
    c1, e0 = 1.0, 0.0
    d2 = 0.1 - r
    d3 = 0.8 - r
    e1 = alpha * e0 + c1
    e2 = alpha * e1 - d2
    e3 = alpha * e2 - d3
    ens = (1 - r - d2) + (0.8 - r - d3)
    # weighted balance residual alpha*d2 + d3 - alpha**2 * c1  (should be 0 at e0=0)
    resid = alpha * d2 + d3 - alpha ** 2 * c1
    return {
        "r": r,
        "c": [c1, 0.0, 0.0],
        "d": [0.0, d2, d3],
        "e0": e0,
        "e": [e1, e2, e3],
        "cyclic_gap": e3 - e0,
        "balance_resid": resid,
        "ENS": ens,
        "ENS_bar": 0.9,
        "LOLH": 1,
        "r_minus_formula": r - 0.8 * 0.1 / 1.9,
    }


# ---------- task 2 ----------

def predict_gamma(S, D1, D2, E, Pc, Pd, eta, delta):
    """Closed form inside the regime; otherwise None (caller uses the solver)."""
    alpha = 1.0 - delta
    if eta <= 0 or alpha <= 0:
        return None
    cap = min(Pc, E / eta)
    # Power-limited: surplus does not bind on [0, r_pred].
    if S + 1e-12 >= cap and Pd + 1e-12 >= alpha * eta * cap and E + 1e-12 >= eta * cap:
        c_star = cap
        early = alpha * eta * c_star
        late = alpha ** 2 * eta * c_star
        if D1 + 1e-12 >= early and D2 <= late + 1e-12 and D2 <= Pd + 1e-12 and D2 > 1e-8:
            # hour-2 not coverable at r=0 (strict) and hour-3 coverable
            if D1 <= early + 1e-9:
                return {"Gamma": 0.0, "regime": "hour2_exactly_coverable", "rA": 0.0, "rB": None}
            rA = D1 - early
            rB = D2 * delta / (2.0 - delta)
            # rA must remain power-feasible: required d2 = early <= Pd, already checked
            return {"Gamma": min(rA, rB), "regime": "power", "rA": rA, "rB": rB, "binding": "A" if rA < rB else "B"}
        if D1 > early + 1e-9 and D2 > late + 1e-9:
            return {"Gamma": 0.0, "regime": "neither_hour_coverable", "rA": None, "rB": None}
        return {"Gamma": 0.0, "regime": "no_conflict_predicted", "rA": None, "rB": None}

    # Surplus-limited: c(r) = S+r up through the predicted r, power/energy caps loose.
    early0 = alpha * eta * S
    late0 = alpha ** 2 * eta * S
    if D1 + 1e-12 >= early0 and D2 <= late0 + 1e-12 and 0 < D2 <= Pd + 1e-12 and Pd + 1e-12 >= early0:
        if D1 <= early0 + 1e-9:
            return {"Gamma": 0.0, "regime": "hour2_exactly_coverable", "rA": 0.0, "rB": None}
        rA = (D1 - early0) / (1.0 + alpha * eta)
        rB = D2 * delta / (1.0 + alpha + (alpha ** 2) * eta)
        r_pred = min(rA, rB)
        if S + r_pred <= cap + 1e-9 and Pd + 1e-12 >= alpha * eta * (S + r_pred):
            return {"Gamma": r_pred, "regime": "surplus", "rA": rA, "rB": rB,
                    "binding": "A" if rA < rB else "B"}
    return None


def task2(ref):
    deltas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    # (name, S, D1, D2, E, Pc, Pd, eta)
    profiles = [
        ("power_wide", 2.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0),
        ("ex1_shape", 1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0),
        ("surplus", 1.0, 1.5, 0.4, 10.0, 10.0, 10.0, 1.0),
        ("eta_0.9", 2.0, 1.0, 0.4, 2.0, 1.0, 1.0, 0.9),
        ("rA_binds", 2.0, 0.92, 0.81, 1.0, 1.0, 1.0, 1.0),
        ("deep_late", 2.0, 1.2, 0.95, 1.0, 1.0, 1.0, 1.0),
    ]
    rows = []
    for name, S0, D1, D2, E, Pc, Pd, eta in profiles:
        for delta in deltas:
            x = [S0, -D1, -D2]
            u = [unit(E, Pc, Pd, eta, delta)]
            t0 = time.perf_counter()
            ev = S.gamma_at_competition(x, u, tol=1e-8)
            pred = predict_gamma(S0, D1, D2, E, Pc, Pd, eta, delta)
            case = {
                "x": x, "units": u,
                "ens_limit": ev["ens_bar"], "lolh_limit": ev["lolh_bar"],
            }
            ref_ok = None
            if ev["ens_bar"] is not None and ev["lolh_bar"] is not None:
                ref_ok = ref_eval(ref, case, tol=1e-7)
            sand = None
            if pred is not None and ev["ens_bar"] is not None and pred["Gamma"] > 1e-12:
                sand = sandwich(x, u, ev["ens_bar"], ev["lolh_bar"], pred["Gamma"], slack_r=1e-6)
            err = None
            if pred is not None and ev["Gamma"] is not None:
                err = ev["Gamma"] - pred["Gamma"]
            row = {
                "profile": name,
                "S": S0, "D1": D1, "D2": D2, "E": E, "Pc": Pc, "Pd": Pd, "eta": eta,
                "delta": delta,
                "ens_bar": ev["ens_bar"], "lolh_bar": ev["lolh_bar"],
                "rE": ev["rE"], "rH": ev["rH"], "rJ": ev["rJ"],
                "Gamma": ev["Gamma"], "Gamma_status": ev["Gamma_status"],
                "pred": pred, "err": err, "sandwich": sand, "ref": ref_ok,
                "sec": time.perf_counter() - t0,
            }
            rows.append(row)
            g = ev["Gamma"]
            pg = None if pred is None else pred["Gamma"]
            print(f"  {name:16} δ={delta:<6} Γ={g} pred={pg} err={err} LOLH={ev['lolh_bar']} {ev['Gamma_status']}",
                  flush=True)
    # limit coefficient along power_wide: Gamma/delta vs D2/2
    coef = []
    for row in rows:
        if row["profile"] == "power_wide" and row["Gamma"] is not None and row["delta"] > 0:
            coef.append({
                "delta": row["delta"],
                "Gamma_over_delta": row["Gamma"] / row["delta"],
                "D2_over_2": 0.25,
                "formula_over_delta": 0.5 / (2.0 - row["delta"]),
            })
    return {"deltas": deltas, "rows": rows, "power_wide_coefficient": coef, "exact_ex1": exact_ex1()}


# ---------- task 3 ----------

def theory_point(delta, T, d_mid, late_frac, c_frac, E_mult=1.0):
    alpha = 1.0 - delta
    c = c_frac / alpha
    E = E_mult * c
    Pc = c
    Pd = 1.0
    L = T - 2
    d_late = late_frac * c_frac * (alpha ** L)
    d_early = max(3.0, alpha * c + 1.0)
    x = [c] + [-d_early] + [-float(d_mid)] * max(0, T - 3) + [-d_late]
    return x, E, Pc, Pd


def eval_ratio(x, E, Pc, Pd, delta, tol=1e-5):
    u = [unit(E, Pc, Pd, 1.0, delta)]
    ev = S.gamma_at_competition(list(map(float, x)), u, tol=tol)
    if ev.get("Gamma") is None or ev.get("Gamma_status") != "optimal":
        return None
    return ev["Gamma"] / float(Pd), ev


def task3(delta, seed=0, n_random=400):
    rng = random.Random(seed)
    best = None
    trials = 0
    top = []

    def consider(tag, x, E, Pc, Pd, tol=1e-5):
        nonlocal best, trials
        trials += 1
        got = eval_ratio(x, E, Pc, Pd, delta, tol=tol)
        if got is None:
            return None
        ratio, ev = got
        rec = {
            "tag": tag, "ratio": ratio, "x": [float(v) for v in x],
            "E": float(E), "Pc": float(Pc), "Pd": float(Pd), "delta": delta,
            "Gamma": ev["Gamma"], "rE": ev["rE"], "rH": ev["rH"], "rJ": ev["rJ"],
            "ens_bar": ev["ens_bar"], "lolh_bar": ev["lolh_bar"],
        }
        top.append(rec)
        top.sort(key=lambda z: -z["ratio"])
        del top[12:]
        if best is None or ratio > best["ratio"] + 1e-12:
            best = rec
            print(f"  new best δ={delta} Γ/Pd={ratio:.6g} tag={tag} T={len(x)}", flush=True)
        return rec

    # Structured family, including rotations and mid-deficit scales.
    mids = [0.0, 0.005, 0.02, 0.05, 0.1, 0.3, 1.0, 5.0]
    lates = [0.5, 0.8, 1.0, 1.2]
    fracs = [0.6, 1.0, 1.5]
    for T in (3, 4, 5, 6):
        for d_mid in mids:
            for late_frac in lates:
                for c_frac in fracs:
                    for E_mult in (1.0, 2.0):
                        x, E, Pc, Pd = theory_point(delta, T, d_mid, late_frac, c_frac, E_mult)
                        consider("family", x, E, Pc, Pd)
                        k = T // 2
                        xr = x[k:] + x[:k]
                        consider("family_rot", xr, E, Pc, Pd)
                        if abs(c_frac - 1.0) < 1e-12 and abs(late_frac - 1.0) < 1e-12 and abs(E_mult - 1.0) < 1e-12:
                            for s in range(1, T):
                                xs = x[s:] + x[:s]
                                consider("family_shift", xs, E, Pc, Pd)

    print(f"  family done, trials={trials}, best={None if best is None else best['ratio']}", flush=True)

    for i in range(n_random):
        T = rng.choice([3, 4, 5, 6])
        x = [rng.uniform(-3.0, 2.0) for _ in range(T)]
        if max(x) <= 0:
            x[rng.randrange(T)] = rng.uniform(0.2, 2.0)
        if min(x) >= 0:
            x[rng.randrange(T)] = -rng.uniform(0.2, 2.0)
        E = 10 ** rng.uniform(-1.0, 1.0)
        Pc = 10 ** rng.uniform(-1.0, 1.0)
        Pd = 1.0
        consider("random", x, E, Pc, Pd)
        if (i + 1) % 100 == 0:
            print(f"  random {i+1}/{n_random} best={best['ratio'] if best else None}", flush=True)

    # Coordinate polish of the current top pool (copy, pool is refreshed).
    seeds = list(top)
    factors = (0.8, 0.9, 1.1, 1.25)
    for rec in seeds:
        x = list(rec["x"])
        E, Pc, Pd = rec["E"], rec["Pc"], rec["Pd"]
        improved = True
        rounds = 0
        while improved and rounds < 4:
            improved = False
            rounds += 1
            base = eval_ratio(x, E, Pc, Pd, delta, tol=1e-5)
            if base is None:
                break
            base_ratio = base[0]
            coords = [("x", j) for j in range(len(x))] + [("E", 0), ("Pc", 0)]
            for kind, j in coords:
                for f in factors:
                    x2, E2, Pc2 = list(x), E, Pc
                    if kind == "x":
                        x2[j] = x[j] * f if abs(x[j]) > 1e-8 else (0.05 if f > 1 else -0.05)
                    elif kind == "E":
                        E2 = max(1e-3, E * f)
                    else:
                        Pc2 = max(1e-3, Pc * f)
                    got = consider("local", x2, E2, Pc2, Pd, tol=1e-5)
                    if got is not None and got["ratio"] > base_ratio + 1e-6:
                        x, E, Pc = x2, E2, Pc2
                        base_ratio = got["ratio"]
                        improved = True
                        break
                if improved:
                    break
    print(f"  polish done trials={trials} best={best['ratio'] if best else None}", flush=True)

    # Tight re-evaluation of the best and a zero-delta control.
    if best is not None:
        tight = eval_ratio(best["x"], best["E"], best["Pc"], best["Pd"], delta, tol=1e-8)
        zero = eval_ratio(best["x"], best["E"], best["Pc"], best["Pd"], 0.0, tol=1e-7)
        best = dict(best)
        if tight is not None:
            best["ratio_tight"] = tight[0]
            best["Gamma_tight"] = tight[1]["Gamma"]
            best["ens_bar"] = tight[1]["ens_bar"]
            best["lolh_bar"] = tight[1]["lolh_bar"]
            best["rE"] = tight[1]["rE"]
            best["rH"] = tight[1]["rH"]
            best["rJ"] = tight[1]["rJ"]
            best["status_tight"] = tight[1]["Gamma_status"]
        best["zero_delta"] = None if zero is None else {
            "Gamma": zero[1]["Gamma"], "ratio": zero[0], "status": zero[1]["Gamma_status"],
            "ens_bar": zero[1]["ens_bar"], "lolh_bar": zero[1]["lolh_bar"],
        }
        # Reference cross-check
        ref = load_ref()
        case = {
            "x": best["x"],
            "units": [unit(best["E"], best["Pc"], best["Pd"], 1.0, delta)],
            "ens_limit": best["ens_bar"],
            "lolh_limit": best["lolh_bar"],
        }
        best["ref"] = ref_eval(ref, case, tol=1e-7)
        best["milp"] = S.milp_evaluate(case, tol=1e-7)
    return {"delta": delta, "trials": trials, "best": best, "top": top}


# ---------- task 4 ----------

def profile_generic():
    # 24h, x>0 surplus. Evening peak normalized near 1.
    return [
        -0.25, -0.20, -0.15, -0.15, -0.15, -0.20,
        -0.45, -0.60, -0.40,
        0.20, 0.70, 1.00, 1.10, 1.00, 0.60, 0.20,
        -0.30, -0.80, -1.00, -1.00, -0.80, -0.50,
        -0.35, -0.30,
    ]


def profile_two_peak():
    # Midday surplus, deep early-evening spike (above Pd=1), smaller late spike.
    return [
        -0.20, -0.15, -0.10, -0.10, -0.10, -0.15,
        -0.30, -0.40, -0.25,
        0.30, 0.80, 1.20, 1.20, 0.90, 0.40, 0.10,
        -0.40, -1.50, -1.50,
        -0.30, -0.20,
        -0.70,
        -0.40, -0.25,
    ]


def repeat_days(day, n, scale_last=1.0):
    out = []
    for i in range(n):
        sc = scale_last if i == n - 1 else 1.0
        out.extend([v * sc if v < 0 else v for v in day])
    return out


def run_case(ref, name, x, units, note):
    t0 = time.perf_counter()
    # Norms via MILP (enumeration is exponential in the number of deficit hours).
    try:
        ens = S.milp_value(x, units, 0.0, objective="ens")
        lolh = S.milp_value(x, units, 0.0, objective="lolh")
    except RuntimeError as exc:
        return {"name": name, "note": note, "status": f"uncertain: {exc}", "x": x, "units": units}
    case = {"x": x, "units": units, "ens_limit": ens, "lolh_limit": lolh}
    try:
        mine = S.milp_evaluate(case, tol=1e-6)
    except RuntimeError as exc:
        mine = {"Gamma": None, "Gamma_status": f"uncertain: {exc}"}
    try:
        refv = ref_eval(ref, case, tol=1e-6)
    except RuntimeError as exc:
        refv = {"error": str(exc)}
    pd = sum(u["Pd"] for u in units)
    pd_tank = sum(u["Pd"] for u in units if u.get("delta", 0) > 0) or pd
    gamma = mine.get("Gamma")
    print(f"  {name}: Γ={gamma} ENS={ens} LOLH={lolh} sec={time.perf_counter()-t0:.1f}", flush=True)
    return {
        "name": name, "note": note, "x": x, "units": units,
        "ens_bar": ens, "lolh_bar": lolh,
        "mine": mine, "ref": refv,
        "Pd_sum": pd, "Pd_tank": pd_tank,
        "Gamma_over_Pd_sum": None if gamma is None else gamma / pd,
        "Gamma_over_Pd_tank": None if gamma is None else gamma / pd_tank,
        "sec": time.perf_counter() - t0,
    }


def task4(ref):
    day = profile_generic()
    two = profile_two_peak()
    rows = []
    for E in (2.0, 3.0, 4.0):
        for delta in (0.005, 0.01, 0.02):
            rows.append(run_case(
                ref, f"tank_generic_E{E}_d{delta}", day,
                [unit(E, 1.0, 1.0, 1.0, delta)],
                "24h generic, Pc=Pd=1, eta=1",
            ))
    for E in (2.0, 3.0, 4.0):
        for delta in (0.005, 0.01, 0.02):
            rows.append(run_case(
                ref, f"tank_twopeak_E{E}_d{delta}", two,
                [unit(E, 1.0, 1.0, 1.0, delta)],
                "24h two-peak, Pc=Pd=1, eta=1",
            ))
    # multi-day, one representative tank
    for n, label in ((2, "48h"), (3, "72h")):
        rows.append(run_case(
            ref, f"tank_generic_{label}", repeat_days(day, n, scale_last=1.1),
            [unit(3.0, 1.0, 1.0, 1.0, 0.01)],
            f"{label} repeated generic, last day deficits x1.1, E=3, δ=0.01",
        ))
    # fleet
    battery = unit(2.0, 1.0, 1.0, 0.9, 0.0)
    tank = unit(3.0, 1.0, 1.0, 1.0, 0.01)
    for label, x in (("generic", day), ("twopeak", two), ("generic72", repeat_days(day, 3))):
        rows.append(run_case(ref, f"batt_{label}", x, [battery], "battery only E=2 Pc=Pd=1 eta=0.9 δ=0"))
        rows.append(run_case(ref, f"tank_{label}", x, [tank], "tank only E=3 Pc=Pd=1 eta=1 δ=0.01"))
        rows.append(run_case(ref, f"fleet_{label}", x, [battery, tank], "battery+tank"))
    return {"generic24": day, "twopeak24": two, "rows": rows}


def phase_check(ref):
    ex1 = json.loads((ROOT / "ex1.json").read_text())
    nod = json.loads((ROOT / "ex1_nodelta.json").read_text())
    out = {"exact_ex1_schedule": exact_ex1()}
    for name, case in (("ex1", ex1), ("ex1_nodelta", nod)):
        enum = S.evaluate_case(case, tol=1e-9)
        alt = S.milp_evaluate(case, tol=1e-8)
        rv = ref_eval(ref, case, tol=1e-8)
        out[name] = {"enum": enum, "milp_alt": alt, "ref": rv}
        print(name, "enum", enum["Gamma"], "alt", alt["Gamma"], "ref", rv["Gamma_lo"], rv["Gamma_hi"], flush=True)
    # a case with rE>0 so the bisection is not trivially zero
    tight = dict(ex1)
    tight["ens_limit"] = 0.5
    tight["lolh_limit"] = 1
    out["ex1_ens_0.5"] = {
        "enum": S.evaluate_case(tight, tol=1e-7),
        "ref": ref_eval(ref, tight, tol=1e-7),
    }
    print("ex1 ens 0.5", out["ex1_ens_0.5"]["enum"], flush=True)
    return out


def canonical_max(delta, big=3.0):
    """T=6 corner: Pd=1, E=Pc=(1-delta)**(-5), late deficit 1, three deep mids.

    Closed form Gamma = alpha * (1 - alpha**4) / (1 + 4 * alpha**4), alpha=1-delta.
    """
    alpha = 1.0 - float(delta)
    E = alpha ** (-5)
    x = [E, -big, -big, -big, -big, -1.0]
    formula = alpha * (1.0 - alpha ** 4) / (1.0 + 4.0 * alpha ** 4)
    return x, E, formula


def phase_refine(ref):
    out = {}
    for delta in (0.01, 0.05):
        x, E, formula = canonical_max(delta)
        u = [unit(E, E, 1.0, 1.0, delta)]
        ev = S.gamma_at_competition(x, u, tol=1e-9)
        case = {"x": x, "units": u, "ens_limit": ev["ens_bar"], "lolh_limit": ev["lolh_bar"]}
        ens_plan, plan = S.min_ens(x, u, formula, cover=(5,), return_plan=True)
        zero_u = [unit(E, E, 1.0, 1.0, 0.0)]
        zero = S.gamma_at_competition(x, zero_u, tol=1e-8)
        out[str(delta)] = {
            "x": x, "E": E, "Pc": E, "Pd": 1.0, "eta": 1.0, "delta": delta,
            "formula": formula,
            "enum": ev,
            "ref": ref_eval(ref, case, tol=1e-8),
            "milp": S.milp_evaluate(case, tol=1e-8),
            "plan_at_formula": {"ENS": ens_plan, "unit": plan[0]},
            "same_profile_delta0": zero,
        }
        print(delta, "formula", formula, "enum", ev["Gamma"], flush=True)
    return out


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    ref = load_ref()
    data = load_out()
    if phase in ("check", "all"):
        data["check"] = phase_check(ref)
        save_out(data)
    if phase in ("task2", "all"):
        print("task2", flush=True)
        data["task2"] = task2(ref)
        save_out(data)
    if phase in ("task3", "all"):
        print("task3 δ=0.01", flush=True)
        data["task3_d001"] = task3(0.01, seed=1, n_random=300)
        save_out(data)
        print("task3 δ=0.05", flush=True)
        data["task3_d005"] = task3(0.05, seed=2, n_random=300)
        save_out(data)
    if phase in ("refine", "all"):
        print("refine", flush=True)
        data["task3_refined"] = phase_refine(ref)
        save_out(data)
    if phase in ("task4", "all"):
        print("task4", flush=True)
        data["task4"] = task4(ref)
        save_out(data)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
