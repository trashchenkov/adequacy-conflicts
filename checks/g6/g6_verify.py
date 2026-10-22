
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
"""MILP brackets for the overlap profile that came closest to 1/4."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_milp import bisect_hd

x = [5.59337, -5.58394, -5.58394, -5.58394, -2.75368, -2.76316,
     4.68582, -4.52565, -5.3833, -5.58394]
E = 5.593365766503407
P = 5.58394454642286
case = {"x": x, "units": [{"E": E, "Pc": max(E, P), "Pd": P, "eta": 1.0}]}
jobs = [
    ("H5", 5, None, True),
    ("D2lin", None, 2, False),
    ("D2cyc", None, 2, True),
    ("Jlin", 5, 2, False),
    ("Jcyc", 5, 2, True),
]
for name, H, D, cyc in jobs:
    lo, hi = bisect_hd(case, H, D, duration_cyclic=cyc, tol=1e-4)
    print(name, round(lo, 5), round(hi, 5))
print("quarter", round(P / 4, 5))
