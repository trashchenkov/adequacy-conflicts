
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
"""Continuation of the two-unit MILP sample. Same menu and stopping rule."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g6_more import fleet

seen, best, rec, over, unexpected, dt = fleet(200, seed=9)
print("fleet2", seen, "best", round(best, 5), "sec", round(dt, 1),
      "over", len(over), "unexpected", len(unexpected))
print("rec", rec)
with open("g6_fleet2.json", "w") as f:
    json.dump({
        "seen": seen, "best": best, "rec": rec, "over": len(over),
        "unexpected": unexpected[:8], "seconds": dt,
    }, f, ensure_ascii=False, indent=2, default=str)
