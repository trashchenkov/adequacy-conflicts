
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
"""Short continuation so the two-unit sample reaches the planned 120."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g6_more import fleet

seen, best, rec, over, unexpected, dt = fleet(40, seed=10)
print("fleet3", seen, "best", round(best, 5), "sec", round(dt, 1),
      "over", len(over), "unexpected", len(unexpected))
with open("g6_fleet3.json", "w") as f:
    json.dump({
        "seen": seen, "best": best, "over": len(over),
        "unexpected": unexpected[:8], "seconds": dt,
    }, f, ensure_ascii=False, indent=2, default=str)
