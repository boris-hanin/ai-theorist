import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for relative in (
    "skills/dmft-derivation/scripts",
    "skills/dmft-graph/scripts",
    "skills/dmft-moe/scripts",
):
    sys.path.insert(0, str(ROOT / relative))
