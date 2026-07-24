import sys
from pathlib import Path

# Make the in-tree `ffjit` package importable without installation.
FRONTEND = Path(__file__).resolve().parents[2]
if str(FRONTEND) not in sys.path:
    sys.path.insert(0, str(FRONTEND))
