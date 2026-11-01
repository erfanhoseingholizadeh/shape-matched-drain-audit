# Puts src/ and the repository root (the stage folders are packages under it) on the import path for the release tests.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
