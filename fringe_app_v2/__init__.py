"""Clean application layer for the fringe projection system."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


__all__ = ["REPO_ROOT", "SRC_ROOT"]
