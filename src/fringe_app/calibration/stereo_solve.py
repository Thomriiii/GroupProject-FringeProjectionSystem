"""Stable stereo solve facade for projector calibration workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fringe_app.calibration_v2.stereo_solve import solve_session


def solve_projector_session_v2(session_dir: str | Path) -> dict[str, Any]:
    """Run deterministic projector stereo solve on a session folder."""
    return solve_session(Path(session_dir))


__all__ = ["solve_projector_session_v2"]
