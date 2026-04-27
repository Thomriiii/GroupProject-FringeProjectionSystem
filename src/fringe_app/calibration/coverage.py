"""Stable coverage facade for calibration workflows."""

from fringe_app.calibration_v2.coverage import (
    init_coverage,
    update_coverage,
    recompute_coverage,
)

__all__ = ["init_coverage", "update_coverage", "recompute_coverage"]
