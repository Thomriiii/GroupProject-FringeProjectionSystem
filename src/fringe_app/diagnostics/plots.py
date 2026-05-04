"""Plot wrappers for calibration diagnostics."""

from fringe_app.calibration_v2.reporting import (
    save_coverage_plot,
    save_reprojection_plot,
    save_residual_histogram,
)

__all__ = [
    "save_coverage_plot",
    "save_reprojection_plot",
    "save_residual_histogram",
]
