"""Diagnostics wrappers separated from calibration/scanning algorithms."""

from .reports import build_solve_report
from .plots import save_coverage_plot, save_reprojection_plot, save_residual_histogram

__all__ = [
    "build_solve_report",
    "save_coverage_plot",
    "save_reprojection_plot",
    "save_residual_histogram",
]
