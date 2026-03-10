"""Checkerboard calibration helpers."""

from .manager import CalibrationConfig, CalibrationManager
from .projector_stereo import (
    create_session as create_projector_session,
    list_views as list_projector_views,
    capture_projector_view,
    calibrate_projector_gamma,
    delete_view as delete_projector_view,
    stereo_calibrate as stereo_calibrate_projector,
    solve_projector_session,
)

__all__ = [
    "CalibrationConfig",
    "CalibrationManager",
    "create_projector_session",
    "list_projector_views",
    "capture_projector_view",
    "calibrate_projector_gamma",
    "delete_projector_view",
    "stereo_calibrate_projector",
    "solve_projector_session",
]
