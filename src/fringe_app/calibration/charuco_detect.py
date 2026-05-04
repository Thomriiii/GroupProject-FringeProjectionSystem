"""Stable Charuco detection facade.

This module delegates to the calibrated v2 Charuco implementation.
"""

from fringe_app.calibration_v2.charuco_detect import (
    CharucoDetection,
    build_charuco_board,
    charuco_object_points,
    detect_charuco,
)

__all__ = [
    "CharucoDetection",
    "build_charuco_board",
    "charuco_object_points",
    "detect_charuco",
]
