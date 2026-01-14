"""
calibration package

Convenience exports for checkerboard and structured-light calibration utilities.
"""

from .camera_calibration import (
    CalibrationResult,
    create_object_points,
    load_calibration,
    load_calibration_images,
    report_dataset_coverage,
    run_calibration_from_folder,
    save_calibration,
    undistort_preview,
)

__all__ = [
    "CalibrationResult",
    "create_object_points",
    "load_calibration",
    "load_calibration_images",
    "report_dataset_coverage",
    "run_calibration_from_folder",
    "save_calibration",
    "undistort_preview",
]
