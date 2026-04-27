"""Calibration package exports.

Use explicit submodule imports for projector calibration implementations to
avoid heavy optional dependencies at package import time.
"""

from .manager import CalibrationConfig, CalibrationManager

__all__ = ["CalibrationConfig", "CalibrationManager"]
