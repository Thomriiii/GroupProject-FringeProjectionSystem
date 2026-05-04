"""Inspection workflows."""

from .known_object import (
    InspectionConfig,
    load_reference_geometry,
    run_known_object_inspection,
)
from .defect2d import (
    DEFECT_2D_MODE,
    Defect2DConfig,
    Defect2DCaptureConfig,
    analyse_image,
    capture_and_run,
    config_from_dict as defect2d_config_from_dict,
    capture_config_from_dict as defect2d_capture_config_from_dict,
    run_existing_image,
)

__all__ = [
    "DEFECT_2D_MODE",
    "Defect2DConfig",
    "Defect2DCaptureConfig",
    "InspectionConfig",
    "analyse_image",
    "capture_and_run",
    "defect2d_config_from_dict",
    "defect2d_capture_config_from_dict",
    "load_reference_geometry",
    "run_existing_image",
    "run_known_object_inspection",
]
