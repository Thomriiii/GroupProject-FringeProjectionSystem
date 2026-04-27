"""Stable scanning wrappers.

This package keeps orchestration boundaries clean: web/CLI call wrappers,
wrappers call the existing numerical pipeline implementation.
"""

from .capture import run_scan_command
from .phase_decode import run_phase_decode_command
from .unwrap import run_unwrap_command
from .uv_map import run_uv_map_command
from .triangulation import run_reconstruction_command
from .reconstruction_quality import (
    ReconstructionQualityParams,
    filter_points_by_confidence,
    smooth_depth_bilateral,
    sweep_reconstruction_quality,
)

__all__ = [
    "run_scan_command",
    "run_phase_decode_command",
    "run_unwrap_command",
    "run_uv_map_command",
    "run_reconstruction_command",
    "ReconstructionQualityParams",
    "filter_points_by_confidence",
    "smooth_depth_bilateral",
    "sweep_reconstruction_quality",
]
