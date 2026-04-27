"""ROI pipeline stage."""

from __future__ import annotations

from typing import Any

import numpy as np

from fringe_app_v2.core.camera import CameraService
from fringe_app_v2.core.projector import ProjectorService
from fringe_app_v2.core.roi import detect_and_save_roi
from fringe_app_v2.pipeline.capture import capture_raw
from fringe_app_v2.utils.io import RunPaths


def run_roi_stage(
    run: RunPaths,
    camera: CameraService,
    projector: ProjectorService,
    config: dict[str, Any],
    source: np.ndarray | None = None,
) -> np.ndarray:
    if source is None:
        projector.show_level(128)
        source = capture_raw(run, camera, name="roi_reference.png", flush_frames=1)
    return detect_and_save_roi(run, source, config)
