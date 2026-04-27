"""Camera driver factory for CLI and web orchestration."""

from __future__ import annotations

from typing import Any

from fringe_app.camera.mock import MockCamera
from fringe_app.camera.picamera2_impl import Picamera2Camera


def create_camera_from_config(cfg: dict[str, Any]):
    """Create configured camera instance using existing implementations."""
    cam_cfg = (cfg.get("camera", {}) or {}) if isinstance(cfg, dict) else {}
    cam_type = str(cam_cfg.get("type", "picamera2"))
    if cam_type == "mock":
        return MockCamera(data_dir=cam_cfg.get("mock_data", "mock_data"))
    if cam_type == "picamera2":
        return Picamera2Camera(
            lores_yuv_format=str(cam_cfg.get("lores_yuv_format", "nv12")),
            lores_uv_swap=bool(cam_cfg.get("lores_uv_swap", False)),
        )
    raise RuntimeError("camera.type must be picamera2 or mock")
