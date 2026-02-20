"""Mock camera that reads images from mock_data/."""

from __future__ import annotations

from pathlib import Path
import itertools

import numpy as np
from PIL import Image

from fringe_app.camera.base import CameraBase
from fringe_app.core.models import ScanParams


class MockCamera(CameraBase):
    """Mock camera that returns frames from disk in sequence."""

    def __init__(self, data_dir: str = "mock_data") -> None:
        self.data_dir = Path(data_dir)
        self._iter = None
        self._files: list[Path] = []

    def start(self, params: ScanParams) -> None:
        if not self.data_dir.exists():
            raise RuntimeError(f"Mock data dir not found: {self.data_dir}")
        self._files = sorted([p for p in self.data_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if not self._files:
            raise RuntimeError(f"No mock images in {self.data_dir}")
        self._iter = itertools.cycle(self._files)

    def capture_pair(self) -> tuple[np.ndarray, np.ndarray]:
        if self._iter is None:
            raise RuntimeError("MockCamera not started")
        path = next(self._iter)
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        return arr, arr

    def stop(self) -> None:
        self._iter = None
        self._files = []

    def get_applied_controls(self) -> dict:
        return {}

    def set_manual_controls(self, exposure_us: int, analogue_gain: float, awb_enable: bool = False) -> None:
        return None
