"""Camera base interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from fringe_app.core.models import ScanParams


class CameraBase(ABC):
    """Abstract camera interface."""

    @abstractmethod
    def start(self, params: ScanParams) -> None:
        pass

    @abstractmethod
    def capture_pair(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Capture a frame pair. Returns (main, preview) uint8 RGB arrays.
        """
        pass

    def get_applied_controls(self) -> dict:
        """Return last applied camera controls/settings."""
        return {}

    def set_manual_controls(self, exposure_us: int, analogue_gain: float, awb_enable: bool = False) -> None:
        """Optional: lock manual camera controls during runtime."""
        return None

    @abstractmethod
    def stop(self) -> None:
        pass
