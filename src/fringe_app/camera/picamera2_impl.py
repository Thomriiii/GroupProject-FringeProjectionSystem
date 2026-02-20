"""Picamera2 implementation."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from fringe_app.camera.base import CameraBase
from fringe_app.core.models import ScanParams


class Picamera2Camera(CameraBase):
    """Camera wrapper using Picamera2/libcamera."""

    def __init__(self, lores_yuv_format: str = "i420", lores_uv_swap: bool = False) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Picamera2 not available. Install picamera2 or use MockCamera."
            ) from exc

        self._picamera2_cls = Picamera2
        self._cam = None
        self._exposure_us: Optional[int] = None
        self._lores_yuv_format = lores_yuv_format.lower()
        self._lores_size: Optional[tuple[int, int]] = None
        self._lores_uv_swap = lores_uv_swap
        self._applied_controls: dict = {}

    def set_exposure(self, exposure_us: int | None) -> None:
        self._exposure_us = exposure_us

    def set_manual_controls(self, exposure_us: int, analogue_gain: float, awb_enable: bool = False) -> None:
        if self._cam is None:
            raise RuntimeError("Camera not started")
        controls = {
            "AeEnable": False,
            "ExposureTime": int(exposure_us),
            "AnalogueGain": float(analogue_gain),
            "AwbEnable": bool(awb_enable),
        }
        try:
            self._cam.set_controls(controls)
        except Exception as exc:
            raise RuntimeError(f"Failed to apply manual camera controls: {exc}") from exc
        self._applied_controls.update(controls)
        try:
            md = self._cam.capture_metadata()
            for k in ("ExposureTime", "AnalogueGain", "AeEnable", "AwbEnable"):
                if k in md:
                    self._applied_controls[f"actual_{k}"] = md[k]
        except Exception:
            pass

    def start(self, params: ScanParams) -> None:
        if self._cam is not None:
            return
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                self._cam = self._picamera2_cls()
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(0.4)
        if self._cam is None:
            raise RuntimeError("Picamera2 failed to initialise; camera may be busy.") from last_exc
        width, height = params.resolution
        lores_size = (max(160, width // 4), max(120, height // 4))
        self._lores_size = lores_size
        config = self._cam.create_still_configuration(
            main={"size": (width, height), "format": "RGB888"},
            lores={"size": lores_size, "format": "YUV420"},
        )
        self._cam.configure(config)
        self._cam.start()
        time.sleep(0.2)
        controls = {}
        requested_exposure = params.exposure_us if params.exposure_us is not None else self._exposure_us
        requested_gain = params.analogue_gain
        if params.ae_enable is not None:
            controls["AeEnable"] = bool(params.ae_enable)
        if requested_exposure is not None or requested_gain is not None:
            # Manual exposure/gain implies deterministic AE-off capture.
            controls["AeEnable"] = False
        if requested_exposure is not None:
            controls["ExposureTime"] = int(requested_exposure)
        if requested_gain is not None:
            controls["AnalogueGain"] = float(requested_gain)
        if params.awb_enable is not None:
            controls["AwbEnable"] = bool(params.awb_enable)
        if params.awb_mode is not None:
            controls["AwbMode"] = str(params.awb_mode)
        if params.iso is not None:
            controls["ISO"] = int(params.iso)
        if controls:
            try:
                self._cam.set_controls(controls)
            except Exception:
                pass
        self._applied_controls = dict(controls)
        try:
            md = self._cam.capture_metadata()
            for k in ("ExposureTime", "AnalogueGain", "AeEnable", "AwbEnable"):
                if k in md:
                    self._applied_controls[f"actual_{k}"] = md[k]
        except Exception:
            pass

    def capture_pair(self) -> tuple[np.ndarray, np.ndarray]:
        if self._cam is None:
            raise RuntimeError("Camera not started")
        main = self._cam.capture_array("main")
        lores_yuv = self._cam.capture_array("lores")
        lores_rgb = self._yuv420_to_rgb(
            lores_yuv,
            self._lores_yuv_format,
            self._lores_size[0] if self._lores_size else None,
            self._lores_size[1] if self._lores_size else None,
            self._lores_uv_swap,
        )
        return main.astype(np.uint8), lores_rgb.astype(np.uint8)

    @staticmethod
    def _yuv420_to_rgb(
        yuv: np.ndarray,
        yuv_format: str,
        width: Optional[int],
        height: Optional[int],
        uv_swap: bool,
    ) -> np.ndarray:
        """
        Convert YUV420 (I420) to RGB using numpy. Expects shape (H*3/2, W).
        """
        if yuv.ndim != 2:
            raise ValueError("Expected YUV420 as 2D array (H*3/2, W)")
        h2, stride = yuv.shape
        h = int(h2 * 2 / 3)
        w = width or stride
        if height is not None:
            h = min(h, height)
        y = yuv[:h, :w]
        if yuv_format == "i420":
            # I420 is planar in bytes: Y (h*w), then U (h/2*w/2), then V (h/2*w/2).
            uv_bytes = yuv[h:, :w].reshape(-1)
            plane_size = (h // 2) * (w // 2)
            if uv_bytes.size < 2 * plane_size:
                raise ValueError("I420 buffer too small for expected UV planes")
            u = uv_bytes[:plane_size].reshape((h // 2, w // 2))
            v = uv_bytes[plane_size:2 * plane_size].reshape((h // 2, w // 2))
        elif yuv_format == "nv12":
            uv = yuv[h:h + h // 2, :w].reshape((h // 2, w))
            u = uv[:, 0::2]
            v = uv[:, 1::2]
        else:
            raise ValueError(f"Unsupported lores_yuv_format: {yuv_format}")

        if uv_swap:
            u, v = v, u

        u = u.repeat(2, axis=0).repeat(2, axis=1)
        v = v.repeat(2, axis=0).repeat(2, axis=1)

        y = y.astype(np.float32)
        u = u.astype(np.float32) - 128.0
        v = v.astype(np.float32) - 128.0

        r = y + 1.402 * v
        g = y - 0.344136 * u - 0.714136 * v
        b = y + 1.772 * u

        rgb = np.stack([r, g, b], axis=2)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def stop(self) -> None:
        if self._cam is None:
            return
        try:
            self._cam.stop()
        except Exception:
            pass
        try:
            close_fn = getattr(self._cam, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass
        time.sleep(0.2)
        self._cam = None

    def get_applied_controls(self) -> dict:
        return dict(self._applied_controls)
