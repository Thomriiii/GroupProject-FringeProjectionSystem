"""Thread-safe camera service used by live preview and pipeline stages."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterator

import numpy as np
from PIL import Image

from fringe_app.camera.mock import MockCamera
from fringe_app.camera.picamera2_impl import Picamera2Camera
from fringe_app.core.models import ScanParams


@dataclass(slots=True)
class CameraSettings:
    camera_type: str
    mock_data: str
    lores_yuv_format: str
    lores_uv_swap: bool
    preview_fps: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CameraSettings":
        cfg = config.get("camera", {}) or {}
        return cls(
            camera_type=str(cfg.get("type", "picamera2")),
            mock_data=str(cfg.get("mock_data", "mock_data")),
            lores_yuv_format=str(cfg.get("lores_yuv_format", "i420")),
            lores_uv_swap=bool(cfg.get("lores_uv_swap", False)),
            preview_fps=float(cfg.get("preview_fps", 10.0)),
        )


def build_scan_params(config: dict[str, Any], orientation: str = "vertical") -> ScanParams:
    scan = config.get("scan", {}) or {}
    freqs = [float(v) for v in scan.get("frequencies", [float(scan.get("frequency", 16.0))])]
    return ScanParams(
        n_steps=int(scan.get("n_steps", 8)),
        frequency=float(freqs[0]),
        frequencies=freqs,
        orientation=orientation,  # type: ignore[arg-type]
        brightness=float(scan.get("brightness", 1.0)),
        resolution=(int(scan.get("width", 1024)), int(scan.get("height", 768))),
        settle_ms=int(scan.get("settle_ms", 180)),
        save_patterns=bool(scan.get("save_patterns", False)),
        preview_fps=float((config.get("camera", {}) or {}).get("preview_fps", scan.get("preview_fps", 10.0))),
        phase_convention="atan2(-S,C)",
        exposure_us=None if scan.get("exposure_us") is None else int(scan.get("exposure_us")),
        analogue_gain=None if scan.get("analogue_gain") is None else float(scan.get("analogue_gain")),
        awb_enable=None if scan.get("awb_enable") is None else bool(scan.get("awb_enable")),
        ae_enable=None if scan.get("ae_enable") is None else bool(scan.get("ae_enable")),
        brightness_offset=float(scan.get("brightness_offset", 0.45)),
        contrast=float(scan.get("contrast", 0.6)),
        min_intensity=float(scan.get("min_intensity", 0.10)),
        frequency_semantics=str(scan.get("frequency_semantics", "cycles_across_dimension")),
        phase_origin_rad=float(scan.get("phase_origin_rad", 0.0)),
        auto_normalise=False,
    )


class CameraService:
    """Single-owner camera access with a background preview loop."""

    def __init__(self, settings: CameraSettings, params: ScanParams) -> None:
        self.settings = settings
        self.params = params
        self._camera = self._build_camera(settings)
        self._lock = threading.RLock()
        self._started = False
        self._preview_thread: threading.Thread | None = None
        self._preview_stop = threading.Event()
        self._latest_frame: np.ndarray | None = None
        self._latest_jpeg: bytes | None = None
        self._last_error: str | None = None

    @staticmethod
    def _build_camera(settings: CameraSettings):
        if settings.camera_type == "mock":
            return MockCamera(data_dir=settings.mock_data)
        return Picamera2Camera(
            lores_yuv_format=settings.lores_yuv_format,
            lores_uv_swap=settings.lores_uv_swap,
        )

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._camera.start(self.params)
            self._started = True

    def stop(self) -> None:
        self.stop_preview()
        with self._lock:
            if self._started:
                self._camera.stop()
                self._started = False

    def set_manual_controls(self, exposure_us: int, analogue_gain: float, awb_enable: bool = False) -> None:
        with self._lock:
            self.start()
            self._camera.set_manual_controls(
                exposure_us=int(exposure_us),
                analogue_gain=float(analogue_gain),
                awb_enable=bool(awb_enable),
            )

    def capture(self, flush_frames: int = 0) -> np.ndarray:
        with self._lock:
            self.start()
            for _ in range(max(0, int(flush_frames))):
                self._camera.capture_pair()
            frame, preview = self._camera.capture_pair()
            self._set_latest(preview)
            return np.ascontiguousarray(frame).copy()

    def flush(self, frame_count: int) -> None:
        with self._lock:
            self.start()
            for _ in range(max(0, int(frame_count))):
                self._camera.capture_pair()

    def start_preview(self) -> None:
        if self._preview_thread and self._preview_thread.is_alive():
            return
        self._preview_stop.clear()
        self._preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self._preview_thread.start()

    def stop_preview(self) -> None:
        self._preview_stop.set()
        if self._preview_thread and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=2.0)

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "type": self.settings.camera_type,
                "started": self._started,
                "latest_frame": None if self._latest_frame is None else list(self._latest_frame.shape),
                "last_error": self._last_error,
                "applied_controls": self._camera.get_applied_controls(),
            }

    def mjpeg_frames(self) -> Iterator[bytes]:
        while True:
            frame = self.latest_jpeg()
            if frame is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(max(0.03, 1.0 / max(self.settings.preview_fps, 1.0)))

    def _preview_loop(self) -> None:
        while not self._preview_stop.is_set():
            try:
                self.capture(flush_frames=0)
                self._last_error = None
            except Exception as exc:
                self._last_error = str(exc)
                time.sleep(0.5)
            time.sleep(max(0.02, 1.0 / max(self.settings.preview_fps, 1.0)))

    def _set_latest(self, frame: np.ndarray) -> None:
        local = np.ascontiguousarray(frame).copy()
        self._latest_frame = local
        self._latest_jpeg = encode_jpeg(local)


def encode_jpeg(frame: np.ndarray, max_width: int = 960, quality: int = 75) -> bytes:
    image = Image.fromarray(frame)
    if image.width > max_width:
        scale = max_width / float(image.width)
        image = image.resize((max_width, max(1, int(image.height * scale))))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue()
