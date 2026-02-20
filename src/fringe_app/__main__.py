"""Entry point for fringe_app."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
import yaml
import sys

from fringe_app.core.logging import setup_logging
from fringe_app.core.models import ScanParams
from fringe_app.web.preview import PreviewBroadcaster
from fringe_app.cli import main as cli_main
from fringe_app.patterns.generator import FringePatternGenerator
from fringe_app.display.pygame_display import PygameProjectorDisplay
from fringe_app.io.run_store import RunStore
from fringe_app.core.controller import ScanController
from fringe_app.web.server import FringeServer


def _load_config() -> dict:
    cfg_path = Path("config/default.yaml")
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text()) or {}


def _create_camera(cfg: dict):
    cam_cfg = cfg.get("camera", {})
    cam_type = cam_cfg.get("type", "picamera2")
    if cam_type == "picamera2":
        from fringe_app.camera.picamera2_impl import Picamera2Camera
        return Picamera2Camera(
            lores_yuv_format=str(cam_cfg.get("lores_yuv_format", "nv12")),
            lores_uv_swap=bool(cam_cfg.get("lores_uv_swap", False)),
        )
    if cam_type == "mock":
        from fringe_app.camera.mock import MockCamera
        return MockCamera(data_dir=cam_cfg.get("mock_data", "mock_data"))
    raise RuntimeError("camera.type must be picamera2 or mock")

def _default_scan_params(cfg: dict) -> ScanParams:
    scan_cfg = cfg.get("scan", {})
    pat_cfg = cfg.get("patterns", {})
    calib_cam_cfg = (cfg.get("calibration", {}) or {}).get("camera", {}) or {}
    width = int(scan_cfg.get("width", 1024))
    height = int(scan_cfg.get("height", 768))
    frequencies = scan_cfg.get("frequencies")
    if not frequencies:
        frequencies = [float(scan_cfg.get("frequency", 8.0))]
    return ScanParams(
        n_steps=int(scan_cfg.get("n_steps", 4)),
        frequency=float(frequencies[0]),
        frequencies=[float(f) for f in frequencies],
        orientation=str(scan_cfg.get("orientation", "vertical")),
        brightness=float(scan_cfg.get("brightness", 1.0)),
        resolution=(width, height),
        settle_ms=int(scan_cfg.get("settle_ms", 50)),
        save_patterns=bool(scan_cfg.get("save_patterns", False)),
        preview_fps=float(scan_cfg.get("preview_fps", 10.0)),
        phase_convention=str(scan_cfg.get("phase_convention", "atan2(-S,C)")),
        # Preview stream should be usable in ambient light, so allow calibration
        # camera defaults to override dark scan defaults.
        exposure_us=calib_cam_cfg.get("exposure_us", scan_cfg.get("exposure_us")),
        analogue_gain=calib_cam_cfg.get("analogue_gain", scan_cfg.get("analogue_gain")),
        awb_mode=scan_cfg.get("awb_mode"),
        awb_enable=calib_cam_cfg.get("awb_enable", scan_cfg.get("awb_enable")),
        ae_enable=calib_cam_cfg.get("ae_enable", scan_cfg.get("ae_enable")),
        iso=scan_cfg.get("iso"),
        brightness_offset=float(pat_cfg.get("brightness_offset", scan_cfg.get("brightness_offset", 0.45))),
        contrast=float(pat_cfg.get("contrast", scan_cfg.get("contrast", 0.6))),
        min_intensity=float(pat_cfg.get("min_intensity", scan_cfg.get("min_intensity", 0.10))),
        auto_normalise=bool(scan_cfg.get("auto_normalise", True)),
        expert_mode=False,
    )
def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit(cli_main())
    cfg = _load_config()
    log = setup_logging()

    display_cfg = cfg.get("display", {})
    driver = display_cfg.get("driver")
    if driver:
        os.environ["SDL_VIDEODRIVER"] = str(driver)
    display = PygameProjectorDisplay()

    camera = _create_camera(cfg)
    run_store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
    generator = FringePatternGenerator()
    preview_broadcaster = PreviewBroadcaster()
    controller = ScanController(
        display=display,
        camera=camera,
        generator=generator,
        store=run_store,
        preview=preview_broadcaster,
        config=cfg,
    )
    controller.start_preview_loop(_default_scan_params(cfg))

    server = FringeServer(controller=controller, run_store=run_store, config=cfg)
    host = cfg.get("web", {}).get("host", "0.0.0.0")
    port = int(cfg.get("web", {}).get("port", 8000))

    log.info("Starting server on %s:%s", host, port)
    uvicorn.run(server.app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
