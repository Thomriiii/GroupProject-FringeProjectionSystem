"""Entrypoint for the clean v2 application."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import fringe_app_v2
from fringe_app_v2.core.calibration import CalibrationService
from fringe_app_v2.core.camera import CameraService, CameraSettings, build_scan_params
from fringe_app_v2.core.patterns import PatternService
from fringe_app_v2.core.projector import ProjectorService, ProjectorSettings
from fringe_app_v2.utils.io import load_yaml
from fringe_app_v2.web.server import create_app


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (Path(__file__).resolve().parent / "config" / "default.yaml")
    config = load_yaml(cfg_path)
    config["_config_path"] = str(cfg_path)
    return config


def build_app(config: dict[str, Any]):
    params = build_scan_params(config)
    camera = CameraService(CameraSettings.from_config(config), params)
    projector = ProjectorService(ProjectorSettings.from_config(config))
    patterns = PatternService()
    calibration = CalibrationService(config)
    camera.start_preview()
    return create_app(config, camera, projector, patterns, calibration)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clean fringe app v2 Flask UI")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    app = build_app(config)
    web = config.get("web", {}) or {}
    host = args.host or str(web.get("host", "0.0.0.0"))
    port = int(args.port or web.get("port", 5000))
    app.run(host=host, port=port, debug=bool(web.get("debug", False)), threaded=True)


if __name__ == "__main__":
    main()
