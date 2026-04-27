"""HTTP routes for the v2 Flask app."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flask import Flask, Response, jsonify, render_template

from fringe_app_v2.core.calibration import CalibrationService
from fringe_app_v2.core.camera import CameraService

if TYPE_CHECKING:
    from fringe_app_v2.web.server import PipelineRunner


def register_routes(
    app: Flask,
    runner: PipelineRunner,
    camera: CameraService,
    calibration: CalibrationService,
) -> None:
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/video")
    def video():
        return Response(camera.mjpeg_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/status")
    def status():
        return jsonify({"pipeline": runner.status(), "camera": camera.status()})

    @app.get("/api/calibration/status")
    def calibration_status():
        return jsonify(calibration.status())

    @app.post("/api/capture_roi")
    def capture_roi():
        ok, state = runner.start("roi", runner.capture_roi)
        return jsonify({"ok": ok, "status": state}), (202 if ok else 409)

    @app.post("/api/scan")
    def scan():
        ok, state = runner.start("structured", runner.structured_scan)
        return jsonify({"ok": ok, "status": state}), (202 if ok else 409)

    @app.post("/api/run_full")
    def run_full():
        ok, state = runner.start("full_pipeline", runner.full_pipeline)
        return jsonify({"ok": ok, "status": state}), (202 if ok else 409)

    @app.post("/api/compute")
    def compute():
        ok, state = runner.start("compute", runner.compute_from_active_run)
        return jsonify({"ok": ok, "status": state}), (202 if ok else 409)
