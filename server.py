# server.py

"""
Flask web server for the structured-light scanner.

Responsibilities:
  - Serve a clean HTML UI from templates/index.html
  - Provide a /video MJPEG live camera stream
  - Provide a /scan endpoint to trigger the scan
  - Provide a /calib_capture endpoint to capture a calibration pose
  - Display scan status and last-scan directory
"""

from __future__ import annotations

import threading

from flask import Flask, Response, render_template

import cv2
import numpy as np


class WebServer:
    """
    Flask server wrapper.
    """

    def __init__(self, camera, scan_controller):
        self.camera = camera
        self.scan_controller = scan_controller

        # Single lock to prevent concurrent scan/calibration
        self.scan_lock = threading.Lock()
        self.scan_status = "Idle"
        self.last_scan_dir = None

        self.app = Flask(
            __name__,
            static_folder="static",
            template_folder="templates"
        )
        self._setup_routes()

    # =====================================================================
    # ROUTES
    # =====================================================================

    def _setup_routes(self):
        app = self.app

        @app.route("/")
        def index():
            return render_template(
                "index.html",
                status=self.scan_status,
                last_dir=self.last_scan_dir,
            )

        @app.route("/video")
        def video():
            return Response(
                self._mjpeg_stream(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.post("/scan")
        def scan():
            if not self._start_scan_thread():
                return render_template(
                    "message.html",
                    title="Busy",
                    message="A scan or calibration is already in progress.",
                    back_url="/"
                )
            return render_template(
                "message.html",
                title="Scan Started",
                message="The scan has started successfully.",
                back_url="/"
            )

        @app.post("/calib_capture")
        def calib_capture():
            if not self._start_calib_thread():
                return render_template(
                    "message.html",
                    title="Busy",
                    message="A scan or calibration is already in progress.",
                    back_url="/"
                )
            return render_template(
                "message.html",
                    title="Calibration Pose Capture Started",
                    message="Calibration pose capture has started. Move the flat board to a new pose for each capture.",
                    back_url="/"
            )

    # =====================================================================
    # MJPEG STREAMING
    # =====================================================================

    def _mjpeg_stream(self):
        while True:
            frame_rgb = self.camera.capture_rgb()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            ok, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                continue

            data = jpg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" +
                data + b"\r\n"
            )

    # =====================================================================
    # SCAN & CALIBRATION THREAD HANDLERS
    # =====================================================================

    def _start_scan_thread(self) -> bool:
        if not self.scan_lock.acquire(blocking=False):
            return False

        self.scan_status = "Scan Running"

        t = threading.Thread(target=self._run_scan_worker, daemon=True)
        t.start()
        return True

    def _run_scan_worker(self):
        try:
            scan_dir = self.scan_controller.run_scan()
            self.last_scan_dir = scan_dir
            self.scan_status = "Scan Complete"
        finally:
            self.scan_lock.release()

    def _start_calib_thread(self) -> bool:
        if not self.scan_lock.acquire(blocking=False):
            return False

        self.scan_status = "Calibration Running"

        t = threading.Thread(target=self._run_calib_worker, daemon=True)
        t.start()
        return True

    def _run_calib_worker(self):
        try:
            pose_dir = self.scan_controller.run_calib_pose()
            self.last_scan_dir = pose_dir
            self.scan_status = "Calibration Pose Complete"
        finally:
            self.scan_lock.release()

    # =====================================================================
    # RUN SERVER
    # =====================================================================

    def run(self, host="0.0.0.0", port=5000):
        print(f"[WEB] Serving at http://{host}:{port}")
        self.app.run(host=host, port=port, threaded=True)
