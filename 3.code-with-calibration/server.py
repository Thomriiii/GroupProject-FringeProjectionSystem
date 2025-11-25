"""
server.py

Flask web server for the structured-light scanner.

Responsibilities:
  - Serve a simple HTML UI
  - Provide a /video MJPEG live camera stream
  - Provide a /scan endpoint to trigger the scan
  - Display scan status and last-scan directory

This module does *not* directly interact with pygame.
Instead, it calls ScanController.run_scan() inside a worker thread.

CameraController:
    server uses camera.capture_rgb() for live MJPEG stream.

ScanController:
    server calls scan_controller.run_scan() in a background worker thread.

Threading:
    A simple lock protects scan_status and last_scan_dir.
"""

from __future__ import annotations

import threading
import time

from flask import Flask, Response, render_template_string

import cv2
import numpy as np


class WebServer:
    """
    Flask server wrapper.

    Attributes
    ----------
    app : Flask
        The Flask application object.
    camera : CameraController
        Passed externally.
    scan_controller : ScanController
        Full scan pipeline manager.
    scan_lock : threading.Lock
        Ensures only one scan runs at a time.
    scan_status : str
        "Idle", "Running", or "Complete".
    last_scan_dir : str or None
        Path to the last completed scan directory.
    """

    def __init__(self, camera, scan_controller):
        self.camera = camera
        self.scan_controller = scan_controller

        self.scan_lock = threading.Lock()
        self.scan_status = "Idle"
        self.last_scan_dir = None

        self.app = Flask(__name__)
        self._setup_routes()

    # =====================================================================
    # ROUTES
    # =====================================================================

    INDEX_HTML = """
<!doctype html>
<html>
<head>
    <title>Fringe Projection Scanner</title>
</head>
<body>
    <h1>Fringe Projection Scanner</h1>

    <h2>Live Camera</h2>
    <img src="/video" style="max-width: 640px;"><br><br>

    <h2>Status: <strong>{{ status }}</strong></h2>
    {% if last_dir %}
        <p>Last scan directory: {{ last_dir }}</p>
    {% endif %}

    <form method="post" action="/scan">
        <button type="submit">Run Scan</button>
    </form>

    <p>Projector output is updating in real time.</p>
</body>
</html>
"""

    def _setup_routes(self):
        app = self.app

        # index
        @app.route("/")
        def index():
            return render_template_string(
                self.INDEX_HTML,
                status=self.scan_status,
                last_dir=self.last_scan_dir,
            )

        # MJPEG stream
        @app.route("/video")
        def video():
            return Response(
                self._mjpeg_stream(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        # scan trigger
        @app.post("/scan")
        def scan():
            if not self._start_scan_thread():
                return "Scan already running. <a href='/'>Back</a>"
            return "Scan started. <a href='/'>Back</a>"

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
    # SCAN THREAD HANDLER
    # =====================================================================

    def _start_scan_thread(self) -> bool:
        if not self.scan_lock.acquire(blocking=False):
            return False

        self.scan_status = "Running"

        t = threading.Thread(target=self._run_scan_worker, daemon=True)
        t.start()
        return True

    def _run_scan_worker(self):
        try:
            scan_dir = self.scan_controller.run_scan()
            self.last_scan_dir = scan_dir
            self.scan_status = "Complete"
        finally:
            self.scan_lock.release()

    # =====================================================================
    # RUN SERVER
    # =====================================================================

    def run(self, host="0.0.0.0", port=5000):
        print(f"[WEB] Serving at http://{host}:{port}")
        self.app.run(host=host, port=port, threaded=True)
