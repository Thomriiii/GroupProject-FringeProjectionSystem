from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from flask import Flask, Response, render_template, request, redirect, url_for

import cv2
import numpy as np

from calibration import run_calibration_from_folder
from calibration.camera_calibration import (
    MIN_BOARD_AREA_FRAC,
    MIN_BOARD_ASPECT,
    MIN_MEAN,
    MAX_MEAN,
    MIN_STD,
    compute_image_stats,
    detect_checkerboard,
)


class WebServer:
    """
    Flask server wrapper for structured-light scanning.
    Handles:
      - /                  → main UI
      - /video             → live MJPEG feed
      - /scan              → run structured-light scan
      - /calib             → checkerboard camera calibration
      - /calib/capture     → capture one checkerboard view
      - /calib/finish      → solve camera intrinsics
    """

    def __init__(self, camera, scan_controller):
        self.camera = camera
        self.scan_controller = scan_controller

        # Scan state
        self.scan_lock = threading.Lock()
        self.scan_status = "Idle"
        self.last_scan_dir = None

        # Checkerboard camera calibration state
        self.cb_checkerboard = (8, 6)        # inner corners (cols, rows) for 9x7 board (10 mm squares)
        self.cb_square_size = 0.010          # 10 mm squares
        self.cb_views = 0
        self.cb_status = "Not calibrated"
        self.cb_last_rms: float | None = None
        self.cb_lock = threading.Lock()
        self.cam_calib_root = Path(self.scan_controller.calib_root) / "camera"
        self.cam_calib_session = self._start_new_calib_session()

        self.app = Flask(
            __name__,
            template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "..", "static"),
        )
        self._setup_routes()

    # ============================================================
    # ROUTES
    # ============================================================

    def _setup_routes(self):
        app = self.app

        @app.route("/")
        def index():
            return render_template(
                "index.html",
                status=self.scan_status,
                last_dir=self.last_scan_dir,
                cb_cols=self.cb_checkerboard[0],
                cb_rows=self.cb_checkerboard[1],
                cb_square_size=self.cb_square_size,
                cb_views=self.cb_views,
                cb_status=self.cb_status,
                cb_session=str(self.cam_calib_session),
            )

        @app.get("/calib")
        def calib():
            return render_template(
                "calib.html",
                cb_cols=self.cb_checkerboard[0],
                cb_rows=self.cb_checkerboard[1],
                cb_square_size=self.cb_square_size,
                cb_views=self.cb_views,
                cb_status=self.cb_status,
                cb_last_rms=self.cb_last_rms,
                cb_session=str(self.cam_calib_session),
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
                return "Scan already running. <a href='/'>Back</a>"
            return "Scan started. <a href='/'>Back</a>"

        @app.post("/calib/capture")
        def calib_capture():
            ok, msg = self._capture_checkerboard_view()
            if not ok:
                print(f"[CAM-CAL] Capture failed: {msg}")
            return redirect(url_for("calib"))

        @app.post("/calib/finish")
        def calib_finish():
            ok, msg = self._finish_camera_calibration()
            if not ok:
                print(f"[CAM-CAL] Calibration failed: {msg}")
            return redirect(url_for("calib"))

    # ============================================================
    # MJPEG STREAMING
    # ============================================================

    def _mjpeg_stream(self):
        while True:
            frame_rgb = self.camera.capture_rgb()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            ok, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                jpg.tobytes() + b"\r\n"
            )

    # ============================================================
    # SCAN THREAD HANDLER
    # ============================================================

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

    # ============================================================
    # CHECKERBOARD CALIBRATION
    # ============================================================

    def _start_new_calib_session(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session = self.cam_calib_root / f"session_{timestamp}"
        session.mkdir(parents=True, exist_ok=True)
        (session / "debug_corners").mkdir(exist_ok=True)
        print(f"[CAM-CAL] New calibration session at {session}")
        return session

    def _capture_checkerboard_view(self):
        with self.cb_lock:
            try:
                square_form = request.form.get("square_size", type=float)
                if square_form:
                    self.cb_square_size = square_form
            except Exception:
                pass

            frame_rgb = self.camera.capture_rgb()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            mean, std, min_val, max_val = compute_image_stats(gray)
            if mean < MIN_MEAN:
                self.cb_status = f"Rejected frame: too dark (mean={mean:.1f})"
                return False, self.cb_status
            if mean > MAX_MEAN:
                self.cb_status = f"Rejected frame: too bright (mean={mean:.1f})"
                return False, self.cb_status
            if std < MIN_STD:
                self.cb_status = f"Rejected frame: low contrast (std={std:.1f})"
                return False, self.cb_status

            dbg_path = self.cam_calib_session / "debug_corners" / f"view_{self.cb_views:03d}.png"
            detection = detect_checkerboard(gray, self.cb_checkerboard, debug_path=dbg_path)
            if detection.corners is None:
                reason = detection.reason or "checkerboard not found"
                self.cb_status = f"Checkerboard NOT found: {reason}"
                return False, self.cb_status

            if detection.area_frac < MIN_BOARD_AREA_FRAC:
                self.cb_status = f"Rejected: board too small in frame (area={detection.area_frac:.3f})"
                return False, self.cb_status

            if detection.aspect_ratio < MIN_BOARD_ASPECT:
                self.cb_status = f"Rejected: board extremely tilted (aspect={detection.aspect_ratio:.3f})"
                return False, self.cb_status

            # Save frame to session directory
            self.cam_calib_session.mkdir(parents=True, exist_ok=True)
            fname = f"view_{self.cb_views:03d}.png"
            out_path = self.cam_calib_session / fname
            cv2.imwrite(str(out_path), frame_bgr)

            self.cb_views += 1
            self.cb_status = f"Captured {self.cb_views} views (saved {fname})."

            print(f"[CAM-CAL] Captured view {self.cb_views} at {out_path} "
                  f"(mean={mean:.1f}, std={std:.1f}, area={detection.area_frac:.3f})")
            return True, self.cb_status

    def _finish_camera_calibration(self):
        with self.cb_lock:
            try:
                square_form = request.form.get("square_size", type=float)
                if square_form:
                    self.cb_square_size = square_form
            except Exception:
                pass

            min_views = 12
            if self.cb_views < min_views:
                msg = f"Need at least {min_views} views. Have {self.cb_views}."
                self.cb_status = msg
                return False, msg

            try:
                print(f"[CAM-CAL] Running calibration on {self.cb_views} views from {self.cam_calib_session} ...")
                calib = run_calibration_from_folder(
                    image_dir=self.cam_calib_session,
                    pattern_size=self.cb_checkerboard,
                    square_size=self.cb_square_size,
                    debug_dir=self.cam_calib_session / "debug_corners",
                    min_images=min_views,
                    max_per_view_error=1.5,
                )
            except Exception as e:
                msg = f"Calibration failed: {e}"
                self.cb_status = msg
                return False, msg

            self.cb_last_rms = calib.rms
            msg = (
                f"Calibration complete. RMS={calib.rms:.3f} px "
                f"(fx={calib.K[0,0]:.1f}, fy={calib.K[1,1]:.1f}). "
                "Saved camera_intrinsics.npz and calibration_report.txt."
            )
            self.cb_status = msg
            return True, msg

    # ============================================================
    # RUN
    # ============================================================

    def run(self, host="0.0.0.0", port=5000):
        print(f"[WEB] Server at http://{host}:{port}")
        self.app.run(host=host, port=port, threaded=True)
