from __future__ import annotations

import os
import threading
import time
from typing import List, Tuple

from flask import Flask, Response, render_template, request, redirect, url_for

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Camera calibration helpers (checkerboard)
# ---------------------------------------------------------------------------

def capture_frame(camera) -> np.ndarray:
    """Grab an RGB frame from the camera and return grayscale."""
    frame_rgb = camera.capture_rgb()
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)


def detect_checkerboard(gray_frame: np.ndarray, pattern_size: Tuple[int, int]):
    """Detect and refine inner-corner chessboard points."""
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray_frame, pattern_size, flags=flags)
    if not found:
        return False, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_sub = cv2.cornerSubPix(
        gray_frame,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria,
    )
    return True, corners_sub


def add_view(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    corners: np.ndarray,
    pattern_size: Tuple[int, int],
    square_size: float,
):
    """Append a detected checkerboard view to calibration buffers."""
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)

    objpoints.append(objp)
    imgpoints.append(corners)


def run_calibration(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    image_size: Tuple[int, int],
):
    """Run cv2.calibrateCamera on accumulated views."""
    return cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)


class WebServer:
    """
    Flask server wrapper focused on PSP-based structured light.
    Handles:
      - /                  → main UI
      - /video             → live MJPEG feed
      - /scan              → run structured-light scan
      - /calib             → checkerboard camera calibration
      - /calib/capture     → capture one checkerboard view
      - /calib/finish      → solve camera intrinsics
      - /calib_psp_start   → PSP projector calibration (dataset + solve)
    """

    def __init__(self, camera, scan_controller):
        self.camera = camera
        self.scan_controller = scan_controller

        # Scan state
        self.scan_lock = threading.Lock()
        self.scan_status = "Idle"
        self.last_scan_dir = None

        # Checkerboard camera calibration state
        self.cb_checkerboard = (8, 6)        # inner corners (cols, rows)
        self.cb_square_size = 0.020          # 20 mm squares
        self.cb_objpoints: list[np.ndarray] = []
        self.cb_imgpoints: list[np.ndarray] = []
        self.cb_image_size = None
        self.cb_views = 0
        self.cb_status = "Not calibrated"
        self.cb_last_rms: float | None = None
        self.cb_lock = threading.Lock()

        # PSP projector calibration state
        self.psp_status = "Idle"
        self.psp_views = 0
        self.psp_lock = threading.Lock()
        self.psp_session: str | None = None
        self.psp_result: dict | None = None

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
                psp_status=self.psp_status,
                psp_views=self.psp_views,
                psp_session=self.psp_session,
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
            )

        @app.get("/proj_calib")
        def proj_calib():
            return render_template(
                "proj_calib.html",
                status=self.psp_status,
                session=self.psp_session,
                views=self.psp_views,
                proj_w=self.scan_controller.proj_w,
                proj_h=self.scan_controller.proj_h,
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

        @app.post("/calib_psp_start")
        def calib_psp_start():
            if not self._start_psp_thread():
                return "Already running. <a href='/'>Back</a>"
            return "PSP calibration started. <a href='/'>Back</a>"

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
    # PSP PROJECTOR CALIBRATION
    # ============================================================

    def _start_psp_thread(self):
        if not self.psp_lock.acquire(blocking=False):
            return False

        t = threading.Thread(target=self._run_psp_worker, daemon=True)
        t.start()
        return True

    def _run_psp_worker(self):
        try:
            self.psp_status = "Running"
            session = self.scan_controller.start_psp_session()
            self.psp_session = session
            self.psp_views = 0

            # Capture a fixed number of calibration poses automatically
            for i in range(10):
                self.psp_status = f"Capturing pose {i + 1}/10"
                self.scan_controller.run_calib_pose_psp()
                self.psp_views += 1

            self.psp_status = "Solving intrinsics"
            out = self.scan_controller.solve_psp_calibration(session)

            rms = out.get("rms", None)
            if rms is not None:
                self.psp_status = f"Complete (RMS={rms:.3f})"
            else:
                self.psp_status = "Complete"
            self.psp_result = out
        except Exception as exc:
            self.psp_status = f"PSP calibration failed: {exc}"
            print(f"[PSP] Calibration failed: {exc}")
        finally:
            self.psp_lock.release()

    # ============================================================
    # CHECKERBOARD CALIBRATION
    # ============================================================

    def _capture_checkerboard_view(self):
        with self.cb_lock:
            try:
                square_form = request.form.get("square_size", type=float)
                if square_form:
                    self.cb_square_size = square_form
            except Exception:
                pass

            gray = capture_frame(self.camera)

            found, corners = detect_checkerboard(gray, self.cb_checkerboard)
            if not found or corners is None:
                self.cb_status = "Checkerboard NOT found. Adjust pose/lighting."
                return False, self.cb_status

            add_view(
                self.cb_objpoints,
                self.cb_imgpoints,
                corners,
                self.cb_checkerboard,
                self.cb_square_size,
            )

            self.cb_image_size = (gray.shape[1], gray.shape[0])
            self.cb_views += 1
            self.cb_status = f"Captured {self.cb_views} views."

            print(f"[CAM-CAL] Captured view {self.cb_views}")
            return True, self.cb_status

    def _finish_camera_calibration(self):
        with self.cb_lock:
            try:
                square_form = request.form.get("square_size", type=float)
                if square_form:
                    self.cb_square_size = square_form
            except Exception:
                pass

            min_views = 8
            if self.cb_views < min_views:
                msg = f"Need at least {min_views} views. Have {self.cb_views}."
                self.cb_status = msg
                return False, msg

            if self.cb_image_size is None:
                msg = "No image size recorded; capture a view first."
                self.cb_status = msg
                return False, msg

            print(f"[CAM-CAL] Running calibration on {self.cb_views} views...")
            rms, K, dist, _, _ = run_calibration(
                self.cb_objpoints,
                self.cb_imgpoints,
                self.cb_image_size,
            )

            print("RMS:", rms)
            print("K:\n", K)
            print("dist:", dist)

            np.savez(
                "camera_intrinsics.npz",
                K=K,
                dist=dist,
                size=self.cb_image_size,
            )

            msg = f"Calibration complete. RMS={rms:.3f}. Saved camera_intrinsics.npz"
            self.cb_last_rms = float(rms)
            self.cb_status = msg
            return True, msg

    # ============================================================
    # RUN
    # ============================================================

    def run(self, host="0.0.0.0", port=5000):
        print(f"[WEB] Server at http://{host}:{port}")
        self.app.run(host=host, port=port, threaded=True)
