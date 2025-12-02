from __future__ import annotations

import os
import threading
import time

from flask import Flask, Response, render_template, request, redirect, url_for

import cv2
import numpy as np

from calibration.checkerboard import (
    capture_frame,
    detect_checkerboard,
    add_view,
    run_calibration,
)
from calibration.patterns_graycode.graycode_generator import generate_graycode_patterns
from calibration.patterns_graycode.graycode_decode import decode_graycode


class WebServer:
    """
    Flask server wrapper.
    Handles:
      - /                 → main web UI (template)
      - /video            → live MJPEG feed
      - /scan             → run structured-light scan
      - /calib            → checkerboard calibration UI
      - /calib/capture    → capture one checkerboard view
      - /calib/finish     → run full calibrateCamera()
    """

    def __init__(self, camera, scan_controller):
        self.camera = camera
        self.scan_controller = scan_controller

        # Scan state
        self.scan_lock = threading.Lock()
        self.scan_status = "Idle"
        self.last_scan_dir = None

        # Checkerboard calibration state
        self.cb_checkerboard = (8, 6)        # inner corners (cols, rows)
        self.cb_square_size = 0.020          # 20 mm squares
        self.cb_objpoints: list[np.ndarray] = []
        self.cb_imgpoints: list[np.ndarray] = []
        self.cb_image_size = None
        self.cb_views = 0
        self.cb_status = "Not calibrated"
        self.cb_last_rms: float | None = None
        self.cb_lock = threading.Lock()

        # Projector calibration state (projected checkerboard)
        self.proj_calib_lock = threading.Lock()
        self.proj_calib_status = "Idle"
        self.proj_calib_session: str | None = None
        self.proj_calib_views = 0
        self.proj_calib_cam_pts: list[np.ndarray] = []
        self.proj_calib_proj_pts: list[np.ndarray] = []
        self.proj_calib_image_size: tuple[int, int] | None = None
        self.proj_cb_pattern = (8, 6)       # inner corners
        self.proj_cb_square_px = 80         # projector pixel size per square
        self.gray_patterns_x = []
        self.gray_patterns_y = []
        self.gray_bits_x = 0
        self.gray_bits_y = 0
        # Gray-code calibration state (new workflow)
        self.graycode_session: str | None = None
        self.graycode_patterns: list[tuple[str, object]] = []
        self.graycode_bits_x = 0
        self.graycode_bits_y = 0
        self.graycode_status = "Idle"
        self.graycode_views = 0
        self.graycode_dataset: str | None = None
        # PSP calibration state
        self.psp_status = "Idle"
        self.psp_views = 0
        self.psp_lock = threading.Lock()


        self.app = Flask(
            __name__,
            template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "..", "static"),
        )
        self._setup_routes()
        self._init_proj_patterns()

    # ============================================================
    # INITIALISE PROJECTOR PATTERNS
    # ============================================================
    def _init_proj_patterns(self):
        px, py, bx, by = generate_graycode_patterns(
            self.scan_controller.proj_w,
            self.scan_controller.proj_h,
        )
        self.graycode_patterns = px + py
        self.graycode_bits_x = bx
        self.graycode_bits_y = by

    # ============================================================
    # ROUTES
    # ============================================================

    def _setup_routes(self):
        app = self.app

        # -------------------------------
        # Index — load from template
        # -------------------------------
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
                graycode_status=self.graycode_status,
                graycode_views=self.graycode_views,
            )

        # -------------------------------
        # Calibration page
        # -------------------------------
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

        # -------------------------------
        # Projector calibration page
        # -------------------------------
        @app.get("/proj_calib")
        def proj_calib():
            return render_template(
                "proj_calib.html",
                status=self.proj_calib_status,
                session=self.proj_calib_session,
                views=self.proj_calib_views,
                proj_w=self.scan_controller.proj_w,
                proj_h=self.scan_controller.proj_h,
                pattern_cols=self.proj_cb_pattern[0],
                pattern_rows=self.proj_cb_pattern[1],
                square_px=self.proj_cb_square_px,
                bits_x=self.gray_bits_x,
                bits_y=self.gray_bits_y,
            )

        # -------------------------------
        # MJPEG camera feed
        # -------------------------------
        @app.route("/video")
        def video():
            return Response(
                self._mjpeg_stream(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        # -------------------------------
        # Structured-light scan
        # -------------------------------
        @app.post("/scan")
        def scan():
            if not self._start_scan_thread():
                return "Scan already running. <a href='/'>Back</a>"
            return "Scan started. <a href='/'>Back</a>"

        # -------------------------------
        # Checkerboard: capture one frame
        # -------------------------------
        @app.post("/calib/capture")
        def calib_capture():
            ok, msg = self._capture_checkerboard_view()
            if not ok:
                print(f"[CAM-CAL] Capture failed: {msg}")
            return redirect(url_for("calib"))

        # -------------------------------
        # Checkerboard: run full calibration
        # -------------------------------
        @app.post("/calib/finish")
        def calib_finish():
            ok, msg = self._finish_camera_calibration()
            if not ok:
                print(f"[CAM-CAL] Calibration failed: {msg}")
            return redirect(url_for("calib"))

        # -------------------------------
        # Projector calibration: capture pose
        # -------------------------------
        @app.post("/proj_calib/capture")
        def proj_calib_capture():
            if not self._start_proj_calib_capture_thread():
                return redirect(url_for("proj_calib"))
            return redirect(url_for("proj_calib"))

        # -------------------------------
        # Projector calibration: solve intrinsics
        # -------------------------------
        @app.post("/proj_calib/solve")
        def proj_calib_solve():
            self._solve_proj_intrinsics()
            return redirect(url_for("proj_calib"))

        # -------------------------------
        # Gray code calibration (new workflow)
        # -------------------------------
        @app.post("/calib_graycode_start")
        def calib_graycode_start():
            self._start_graycode_session()
            return redirect(url_for("index"))

        @app.post("/calib_graycode_capture")
        def calib_graycode_capture():
            self._capture_graycode_pose()
            return redirect(url_for("index"))

        @app.post("/calib_graycode_finish")
        def calib_graycode_finish():
            self._finish_graycode_calibration()
            return redirect(url_for("index"))
        
        # -------------------------------
        # PSP Calibration (placeholder)
        # -------------------------------
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
    # PROJECTOR CALIBRATION HELPERS
    # ============================================================

    def _start_proj_calib_capture_thread(self) -> bool:
        """
        Start a background thread to capture one projector calibration view using Gray code.
        """
        if self.scan_lock.locked():
            self.proj_calib_status = "Scan running; wait until it completes."
            return False
        if not self.proj_calib_lock.acquire(blocking=False):
            self.proj_calib_status = "Projector calibration already running."
            return False

        self.proj_calib_status = "Capturing checkerboard view..."
        t = threading.Thread(target=self._run_proj_cb_capture_worker, daemon=True)
        t.start()
        return True

    def _run_proj_cb_capture_worker(self):
        try:
            # Ensure session dir
            if self.proj_calib_session is None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                self.proj_calib_session = os.path.join("calib_proj", f"session_{ts}")
                os.makedirs(self.proj_calib_session, exist_ok=True)

            # Capture Gray code sequences
            frames_x = []
            frames_y = []

            # X patterns
            for surf in self.gray_patterns_x:
                self.scan_controller.set_surface_callback(surf)
                time.sleep(self.scan_controller.pattern_settle_time)
                gray = self.scan_controller.camera.capture_gray().astype(np.float32)
                frames_x.append(gray)

            # Y patterns
            for surf in self.gray_patterns_y:
                self.scan_controller.set_surface_callback(surf)
                time.sleep(self.scan_controller.pattern_settle_time)
                gray = self.scan_controller.camera.capture_gray().astype(np.float32)
                frames_y.append(gray)

            # Decode projector coordinates
            proj_x, proj_y, mask_gray = decode_graycode(
                frames_x, frames_y,
                width=self.scan_controller.proj_w,
                height=self.scan_controller.proj_h,
                bits_x=self.gray_bits_x,
                bits_y=self.gray_bits_y,
            )

            # Detect printed checkerboard on captured image (use last frame)
            frame_gray = frames_x[0].astype(np.uint8)
            found, corners = detect_checkerboard(frame_gray, (8, 6))
            if found and corners is not None:
                cb_mask = np.zeros_like(frame_gray, dtype=np.uint8)
                cv2.fillConvexPoly(cb_mask, corners.astype(np.int32), 1)
                mask = (cb_mask.astype(bool)) & mask_gray
            else:
                mask = mask_gray

            v_idx, u_idx = np.nonzero(mask)
            if v_idx.size < 4:
                self.proj_calib_status = "Too few valid correspondences; adjust pose/lighting."
                return

            cam_pts = np.stack([u_idx.astype(np.float32), v_idx.astype(np.float32)], axis=1)
            proj_pts = np.stack([
                proj_x[v_idx, u_idx].astype(np.float32),
                proj_y[v_idx, u_idx].astype(np.float32)
            ], axis=1)

            # Optional subsample for speed
            cam_pts = cam_pts[::2]
            proj_pts = proj_pts[::2]

            self.proj_calib_cam_pts.append(cam_pts)
            self.proj_calib_proj_pts.append(proj_pts)
            self.proj_calib_views += 1
            self.proj_calib_image_size = (frame_gray.shape[1], frame_gray.shape[0])

            self.proj_calib_status = f"Captured Gray code view {self.proj_calib_views} with {cam_pts.shape[0]} points."
        except Exception as exc:
            self.proj_calib_status = f"Capture failed: {exc}"
            print(f"[PROJ-CAL] Capture failed: {exc}")
        finally:
            self.proj_calib_lock.release()

    def _save_proj_dataset(self) -> str:
        """
        Persist current projector calibration correspondences to an npz and return the path.
        """
        if self.proj_calib_session is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.proj_calib_session = os.path.join("calib_proj", f"session_{ts}")
            os.makedirs(self.proj_calib_session, exist_ok=True)

        ds_path = os.path.join(self.proj_calib_session, "calib_dataset_proj_graycode.npz")
        np.savez(
            ds_path,
            cam_points_list=np.array(self.proj_calib_cam_pts, dtype=object),
            proj_points_list=np.array(self.proj_calib_proj_pts, dtype=object),
            proj_size=(self.scan_controller.proj_w, self.scan_controller.proj_h),
            image_size_cam=self.proj_calib_image_size,
        )
        return ds_path
    
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

        # Capture 10 poses automatically
            for i in range(10):
                self.psp_status = f"Capturing pose {i+1}/10"
                pose_dir = self.scan_controller.run_calib_pose_psp()
                self.psp_views += 1

            self.psp_status = "Solving intrinsics"
            out = self.scan_controller.solve_psp_calibration(session)

            self.psp_status = f"Complete: {out}"
        finally:
            self.psp_lock.release()



    # ============================================================
    # GRAY CODE CALIBRATION (new workflow)
    # ============================================================
    def _start_graycode_session(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.graycode_session = os.path.join("calib_graycode", f"session_{ts}")
        os.makedirs(self.graycode_session, exist_ok=True)
        from calibration.patterns_graycode.graycode_generator import generate_graycode_patterns

        px, py, bx, by = generate_graycode_patterns(
            self.scan_controller.proj_w,
            self.scan_controller.proj_h,
            out_dir=os.path.join(self.graycode_session, "patterns"),
        )
        self.graycode_patterns = px + py
        self.graycode_bits_x = bx
        self.graycode_bits_y = by
        self.graycode_views = 0
        self.graycode_status = f"Session started: {self.graycode_session}"
        print(f"[GRAYCODE] Session started at {self.graycode_session}, bits X={bx}, Y={by}")

    def _capture_graycode_pose(self):
        if self.graycode_session is None:
            self._start_graycode_session()

        pose_dir = self.scan_controller.run_graycode_pose(
            patterns_gray=self.graycode_patterns,
            session_root=self.graycode_session,
        )
        self.graycode_views += 1
        self.graycode_status = f"Captured Gray code pose {self.graycode_views} at {pose_dir}"
        print(f"[GRAYCODE] Captured {pose_dir}")

    def _finish_graycode_calibration(self):
        if self.graycode_session is None:
            self.graycode_status = "No session started."
            return

        try:
            from calibration.build_graycode_dataset import build_dataset
            from calibration.calibrate_projector_graycode import calibrate as calib_gray

            ds_path = build_dataset(
                session_root=self.graycode_session,
                proj_width=self.scan_controller.proj_w,
                proj_height=self.scan_controller.proj_h,
                bits_x=self.graycode_bits_x,
                bits_y=self.graycode_bits_y,
                subsample=2,
            )
            self.graycode_dataset = ds_path
            calib_gray(dataset_path=ds_path, cam_intr_path="camera_intrinsics.npz")
            self.graycode_status = "Gray code calibration complete. Saved projector_intrinsics_graycode.npz and stereo_params_graycode.npz"
        except Exception as exc:
            self.graycode_status = f"Gray code calibration failed: {exc}"
            print(f"[GRAYCODE] Calibration failed: {exc}")

    def _solve_proj_intrinsics(self):
        """
        Solve projector intrinsics/extrinsics using captured checkerboard views.
        """
        if self.proj_calib_views < 3:
            self.proj_calib_status = f"Need at least 3 views. Have {self.proj_calib_views}."
            return
        if self.proj_calib_image_size is None:
            self.proj_calib_status = "No captures yet; capture first."
            return

        try:
            from calibration.projector_intrinsics import calibrate as calibrate_proj

            ds_path = self._save_proj_dataset()

            calibrate_proj(
                dataset_path=ds_path,
                cam_intr_path="camera_intrinsics.npz",
            )
            self.proj_calib_status = "Projector calibration complete. Saved projector_intrinsics.npz and stereo_params.npz"
        except Exception as exc:
            self.proj_calib_status = f"Projector calibration failed: {exc}"
            print(f"[PROJ-CAL] Calibration failed: {exc}")

    # ============================================================
    # CHECKERBOARD CALIBRATION
    # ============================================================

    def _capture_checkerboard_view(self):
        with self.cb_lock:
            # Update square size if provided in the form
            try:
                square_form = request.form.get("square_size", type=float)
                if square_form:
                    self.cb_square_size = square_form
            except Exception:
                pass

            # Capture camera image
            gray = capture_frame(self.camera)

            found, corners = detect_checkerboard(gray, self.cb_checkerboard)
            if not found or corners is None:
                self.cb_status = "Checkerboard NOT found. Adjust pose/lighting."
                return False, self.cb_status

            # Append calibration view
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
            # Update square size if provided
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

            print(f"[CAM-CAL] Running calibration on {self.cb_views} views…")
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
