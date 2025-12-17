from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from flask import Flask, Response, render_template, request, redirect, url_for

import cv2
import numpy as np
import pygame
import shutil
import time as time_module

from calibration import run_calibration_from_folder
from calibration.camera_calibration import (
    MIN_BOARD_AREA_FRAC,
    MIN_BOARD_ASPECT,
    MIN_MEAN,
    MAX_MEAN,
    MIN_STD,
    compute_image_stats,
    detect_checkerboard,
    create_object_points,
)
from calibration.projector_calibration import run_from_session as run_projector_calibration
from core.graycode import decode_with_cleaning, GrayCodeSet, generate_midgray_surface
from core.triangulation import reconstruct_3d_from_scan


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
      - /proj_calib        → projector calibration UI
      - /proj_calib/capture→ capture one projector pose (GrayCode + checkerboard)
      - /proj_calib/finish → solve projector intrinsics + stereo extrinsics
    """

    def __init__(self, camera, scan_controller, set_surface_callback, graycode_set: GrayCodeSet, midgrey_surface):
        self.camera = camera
        self.scan_controller = scan_controller
        self.set_surface_callback = set_surface_callback
        self.graycode_set = graycode_set
        self.midgrey_surface = midgrey_surface

        # Scan state
        self.scan_lock = threading.Lock()
        self.scan_status = "Idle"
        self.last_scan_dir = None
        self.recon_status = "Idle"
        self.next_scan_polished: bool = False

        # Projector calibration state
        self.proj_lock = threading.Lock()
        self.proj_status = "Not calibrated"
        self.proj_views = 0
        self.proj_last_rms: float | None = None
        self.proj_session: Path | None = None
        self.proj_calib_root = Path(self.scan_controller.calib_root) / "projector"
        self.proj_calib_root.mkdir(parents=True, exist_ok=True)
        self.proj_session = self._start_new_proj_session()
        self.proj_max_error = 2.0  # drop poses with per-view RMS above this
        self.pattern_settle_time = 0.12
        self.proj_w = graycode_set.width
        self.proj_h = graycode_set.height

        self.white_surface = self._make_uniform_surface(level=1.0)
        self.graycode_midgray = generate_midgray_surface(self.proj_w, self.proj_h, level=0.5, gamma_proj=None)

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
                proj_status=self.proj_status,
                proj_views=self.proj_views,
                proj_session=str(self.proj_session),
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

        @app.get("/proj_calib")
        def proj_calib():
            sessions = sorted(self.proj_calib_root.glob("session_*"))
            return render_template(
                "proj_calib.html",
                status=self.proj_status,
                session=str(self.proj_session),
                views=self.proj_views,
                proj_w=self.proj_w,
                proj_h=self.proj_h,
                sessions=sessions,
                max_error=self.proj_max_error,
            )

        @app.route("/video")
        def video():
            return Response(
                self._mjpeg_stream(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.post("/scan")
        def scan():
            polished_flag = str(request.form.get("polished", "false")).lower() in ("1", "true", "on", "yes")
            if not self._start_scan_thread(polished=polished_flag):
                return "Scan already running. <a href='/'>Back</a>"
            return "Scan started. <a href='/'>Back</a>"

        @app.post("/scan/reconstruct")
        def scan_reconstruct():
            msg = self._run_reconstruct_latest()
            return msg

        @app.post("/scan/reconstruct_previous")
        def scan_reconstruct_previous():
            scan_path = request.form.get("scan_path", "").strip()
            if not scan_path:
                return "No scan path provided. <a href='/'>Back</a>"
            msg = self._run_reconstruct_scan(scan_path)
            return msg

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

        @app.post("/proj_calib/capture")
        def proj_calib_capture():
            ok, msg = self._capture_projector_pose()
            if not ok:
                print(f"[PROJ-CALIB] Capture failed: {msg}")
            return redirect(url_for("proj_calib"))

        @app.post("/proj_calib/finish")
        def proj_calib_finish():
            try:
                val = request.form.get("max_error", type=float)
                if val is not None:
                    self.proj_max_error = val
            except Exception:
                pass
            ok, msg = self._finish_projector_calibration()
            if not ok:
                print(f"[PROJ-CALIB] Finish failed: {msg}")
            return redirect(url_for("proj_calib"))

        @app.post("/proj_calib/select")
        def proj_calib_select():
            session_path = request.form.get("session_path", "")
            if session_path:
                self._set_proj_session(Path(session_path))
            return redirect(url_for("proj_calib"))

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

    def _start_scan_thread(self, polished: bool = False) -> bool:
        if not self.scan_lock.acquire(blocking=False):
            return False

        self.next_scan_polished = polished
        self.scan_status = "Running"
        t = threading.Thread(target=self._run_scan_worker, daemon=True)
        t.start()
        return True

    def _run_scan_worker(self):
        try:
            scan_dir = self.scan_controller.run_scan(polished=self.next_scan_polished)
            self.last_scan_dir = scan_dir
            self.scan_status = "Complete"
        finally:
            self.scan_lock.release()

    # ============================================================
    # RECONSTRUCTION
    # ============================================================

    def _run_reconstruct_latest(self) -> str:
        if self.last_scan_dir is None:
            return "No scan available. Run a scan first."

        try:
            suffix = "calibrated" if Path("stereo_params.npz").exists() else "fake"
            points, _ = reconstruct_3d_from_scan(
                scan_dir=self.last_scan_dir,
                proj_size=(self.scan_controller.proj_w, self.scan_controller.proj_h),
            )
            msg = (
                f"Reconstruction complete ({suffix}). "
                f"Saved {len(points)} points to {self.last_scan_dir}/points_{suffix}.ply"
            )
            self.recon_status = msg
            return msg + " <a href='/'>Back</a>"
        except Exception as e:
            msg = f"Reconstruction failed: {e}"
            self.recon_status = msg
            return msg + " <a href='/'>Back</a>"

    def _run_reconstruct_scan(self, scan_path: str) -> str:
        scan_dir = Path(scan_path)
        if not scan_dir.exists():
            return f"Scan directory not found: {scan_dir} <a href='/'>Back</a>"
        try:
            suffix = "calibrated" if Path("stereo_params.npz").exists() else "fake"
            points, _ = reconstruct_3d_from_scan(
                scan_dir=scan_dir,
                proj_size=(self.scan_controller.proj_w, self.scan_controller.proj_h),
            )
            msg = (
                f"Reconstruction complete ({suffix}). "
                f"Saved {len(points)} points to {scan_dir}/points_{suffix}.ply"
            )
            self.recon_status = msg
            return msg + " <a href='/'>Back</a>"
        except Exception as e:
            msg = f"Reconstruction failed: {e}"
            self.recon_status = msg
            return msg + " <a href='/'>Back</a>"

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
    # PROJECTOR CALIBRATION HELPERS
    # ============================================================

    def _make_uniform_surface(self, level: float) -> pygame.Surface:
        img = np.full((self.proj_h, self.proj_w), np.clip(level, 0.0, 1.0), dtype=np.float32)
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        rgb = np.dstack([img_u8] * 3)
        return pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))

    def _start_new_proj_session(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session = self.proj_calib_root / f"session_{timestamp}"
        session.mkdir(parents=True, exist_ok=True)
        print(f"[PROJ-CALIB] New projector session at {session}")
        return session

    def _ensure_proj_session(self) -> Path:
        if self.proj_session is None:
            self.proj_session = self._start_new_proj_session()
        return self.proj_session

    def _set_proj_session(self, session_path: Path):
        """
        Load an existing projector calibration session so the user can continue capturing
        or re-run finish.
        """
        session_path = session_path.resolve()
        if not session_path.exists():
            self.proj_status = f"Session not found: {session_path}"
            return
        if session_path.parent != self.proj_calib_root.resolve():
            self.proj_status = f"Session must be under {self.proj_calib_root}"
            return
        self.proj_session = session_path
        # set next pose index based on existing pose_* folders
        pose_dirs = sorted(session_path.glob("pose_*"))
        next_idx = 0
        if pose_dirs:
            # parse max existing index
            def _idx(p):
                try:
                    return int(p.name.split("_")[1])
                except Exception:
                    return -1
            next_idx = max(_idx(p) for p in pose_dirs) + 1
        self.proj_views = next_idx
        self.proj_status = f"Loaded session {session_path.name} (next pose {next_idx:03d})"

    def _make_proj_pose_dir(self) -> Path:
        session = self._ensure_proj_session()
        pose_dir = session / f"pose_{self.proj_views:03d}"
        pose_dir.mkdir(parents=True, exist_ok=True)
        return pose_dir

    def _bilinear_sample(self, img: np.ndarray, pts: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        Bilinear sample img at floating-point pixel coords pts (Nx2).
        Returns Nx array with NaN for invalid samples (outside or invalid mask).
        """
        h, w = img.shape
        xs = pts[:, 0]
        ys = pts[:, 1]

        x0 = np.floor(xs).astype(int)
        y0 = np.floor(ys).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        wx = xs - x0
        wy = ys - y0

        valid = (
            (x0 >= 0) & (y0 >= 0) &
            (x1 < w) & (y1 < h)
        )

        # Require mask validity in surrounding pixels
        valid &= valid_mask[y0, x0] & valid_mask[y0, x1] & valid_mask[y1, x0] & valid_mask[y1, x1]

        samples = np.full(xs.shape, np.nan, dtype=np.float32)
        if not valid.any():
            return samples

        x0_c = x0[valid]
        y0_c = y0[valid]
        x1_c = x1[valid]
        y1_c = y1[valid]
        wx_c = wx[valid]
        wy_c = wy[valid]

        v00 = img[y0_c, x0_c]
        v01 = img[y0_c, x1_c]
        v10 = img[y1_c, x0_c]
        v11 = img[y1_c, x1_c]

        top = v00 * (1 - wx_c) + v01 * wx_c
        bot = v10 * (1 - wx_c) + v11 * wx_c
        val = top * (1 - wy_c) + bot * wy_c
        samples[valid] = val.astype(np.float32)
        return samples

    def _capture_projector_pose(self):
        with self.proj_lock:
            if not Path("camera_intrinsics.npz").exists():
                msg = "camera_intrinsics.npz missing. Calibrate camera first."
                self.proj_status = msg
                return False, msg

            if self.scan_lock.locked():
                msg = "Scan running; wait until complete."
                self.proj_status = msg
                return False, msg

            pose_dir = self._make_proj_pose_dir()

            # Auto exposure on mid-grey
            def set_midgrey():
                self.set_surface_callback(self.graycode_midgray)

            try:
                exp, gain = self.camera.auto_expose_with_midgrey(
                    set_midgrey_surface_callback=set_midgrey,
                    target_mean=120,
                    tolerance=6,
                    max_iters=6,
                )
                print(f"[PROJ-CALIB] AE locked for pose {self.proj_views}: exp={exp}us gain={gain}")
            except Exception as e:
                msg = f"Auto-exposure failed: {e}"
                self.proj_status = msg
                return False, msg

            frames: list[np.ndarray] = []
            for idx, pat in enumerate(self.graycode_set.patterns):
                self.set_surface_callback(pat.surface)
                time_module.sleep(self.pattern_settle_time)
                gray = self.camera.capture_gray().astype(np.float32)
                frames.append(gray)
                fname = f"gray_{pat.name}_{idx:03d}.png"
                cv2.imwrite(str(pose_dir / fname), gray)

            # Bright frame for checkerboard
            self.set_surface_callback(self.white_surface)
            time_module.sleep(self.pattern_settle_time)
            bright = self.camera.capture_gray().astype(np.float32)
            cv2.imwrite(str(pose_dir / "bright.png"), bright)

            # Return to midgrey
            self.set_surface_callback(self.midgrey_surface)

            bright_u8 = np.clip(bright, 0, 255).astype(np.uint8)
            mean, std, min_val, max_val = compute_image_stats(bright_u8)
            if mean < MIN_MEAN or mean > MAX_MEAN or std < MIN_STD:
                msg = f"Bright frame rejected: mean={mean:.1f}, std={std:.1f}, min={min_val:.1f}, max={max_val:.1f}"
                self.proj_status = msg
                return False, msg

            dbg_path = pose_dir / "checkerboard_debug.png"
            detection = detect_checkerboard(bright_u8, self.cb_checkerboard, debug_path=dbg_path)
            if detection.corners is None:
                msg = detection.reason or "Checkerboard not found"
                self.proj_status = msg
                return False, msg

            if detection.area_frac < MIN_BOARD_AREA_FRAC or detection.aspect_ratio < MIN_BOARD_ASPECT:
                msg = f"Checkerboard invalid: area={detection.area_frac:.3f}, aspect={detection.aspect_ratio:.3f}"
                self.proj_status = msg
                return False, msg

            # Decode GrayCode
            debug_dir = pose_dir / "graycode_debug"
            proj_u, proj_v, valid_mask, debug = decode_with_cleaning(
                frames,
                self.graycode_set,
                min_contrast=3.0,
                norm_thresh=0.02,
                min_fraction_bits=0.6,
                morph_kernel=3,
                min_component=200,
                debug_dir=debug_dir,
            )
            np.save(pose_dir / "proj_u.npy", proj_u.astype(np.float32))
            np.save(pose_dir / "proj_v.npy", proj_v.astype(np.float32))
            np.save(pose_dir / "valid_mask.npy", valid_mask.astype(np.uint8))
            # Always save debug overlays to inspect failures
            self._save_proj_debug(pose_dir, bright_u8, detection, proj_u, proj_v, valid_mask, debug)

            valid_fraction_total = float(valid_mask.mean())
            if valid_fraction_total < 0.6:
                msg = f"Decoded coverage too low ({valid_fraction_total*100:.1f}%). Check projector brightness/alignment."
                self.proj_status = msg
                return False, msg

            H, W = bright_u8.shape
            obj_points = create_object_points(self.cb_checkerboard, self.cb_square_size).astype(np.float32)
            image_points_cam = detection.corners.reshape(-1, 2).astype(np.float32)

            u_samples = self._bilinear_sample(proj_u, image_points_cam, valid_mask)
            v_samples = self._bilinear_sample(proj_v, image_points_cam, valid_mask)
            image_points_proj = np.stack([u_samples, v_samples], axis=1)

            valid_corners = np.isfinite(image_points_proj).all(axis=1)
            valid_frac = float(valid_corners.mean()) if valid_corners.size > 0 else 0.0

            if valid_frac < 0.4:
                msg = f"Pose rejected: only {valid_frac*100:.1f}% corners decoded."
                self.proj_status = msg
                return False, msg

            np.savez_compressed(
                pose_dir / "pose_data.npz",
                object_points=obj_points,
                image_points_cam=image_points_cam,
                image_points_proj=image_points_proj,
                valid_fraction=np.array([valid_frac], dtype=np.float32),
                proj_size=np.array([self.proj_w, self.proj_h], dtype=np.int32),
                image_size_cam=np.array([W, H], dtype=np.int32),
            )

            self.proj_views += 1
            self.proj_status = f"Captured pose {self.proj_views} (valid corners {valid_frac*100:.1f}%)."
            print(f"[PROJ-CALIB] Pose {self.proj_views} saved to {pose_dir} | valid corners {valid_frac*100:.1f}%")
            return True, self.proj_status

    def _save_proj_debug(self, pose_dir: Path, bright: np.ndarray, detection, proj_u: np.ndarray, proj_v: np.ndarray, mask: np.ndarray, debug: dict | None = None):
        pose_dir.mkdir(parents=True, exist_ok=True)
        vis = cv2.cvtColor(bright, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, self.cb_checkerboard, detection.corners, True)
        cv2.imwrite(str(pose_dir / "checkerboard_overlay.png"), vis)

        def _colorize(arr: np.ndarray, max_val: float) -> np.ndarray:
            arr_disp = arr.copy()
            arr_disp[~np.isfinite(arr_disp)] = 0
            arr_disp = np.clip(arr_disp, 0, max_val)
            norm = (arr_disp / (max_val + 1e-6) * 255.0).astype(np.uint8)
            return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)

        uv_vis = _colorize(proj_u, self.proj_w - 1)
        vv_vis = _colorize(proj_v, self.proj_h - 1)
        cv2.imwrite(str(pose_dir / "proj_u_debug.png"), uv_vis)
        cv2.imwrite(str(pose_dir / "proj_v_debug.png"), vv_vis)

        mask_img = (mask.astype(np.uint8) * 255)
        cv2.imwrite(str(pose_dir / "mask.png"), mask_img)

        if debug is not None and "diff_u" in debug and "diff_v" in debug:
            def _norm_diff(d):
                d = np.clip((d - d.min()) / (d.max() - d.min() + 1e-6), 0, 1)
                img = (d * 255).astype(np.uint8)
                return cv2.applyColorMap(img, cv2.COLORMAP_BONE)
            cv2.imwrite(str(pose_dir / "diff_u.png"), _norm_diff(debug["diff_u"]))
            cv2.imwrite(str(pose_dir / "diff_v.png"), _norm_diff(debug["diff_v"]))

            dbg_dir = pose_dir / "graycode_debug"
            dbg_dir.mkdir(exist_ok=True)
            # Save histograms
            hist, _ = np.histogram(bright, bins=256, range=(0, 255))
            np.savetxt(dbg_dir / "intensity_hist.csv", hist, fmt="%d")
            if "valid" in debug:
                valid_mask = debug["valid"]
                cv2.imwrite(str(dbg_dir / "valid_mask_raw.png"), (valid_mask.astype(np.uint8) * 255))

    def _finish_projector_calibration(self):
        with self.proj_lock:
            min_views = 6
            if self.proj_views < min_views:
                msg = f"Need at least {min_views} poses. Have {self.proj_views}."
                self.proj_status = msg
                return False, msg

            session = self._ensure_proj_session()
            try:
                calib = run_projector_calibration(
                    session_dir=session,
                    camera_calib_path=Path("camera_intrinsics.npz"),
                    min_views=min_views,
                    max_proj_error=self.proj_max_error,
                )
            except Exception as e:
                msg = f"Projector calibration failed: {e}"
                self.proj_status = msg
                return False, msg

            # Copy results to root for reconstruction
            intr_src = session / "projector_intrinsics.npz"
            stereo_src = session / "stereo_params.npz"
            if intr_src.exists():
                shutil.copy(intr_src, Path("projector_intrinsics.npz"))
            if stereo_src.exists():
                shutil.copy(stereo_src, Path("stereo_params.npz"))

            self.proj_last_rms = calib.rms_stereo
            msg = (
                f"Projector calibration complete. proj RMS={calib.rms_proj:.3f} px, "
                f"stereo RMS={calib.rms_stereo:.3f} px, baseline={calib.baseline_m:.3f} m. "
                "Saved projector_intrinsics.npz and stereo_params.npz."
            )
            self.proj_status = msg
            return True, msg

    # ============================================================
    # RUN
    # ============================================================

    def run(self, host="0.0.0.0", port=5000):
        print(f"[WEB] Server at http://{host}:{port}")
        self.app.run(host=host, port=port, threaded=True)
