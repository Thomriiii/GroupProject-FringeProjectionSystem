"""HTTP routes for the v2 Flask app."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from flask import Flask, Response, jsonify, render_template, request, send_file

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
        runner.refresh_turntable_pos()
        return jsonify({
            "pipeline": runner.status(),
            "camera": camera.status(),
            "turntable": runner.turntable_status(),
        })

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

    # ── turntable connection ───────────────────────────────────────────────────

    @app.post("/api/turntable/connect")
    def turntable_connect():
        """
        Connect to a turntable by IP.
        Body: {"ip": "192.168.0.x"}
        """
        body = request.get_json(silent=True) or {}
        ip = body.get("ip", "").strip()
        if not ip:
            return jsonify({"error": "missing field: ip"}), 400
        state = runner.connect_turntable(ip)
        ok = state.get("status") == "connected"
        return jsonify(state), (200 if ok else 503)

    @app.post("/api/turntable/discover")
    def turntable_discover():
        """Trigger nmap-based auto-discovery in the background."""
        runner.discover_turntable_async()
        return jsonify(runner.turntable_status()), 202

    # ── turntable control ──────────────────────────────────────────────────────

    @app.post("/api/turntable/rotate")
    def turntable_rotate():
        """Body: {"degrees": float, "relative": bool (default true)}"""
        tt = runner.turntable
        if tt is None:
            return jsonify({"error": "turntable not connected"}), 503
        body = request.get_json(silent=True) or {}
        if "degrees" not in body:
            return jsonify({"error": "missing field: degrees"}), 400
        try:
            resp = tt.rotate(float(body["degrees"]), bool(body.get("relative", True)))
            return jsonify(resp)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 503

    @app.post("/api/turntable/home")
    def turntable_home():
        tt = runner.turntable
        if tt is None:
            return jsonify({"error": "turntable not connected"}), 503
        try:
            pos = tt.home()
            return jsonify({"ok": True, "degrees": pos})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 503

    @app.post("/api/multi_scan")
    def multi_scan():
        ok, state = runner.start("multi_scan", runner.multi_scan)
        return jsonify({"ok": ok, "status": state}), (202 if ok else 409)

    # ── turntable calibration ──────────────────────────────────────────────────

    @app.get("/turntable-calibration")
    def turntable_calibration_page():
        return render_template("turntable.html")

    def _tt_cfg() -> dict:
        return runner.config.get("turntable_calibration") or {}

    def _tt_storage() -> Path:
        return Path(_tt_cfg().get("storage_root", "data/turntable/sessions"))

    def _tt_apply_camera_settings() -> None:
        """Apply bright exposure for turntable/camera calibration captures."""
        import time as _time
        cam_cfg = (_tt_cfg().get("camera") or {})
        if not cam_cfg:
            return
        camera.set_manual_controls(
            exposure_us=int(cam_cfg.get("exposure_us", 15000)),
            analogue_gain=float(cam_cfg.get("analogue_gain", 4.0)),
            awb_enable=bool(cam_cfg.get("awb_enable", True)),
        )
        settle_ms = int(cam_cfg.get("settle_ms", 300))
        if settle_ms > 0:
            _time.sleep(settle_ms / 1000.0)

    def _tt_intrinsics():
        """Return (K, D) from camera intrinsics JSON, or (None, None)."""
        import numpy as np
        cfg = runner.config
        path_str = (cfg.get("calibration") or {}).get(
            "camera_intrinsics_path",
            "data/calibration/camera/intrinsics_latest.json",
        )
        p = Path(path_str)
        if not p.exists():
            p = Path("data/calibration/camera/intrinsics_latest.json")
        if not p.exists():
            return None, None
        data = json.loads(p.read_text())
        K = np.array(data["camera_matrix"], dtype=np.float64)
        D = np.array(
            data.get("dist_coeffs", data.get("distortion_coefficients", [[0, 0, 0, 0, 0]])),
            dtype=np.float64,
        ).reshape(-1)
        return K, D

    @app.post("/api/turntable-cal/session/new")
    def tt_cal_new_session():
        from fringe_app_v2.turntable.session import new_session
        body = request.get_json(silent=True) or {}
        step = float(body.get("nominal_step_deg", _tt_cfg().get("nominal_step_deg", 15.0)))
        session = new_session(_tt_storage(), step)
        return jsonify({"ok": True, "session_id": session.session_id})

    @app.get("/api/turntable-cal/sessions")
    def tt_cal_list_sessions():
        from fringe_app_v2.turntable.session import list_sessions
        return jsonify({"sessions": list_sessions(_tt_storage())})

    @app.get("/api/turntable-cal/session/<session_id>")
    def tt_cal_get_session(session_id: str):
        from fringe_app_v2.turntable.session import load_session
        try:
            session = load_session(_tt_storage(), session_id)
            return jsonify(session.to_dict())
        except FileNotFoundError:
            return jsonify({"error": "not found"}), 404

    @app.post("/api/turntable-cal/session/<session_id>/capture")
    def tt_cal_capture(session_id: str):
        from fringe_app_v2.turntable.session import load_session, add_frame
        from fringe_app_v2.turntable.charuco_pose import process_frame
        import cv2
        import numpy as np

        body = request.get_json(silent=True) or {}
        angle_deg = float(body.get("angle_deg", 0.0))
        note = str(body.get("note", ""))
        settle_ms = int((_tt_cfg().get("capture") or {}).get("settle_ms_after_rotation", 300))

        session = load_session(_tt_storage(), session_id)
        label = f"angle_{int(round(angle_deg)):03d}"
        frame_dir = session.frame_dir(label)
        frame_dir.mkdir(parents=True, exist_ok=True)

        import time
        _tt_apply_camera_settings()
        if settle_ms > 0:
            time.sleep(settle_ms / 1000.0)
        image = camera.capture(flush_frames=1)
        cv2.imwrite(str(frame_dir / "image.png"), image)

        K, D = _tt_intrinsics()
        if K is not None:
            charuco_cfg = _tt_cfg().get("charuco") or {}
            min_corners = int(charuco_cfg.get("min_corners", 6))
            pose = process_frame(frame_dir, image, charuco_cfg, K, D, min_corners=min_corners)
        else:
            pose = {"ok": False, "n_corners": 0, "reprojection_error_px": None}

        rec = add_frame(session, angle_deg, str(frame_dir / "image.png"), note=note)
        rec.charuco_ok = pose.get("ok", False) or (pose.get("n_corners", 0) > 0)
        rec.n_corners = int(pose.get("n_corners") or 0)
        rec.pose_ok = bool(pose.get("ok"))
        rec.reprojection_error_px = float(pose.get("reprojection_error_px") or 0.0)
        session.save()

        return jsonify({
            "ok": True,
            "angle_deg": angle_deg,
            "status": rec.status,
            "n_corners": rec.n_corners,
            "reprojection_error_px": rec.reprojection_error_px,
            "pose_ok": rec.pose_ok,
        })

    @app.post("/api/turntable-cal/session/<session_id>/analyse")
    def tt_cal_analyse(session_id: str):
        from fringe_app_v2.turntable.session import load_session
        from fringe_app_v2.turntable.axis_fit import run_analysis
        from fringe_app_v2.turntable.alignment import run_alignment
        from fringe_app_v2.turntable.report import write_report

        session = load_session(_tt_storage(), session_id)
        axis = run_analysis(session.root, nominal_step_deg=session.nominal_step_deg)
        align = run_alignment(session.root)
        session.analysed = True
        session.save()
        report_path = write_report(session.root)

        return jsonify({
            "ok": True,
            "axis_fit": axis,
            "alignment": {"n_pairs": align.get("n_pairs"), "n_ok": align.get("n_ok")},
            "report_path": str(report_path),
        })

    @app.get("/api/turntable-cal/session/<session_id>/frame/<angle_label>/image")
    def tt_cal_frame_image(session_id: str, angle_label: str):
        from fringe_app_v2.turntable.session import load_session
        session = load_session(_tt_storage(), session_id)
        img_path = session.frame_dir(angle_label) / "image.png"
        if not img_path.exists():
            return jsonify({"error": "not found"}), 404
        return send_file(str(img_path), mimetype="image/png")

    @app.get("/api/turntable-cal/session/<session_id>/frame/<angle_label>/overlay")
    def tt_cal_frame_overlay(session_id: str, angle_label: str):
        from fringe_app_v2.turntable.session import load_session
        session = load_session(_tt_storage(), session_id)
        ov_path = session.frame_dir(angle_label) / "overlay.png"
        fallback = session.frame_dir(angle_label) / "image.png"
        p = ov_path if ov_path.exists() else fallback
        if not p.exists():
            return jsonify({"error": "not found"}), 404
        return send_file(str(p), mimetype="image/png")

    @app.get("/api/turntable-cal/session/<session_id>/report")
    def tt_cal_report(session_id: str):
        from fringe_app_v2.turntable.session import load_session
        session = load_session(_tt_storage(), session_id)
        report_path = session.calibration_dir / "calibration_report.md"
        if not report_path.exists():
            return jsonify({"error": "report not generated yet"}), 404
        return send_file(str(report_path), mimetype="text/markdown",
                         download_name="calibration_report.md", as_attachment=True)

    # ── auto capture (background) ──────────────────────────────────────────────

    _auto_jobs: dict[str, dict] = {}   # session_id → {state, done, frames, error}

    @app.post("/api/turntable-cal/session/<session_id>/capture-auto")
    def tt_cal_capture_auto(session_id: str):
        from fringe_app_v2.turntable.session import load_session
        from fringe_app_v2.turntable.auto_capture import run_auto_capture
        import numpy as np

        if session_id in _auto_jobs and not _auto_jobs[session_id].get("done"):
            return jsonify({"error": "auto-capture already running for this session"}), 409

        if runner.turntable is None:
            return jsonify({"error": "turntable not connected — connect it first"}), 503

        try:
            session = load_session(_tt_storage(), session_id)
        except FileNotFoundError:
            return jsonify({"error": "session not found"}), 404

        step = float(_tt_cfg().get("nominal_step_deg", 15.0))
        total = float(_tt_cfg().get("total_degrees", 360.0))
        settle_ms = int((_tt_cfg().get("capture") or {}).get("settle_ms_after_rotation", 600))
        charuco_cfg = _tt_cfg().get("charuco") or {}
        K, D = _tt_intrinsics()

        job = {"done": False, "frame": 0, "total": int(round(total / step)), "error": None}
        _auto_jobs[session_id] = job

        def _progress(idx, n, rec):
            job["frame"] = idx

        def _run():
            try:
                _tt_apply_camera_settings()
                run_auto_capture(
                    session=session,
                    turntable=runner.turntable,
                    camera=camera,
                    charuco_cfg=charuco_cfg,
                    K=K, D=D,
                    step_deg=step,
                    total_deg=total,
                    settle_ms=settle_ms,
                    on_frame=_progress,
                )
                # Auto-analyse on completion
                from fringe_app_v2.turntable.axis_fit import run_analysis
                from fringe_app_v2.turntable.alignment import run_alignment
                from fringe_app_v2.turntable.report import write_report
                run_analysis(session.root, nominal_step_deg=step)
                run_alignment(session.root)
                session.analysed = True
                session.save()
                write_report(session.root)
            except Exception as exc:
                job["error"] = str(exc)
            finally:
                job["done"] = True

        threading.Thread(target=_run, daemon=True).start()
        return jsonify({"ok": True, "total_frames": job["total"]}), 202

    @app.get("/api/turntable-cal/session/<session_id>/capture-auto/status")
    def tt_cal_capture_auto_status(session_id: str):
        job = _auto_jobs.get(session_id)
        if job is None:
            return jsonify({"running": False, "frame": 0, "total": 0})
        return jsonify({
            "running": not job["done"],
            "done": job["done"],
            "frame": job["frame"],
            "total": job["total"],
            "error": job["error"],
        })

    # ── Camera calibration ─────────────────────────────────────────────────────

    @app.get("/camera-calibration")
    def camera_cal_page():
        return render_template("calibration_camera.html")

    def _cam_cal_root() -> Path:
        from fringe_app_v2.calibration.camera.session import _camera_root
        return _camera_root(runner.config)

    @app.post("/api/calibration/camera/session/new")
    def cam_cal_new_session():
        from fringe_app_v2.calibration.camera.session import create_session
        try:
            session_dir, payload = create_session(runner.config)
            return jsonify({"ok": True, "session_id": str(payload["session_id"])})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.get("/api/calibration/camera/sessions")
    def cam_cal_list_sessions():
        from fringe_app_v2.calibration.camera.session import list_sessions
        return jsonify({"sessions": list_sessions(runner.config)})

    @app.get("/api/calibration/camera/session/<session_id>")
    def cam_cal_get_session(session_id: str):
        from fringe_app_v2.calibration.camera.session import get_session_dir, load_session
        sdir = get_session_dir(runner.config, session_id)
        try:
            sess = load_session(sdir)
            captures = sess.get("captures", []) or []
            return jsonify({
                "session_id": session_id,
                "n_captures": len(captures),
                "n_good": len([c for c in captures if c.get("found")]),
                "solved": sess.get("calibration") is not None,
                "calibration": sess.get("calibration"),
                "captures": captures,
            })
        except FileNotFoundError:
            return jsonify({"error": "not found"}), 404

    @app.post("/api/calibration/camera/session/<session_id>/capture")
    def cam_cal_capture(session_id: str):
        from fringe_app_v2.calibration.camera.session import get_session_dir, add_capture, load_session

        calib_cfg = runner.config.get("calibration", {}) or {}
        cam_cfg = calib_cfg.get("camera", {}) or {}
        if cam_cfg:
            try:
                camera.apply_controls({
                    "ExposureTime": int(cam_cfg.get("exposure_us", 12000)),
                    "AnalogueGain": float(cam_cfg.get("analogue_gain", 3.0)),
                    "AwbEnable": bool(cam_cfg.get("awb_enable", True)),
                    "AeEnable": False,
                })
                import time as _time
                _time.sleep(0.3)
            except Exception:
                pass

        image = camera.capture(flush_frames=int(calib_cfg.get("capture_flush_frames", 1)))
        charuco_cfg = calib_cfg.get("charuco") or {}

        sdir = get_session_dir(runner.config, session_id)
        try:
            rec = add_capture(sdir, image, charuco_cfg)
            return jsonify({"ok": True, **rec})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.post("/api/calibration/camera/session/<session_id>/solve")
    def cam_cal_solve(session_id: str):
        from fringe_app_v2.calibration.camera.session import get_session_dir, solve_session
        sdir = get_session_dir(runner.config, session_id)
        try:
            result = solve_session(sdir, output_root=_cam_cal_root())
            return jsonify({"ok": True, **result})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.get("/api/calibration/camera/session/<session_id>/capture/<capture_id>/image")
    def cam_cal_frame_image(session_id: str, capture_id: str):
        from fringe_app_v2.calibration.camera.session import get_session_dir
        sdir = get_session_dir(runner.config, session_id)
        p = sdir / "captures" / capture_id / "image.png"
        if not p.exists():
            return jsonify({"error": "not found"}), 404
        return send_file(str(p), mimetype="image/png")

    @app.get("/api/calibration/camera/session/<session_id>/capture/<capture_id>/overlay")
    def cam_cal_frame_overlay(session_id: str, capture_id: str):
        from fringe_app_v2.calibration.camera.session import get_session_dir
        sdir = get_session_dir(runner.config, session_id)
        ov = sdir / "captures" / capture_id / "overlay.png"
        img = sdir / "captures" / capture_id / "image.png"
        p = ov if ov.exists() else img
        if not p.exists():
            return jsonify({"error": "not found"}), 404
        return send_file(str(p), mimetype="image/png")

    @app.delete("/api/calibration/camera/session/<session_id>/capture/<capture_id>")
    def cam_cal_delete_capture(session_id: str, capture_id: str):
        from fringe_app_v2.calibration.camera.session import get_session_dir, delete_capture
        sdir = get_session_dir(runner.config, session_id)
        try:
            delete_capture(sdir, capture_id)
            return jsonify({"ok": True})
        except FileNotFoundError:
            return jsonify({"error": "not found"}), 404

    # ── Projector calibration ──────────────────────────────────────────────────

    @app.get("/projector-calibration")
    def proj_cal_page():
        return render_template("calibration_projector.html")

    _proj_cal_jobs: dict[str, dict] = {}

    def _proj_cal_session_path(session_id: str) -> Path:
        from fringe_app_v2.calibration.projector.session import _proj_root
        return _proj_root(runner.config) / session_id

    @app.post("/api/calibration/projector/session/new")
    def proj_cal_new_session():
        from fringe_app_v2.calibration.projector.session import create_session
        try:
            session_path, payload = create_session(runner.config)
            return jsonify({"ok": True, "session_id": str(payload["session_id"])})
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 412
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.get("/api/calibration/projector/sessions")
    def proj_cal_list_sessions():
        from fringe_app_v2.calibration.projector.session import list_sessions
        return jsonify({"sessions": list_sessions(runner.config)})

    @app.get("/api/calibration/projector/session/<session_id>")
    def proj_cal_get_session(session_id: str):
        from fringe_app_v2.calibration.projector.session import build_session_payload
        sdir = _proj_cal_session_path(session_id)
        try:
            return jsonify(build_session_payload(sdir, api_prefix="/api/calibration/projector"))
        except FileNotFoundError:
            return jsonify({"error": "not found"}), 404

    @app.post("/api/calibration/projector/session/<session_id>/capture")
    def proj_cal_capture(session_id: str):
        from fringe_app_v2.calibration.projector.session import (
            load_session, allocate_view_id, register_view
        )
        from fringe_app_v2.calibration.projector.view_capture import capture_calibration_view

        sdir = _proj_cal_session_path(session_id)
        try:
            session = load_session(sdir)
        except FileNotFoundError:
            return jsonify({"error": "session not found"}), 404

        snap = (session.get("config_snapshot") or {})
        v2_cfg = snap.get("calibration_v2") or {}
        charuco_cfg = v2_cfg.get("charuco") or {}
        gating_cfg = v2_cfg.get("gating") or {}
        uv_refine_cfg = v2_cfg.get("uv_refine") or {}

        import json as _json
        intr_path = Path(str(snap.get("camera_intrinsics_path", "")))
        K, D = None, None
        if intr_path.exists():
            import numpy as _np
            payload = _json.loads(intr_path.read_text())
            K = _np.asarray(payload["camera_matrix"], dtype=_np.float64)
            D = _np.asarray(payload["dist_coeffs"], dtype=_np.float64).reshape(-1, 1)

        scan_cfg = runner.config.get("scan", {}) or {}
        proj_w = int(scan_cfg.get("width", 1024))
        proj_h = int(scan_cfg.get("height", 768))
        projector_size = (proj_w, proj_h)

        view_id, session = allocate_view_id(sdir, session)
        view_dir = sdir / "views" / view_id

        try:
            view_record = capture_calibration_view(
                view_dir=view_dir,
                camera=camera,
                projector=runner.projector,
                patterns=runner.patterns,
                params=runner.params,
                config=runner.config,
                charuco_cfg=charuco_cfg,
                K=K,
                D=D,
                projector_size=projector_size,
                uv_refine_cfg=uv_refine_cfg,
                gating_cfg=gating_cfg,
            )
        except Exception as exc:
            import traceback
            return jsonify({"ok": False, "error": str(exc), "traceback": traceback.format_exc()}), 500

        view_record["view_id"] = view_id
        register_view(sdir, view_record)
        return jsonify({"ok": True, "view_id": view_id, "status": view_record.get("status"), "metrics": view_record.get("metrics"), "hints": view_record.get("hints")})

    @app.post("/api/calibration/projector/session/<session_id>/solve")
    def proj_cal_solve(session_id: str):
        from fringe_app_v2.calibration.projector.solve import solve_session
        sdir = _proj_cal_session_path(session_id)
        scan_cfg = runner.config.get("scan", {}) or {}
        proj_size = (int(scan_cfg.get("width", 1024)), int(scan_cfg.get("height", 768)))
        try:
            result = solve_session(sdir, projector_size=proj_size)
            return jsonify({"ok": True, **result})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.delete("/api/calibration/projector/session/<session_id>/view/<view_id>")
    def proj_cal_delete_view(session_id: str, view_id: str):
        from fringe_app_v2.calibration.projector.session import delete_view
        sdir = _proj_cal_session_path(session_id)
        try:
            delete_view(sdir, view_id)
            return jsonify({"ok": True})
        except FileNotFoundError:
            return jsonify({"error": "not found"}), 404

    @app.get("/api/calibration/projector/session/<session_id>/view/<view_id>/image")
    def proj_cal_view_image(session_id: str, view_id: str):
        p = _proj_cal_session_path(session_id) / "views" / view_id / "image.png"
        return send_file(str(p), mimetype="image/png") if p.exists() else (jsonify({"error": "not found"}), 404)

    @app.get("/api/calibration/projector/session/<session_id>/view/<view_id>/overlay")
    def proj_cal_view_overlay(session_id: str, view_id: str):
        vdir = _proj_cal_session_path(session_id) / "views" / view_id
        p = vdir / "overlay.png" if (vdir / "overlay.png").exists() else vdir / "image.png"
        return send_file(str(p), mimetype="image/png") if p.exists() else (jsonify({"error": "not found"}), 404)

    @app.get("/api/calibration/projector/session/<session_id>/view/<view_id>/uv_overlay")
    def proj_cal_view_uv_overlay(session_id: str, view_id: str):
        p = _proj_cal_session_path(session_id) / "views" / view_id / "uv" / "uv_overlay.png"
        return send_file(str(p), mimetype="image/png") if p.exists() else (jsonify({"error": "not found"}), 404)

    @app.get("/api/calibration/projector/session/<session_id>/stereo_download")
    def proj_cal_download_stereo(session_id: str):
        p = _proj_cal_session_path(session_id) / "export" / "stereo.json"
        if not p.exists():
            return jsonify({"error": "not solved yet"}), 404
        return send_file(str(p), mimetype="application/json",
                         download_name="stereo.json", as_attachment=True)
