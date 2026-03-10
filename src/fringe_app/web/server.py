"""FastAPI server for fringe app."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import functools
import json
import math
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import logging
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from fringe_app.calibration import CalibrationConfig, CalibrationManager
from fringe_app.calibration.projector_stereo import (
    create_session as create_projector_session,
    list_views as list_projector_views,
    capture_projector_view,
    calibrate_projector_gamma,
    delete_view as delete_projector_view,
    solve_projector_session,
)
from fringe_app.calibration_v2 import (
    create_projector_v2_session,
    list_projector_v2_sessions,
    get_projector_v2_session,
    capture_projector_v2_view,
    delete_projector_v2_view,
    reset_projector_v2_session,
    solve_projector_v2_session,
    download_projector_v2_session_zip,
)
from fringe_app.core.models import ScanParams
from fringe_app.core.controller import ScanController
from fringe_app.io.run_store import RunStore
from fringe_app.phase.psp import PhaseShiftProcessor, PhaseThresholds
from fringe_app.phase.masking_post import cleanup_mask
from fringe_app.phase.metrics import score_mask
from fringe_app.vision.object_roi import detect_object_roi, ObjectRoiConfig
from fringe_app.overlays.generate import overlay_masks
from fringe_app.unwrap.temporal import unwrap_multi_frequency, save_unwrap_outputs
from fringe_app.cli import cmd_phase, cmd_unwrap, cmd_score, cmd_pipeline_run_3d
from fringe_app.recon import load_stereo_model, reconstruct_uv_run, save_reconstruction_outputs
from fringe_app.web.routes.projector_calibration import (
    build_capture_response_payload,
    build_projector_session_payload,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class FringeServer:
    """FastAPI wrapper with REST and WebSocket endpoints."""

    def __init__(self, controller: ScanController, run_store: RunStore, config: Dict[str, Any]) -> None:
        self.controller = controller
        self.run_store = run_store
        self.config = config
        calib_cfg = self.config.get("calibration", {}) or {}
        camera_calib_cfg = (calib_cfg.get("camera_calibration", {}) or {})
        camera_cb_cfg = (camera_calib_cfg.get("checkerboard", {}) or {})
        legacy_cb_cfg = (calib_cfg.get("checkerboard", {}) or {})

        squares_x = int(camera_cb_cfg.get("squares_x", camera_calib_cfg.get("squares_x", 0) or 0))
        squares_y = int(camera_cb_cfg.get("squares_y", camera_calib_cfg.get("squares_y", 0) or 0))
        default_cols = max(2, squares_x - 1) if squares_x > 0 else int(legacy_cb_cfg.get("cols", 9))
        default_rows = max(2, squares_y - 1) if squares_y > 0 else int(legacy_cb_cfg.get("rows", 6))

        camera_checkerboard_cols = int(camera_cb_cfg.get("cols", camera_calib_cfg.get("cols", default_cols)))
        camera_checkerboard_rows = int(camera_cb_cfg.get("rows", camera_calib_cfg.get("rows", default_rows)))
        camera_square_size_mm = float(
            camera_cb_cfg.get(
                "square_size_mm",
                camera_calib_cfg.get("square_size_mm", legacy_cb_cfg.get("square_size_mm", 25.0)),
            )
        )
        camera_board_type = str(camera_calib_cfg.get("board_type", calib_cfg.get("board_type", "checkerboard")))
        camera_charuco_cfg = (camera_calib_cfg.get("charuco", calib_cfg.get("charuco", {}) or {}))
        camera_min_valid_detections = int(
            camera_calib_cfg.get("min_valid_detections", calib_cfg.get("min_valid_detections", 10))
        )
        camera_root = self._camera_calibration_root()
        self.calibration = CalibrationManager(
            CalibrationConfig(
                root=str(camera_root),
                checkerboard_cols=camera_checkerboard_cols,
                checkerboard_rows=camera_checkerboard_rows,
                square_size_mm=camera_square_size_mm,
                min_valid_detections=camera_min_valid_detections,
                board_type=camera_board_type,
                charuco=camera_charuco_cfg,
            )
        )
        self._pipeline_lock = asyncio.Lock()
        self._projector_capture_lock = asyncio.Lock()
        self._projector_calibrate_lock = asyncio.Lock()
        self._projector_display_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="projector_display",
        )
        self._projector_solve_tasks: Dict[str, asyncio.Task] = {}
        self._projector_solve_status: Dict[str, Dict[str, Any]] = {}
        self._pipeline_task: asyncio.Task | None = None
        self._pipeline_state: str = "idle"
        self._pipeline_run_id: str | None = None
        self._pipeline_error: str | None = None
        self.app = FastAPI()
        self._configure_routes()

    def _configure_routes(self) -> None:
        static_dir = Path(__file__).parent / "static"
        log = logging.getLogger("fringe_app")

        @self.app.get("/api/status")
        async def get_status():
            return self.controller.get_status()

        @self.app.post("/api/scan/start")
        async def start_scan(payload: Dict[str, Any]):
            params = self._build_params(payload)
            run_id = self.controller.start_scan(params)
            return {"ok": True, "run_id": run_id}

        @self.app.post("/api/scan/stop")
        async def stop_scan():
            self.controller.stop_scan()
            return {"ok": True}

        @self.app.post("/api/pipeline/start")
        async def start_pipeline(payload: Dict[str, Any] | None = None):
            if self._pipeline_task is not None and not self._pipeline_task.done():
                return JSONResponse({"ok": False, "error": "pipeline already running"}, status_code=409)
            payload = payload or {}
            force = bool(payload.get("force", False))
            self._pipeline_state = "running"
            self._pipeline_error = None
            self._pipeline_run_id = None
            self._pipeline_task = asyncio.create_task(self._run_pipeline(force=force))
            return {"ok": True, "force": force}

        @self.app.get("/api/pipeline/status")
        async def pipeline_status():
            return {
                "state": self._pipeline_state,
                "run_id": self._pipeline_run_id,
                "error": self._pipeline_error,
            }

        @self.app.get("/api/runs")
        async def list_runs():
            return [m.to_dict() for m in self.run_store.list_runs()]

        @self.app.get("/api/runs/{run_id}/download")
        async def download_run(run_id: str):
            zip_path = self.run_store.zip_run(run_id)
            return FileResponse(zip_path, filename=f"{run_id}.zip")

        @self.app.get("/api/runs/{run_id}/quality")
        async def quality_status(run_id: str):
            qpath = Path(self.run_store.root) / run_id / "quality_report.json"
            if not qpath.exists():
                return {"exists": False}
            return {"exists": True, "report": json.loads(qpath.read_text())}

        @self.app.post("/api/runs/{run_id}/phase/compute")
        async def compute_phase(run_id: str):
            try:
                log.info("Phase compute start: %s", run_id)
                thresholds = self._build_phase_thresholds()
                params = self._params_from_meta(run_id)
                processor = PhaseShiftProcessor()
                roi_cfg = self._build_roi_config()
                ref = self.run_store.load_reference_image(run_id, ref_method=roi_cfg.ref_method)
                roi_res = detect_object_roi(ref, roi_cfg)
                from dataclasses import asdict as _asdict
                self.run_store.save_roi(
                    run_id,
                    roi_res.roi_dilated_mask,
                    roi_res.bbox_dilated,
                    {
                        "cfg": _asdict(roi_cfg),
                        "bbox_core": roi_res.bbox_core,
                        "bbox_dilated": roi_res.bbox_dilated,
                        "area_ratio_core": float(np.count_nonzero(roi_res.roi_core_mask) / roi_res.roi_core_mask.size),
                        "area_ratio_dilated": float(np.count_nonzero(roi_res.roi_dilated_mask) / roi_res.roi_dilated_mask.size),
                        "roi_gate_mask": "core",
                        "debug": {
                            **roi_res.debug,
                            "ref_method": roi_cfg.ref_method,
                            "ref_stats": {
                                "min": int(np.min(ref)),
                                "mean": float(np.mean(ref)),
                                "max": int(np.max(ref)),
                            },
                            "bbox_core": roi_res.bbox_core,
                            "bbox_dilated": roi_res.bbox_dilated,
                            "area_ratio_core": float(np.count_nonzero(roi_res.roi_core_mask) / roi_res.roi_core_mask.size),
                            "area_ratio_dilated": float(np.count_nonzero(roi_res.roi_dilated_mask) / roi_res.roi_dilated_mask.size),
                            "roi_gate_mask": "core",
                        },
                    },
                    roi_raw=roi_res.raw_mask,
                    roi_post=roi_res.post_mask,
                    roi_core=roi_res.roi_core_mask,
                    roi_dilated=roi_res.roi_dilated_mask,
                    bbox_core=roi_res.bbox_core,
                    bbox_dilated=roi_res.bbox_dilated,
                )
                roi_mask_for_score = None if roi_res.debug.get("roi_fallback") else roi_res.roi_core_mask
                roi_mask_for_clip = None if roi_res.debug.get("roi_fallback") else roi_res.roi_dilated_mask
                post_cfg = self.config.get("phase", {}).get("postmask_cleanup", {}) or {}
                post_enabled = bool(post_cfg.get("enabled", True))
                min_component_area = int(post_cfg.get("min_component_area", 200))
                fill_small_holes = bool(post_cfg.get("fill_small_holes", True))
                max_hole_area = int(post_cfg.get("max_hole_area", 200))
                freqs = params.get_frequencies()
                last_debug = None
                for freq in freqs:
                    images = self.run_store.iter_captures(run_id, freq=freq) if len(freqs) > 1 else self.run_store.iter_captures(run_id)
                    result = processor.compute_phase(images, params, thresholds, roi_mask=roi_mask_for_clip)
                    raw_mask = result.mask_raw.copy()
                    if post_enabled:
                        cleaned = cleanup_mask(
                            raw_mask,
                            roi_mask_for_score,
                            min_component_area=min_component_area,
                            fill_small_holes=fill_small_holes,
                            max_hole_area=max_hole_area,
                        )
                        cleaned &= (~result.clipped_any_map)
                        result.mask_clean = cleaned
                        result.mask = cleaned
                        result.debug["postmask_cleanup"] = {
                            "enabled": True,
                            "min_component_area": min_component_area,
                            "fill_small_holes": fill_small_holes,
                            "max_hole_area": max_hole_area,
                            "raw_valid_ratio": float(np.count_nonzero(raw_mask) / raw_mask.size),
                            "clean_valid_ratio": float(np.count_nonzero(cleaned) / cleaned.size),
                        }
                    else:
                        result.mask_clean = raw_mask
                        result.mask = raw_mask
                        result.debug["postmask_cleanup"] = {"enabled": False}
                    score = score_mask(result.mask, result.B, roi_mask=roi_mask_for_score)
                    result.debug.update({
                        "roi_valid_ratio": score.roi_valid_ratio,
                        "roi_largest_component_ratio": score.roi_largest_component_ratio,
                        "roi_edge_noise_ratio": score.roi_edge_noise_ratio,
                        "roi_b_median": score.roi_b_median,
                        "roi_score": score.roi_score,
                        "roi_fallback": roi_res.debug.get("roi_fallback", False),
                    })
                    self.run_store.save_phase_outputs(run_id, result, freq=freq if len(freqs) > 1 else None)
                    phase_dir = Path(self.run_store.root) / run_id / "phase"
                    if len(freqs) > 1:
                        phase_dir = phase_dir / self.run_store._freq_tag(freq)
                    from fringe_app.phase import visualize as _viz
                    valid_in_roi = result.mask & roi_res.roi_mask
                    try:
                        _viz.save_mask_png(valid_in_roi, str(phase_dir / "valid_in_roi.png"))
                    except Exception:
                        pass
                    last_debug = result.debug

                overlays_dir = Path(self.run_store.root) / run_id / "overlays"
                overlays_dir.mkdir(exist_ok=True)
                try:
                    empty = np.zeros_like(roi_res.roi_mask, dtype=bool)
                    overlay_masks(ref, roi_res.roi_mask, empty, overlays_dir / "roi_overlay.png")
                except Exception:
                    pass
                log.info("Phase compute complete: %s", run_id)
                return {"ok": True, "run_id": run_id, "stats": last_debug}
            except Exception as exc:
                log.error("Phase compute failed for %s: %s", run_id, exc)
                log.error(traceback.format_exc())
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @self.app.get("/api/runs/{run_id}/phase/status")
        async def phase_status(run_id: str):
            phase_dir = Path(self.run_store.root) / run_id / "phase"
            phase_dir = self._select_phase_dir(phase_dir)
            meta_path = phase_dir / "phase_meta.json"
            if not meta_path.exists():
                return {"exists": False}
            return {"exists": True, "meta": json.loads(meta_path.read_text())}

        @self.app.get("/api/runs/{run_id}/phase/preview")
        async def phase_preview(run_id: str):
            phase_dir = Path(self.run_store.root) / run_id / "phase"
            phase_dir = self._select_phase_dir(phase_dir)
            fixed = phase_dir / "phi_debug_fixed.png"
            auto = phase_dir / "phi_debug_autoscale.png"
            if fixed.exists():
                return FileResponse(fixed)
            if auto.exists():
                return FileResponse(auto)
            return JSONResponse({"error": "phase preview not found"}, status_code=404)

        @self.app.get("/api/runs/{run_id}/roi/preview")
        async def roi_preview(run_id: str):
            overlays = Path(self.run_store.root) / run_id / "overlays"
            roi_overlay = overlays / "roi_overlay.png"
            if roi_overlay.exists():
                return FileResponse(roi_overlay)
            return JSONResponse({"error": "roi preview not found"}, status_code=404)

        @self.app.get("/api/runs/{run_id}/phase/overlay")
        async def phase_overlay(run_id: str):
            overlays = Path(self.run_store.root) / run_id / "overlays"
            overlay = overlays / "valid_in_roi_overlay.png"
            if overlay.exists():
                return FileResponse(overlay)
            return JSONResponse({"error": "phase overlay not found"}, status_code=404)

        @self.app.get("/api/runs/{run_id}/capture/preview")
        async def capture_preview(run_id: str):
            run_dir = Path(self.run_store.root) / run_id
            cap_dir = run_dir / "captures"
            first = sorted(cap_dir.glob("frame_*.png"))
            if first:
                return FileResponse(first[0])
            return JSONResponse({"error": "capture not found"}, status_code=404)

        @self.app.post("/api/runs/{run_id}/unwrap/compute")
        async def unwrap_compute(run_id: str):
            try:
                params = self._params_from_meta(run_id)
                freqs = params.get_frequencies()
                if len(freqs) < 2:
                    return JSONResponse({"ok": False, "error": "Need at least two frequencies"}, status_code=400)
                phases = []
                masks = []
                for freq in freqs:
                    phase_dir = Path(self.run_store.root) / run_id / "phase" / self.run_store._freq_tag(freq)
                    phases.append(np.load(phase_dir / "phi_wrapped.npy"))
                    mask_path = phase_dir / "mask_for_unwrap.npy"
                    if not mask_path.exists():
                        mask_path = phase_dir / "mask_clean.npy"
                    if not mask_path.exists():
                        mask_path = phase_dir / "mask.npy"
                    masks.append(np.load(mask_path))
                roi_mask = None
                roi_path = Path(self.run_store.root) / run_id / "roi" / "roi_mask.png"
                if roi_path.exists():
                    roi_mask = np.array(Image.open(roi_path)) > 0
                phi_abs, mask_unwrap, meta, residual = unwrap_multi_frequency(
                    phases,
                    masks,
                    freqs,
                    roi_mask=roi_mask,
                    use_roi=True,
                )
                save_unwrap_outputs(
                    Path(self.run_store.root) / run_id,
                    phi_abs,
                    mask_unwrap,
                    meta,
                    f_max=max(freqs),
                    residual=residual,
                )
                return {"ok": True, "run_id": run_id, "meta": meta}
            except Exception as exc:
                log.error("Unwrap compute failed for %s: %s", run_id, exc)
                log.error(traceback.format_exc())
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @self.app.get("/api/runs/{run_id}/unwrap/preview")
        async def unwrap_preview(run_id: str):
            unwrap_dir = Path(self.run_store.root) / run_id / "unwrap"
            fixed = unwrap_dir / "phi_abs_debug_fixed.png"
            auto = unwrap_dir / "phi_abs_debug_autoscale.png"
            if fixed.exists():
                return FileResponse(fixed)
            if auto.exists():
                return FileResponse(auto)
            return JSONResponse({"error": "unwrap preview not found"}, status_code=404)

        @self.app.get("/api/runs/{run_id}/unwrap/status")
        async def unwrap_status(run_id: str):
            unwrap_dir = Path(self.run_store.root) / run_id / "unwrap"
            exists = (unwrap_dir / "phi_abs_debug_fixed.png").exists() or (unwrap_dir / "phi_abs_debug_autoscale.png").exists()
            return {"exists": exists}

        @self.app.post("/api/calibration/sessions")
        async def calibration_create_session():
            session = self.calibration.create_session()
            # Pre-apply calibration camera controls so preview/captures are not underexposed.
            calib_cfg = self.config.get("calibration", {}) or {}
            cam_cfg = calib_cfg.get("camera", {}) or {}
            try:
                self.controller.capture_single_frame(
                    flush_frames=0,
                    exposure_us=cam_cfg.get("exposure_us"),
                    analogue_gain=cam_cfg.get("analogue_gain"),
                    awb_enable=cam_cfg.get("awb_enable"),
                    settle_ms=int(cam_cfg.get("settle_ms", 120)),
                )
            except Exception:
                pass
            return {"ok": True, "session": session}

        @self.app.post("/api/calibration/camera/apply")
        async def calibration_apply_camera():
            calib_cfg = self.config.get("calibration", {}) or {}
            cam_cfg = calib_cfg.get("camera", {}) or {}
            try:
                frame = self.controller.capture_single_frame(
                    flush_frames=0,
                    exposure_us=cam_cfg.get("exposure_us"),
                    analogue_gain=cam_cfg.get("analogue_gain"),
                    awb_enable=cam_cfg.get("awb_enable"),
                    settle_ms=int(cam_cfg.get("settle_ms", 120)),
                )
                if frame.ndim == 3:
                    gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
                else:
                    gray = frame
                return {
                    "ok": True,
                    "mean_dn": float(np.mean(gray)),
                    "min_dn": float(np.min(gray)),
                    "max_dn": float(np.max(gray)),
                }
            except RuntimeError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)
            except Exception as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @self.app.get("/api/calibration/sessions")
        async def calibration_list_sessions():
            return {"ok": True, "sessions": self.calibration.list_sessions()}

        @self.app.get("/api/calibration/sessions/{session_id}")
        async def calibration_get_session(session_id: str):
            try:
                return {"ok": True, "session": self.calibration.load_session(session_id)}
            except FileNotFoundError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)

        @self.app.post("/api/calibration/sessions/{session_id}/capture")
        async def calibration_capture(session_id: str):
            try:
                calib_cfg = self.config.get("calibration", {}) or {}
                flush_frames = int(calib_cfg.get("capture_flush_frames", 1))
                cam_cfg = calib_cfg.get("camera", {}) or {}
                frame = self.controller.capture_single_frame(
                    flush_frames=flush_frames,
                    exposure_us=cam_cfg.get("exposure_us"),
                    analogue_gain=cam_cfg.get("analogue_gain"),
                    awb_enable=cam_cfg.get("awb_enable"),
                    settle_ms=int(cam_cfg.get("settle_ms", 120)),
                )
                result = self.calibration.capture(session_id, frame)
                session = self.calibration.load_session(session_id)
                found_count = int(sum(1 for c in session.get("captures", []) if c.get("found")))
                return {
                    "ok": True,
                    "capture": result["record"],
                    "detection": result["detection"],
                    "found_count": found_count,
                    "total_count": len(session.get("captures", [])),
                }
            except FileNotFoundError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
            except RuntimeError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)
            except Exception as exc:
                log.error("Calibration capture failed: %s", exc)
                log.error(traceback.format_exc())
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @self.app.post("/api/calibration/sessions/{session_id}/calibrate")
        async def calibration_run(session_id: str):
            try:
                intrinsics = self.calibration.calibrate(session_id)
                return {
                    "ok": True,
                    "intrinsics": intrinsics,
                    "rms": float(intrinsics.get("rms", 0.0)),
                }
            except ValueError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
            except FileNotFoundError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
            except Exception as exc:
                log.error("Calibration failed: %s", exc)
                log.error(traceback.format_exc())
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @self.app.get("/api/calibration/sessions/{session_id}/captures/{capture_id}/image")
        async def calibration_capture_image(session_id: str, capture_id: str):
            p = self.calibration.capture_image_path(session_id, capture_id)
            if not p.exists():
                return JSONResponse({"ok": False, "error": "image not found"}, status_code=404)
            return FileResponse(p)

        @self.app.get("/api/calibration/sessions/{session_id}/captures/{capture_id}/overlay")
        async def calibration_capture_overlay(session_id: str, capture_id: str):
            p = self.calibration.overlay_image_path(session_id, capture_id)
            if not p.exists():
                return JSONResponse({"ok": False, "error": "overlay not found"}, status_code=404)
            return FileResponse(p)

        @self.app.get("/api/calibration/sessions/{session_id}/captures/{capture_id}/detection")
        async def calibration_capture_detection(session_id: str, capture_id: str):
            p = self.calibration.detection_path(session_id, capture_id)
            if not p.exists():
                return JSONResponse({"ok": False, "error": "detection not found"}, status_code=404)
            return JSONResponse(json.loads(p.read_text()))

        @self.app.get("/api/calibration/sessions/{session_id}/intrinsics")
        async def calibration_intrinsics(session_id: str):
            p = self.calibration.intrinsics_path(session_id)
            if not p.exists():
                return JSONResponse({"ok": False, "error": "intrinsics not found"}, status_code=404)
            return JSONResponse(json.loads(p.read_text()))

        # Projector calibration mode split:
        #   Capture mode: per-view capture/UV/corner sampling only (fast feedback).
        #   Solve mode: explicit batch stereo solve + reports under results/.
        @self.app.post("/api/calibration/projector/session/start")
        async def projector_calibration_start_session():
            try:
                root = self._calibration_root()
                session_id = create_projector_session(root, self.config)
                self._projector_solve_status[session_id] = {
                    "session_id": session_id,
                    "state": "idle",
                    "started_at": None,
                    "ended_at": None,
                    "error": None,
                    "result": None,
                }
                return {"ok": True, "session_id": session_id}
            except Exception as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @self.app.post("/api/calibration/projector/session/new")
        async def projector_calibration_new_session():
            # Backward-compatible alias.
            return await projector_calibration_start_session()

        @self.app.get("/api/calibration/projector/sessions")
        async def projector_calibration_list_sessions():
            root = self._calibration_root()
            sessions_dir = root / "projector" / "sessions"
            sessions: list[dict[str, Any]] = []
            if sessions_dir.exists():
                for p in sorted(sessions_dir.iterdir(), reverse=True):
                    if not p.is_dir():
                        continue
                    sess_path = p / "session.json"
                    if not sess_path.exists():
                        continue
                    try:
                        sess = json.loads(sess_path.read_text())
                    except Exception:
                        continue
                    views = sess.get("views", []) or []
                    valid = sum(1 for v in views if v.get("status") == "valid")
                    sessions.append({
                        "session_id": str(sess.get("session_id", p.name)),
                        "created_at": sess.get("created_at"),
                        "views_total": int(len(views)),
                        "views_valid": int(valid),
                    })
            return {"ok": True, "sessions": sessions}

        @self.app.post("/api/calibration/projector/illumination")
        async def projector_calibration_illumination(payload: Dict[str, Any] | None = None):
            payload = payload or {}
            enabled = bool(payload.get("enabled", True))
            pcal = self.config.get("projector_calibration", {}) or {}
            cap = pcal.get("capture", {}) or {}
            dn = int(payload.get("dn", cap.get("checkerboard_white_dn", 230)))
            try:
                async with self._projector_capture_lock:
                    await self._run_projector_display(
                        self.controller.set_projector_calibration_light,
                        enabled=enabled,
                        dn=dn,
                    )
            except RuntimeError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)
            except Exception as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
            return {"ok": True, "enabled": enabled, "dn": dn}

        @self.app.get("/api/calibration/projector/session/{session_id}")
        async def projector_calibration_get_session(session_id: str):
            root = self._calibration_root()
            try:
                payload = build_projector_session_payload(root, session_id, list_projector_views)
            except Exception as exc:
                err = str(exc)
                status = 404 if "not found" in err.lower() else 500
                return JSONResponse({"ok": False, "error": err}, status_code=status)
            payload["solve_status"] = self._projector_session_status(session_id)
            return {"ok": True, **payload}

        @self.app.get("/api/calibration/projector/session/{session_id}/status")
        async def projector_calibration_session_status(session_id: str):
            root = self._calibration_root()
            sdir = root / "projector" / "sessions" / session_id
            if not sdir.exists():
                return JSONResponse({"ok": False, "error": "session not found"}, status_code=404)
            return {"ok": True, "status": self._projector_session_status(session_id)}

        @self.app.get("/api/calibration/projector/session/{session_id}/views")
        async def projector_calibration_session_views(session_id: str):
            root = self._calibration_root()
            sdir = root / "projector" / "sessions" / session_id
            if not sdir.exists():
                return JSONResponse({"ok": False, "error": "session not found"}, status_code=404)
            views = list_projector_views(sdir)
            valid = sum(1 for v in views if v.get("status") == "valid")
            payload = {
                "ok": True,
                "session_id": session_id,
                "views_total": int(len(views)),
                "views_valid": int(valid),
                "views": views,
            }
            return JSONResponse(_json_safe(payload))

        @self.app.get("/api/calibration/projector/session/{session_id}/results")
        async def projector_calibration_session_results(session_id: str):
            root = self._calibration_root()
            sdir = root / "projector" / "sessions" / session_id
            if not sdir.exists():
                return JSONResponse({"ok": False, "error": "session not found"}, status_code=404)
            res_dir = sdir / "results"
            stereo_candidates = [
                "stereo_dense_pruned.json",
                "stereo_dense.json",
                "stereo_refined_pruned.json",
                "stereo_refined.json",
                "stereo_pruned.json",
                "stereo.json",
            ]
            stereo_path = None
            for name in stereo_candidates:
                p = res_dir / name
                if p.exists():
                    stereo_path = p
                    break
            coverage_path = res_dir / "coverage.json"
            session_report_path = res_dir / "session_report.json"
            payload: Dict[str, Any] = {
                "ok": True,
                "session_id": session_id,
                "has_results": bool(stereo_path is not None),
                "solve_status": self._projector_session_status(session_id),
            }
            if stereo_path is not None:
                payload["stereo"] = json.loads(stereo_path.read_text())
                payload["stereo_path"] = str(stereo_path.relative_to(sdir))
            if coverage_path.exists():
                payload["coverage"] = json.loads(coverage_path.read_text())
            if session_report_path.exists():
                payload["session_report"] = json.loads(session_report_path.read_text())
            return JSONResponse(_json_safe(payload))

        @self.app.post("/api/calibration/projector/session/{session_id}/capture")
        async def projector_calibration_capture_view(session_id: str):
            root = self._calibration_root()
            sdir = root / "projector" / "sessions" / session_id
            if not sdir.exists():
                return JSONResponse({"ok": False, "error": "session not found"}, status_code=404)
            try:
                sess = json.loads((sdir / "session.json").read_text())
                cov = (sess.get("coverage_map", {}) or {})
                cond_cfg = ((sess.get("config", {}) or {}).get("conditioning", {}) or {})
                if bool(cond_cfg.get("stop_when_sufficient", True)) and bool(cov.get("sufficient", False)):
                    return JSONResponse(
                        {
                            "ok": False,
                            "error": "Coverage conditioning target reached; run Solve or disable stop_when_sufficient.",
                        },
                        status_code=409,
                    )
            except Exception:
                pass
            if self._projector_session_status(session_id).get("state") == "solving":
                return JSONResponse(
                    {
                        "ok": False,
                        "error": "Solve is in progress; wait for completion before capturing more views.",
                    },
                    status_code=409,
                )
            try:
                async with self._projector_capture_lock:
                    self._projector_solve_status[session_id] = {
                        **self._projector_session_status(session_id),
                        "state": "capturing",
                        "error": None,
                    }
                    summary = await self._run_projector_display(
                        capture_projector_view,
                        session_id,
                        self.config,
                        self.controller,
                        root_dir=root,
                    )
                self._projector_solve_status[session_id] = {
                    **self._projector_session_status(session_id),
                    "state": "idle",
                    "error": None,
                }
                return {"ok": True, **build_capture_response_payload(summary)}
            except Exception as exc:
                self._projector_solve_status[session_id] = {
                    **self._projector_session_status(session_id),
                    "state": "error",
                    "error": str(exc),
                }
                log.error("Projector calibration capture failed: %s", exc)
                log.error(traceback.format_exc())
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @self.app.delete("/api/calibration/projector/session/{session_id}/view/{view_id}")
        async def projector_calibration_delete_view(session_id: str, view_id: str):
            root = self._calibration_root()
            sdir = root / "projector" / "sessions" / session_id
            if not sdir.exists():
                return JSONResponse({"ok": False, "error": "session not found"}, status_code=404)
            try:
                delete_projector_view(sdir, view_id)
                return {"ok": True}
            except Exception as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @self.app.post("/api/calibration/projector/session/{session_id}/solve")
        async def projector_calibration_solve(session_id: str, payload: Dict[str, Any] | None = None):
            root = self._calibration_root()
            sdir = root / "projector" / "sessions" / session_id
            if not sdir.exists():
                return JSONResponse({"ok": False, "error": "session not found"}, status_code=404)
            state = self._projector_session_status(session_id)
            if state.get("state") == "solving":
                return JSONResponse({"ok": False, "error": "solve already running"}, status_code=409)
            if state.get("state") == "capturing":
                return JSONResponse({"ok": False, "error": "capture in progress"}, status_code=409)

            payload = payload or {}
            prune = bool(payload.get("prune", False))
            background = bool(payload.get("background", True))
            try:
                intr_path = self._find_camera_intrinsics_latest()
            except Exception as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)

            async def _solve_task():
                started_at = time.time()
                self._projector_solve_status[session_id] = {
                    "session_id": session_id,
                    "state": "solving",
                    "started_at": started_at,
                    "ended_at": None,
                    "error": None,
                    "result": None,
                    "prune": prune,
                }
                try:
                    async with self._projector_calibrate_lock:
                        result = await asyncio.to_thread(
                            solve_projector_session,
                            session_id,
                            self.config,
                            root_dir=root,
                            camera_intrinsics_path=intr_path,
                            prune=prune,
                        )
                    self._projector_solve_status[session_id] = {
                        "session_id": session_id,
                        "state": "done",
                        "started_at": started_at,
                        "ended_at": time.time(),
                        "error": None,
                        "result": {
                            "rms_projector_intrinsics": result.get("rms_projector_intrinsics"),
                            "rms_stereo": result.get("rms_stereo"),
                            "views_used": result.get("views_used"),
                        },
                        "prune": prune,
                    }
                    return result
                except Exception as exc:
                    self._projector_solve_status[session_id] = {
                        "session_id": session_id,
                        "state": "error",
                        "started_at": started_at,
                        "ended_at": time.time(),
                        "error": str(exc),
                        "result": None,
                        "prune": prune,
                    }
                    raise
                finally:
                    task = self._projector_solve_tasks.get(session_id)
                    if task is not None and task.done():
                        self._projector_solve_tasks.pop(session_id, None)

            if background:
                task = asyncio.create_task(_solve_task())
                self._projector_solve_tasks[session_id] = task
                return {
                    "ok": True,
                    "session_id": session_id,
                    "started": True,
                    "background": True,
                    "status": self._projector_session_status(session_id),
                }
            try:
                result = await _solve_task()
                return {"ok": True, "result": result, "status": self._projector_session_status(session_id)}
            except ValueError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
            except FileNotFoundError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
            except Exception as exc:
                log.error("Projector calibration solve failed: %s", exc)
                log.error(traceback.format_exc())
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @self.app.post("/api/calibration/projector/session/{session_id}/gamma")
        async def projector_calibration_gamma(session_id: str):
            root = self._calibration_root()
            sdir = root / "projector" / "sessions" / session_id
            if not sdir.exists():
                return JSONResponse({"ok": False, "error": "session not found"}, status_code=404)
            try:
                async with self._projector_capture_lock:
                    result = await self._run_projector_display(
                        calibrate_projector_gamma,
                        session_id,
                        self.config,
                        self.controller,
                        root_dir=root,
                    )
                return {"ok": True, "result": result}
            except Exception as exc:
                log.error("Projector gamma calibration failed: %s", exc)
                log.error(traceback.format_exc())
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @self.app.post("/api/calibration/projector/session/{session_id}/calibrate")
        async def projector_calibration_calibrate(session_id: str):
            # Backward-compatible alias (synchronous solve).
            return await projector_calibration_solve(session_id, payload={"background": False})

        @self.app.get("/api/calibration/projector/session/{session_id}/view/{view_id}/image")
        async def projector_calibration_view_image(session_id: str, view_id: str):
            root = self._calibration_root()
            p = root / "projector" / "sessions" / session_id / "views" / view_id / "camera.png"
            if not p.exists():
                return JSONResponse({"ok": False, "error": "image not found"}, status_code=404)
            return FileResponse(p)

        @self.app.get("/api/calibration/projector/session/{session_id}/view/{view_id}/overlay")
        async def projector_calibration_view_overlay(session_id: str, view_id: str):
            root = self._calibration_root()
            base = root / "projector" / "sessions" / session_id / "views" / view_id
            candidates = [
                base / "corner_validity_overlay.png",
                base / "overlay.png",
                base / "view_quality_overlay.png",
                base / "view_charuco_overlay.png",
                base / "camera_overlay.png",
                base / "camera.png",
            ]
            p = None
            for cand in candidates:
                if cand.exists():
                    p = cand
                    break
            if p is None:
                return JSONResponse({"ok": False, "error": "overlay not found"}, status_code=404)
            return FileResponse(p)

        @self.app.get("/api/calibration/projector/session/{session_id}/view/{view_id}/diag")
        async def projector_calibration_view_diag(session_id: str, view_id: str):
            root = self._calibration_root()
            p = root / "projector" / "sessions" / session_id / "views" / view_id / "view_diag.json"
            if not p.exists():
                return JSONResponse({"ok": False, "error": "view_diag not found"}, status_code=404)
            return JSONResponse(json.loads(p.read_text()))

        @self.app.get("/api/calibration/projector/session/{session_id}/view/{view_id}/uv_preview")
        async def projector_calibration_view_uv_preview(session_id: str, view_id: str):
            root = self._calibration_root()
            p = root / "projector" / "sessions" / session_id / "views" / view_id / "uv" / "uv_overlay.png"
            if not p.exists():
                return JSONResponse({"ok": False, "error": "uv overlay not found"}, status_code=404)
            return FileResponse(p)

        @self.app.get("/api/calibration/projector/session/{session_id}/coverage.png")
        async def projector_calibration_coverage_png(session_id: str):
            root = self._calibration_root()
            p = root / "projector" / "sessions" / session_id / "results" / "coverage.png"
            if not p.exists():
                return JSONResponse({"ok": False, "error": "coverage heatmap not found"}, status_code=404)
            return FileResponse(p)

        @self.app.post("/api/calibration/projector_v2/session/new")
        async def projector_calibration_v2_new_session():
            try:
                payload = create_projector_v2_session(self.config)
                return JSONResponse({"ok": True, **_json_safe(payload)})
            except FileNotFoundError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
            except Exception as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @self.app.get("/api/calibration/projector_v2/sessions")
        async def projector_calibration_v2_list_sessions():
            try:
                payload = list_projector_v2_sessions(self.config)
                return JSONResponse({"ok": True, **_json_safe(payload)})
            except Exception as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @self.app.get("/api/calibration/projector_v2/session/{session_id}")
        async def projector_calibration_v2_get_session(session_id: str):
            try:
                payload = get_projector_v2_session(self.config, session_id)
                return JSONResponse({"ok": True, **_json_safe(payload)})
            except FileNotFoundError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
            except Exception as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @self.app.post("/api/calibration/projector_v2/session/{session_id}/capture")
        async def projector_calibration_v2_capture(session_id: str):
            try:
                async with self._projector_capture_lock:
                    payload = await self._run_projector_display(
                        capture_projector_v2_view,
                        self.config,
                        self.controller,
                        session_id,
                    )
                return JSONResponse({"ok": True, **_json_safe(payload)})
            except FileNotFoundError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
            except ValueError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
            except Exception as exc:
                log.error("Projector calibration v2 capture failed: %s", exc)
                log.error(traceback.format_exc())
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @self.app.post("/api/calibration/projector_v2/session/{session_id}/delete_view/{view_id}")
        async def projector_calibration_v2_delete_view(session_id: str, view_id: str):
            try:
                payload = delete_projector_v2_view(self.config, session_id, view_id)
                return JSONResponse({"ok": True, **_json_safe(payload)})
            except FileNotFoundError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
            except Exception as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @self.app.post("/api/calibration/projector_v2/session/{session_id}/reset")
        async def projector_calibration_v2_reset(session_id: str):
            try:
                payload = reset_projector_v2_session(self.config, session_id)
                return JSONResponse({"ok": True, **_json_safe(payload)})
            except FileNotFoundError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
            except Exception as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @self.app.post("/api/calibration/projector_v2/session/{session_id}/solve")
        async def projector_calibration_v2_solve(session_id: str):
            try:
                async with self._projector_calibrate_lock:
                    payload = await asyncio.to_thread(
                        solve_projector_v2_session,
                        self.config,
                        session_id,
                    )
                return JSONResponse({"ok": True, **_json_safe(payload)})
            except FileNotFoundError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
            except ValueError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
            except Exception as exc:
                log.error("Projector calibration v2 solve failed: %s", exc)
                log.error(traceback.format_exc())
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @self.app.get("/api/calibration/projector_v2/session/{session_id}/download_zip")
        async def projector_calibration_v2_download_zip(session_id: str):
            try:
                zip_path = await asyncio.to_thread(
                    download_projector_v2_session_zip,
                    self.config,
                    session_id,
                )
                return FileResponse(zip_path, filename=f"{session_id}.zip")
            except FileNotFoundError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
            except Exception as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @self.app.get("/api/calibration/projector_v2/session/{session_id}/view/{view_id}/image")
        async def projector_calibration_v2_view_image(session_id: str, view_id: str):
            p = self._calibration_root() / "projector_v2" / session_id / "views" / view_id / "camera.png"
            if not p.exists():
                return JSONResponse({"ok": False, "error": "image not found"}, status_code=404)
            return FileResponse(p)

        @self.app.get("/api/calibration/projector_v2/session/{session_id}/view/{view_id}/overlay")
        async def projector_calibration_v2_view_overlay(session_id: str, view_id: str):
            base = self._calibration_root() / "projector_v2" / session_id / "views" / view_id
            candidates = [base / "overlay.png", base / "camera.png"]
            for cand in candidates:
                if cand.exists():
                    return FileResponse(cand)
            return JSONResponse({"ok": False, "error": "overlay not found"}, status_code=404)

        @self.app.get("/api/calibration/projector_v2/session/{session_id}/view/{view_id}/uv_overlay")
        async def projector_calibration_v2_view_uv_overlay(session_id: str, view_id: str):
            p = self._calibration_root() / "projector_v2" / session_id / "views" / view_id / "uv" / "uv_overlay.png"
            if not p.exists():
                return JSONResponse({"ok": False, "error": "uv overlay not found"}, status_code=404)
            return FileResponse(p)

        @self.app.get("/api/calibration/projector_v2/session/{session_id}/plot/{plot_name}")
        async def projector_calibration_v2_plot(session_id: str, plot_name: str):
            allowed = {"reproj_cam.png", "reproj_proj.png", "coverage.png", "residual_hist.png"}
            if plot_name not in allowed:
                return JSONResponse({"ok": False, "error": "invalid plot name"}, status_code=400)
            p = self._calibration_root() / "projector_v2" / session_id / "solve" / "plots" / plot_name
            if not p.exists():
                return JSONResponse({"ok": False, "error": "plot not found"}, status_code=404)
            return FileResponse(p)

        @self.app.get("/api/reconstruction/runs")
        async def reconstruction_runs():
            runs: list[dict[str, Any]] = []
            for meta in self.run_store.list_runs():
                run_id = meta.run_id
                run_dir = Path(self.run_store.root) / run_id
                if not (run_dir / "projector_uv" / "u.npy").exists():
                    continue
                runs.append(
                    {
                        "run_id": run_id,
                        "status": meta.status,
                        "started_at": meta.started_at,
                        "has_reconstruction": (run_dir / "reconstruction" / "reconstruction_meta.json").exists(),
                    }
                )
            return {"ok": True, "runs": runs}

        @self.app.post("/api/reconstruction/run")
        async def reconstruction_run(payload: Dict[str, Any]):
            run_id = str((payload or {}).get("run_id", "")).strip()
            if not run_id:
                return JSONResponse({"ok": False, "error": "run_id is required"}, status_code=400)
            run_dir = Path(self.run_store.root) / run_id
            if not run_dir.exists():
                return JSONResponse({"ok": False, "error": f"run not found: {run_id}"}, status_code=404)
            if not (run_dir / "projector_uv" / "u.npy").exists():
                return JSONResponse({"ok": False, "error": "run does not contain projector_uv outputs"}, status_code=400)
            try:
                summary = await asyncio.to_thread(self._run_reconstruction, run_id)
                return {"ok": True, "run_id": run_id, "summary": summary}
            except FileNotFoundError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
            except ValueError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
            except Exception as exc:
                log.error("Reconstruction failed for %s: %s", run_id, exc)
                log.error(traceback.format_exc())
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @self.app.get("/api/reconstruction/status")
        async def reconstruction_status(run_id: str):
            run_dir = Path(self.run_store.root) / run_id
            meta_path = run_dir / "reconstruction" / "reconstruction_meta.json"
            if not meta_path.exists():
                return {"ok": True, "exists": False}
            return {"ok": True, "exists": True, "meta": json.loads(meta_path.read_text())}

        @self.app.get("/api/reconstruction/outputs")
        async def reconstruction_outputs(run_id: str):
            run_dir = Path(self.run_store.root) / run_id
            rec_dir = run_dir / "reconstruction"
            if not rec_dir.exists():
                return {"ok": True, "exists": False, "files": []}
            files = []
            for p in sorted(rec_dir.rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(rec_dir).as_posix()
                files.append(
                    {
                        "name": rel,
                        "size_bytes": int(p.stat().st_size),
                        "url": f"/api/reconstruction/file/{run_id}/{rel}",
                    }
                )
            meta = {}
            meta_path = rec_dir / "reconstruction_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
            return {"ok": True, "exists": True, "files": files, "meta": meta}

        @self.app.get("/api/reconstruction/file/{run_id}/{file_path:path}")
        async def reconstruction_file(run_id: str, file_path: str):
            run_dir = Path(self.run_store.root) / run_id
            rec_dir = (run_dir / "reconstruction").resolve()
            target = (rec_dir / file_path).resolve()
            try:
                target.relative_to(rec_dir)
            except ValueError:
                return JSONResponse({"ok": False, "error": "invalid path"}, status_code=400)
            if not target.exists() or not target.is_file():
                return JSONResponse({"ok": False, "error": "file not found"}, status_code=404)
            return FileResponse(target)

        @self.app.websocket("/ws/preview")
        async def preview_ws(websocket: WebSocket):
            await websocket.accept()
            last_sent = None
            try:
                while True:
                    latest = self.controller.preview.get_latest()
                    if latest is not None:
                        data, meta = latest
                        if data is not last_sent:
                            await websocket.send_bytes(data)
                            last_sent = data
                    await asyncio.sleep(0.05)
            except WebSocketDisconnect:
                return

        @self.app.get("/")
        async def index():
            return FileResponse(static_dir / "index.html")

        @self.app.get("/calibration")
        async def calibration_page():
            page = (static_dir / "calibration.html").read_text()
            return HTMLResponse(
                page,
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                },
            )

        @self.app.get("/projector-calibration")
        async def projector_calibration_page():
            page = (static_dir / "projector_calibration.html").read_text()
            return HTMLResponse(
                page,
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                },
            )

        @self.app.get("/calibration/projector-v2")
        async def projector_calibration_v2_page():
            page = (static_dir / "projector_calibration_v2.html").read_text()
            return HTMLResponse(
                page,
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                },
            )

        @self.app.get("/reconstruction")
        async def reconstruction_page():
            page = (static_dir / "reconstruction.html").read_text()
            return HTMLResponse(
                page,
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                },
            )

        self.app.mount("/static", StaticFiles(directory=static_dir), name="static")

    async def _run_pipeline(self, force: bool = False) -> None:
        async with self._pipeline_lock:
            preview_params = getattr(self.controller, "_preview_params", None)
            preview_thread = getattr(self.controller, "_preview_thread", None)
            preview_was_running = bool(preview_thread and preview_thread.is_alive())
            try:
                if preview_was_running:
                    # cmd_pipeline_run_3d creates its own camera instance; release this one first.
                    self.controller.stop_preview_loop()

                run_root = Path(self.run_store.root)
                t0 = time.time()
                rc = await asyncio.to_thread(
                    cmd_pipeline_run_3d,
                    argparse.Namespace(
                        force=force,
                        print_hints=True,
                    ),
                )
                latest: tuple[float, str] | None = None
                for run_dir in run_root.iterdir():
                    if not run_dir.is_dir():
                        continue
                    if not (run_dir / "meta.json").exists():
                        continue
                    mtime = run_dir.stat().st_mtime
                    if mtime < t0:
                        continue
                    if latest is None or mtime > latest[0]:
                        latest = (mtime, run_dir.name)
                if latest is not None:
                    self._pipeline_run_id = latest[1]
                if rc != 0:
                    raise RuntimeError(f"pipeline-run-3d failed rc={rc}")
                self._pipeline_state = "idle"
            except Exception as exc:
                self._pipeline_state = "error"
                self._pipeline_error = str(exc)
            finally:
                if preview_was_running and preview_params is not None:
                    try:
                        self.controller.start_preview_loop(preview_params)
                    except Exception:
                        pass

    def _build_params(self, payload: Dict[str, Any]) -> ScanParams:
        defaults = self.config.get("scan", {})
        pat_defaults = self.config.get("patterns", {})
        width = int(payload.get("width", defaults.get("width", 1024)))
        height = int(payload.get("height", defaults.get("height", 768)))
        frequencies = payload.get("frequencies", defaults.get("frequencies"))
        if frequencies:
            frequencies = [float(f) for f in frequencies]
        else:
            frequencies = [float(payload.get("frequency", defaults.get("frequency", 8.0)))]
        params = ScanParams(
            n_steps=int(payload.get("n_steps", defaults.get("n_steps", 4))),
            frequency=float(frequencies[0]),
            frequencies=frequencies,
            orientation=str(payload.get("orientation", defaults.get("orientation", "vertical"))),
            brightness=float(payload.get("brightness", defaults.get("brightness", 1.0))),
            resolution=(width, height),
            settle_ms=int(payload.get("settle_ms", defaults.get("settle_ms", 50))),
            save_patterns=bool(payload.get("save_patterns", defaults.get("save_patterns", False))),
            preview_fps=float(payload.get("preview_fps", defaults.get("preview_fps", 10.0))),
            projector_screen_index=payload.get("projector_screen_index"),
            phase_convention=str(payload.get("phase_convention", defaults.get("phase_convention", "atan2(-S,C)"))),
            exposure_us=payload.get("exposure_us"),
            analogue_gain=payload.get("analogue_gain"),
            awb_mode=payload.get("awb_mode"),
            awb_enable=payload.get("awb_enable"),
            ae_enable=payload.get("ae_enable", defaults.get("ae_enable")),
            iso=payload.get("iso"),
            brightness_offset=float(payload.get("brightness_offset", pat_defaults.get("brightness_offset", defaults.get("brightness_offset", 0.45)))),
            contrast=float(payload.get("contrast", pat_defaults.get("contrast", defaults.get("contrast", 0.6)))),
            min_intensity=float(payload.get("min_intensity", pat_defaults.get("min_intensity", defaults.get("min_intensity", 0.10)))),
            frequency_semantics=str(payload.get("frequency_semantics", defaults.get("frequency_semantics", "cycles_across_dimension"))),
            phase_origin_rad=float(payload.get("phase_origin_rad", defaults.get("phase_origin_rad", 0.0))),
            auto_normalise=bool(payload.get("auto_normalise", defaults.get("auto_normalise", True))),
            expert_mode=bool(payload.get("expert_mode", False)),
        )
        return params

    def _build_phase_thresholds(self) -> PhaseThresholds:
        cfg = self.config.get("phase", {})
        perc = cfg.get("debug_percentiles", [1, 99])
        return PhaseThresholds(
            sat_low=float(cfg.get("sat_low", 0)),
            sat_high=float(cfg.get("sat_high", 250)),
            B_thresh=float(cfg.get("B_thresh", 7)),
            A_min=float(cfg.get("A_min", 15)),
            debug_percentiles=(float(perc[0]), float(perc[1])),
        )

    def _build_roi_config(self) -> ObjectRoiConfig:
        cfg = self.config.get("roi", {})
        return ObjectRoiConfig(
            downscale_max_w=int(cfg.get("downscale_max_w", 640)),
            blur_ksize=int(cfg.get("blur_ksize", 7)),
            black_bg_percentile=float(cfg.get("black_bg_percentile", 70.0)),
            threshold_offset=float(cfg.get("threshold_offset", 10.0)),
            min_area_ratio=float(cfg.get("min_area_ratio", 0.01)),
            max_area_ratio=float(cfg.get("max_area_ratio", 0.95)),
            close_iters=int(cfg.get("close_iters", 2)),
            open_iters=int(cfg.get("open_iters", 1)),
            fill_holes=bool(cfg.get("fill_holes", True)),
            ref_method=str(cfg.get("ref_method", "median_over_frames")),
        )

    def _params_from_meta(self, run_id: str) -> ScanParams:
        run_dir = Path(self.run_store.root) / run_id
        meta = json.loads((run_dir / "meta.json").read_text())
        p = meta.get("params", {})
        width, height = p.get("resolution", (1024, 768))
        frequencies = p.get("frequencies")
        if not frequencies:
            frequencies = [float(p.get("frequency", 8.0))]
        return ScanParams(
            n_steps=int(p.get("n_steps", p.get("N", 4))),
            frequency=float(frequencies[0]),
            frequencies=[float(f) for f in frequencies],
            orientation=str(p.get("orientation", "vertical")),
            brightness=float(p.get("brightness", 1.0)),
            resolution=(int(width), int(height)),
            settle_ms=int(p.get("settle_ms", 50)),
            save_patterns=bool(p.get("save_patterns", False)),
            preview_fps=float(p.get("preview_fps", 10.0)),
            projector_screen_index=p.get("projector_screen_index"),
            phase_convention=str(p.get("phase_convention", "atan2(-S,C)")),
            exposure_us=p.get("exposure_us"),
            analogue_gain=p.get("analogue_gain"),
            awb_mode=p.get("awb_mode"),
            awb_enable=p.get("awb_enable"),
            ae_enable=p.get("ae_enable"),
            iso=p.get("iso"),
            brightness_offset=float(p.get("brightness_offset", 0.5)),
            contrast=float(p.get("contrast", 1.0)),
            min_intensity=float(p.get("min_intensity", 0.10)),
            frequency_semantics=str(p.get("frequency_semantics", "cycles_across_dimension")),
            phase_origin_rad=float(p.get("phase_origin_rad", 0.0)),
            auto_normalise=bool(p.get("auto_normalise", True)),
            expert_mode=bool(p.get("expert_mode", False)),
            quality_retry_count=int(p.get("quality_retry_count", 0)),
        )

    def _calibration_root(self) -> Path:
        cfg = self.config.get("calibration", {}) or {}
        return Path(str(cfg.get("root", "data/calibration")))

    def _camera_calibration_root(self) -> Path:
        cfg = self.config.get("calibration", {}) or {}
        camera_root = cfg.get("camera_root")
        if camera_root:
            return Path(str(camera_root))
        return self._calibration_root() / "camera"

    def _find_camera_intrinsics_latest(self) -> Path:
        camera_root = self._camera_calibration_root()
        candidates = [
            camera_root / "intrinsics_latest.json",
            # Legacy fallbacks:
            self._calibration_root() / "camera_intrinsics" / "intrinsics_latest.json",
            self._calibration_root() / "intrinsics_latest.json",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(
            "Camera intrinsics not found. Expected "
            "data/calibration/camera/intrinsics_latest.json"
        )

    def _find_projector_stereo_latest(self) -> Path:
        root = self._calibration_root()
        candidates = [
            root / "projector" / "stereo_latest.json",
            root / "projector" / "results" / "stereo_latest.json",  # legacy
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(
            "Projector stereo calibration not found. Expected "
            "data/calibration/projector/stereo_latest.json"
        )

    def _projector_session_status(self, session_id: str) -> Dict[str, Any]:
        status = dict(self._projector_solve_status.get(session_id, {}))
        if not status:
            status = {
                "session_id": session_id,
                "state": "idle",
                "started_at": None,
                "ended_at": None,
                "error": None,
                "result": None,
            }
        task = self._projector_solve_tasks.get(session_id)
        if task is not None:
            if task.done():
                self._projector_solve_tasks.pop(session_id, None)
            else:
                status["state"] = "solving"
        return status

    async def _run_projector_display(self, fn, *args, **kwargs):
        """
        Execute projector/display-touching work on a single dedicated thread.
        SDL/EGL contexts are thread-affine; crossing threads can crash the process.
        """
        loop = asyncio.get_running_loop()
        call = functools.partial(fn, *args, **kwargs)
        return await loop.run_in_executor(self._projector_display_executor, call)

    def _run_reconstruction(self, run_id: str) -> dict[str, Any]:
        run_dir = Path(self.run_store.root) / run_id
        camera_intr = self._find_camera_intrinsics_latest()
        stereo = self._find_projector_stereo_latest()
        model = load_stereo_model(camera_intr, stereo)
        result = reconstruct_uv_run(
            run_dir,
            model,
            recon_cfg=self.config.get("reconstruction", {}) or {},
        )
        out_dir = run_dir / "reconstruction"
        meta = save_reconstruction_outputs(
            out_dir,
            result,
            recon_cfg=self.config.get("reconstruction", {}) or {},
        )
        return {
            "run_id": run_id,
            "output_dir": str(out_dir),
            "camera_intrinsics": str(camera_intr),
            "projector_stereo": str(stereo),
            "valid_recon_points": int(meta.get("valid_recon_points", 0)),
            "exported_points": int(meta.get("exported_points", 0)),
            "depth_median_m": meta.get("depth_median_m"),
            "reproj_err_cam_median": meta.get("reproj_err_cam_median"),
            "reproj_err_proj_median": meta.get("reproj_err_proj_median"),
        }

    def _select_phase_dir(self, phase_dir: Path) -> Path:
        if not phase_dir.exists():
            return phase_dir
        if (phase_dir / "phi_debug_fixed.png").exists() or (phase_dir / "phi_debug_autoscale.png").exists():
            return phase_dir
        subdirs = [p for p in phase_dir.iterdir() if p.is_dir() and p.name.startswith("f_")]
        if not subdirs:
            return phase_dir
        def _tag_to_freq(name: str) -> float:
            val = name.replace("f_", "").replace("p", ".")
            try:
                return float(val)
            except Exception:
                return 0.0
        subdirs.sort(key=lambda p: _tag_to_freq(p.name))
        return subdirs[-1]
