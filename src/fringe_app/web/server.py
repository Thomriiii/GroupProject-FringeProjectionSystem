"""FastAPI server for fringe app."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import logging
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from fringe_app.core.models import ScanParams
from fringe_app.core.controller import ScanController
from fringe_app.io.run_store import RunStore
from fringe_app.phase.psp import PhaseShiftProcessor, PhaseThresholds
from fringe_app.phase.masking_post import cleanup_mask
from fringe_app.phase.metrics import score_mask
from fringe_app.vision.object_roi import detect_object_roi, ObjectRoiConfig
from fringe_app.overlays.generate import overlay_masks
from fringe_app.unwrap.temporal import unwrap_multi_frequency, save_unwrap_outputs
from fringe_app.cli import cmd_phase, cmd_unwrap, cmd_score


class FringeServer:
    """FastAPI wrapper with REST and WebSocket endpoints."""

    def __init__(self, controller: ScanController, run_store: RunStore, config: Dict[str, Any]) -> None:
        self.controller = controller
        self.run_store = run_store
        self.config = config
        self._pipeline_lock = asyncio.Lock()
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
        async def start_pipeline():
            if self._pipeline_task is not None and not self._pipeline_task.done():
                return JSONResponse({"ok": False, "error": "pipeline already running"}, status_code=409)
            self._pipeline_state = "running"
            self._pipeline_error = None
            self._pipeline_run_id = None
            self._pipeline_task = asyncio.create_task(self._run_pipeline())
            return {"ok": True}

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
                    roi_res.roi_mask,
                    roi_res.bbox,
                    {
                        "cfg": _asdict(roi_cfg),
                        "debug": {
                            **roi_res.debug,
                            "ref_method": roi_cfg.ref_method,
                            "ref_stats": {
                                "min": int(np.min(ref)),
                                "mean": float(np.mean(ref)),
                                "max": int(np.max(ref)),
                            },
                        },
                    },
                )
                roi_mask_for_score = None if roi_res.debug.get("roi_fallback") else roi_res.roi_mask
                post_cfg = self.config.get("phase", {}).get("postmask_cleanup", {}) or {}
                post_enabled = bool(post_cfg.get("enabled", True))
                min_component_area = int(post_cfg.get("min_component_area", 200))
                fill_small_holes = bool(post_cfg.get("fill_small_holes", True))
                max_hole_area = int(post_cfg.get("max_hole_area", 200))
                freqs = params.get_frequencies()
                last_debug = None
                for freq in freqs:
                    images = self.run_store.iter_captures(run_id, freq=freq) if len(freqs) > 1 else self.run_store.iter_captures(run_id)
                    result = processor.compute_phase(images, params, thresholds, roi_mask=roi_mask_for_score)
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

        self.app.mount("/static", StaticFiles(directory=static_dir), name="static")

    async def _run_pipeline(self) -> None:
        async with self._pipeline_lock:
            try:
                params = self._build_params({})
                run_id = self.controller.start_scan(params)
                self._pipeline_run_id = run_id
                while True:
                    status = self.controller.get_status()
                    if status["state"] in ("IDLE", "ERROR"):
                        if status["state"] == "ERROR":
                            raise RuntimeError(status.get("error") or "scan failed")
                        break
                    await asyncio.sleep(0.2)

                rc_phase = await asyncio.to_thread(cmd_phase, argparse.Namespace(run=run_id))
                if rc_phase != 0:
                    raise RuntimeError(f"phase failed rc={rc_phase}")
                rc_unwrap = await asyncio.to_thread(
                    cmd_unwrap, argparse.Namespace(run=run_id, use_roi="auto")
                )
                if rc_unwrap != 0:
                    raise RuntimeError(f"unwrap failed rc={rc_unwrap}")
                rc_score = await asyncio.to_thread(cmd_score, argparse.Namespace(run=run_id))
                if rc_score != 0:
                    raise RuntimeError(f"score failed rc={rc_score}")
                self._pipeline_state = "idle"
            except Exception as exc:
                self._pipeline_state = "error"
                self._pipeline_error = str(exc)

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
        @self.app.post("/api/runs/{run_id}/unwrap/compute")
        async def unwrap_compute(run_id: str):
            try:
                params = self._params_from_meta(run_id)
                freqs = params.get_frequencies()
                if len(freqs) < 2:
                    return JSONResponse({"ok": False, "error": "Need at least two frequencies"}, status_code=400)
                phases = []
                masks = []
                for f in freqs:
                    phase_dir = Path(self.run_store.root) / run_id / "phase" / self.run_store._freq_tag(f)
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
                    from PIL import Image
                    roi_mask = np.array(Image.open(roi_path)) > 0
                phi_abs, mask_unwrap, meta, residual = unwrap_multi_frequency(phases, masks, freqs, roi_mask=roi_mask, use_roi=True)
                save_unwrap_outputs(Path(self.run_store.root) / run_id, phi_abs, mask_unwrap, meta, f_max=max(freqs), residual=residual)
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
