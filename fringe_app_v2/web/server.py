"""Flask application factory for fringe_app_v2."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import faulthandler
import signal
from typing import Any, Callable

import numpy as np
from flask import Flask

from fringe_app_v2.core.calibration import CalibrationService
from fringe_app_v2.core.camera import CameraService, build_scan_params
from fringe_app_v2.core.patterns import PatternService
from fringe_app_v2.core.projector import ProjectorService
from fringe_app_v2.pipeline.capture import capture_raw, create_pipeline_run, lock_exposure_mid_gray
from fringe_app_v2.pipeline.phase import run_phase_stage
from fringe_app_v2.pipeline.reconstruct import run_reconstruct_stage
from fringe_app_v2.pipeline.roi_stage import run_roi_stage
from fringe_app_v2.pipeline.structured_capture import run_structured_capture
from fringe_app_v2.pipeline.unwrap import run_unwrap_stage
from fringe_app_v2.utils.io import RunPaths, write_json


@dataclass(slots=True)
class PipelineState:
    state: str = "idle"
    stage: str = "idle"
    run_id: str | None = None
    run_dir: str | None = None
    error: str | None = None


class PipelineRunner:
    def __init__(
        self,
        config: dict[str, Any],
        camera: CameraService,
        projector: ProjectorService,
        patterns: PatternService,
        calibration: CalibrationService,
    ) -> None:
        self.config = config
        self.camera = camera
        self.projector = projector
        self.patterns = patterns
        self.calibration = calibration
        self.params = build_scan_params(config)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._state = PipelineState()
        self._active_run: RunPaths | None = None
        self._roi_mask: np.ndarray | None = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state.state,
                "stage": self._state.stage,
                "run_id": self._state.run_id,
                "run_dir": self._state.run_dir,
                "error": self._state.error,
            }

    def start(self, name: str, target: Callable[[], None]) -> tuple[bool, dict[str, Any]]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False, self.status()
            self._state = PipelineState(state="running", stage=name, error=None)
            self._thread = threading.Thread(target=self._run_job, args=(target,), daemon=True)
            self._thread.start()
            return True, self.status()

    def capture_roi(self) -> None:
        run = create_pipeline_run(self.config)
        self._set_run(run, "exposure_lock")
        self.camera.stop_preview()
        try:
            lock_exposure_mid_gray(run, self.camera, self.projector, self.config)
            self._set_run(run, "roi")
            frame = capture_raw(run, self.camera, name="roi_capture.png", flush_frames=1)
            self._roi_mask = run_roi_stage(run, self.camera, self.projector, self.config, source=frame)
            self._active_run = run
            self._mark_run(run, "roi_captured")
        finally:
            self._show_idle_light()
            self.camera.start_preview()

    def structured_scan(self) -> None:
        run = self._active_run or create_pipeline_run(self.config)
        self._active_run = run
        self._set_run(run, "structured")
        self.camera.stop_preview()
        try:
            if self._roi_mask is None:
                lock_exposure_mid_gray(run, self.camera, self.projector, self.config)
                frame = capture_raw(run, self.camera, name="roi_capture.png", flush_frames=1)
                self._roi_mask = run_roi_stage(run, self.camera, self.projector, self.config, source=frame)
            run_structured_capture(run, self.camera, self.projector, self.patterns, self.params, self.config)
            self._mark_run(run, "structured_captured")
        finally:
            self._show_idle_light()
            self.camera.start_preview()

    def full_pipeline(self) -> None:
        run = create_pipeline_run(self.config)
        self._active_run = run
        self._roi_mask = None
        self.camera.stop_preview()
        try:
            self._set_run(run, "exposure_lock")
            lock_exposure_mid_gray(run, self.camera, self.projector, self.config)
            self._set_run(run, "roi")
            frame = capture_raw(run, self.camera, name="roi_capture.png", flush_frames=1)
            self._roi_mask = run_roi_stage(run, self.camera, self.projector, self.config, source=frame)
            self._set_run(run, "structured")
            run_structured_capture(run, self.camera, self.projector, self.patterns, self.params, self.config)
            self.compute_from_active_run()
            self._mark_run(run, "completed")
        finally:
            self._show_idle_light()
            self.camera.start_preview()

    def compute_from_active_run(self) -> None:
        if self._active_run is None:
            raise RuntimeError("No active run available")
        run = self._active_run
        if self._roi_mask is None and (run.roi / "roi_mask.npy").exists():
            self._roi_mask = np.load(run.roi / "roi_mask.npy").astype(bool)
        self._set_run(run, "phase")
        run_phase_stage(run, self.params, self.config, self._roi_mask)
        self._set_run(run, "unwrap")
        run_unwrap_stage(run, self.config, self._roi_mask)
        self._set_run(run, "reconstruct")
        run_reconstruct_stage(run, self.calibration, self.config, self._roi_mask)

    def _run_job(self, target: Callable[[], None]) -> None:
        try:
            target()
            with self._lock:
                self._state.state = "idle"
                if self._state.stage != "completed":
                    self._state.stage = "done"
        except Exception as exc:
            with self._lock:
                self._state.state = "error"
                self._state.error = str(exc)
        finally:
            self._show_idle_light()
            self.camera.start_preview()

    def _set_run(self, run: RunPaths, stage: str) -> None:
        with self._lock:
            self._state.run_id = run.run_id
            self._state.run_dir = str(run.root)
            self._state.stage = stage

    @staticmethod
    def _mark_run(run: RunPaths, status: str) -> None:
        write_json(run.root / "run.json", {"run_id": run.run_id, "status": status, "run_dir": str(run.root)})

    def _show_idle_light(self) -> None:
        try:
            self.projector.show_idle(self.config)
        except Exception as exc:
            with self._lock:
                if self._state.error is None:
                    self._state.error = f"idle projector light failed: {exc}"


def create_app(
    config: dict[str, Any],
    camera: CameraService,
    projector: ProjectorService,
    patterns: PatternService,
    calibration: CalibrationService,
) -> Flask:
    # Enable faulthandler to dump native/Python tracebacks on fatal signals.
    try:
        faulthandler.enable(all_threads=True)
        faulthandler.register(signal.SIGSEGV, chain=False, all_threads=True)
        faulthandler.register(signal.SIGABRT, chain=False, all_threads=True)
    except Exception:
        # Best-effort; don't fail app startup if faulthandler registration isn't allowed.
        pass

    app = Flask(__name__)
    runner = PipelineRunner(config, camera, projector, patterns, calibration)
    runner._show_idle_light()
    register_routes(app, runner, camera, calibration)
    return app


def register_routes(
    app: Flask,
    runner: PipelineRunner,
    camera: CameraService,
    calibration: CalibrationService,
) -> None:
    from .routes import register_routes as _register

    _register(app, runner, camera, calibration)
