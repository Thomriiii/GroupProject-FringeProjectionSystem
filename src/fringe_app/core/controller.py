"""Scan controller coordinating display, camera, and storage."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any, Literal

from fringe_app.core.models import ScanParams, RunMeta
from fringe_app.core.logging import setup_logging
from fringe_app.patterns.generator import FringePatternGenerator
from fringe_app.display.pygame_display import PygameProjectorDisplay
from fringe_app.camera.base import CameraBase
from fringe_app.io.run_store import RunStore
from fringe_app.web.preview import PreviewBroadcaster


State = Literal["IDLE", "RUNNING", "STOPPING", "ERROR"]


class ScanController:
    """
    Orchestrates pattern display, camera capture, and filesystem output.
    """

    def __init__(
        self,
        display: PygameProjectorDisplay,
        camera: CameraBase,
        generator: FringePatternGenerator,
        store: RunStore,
        preview: PreviewBroadcaster,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.display = display
        self.camera = camera
        self.generator = generator
        self.store = store
        self.preview = preview
        self.config = config or {}
        self.log = setup_logging()

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._run_id_ready = threading.Event()
        self._preview_thread: Optional[threading.Thread] = None
        self._preview_stop = threading.Event()
        self._preview_params: Optional[ScanParams] = None
        self._preview_running = False
        self._camera_lock = threading.Lock()
        self._camera_started = False

        self._state: State = "IDLE"
        self._run_id: Optional[str] = None
        self._step_index = 0
        self._total_steps = 0
        self._last_error: Optional[str] = None
        self._started_at: Optional[str] = None
        self._ended_at: Optional[str] = None

    def start_scan(self, params: ScanParams) -> str:
        with self._lock:
            if self._state == "RUNNING":
                raise RuntimeError("Scan already running")
            self._state = "RUNNING"
            self._stop_event.clear()
            self._step_index = 0
            self._total_steps = 0
            self._last_error = None
            self._started_at = datetime.now().isoformat()
            self._ended_at = None
            self._run_id_ready.clear()

        self._worker = threading.Thread(target=self._scan_worker, args=(params,), daemon=True)
        self._worker.start()
        self._run_id_ready.wait(timeout=1.0)
        return self._run_id or "pending"

    def stop_scan(self) -> None:
        with self._lock:
            if self._state in ("IDLE", "ERROR"):
                return
            self._state = "STOPPING"
        self._stop_event.set()

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "run_id": self._run_id,
                "step_index": self._step_index,
                "total_steps": self._total_steps,
                "progress": self._step_index,
                "total": self._total_steps,
                "last_error": self._last_error,
                "started_at": self._started_at,
                "ended_at": self._ended_at,
            }

    def start_preview_loop(self, params: ScanParams) -> None:
        self._preview_params = params
        if self._preview_thread and self._preview_thread.is_alive():
            return
        self._preview_stop.clear()
        self._preview_thread = threading.Thread(target=self._preview_worker, daemon=True)
        self._preview_thread.start()

    def stop_preview_loop(self) -> None:
        self._preview_stop.set()
        if self._preview_thread and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=2.0)

    def _scan_worker(self, params: ScanParams) -> None:
        run_id = None
        run_dir = None
        captured = 0
        status = "error"
        error_msg = None
        try:
            device_info = {
                "camera": type(self.camera).__name__,
                "applied_controls": getattr(self.camera, "get_applied_controls", lambda: {})(),
            }
            run_id, run_dir, meta = self.store.create_run(params, device_info, True)
            with self._lock:
                self._run_id = run_id
                self._run_id_ready.set()

            screen_index = params.projector_screen_index
            if screen_index is None:
                screen_index = self.config.get("display", {}).get("screen_index")

            self.display.open(fullscreen=True, screen_index=screen_index)
            if not self._preview_running:
                with self._camera_lock:
                    self.camera.start(params)
                    self._camera_started = True

            patterns = self.generator.generate_sequence(params)
            total_steps = len(patterns)
            with self._lock:
                self._total_steps = total_steps

            for k, pattern in enumerate(patterns):
                if self._stop_event.is_set():
                    status = "stopped"
                    break

                self.display.show_gray(pattern)
                self.display.pump()
                time.sleep(params.settle_ms / 1000.0)

                with self._camera_lock:
                    try:
                        main_frame, preview_frame = self.camera.capture_pair()
                    except AttributeError:
                        main_frame = self.camera.capture()  # type: ignore[attr-defined]
                        preview_frame = main_frame

                self.store.save_capture(run_dir, k, main_frame)
                self.preview.update(preview_frame, run_id=run_id, step=k + 1, total=total_steps)

                captured += 1
                with self._lock:
                    self._step_index = k + 1

            if status != "stopped":
                status = "completed"
        except Exception as exc:
            error_msg = str(exc)
            self.log.exception("Scan failed")
            with self._lock:
                self._state = "ERROR"
                self._last_error = error_msg
        finally:
            if not self._preview_running:
                try:
                    with self._camera_lock:
                        if self._camera_started:
                            self.camera.stop()
                        self._camera_started = False
                except Exception:
                    pass
            time.sleep(0.2)
            try:
                self.display.close()
            except Exception:
                pass

            self._ended_at = datetime.now().isoformat()
            if run_dir is not None and run_id is not None:
                meta = RunMeta(
                    run_id=run_id,
                    params=params.to_dict(),
                    started_at=self._started_at or "",
                    finished_at=self._ended_at,
                    status=status,
                    error=error_msg,
                    device_info=device_info,
                    total_frames=self._total_steps,
                    saved_frames=captured,
                    preview_enabled=True,
                )
                self.store.save_meta(run_dir, meta)

            with self._lock:
                if self._state != "ERROR":
                    self._state = "IDLE"
                self._ended_at = self._ended_at

    def _preview_worker(self) -> None:
        if self._preview_params is None:
            return
        params = self._preview_params
        self._preview_running = True
        try:
            with self._camera_lock:
                if not self._camera_started:
                    self.camera.start(params)
                    self._camera_started = True
            while not self._preview_stop.is_set():
                with self._lock:
                    state = self._state
                if state != "IDLE":
                    time.sleep(0.05)
                    continue
                with self._camera_lock:
                    try:
                        _, preview_frame = self.camera.capture_pair()
                    except AttributeError:
                        preview_frame = self.camera.capture()  # type: ignore[attr-defined]
                self.preview.update(preview_frame, run_id=self._run_id or "idle", step=0, total=0)
                time.sleep(max(0.05, 1.0 / max(params.preview_fps, 1.0)))
        finally:
            with self._camera_lock:
                if self._camera_started:
                    try:
                        self.camera.stop()
                    except Exception:
                        pass
                    self._camera_started = False
            self._preview_running = False
