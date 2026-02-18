"""Scan controller coordinating display, camera, and storage."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any, Literal

import numpy as np

from fringe_app.core.models import ScanParams, RunMeta
from fringe_app.core.logging import setup_logging
from fringe_app.core.auto_normalise import NormaliseConfig, auto_normalise_capture
from fringe_app.core.step_sanity import StepSanityThresholds, check_step_stack
from fringe_app.patterns.generator import FringePatternGenerator
from fringe_app.display.pygame_display import PygameProjectorDisplay
from fringe_app.camera.base import CameraBase
from fringe_app.io.run_store import RunStore
from fringe_app.web.preview import PreviewBroadcaster
from fringe_app.vision.object_roi import ObjectRoiConfig, detect_object_roi, build_reference_from_stack


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

    def capture_single_frame(self, flush_frames: int = 1) -> np.ndarray:
        """
        Capture one full-resolution frame while idle.
        Used by web calibration flow.
        """
        with self._lock:
            if self._state == "RUNNING":
                raise RuntimeError("Cannot capture calibration frame while scan is running")
        params = self._preview_params
        if params is None:
            raise RuntimeError("Preview camera parameters not initialised")
        with self._camera_lock:
            if not self._camera_started:
                self.camera.start(params)
                self._camera_started = True
            for _ in range(max(0, int(flush_frames))):
                try:
                    self.camera.capture_pair()
                except Exception:
                    break
            main_frame, preview_frame = self.camera.capture_pair()
        self.preview.update(preview_frame, run_id=self._run_id or "idle", step=0, total=0)
        return main_frame

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
            warnings: list[str] = []
            if (
                not bool(params.auto_normalise)
                and params.exposure_us is None
                and (params.ae_enable is None or params.ae_enable)
            ):
                msg = "Auto-exposure enabled; clipping likely during phase shifting. Prefer manual exposure."
                self.log.warning(msg)
                warnings.append(msg)
            device_info = {
                "camera": type(self.camera).__name__,
                "applied_controls": getattr(self.camera, "get_applied_controls", lambda: {})(),
                "warnings": warnings,
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

            scan_cfg = self.config.get("scan", {})
            settle_ms = int(scan_cfg.get("settle_ms", params.settle_ms))
            settle_ms_first_step = int(scan_cfg.get("settle_ms_first_step", 350))
            settle_ms_freq_switch = int(scan_cfg.get("settle_ms_freq_switch", 350))
            freq_switch_warmup_ms = int(scan_cfg.get("freq_switch_warmup_ms", 250))
            flush_frames_per_step = int(scan_cfg.get("flush_frames_per_step", 1))
            flush_frames_on_freq_switch = int(scan_cfg.get("flush_frames_on_freq_switch", 2))
            max_freq_retries = int(scan_cfg.get("max_freq_retries", 2))
            step_thr = StepSanityThresholds(
                min_step_mean_dn=float(scan_cfg.get("step_sanity", {}).get("min_step_mean_dn", 5.0)),
                min_step_mean_ratio=float(scan_cfg.get("step_sanity", {}).get("min_step_mean_ratio", 0.15)),
                max_step_mean_ratio=float(scan_cfg.get("step_sanity", {}).get("max_step_mean_ratio", 2.5)),
                low_light_dn=float(scan_cfg.get("step_sanity", {}).get("low_light_dn", 20.0)),
                min_step_mean_dn_low_light=float(scan_cfg.get("step_sanity", {}).get("min_step_mean_dn_low_light", 1.0)),
                use_center_patch=bool(scan_cfg.get("step_sanity", {}).get("use_center_patch", True)),
                center_patch_frac=float(scan_cfg.get("step_sanity", {}).get("center_patch_frac", 0.30)),
            )
            roi_cfg_d = self.config.get("roi", {})
            roi_post_d = roi_cfg_d.get("post", {}) or {}
            roi_cfg = ObjectRoiConfig(
                downscale_max_w=int(roi_cfg_d.get("downscale_max_w", 640)),
                blur_ksize=int(roi_cfg_d.get("blur_ksize", 7)),
                black_bg_percentile=float(roi_cfg_d.get("black_bg_percentile", 70.0)),
                threshold_offset=float(roi_cfg_d.get("threshold_offset", 10.0)),
                min_area_ratio=float(roi_cfg_d.get("min_area_ratio", 0.01)),
                max_area_ratio=float(roi_cfg_d.get("max_area_ratio", 0.95)),
                close_iters=int(roi_cfg_d.get("close_iters", 2)),
                open_iters=int(roi_cfg_d.get("open_iters", 1)),
                fill_holes=bool(roi_cfg_d.get("fill_holes", True)),
                ref_method=str(roi_cfg_d.get("ref_method", "median_over_frames")),  # type: ignore[arg-type]
                post_enabled=bool(roi_post_d.get("enabled", True)),
                post_keep_largest_component=bool(roi_post_d.get("keep_largest_component", True)),
                post_fill_small_holes=bool(roi_post_d.get("fill_small_holes", True)),
                post_max_hole_area=int(roi_post_d.get("max_hole_area", 2000)),
                post_dilate_radius_px=int(roi_post_d.get("dilate_radius_px", 10)),
            )

            # Auto-normalise operating point before scanning to avoid clipping drift.
            if bool(params.auto_normalise):
                norm_cfg_d = self.config.get("normalise", {})
                norm_cfg = NormaliseConfig(
                    enabled=bool(norm_cfg_d.get("enabled", True)),
                    target_A_mean=float(norm_cfg_d.get("target_A_mean", 120.0)),
                    target_A_tolerance=float(norm_cfg_d.get("target_A_tolerance", 10.0)),
                    sat_high=int(norm_cfg_d.get("sat_high", 250)),
                    max_clip_roi=float(norm_cfg_d.get("max_clip_roi", 0.01)),
                    exposure_min_us=int(norm_cfg_d.get("exposure_min_us", 500)),
                    gain_min=float(norm_cfg_d.get("gain_min", 1.0)),
                    exposure_max_safe_us=int(norm_cfg_d.get("exposure_max_safe_us", norm_cfg_d.get("exposure_max_us", 4000))),
                    gain_max_safe=float(norm_cfg_d.get("gain_max_safe", norm_cfg_d.get("gain_max", 2.0))),
                    allow_extend=bool(norm_cfg_d.get("allow_extend", True)),
                    exposure_max_extended_us=int(norm_cfg_d.get("exposure_max_extended_us", 12000)),
                    gain_max_extended=float(norm_cfg_d.get("gain_max_extended", 4.0)),
                    extend_trigger_roi_mean_dn=float(norm_cfg_d.get("extend_trigger_roi_mean_dn", 20.0)),
                    extend_requires_clip_below=float(norm_cfg_d.get("extend_requires_clip_below", 0.0)),
                    max_iters=int(norm_cfg_d.get("max_iters", 10)),
                    allow_pattern_adjust=bool(norm_cfg_d.get("allow_pattern_adjust", True)),
                    contrast_min=float(norm_cfg_d.get("contrast_min", 0.4)),
                    contrast_max=float(norm_cfg_d.get("contrast_max", 0.8)),
                    brightness_offset_min=float(norm_cfg_d.get("brightness_offset_min", 0.40)),
                    brightness_offset_max=float(norm_cfg_d.get("brightness_offset_max", 0.55)),
                    min_intensity=float(norm_cfg_d.get("min_intensity", 0.10)),
                    warmup_ms=int(norm_cfg_d.get("warmup_ms", 250)),
                    settle_ms=int(norm_cfg_d.get("settle_ms", 200)),
                    flush_frames=int(norm_cfg_d.get("flush_frames", 2)),
                    calib_white_dn=int(norm_cfg_d.get("calib_white_dn", 230)),
                )
                normalise_res = auto_normalise_capture(
                    self.display,
                    self.camera,
                    norm_cfg,
                    params,
                    roi_cfg=roi_cfg,
                    debug_dir=run_dir / "normalise",
                )
                params.exposure_us = normalise_res.exposure_us
                params.analogue_gain = normalise_res.analogue_gain
                params.contrast = normalise_res.contrast
                params.brightness_offset = normalise_res.brightness_offset
                params.awb_enable = False
                params.ae_enable = False
                self.store.save_normalise(run_dir, normalise_res.to_dict())
                device_info["normalise"] = normalise_res.to_dict()

            freqs = params.get_frequencies()
            total_steps = len(freqs) * params.N
            with self._lock:
                self._total_steps = total_steps

            step_counter = 0
            for freq_idx, freq in enumerate(freqs):
                patterns = self.generator.generate_sequence(params, frequency=freq)
                attempt = 0
                freq_ok = False
                while attempt <= max_freq_retries and not freq_ok:
                    if self._stop_event.is_set():
                        status = "stopped"
                        break

                    # Frequency switch warmup: project neutral gray and flush stale frames.
                    if freq_idx > 0 or attempt > 0:
                        neutral = np.full((params.resolution[1], params.resolution[0]), 128, dtype=np.uint8)
                        self.display.show_gray(neutral)
                        self.display.pump()
                        time.sleep(max(freq_switch_warmup_ms, settle_ms_freq_switch) / 1000.0)
                        self._flush_frames(flush_frames_on_freq_switch)

                    frames_main: list[np.ndarray] = []
                    for k, pattern in enumerate(patterns):
                        if self._stop_event.is_set():
                            status = "stopped"
                            break

                        self.display.show_gray(pattern)
                        self.display.pump()
                        if k == 0:
                            time.sleep(settle_ms_first_step / 1000.0)
                        else:
                            time.sleep(settle_ms / 1000.0)

                        self._flush_frames(flush_frames_per_step)
                        main_frame, preview_frame = self._capture_pair()
                        frames_main.append(main_frame)
                        self.preview.update(preview_frame, run_id=run_id, step=step_counter + k + 1, total=total_steps)

                    if status == "stopped":
                        break
                    if len(frames_main) != len(patterns):
                        attempt += 1
                        continue

                    # Step sanity gate on the just-captured stack for this frequency.
                    gray_stack = np.stack([self._to_gray_u8(f) for f in frames_main], axis=0)
                    ref = build_reference_from_stack(gray_stack, ref_method=roi_cfg.ref_method)
                    roi_res = detect_object_roi(ref, roi_cfg)
                    if roi_res.debug.get("roi_fallback"):
                        roi_mask = self._adaptive_bright_roi(ref)
                    else:
                        roi_mask = roi_res.roi_mask
                    sanity = check_step_stack(gray_stack, roi_mask, step_thr)
                    sanity_report = {
                        **sanity.to_dict(),
                        "attempt": attempt,
                        "frequency": freq,
                        "roi_fallback": bool(roi_res.debug.get("roi_fallback", False)),
                    "roi_ref_method": roi_cfg.ref_method,
                }
                    self.store.save_step_sanity(run_dir, freq, sanity_report)
                    if not sanity.ok:
                        self.log.warning(
                            "Step sanity failed for f=%s attempt=%s: %s",
                            freq,
                            attempt,
                            "; ".join(sanity.reasons),
                        )
                        attempt += 1
                        continue

                    # Commit frames only after sanity passes.
                    for k, main_frame in enumerate(frames_main):
                        self.store.save_capture(run_dir, k, main_frame, freq=freq)
                        captured += 1
                        step_counter += 1
                        with self._lock:
                            self._step_index = step_counter
                    freq_ok = True

                if status == "stopped":
                    break
                if not freq_ok:
                    status = "failed"
                    error_msg = f"Step stack sanity failed for f={freq} after {max_freq_retries + 1} attempts"
                    self.log.error(error_msg)
                    break

            if status != "stopped":
                status = "completed" if status != "failed" else "failed"
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
                device_info["applied_controls"] = getattr(self.camera, "get_applied_controls", lambda: {})()
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
                    if status == "failed":
                        self._state = "ERROR"
                        self._last_error = error_msg or "Capture step sanity failed"
                    else:
                        self._state = "IDLE"
                self._ended_at = self._ended_at

    def _flush_frames(self, count: int) -> None:
        if count <= 0:
            return
        with self._camera_lock:
            for _ in range(count):
                try:
                    self.camera.capture_pair()
                except Exception:
                    break

    def _capture_pair(self) -> tuple[np.ndarray, np.ndarray]:
        with self._camera_lock:
            try:
                main_frame, preview_frame = self.camera.capture_pair()
            except AttributeError:
                main_frame = self.camera.capture()  # type: ignore[attr-defined]
                preview_frame = main_frame
        return main_frame, preview_frame

    @staticmethod
    def _to_gray_u8(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img.astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 3:
            f = img.astype(np.float32)
            g = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
            return np.clip(np.rint(g), 0, 255).astype(np.uint8)
        raise ValueError("Unsupported image format for gray conversion")

    @staticmethod
    def _adaptive_bright_roi(gray_u8: np.ndarray) -> np.ndarray:
        # Build a stable ROI from bright pixels when object detector falls back.
        p = float(np.percentile(gray_u8, 97.0))
        t = max(5.0, p)
        roi = gray_u8 >= t
        min_area = max(128, int(gray_u8.size * 0.002))
        if int(np.count_nonzero(roi)) < min_area:
            roi = gray_u8 >= max(3.0, float(np.percentile(gray_u8, 92.0)))
        if int(np.count_nonzero(roi)) < min_area:
            roi = np.ones_like(gray_u8, dtype=bool)
        return roi

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
