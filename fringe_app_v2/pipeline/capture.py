"""Image capture and exposure-lock stage."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from fringe_app_v2.core.camera import CameraService
from fringe_app_v2.core.projector import ProjectorService
from fringe_app_v2.utils.io import RunPaths, create_run, save_image, write_json
from fringe_app_v2.utils.math_utils import to_gray_u8


def _meter_exposure(gray: np.ndarray, cfg: dict[str, Any]) -> float:
    mode = str(cfg.get("meter", "top_fraction_mean")).strip().lower()
    arr = np.asarray(gray, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    if mode in {"mean", "full_frame_mean"}:
        return float(np.mean(arr))
    if mode in {"percentile", "pctl"}:
        return float(np.percentile(arr, float(cfg.get("meter_percentile", 90.0))))

    fraction = float(cfg.get("meter_top_fraction", 0.10))
    fraction = min(1.0, max(0.001, fraction))
    flat = arr.ravel()
    n = max(1, int(round(flat.size * fraction)))
    if n >= flat.size:
        return float(np.mean(flat))
    top = np.partition(flat, flat.size - n)[-n:]
    return float(np.mean(top))


def create_pipeline_run(config: dict[str, Any]) -> RunPaths:
    from pathlib import Path

    root = Path(str((config.get("storage", {}) or {}).get("run_root", "fringe_app_v2/runs")))
    return create_run(root)


def capture_raw(run: RunPaths, camera: CameraService, name: str = "capture.png", flush_frames: int = 1) -> np.ndarray:
    frame = camera.capture(flush_frames=flush_frames)
    save_image(run.raw / name, frame)
    return frame


def lock_exposure_mid_gray(
    run: RunPaths,
    camera: CameraService,
    projector: ProjectorService,
    config: dict[str, Any],
) -> dict[str, Any]:
    cfg = config.get("exposure_lock", {}) or {}
    scan = config.get("scan", {}) or {}
    if not bool(cfg.get("enabled", True)):
        payload = {"enabled": False}
        write_json(run.raw / "exposure_lock.json", payload)
        return payload

    target = float(cfg.get("target_dn", 128.0))
    tolerance = float(cfg.get("tolerance_dn", 8.0))
    exposure = int(scan.get("exposure_us", 2000))
    gain = float(scan.get("analogue_gain", 1.0))
    min_exp = int(cfg.get("min_exposure_us", 500))
    max_exp = int(cfg.get("max_exposure_us", 12000))
    min_gain = float(cfg.get("min_gain", 1.0))
    max_gain = float(cfg.get("max_gain", 4.0))
    settle_s = float(cfg.get("settle_ms", 180)) / 1000.0
    flush = int(cfg.get("flush_frames", 2))
    trace: list[dict[str, Any]] = []

    projector.show_level(int(round(target)))
    time.sleep(settle_s)
    for idx in range(max(1, int(cfg.get("max_iters", 8)))):
        camera.set_manual_controls(exposure, gain, awb_enable=False)
        time.sleep(settle_s)
        frame = camera.capture(flush_frames=flush)
        gray = to_gray_u8(frame)
        mean_dn = float(np.mean(gray))
        meter_dn = _meter_exposure(gray, cfg)
        clip_pct = float(np.mean(gray >= 250))
        trace.append(
            {
                "iter": idx + 1,
                "exposure_us": exposure,
                "analogue_gain": gain,
                "mean_dn": mean_dn,
                "meter_dn": meter_dn,
                "clip_pct": clip_pct,
            }
        )
        if abs(meter_dn - target) <= tolerance and clip_pct < 0.01:
            break
        ratio = float(np.clip(target / max(meter_dn, 1.0), 0.55, 1.8))
        proposed_exp = int(np.clip(round(exposure * ratio), min_exp, max_exp))
        if proposed_exp != exposure:
            exposure = proposed_exp
            continue
        proposed_gain = float(np.clip(gain * ratio, min_gain, max_gain))
        if abs(proposed_gain - gain) < 0.01:
            break
        gain = proposed_gain

    camera.set_manual_controls(exposure, gain, awb_enable=False)
    final_meter = float(trace[-1].get("meter_dn", trace[-1]["mean_dn"])) if trace else 0.0
    target_ok = bool(trace and abs(final_meter - target) <= tolerance)
    at_limits = exposure >= max_exp and gain >= max_gain - 1e-6
    clip_ok = bool(trace and float(trace[-1].get("clip_pct", 1.0)) < float(cfg.get("max_clip_pct", 0.01)))
    under_target_at_limits_ok = (
        bool(cfg.get("accept_under_target_at_limits", True))
        and at_limits
        and clip_ok
        and final_meter >= float(cfg.get("min_usable_meter_dn", 45.0))
    )
    status = "locked" if target_ok else ("accepted_at_limits" if under_target_at_limits_ok else "failed")

    payload = {
        "enabled": True,
        "target_dn": target,
        "tolerance_dn": tolerance,
        "meter": str(cfg.get("meter", "top_fraction_mean")),
        "meter_top_fraction": float(cfg.get("meter_top_fraction", 0.10)),
        "meter_percentile": float(cfg.get("meter_percentile", 90.0)),
        "min_usable_meter_dn": float(cfg.get("min_usable_meter_dn", 45.0)),
        "exposure_us": int(exposure),
        "analogue_gain": float(gain),
        "trace": trace,
        "ok": bool(target_ok or under_target_at_limits_ok),
        "target_ok": target_ok,
        "accepted_at_limits": under_target_at_limits_ok,
        "status": status,
    }
    write_json(run.raw / "exposure_lock.json", payload)
    if bool(cfg.get("fail_on_error", True)) and not payload["ok"]:
        raise RuntimeError(
            "Exposure lock failed: "
            f"meter={final_meter:.1f} DN, "
            f"target={target:.1f} DN at exposure={exposure} us, gain={gain:.2f}"
        )
    return payload
