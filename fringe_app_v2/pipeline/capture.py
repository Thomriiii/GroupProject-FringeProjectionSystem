"""Image capture and exposure-lock stage."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from fringe_app_v2.core.camera import CameraService
from fringe_app_v2.core.projector import ProjectorService
from fringe_app_v2.utils.io import RunPaths, create_run, save_image, write_json
from fringe_app_v2.utils.math_utils import to_gray_u8


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
        clip_pct = float(np.mean(gray >= 250))
        trace.append(
            {
                "iter": idx + 1,
                "exposure_us": exposure,
                "analogue_gain": gain,
                "mean_dn": mean_dn,
                "clip_pct": clip_pct,
            }
        )
        if abs(mean_dn - target) <= tolerance and clip_pct < 0.01:
            break
        ratio = float(np.clip(target / max(mean_dn, 1.0), 0.55, 1.8))
        proposed_exp = int(np.clip(round(exposure * ratio), min_exp, max_exp))
        if proposed_exp != exposure:
            exposure = proposed_exp
            continue
        proposed_gain = float(np.clip(gain * ratio, min_gain, max_gain))
        if abs(proposed_gain - gain) < 0.01:
            break
        gain = proposed_gain

    camera.set_manual_controls(exposure, gain, awb_enable=False)
    payload = {
        "enabled": True,
        "target_dn": target,
        "tolerance_dn": tolerance,
        "exposure_us": int(exposure),
        "analogue_gain": float(gain),
        "trace": trace,
        "ok": bool(trace and abs(float(trace[-1]["mean_dn"]) - target) <= tolerance),
    }
    write_json(run.raw / "exposure_lock.json", payload)
    return payload
