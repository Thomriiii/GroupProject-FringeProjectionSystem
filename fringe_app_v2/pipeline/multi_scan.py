"""
Multi-angle scan pipeline: rotate turntable, scan at each position, merge clouds.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fringe_app_v2.core.turntable import TurntableClient
from fringe_app_v2.pipeline.capture import capture_raw, create_pipeline_run, lock_exposure_mid_gray
from fringe_app_v2.pipeline.merge_clouds import merge_run_dirs
from fringe_app_v2.pipeline.phase import run_phase_stage
from fringe_app_v2.pipeline.reconstruct import run_reconstruct_stage
from fringe_app_v2.pipeline.roi_stage import run_roi_stage
from fringe_app_v2.pipeline.structured_capture import run_structured_capture
from fringe_app_v2.pipeline.unwrap import run_unwrap_stage
from fringe_app_v2.utils.io import write_json


def run_multi_scan(
    turntable: TurntableClient,
    camera,
    projector,
    patterns,
    params,
    calibration,
    config: dict[str, Any],
    set_stage_cb=None,
) -> dict[str, Any]:
    """
    Capture scans at multiple turntable angles and merge the point clouds.

    The merged run directory is created last so it has the newest timestamp
    and appears as the latest run in the runs folder.

    Returns:
        Summary dict including run_id, run_dir, total_points, and ply_path.
    """
    ms_cfg = config.get("multi_scan") or {}
    angles = [float(a) for a in ms_cfg.get("angles_deg", [0, 90, 180, 270])]
    settle_s = float((config.get("turntable") or {}).get("settle_ms", 500)) / 1000.0

    def _stage(s: str) -> None:
        if set_stage_cb:
            set_stage_cb(s)

    # ── exposure lock (use a temp run; its data is not important) ─────────────
    _stage("exposure_lock")
    exp_run = create_pipeline_run(config)
    lock_exposure_mid_gray(exp_run, camera, projector, config)

    # ── per-angle scans ───────────────────────────────────────────────────────
    scan_run_dirs: list[tuple[float, Path]] = []

    for angle in angles:
        _stage(f"turntable_{angle:.0f}deg")
        turntable.go_to(angle, settle_s=settle_s)

        sub_run = create_pipeline_run(config)
        write_json(sub_run.root / "angle.json", {"angle_deg": angle})

        _stage(f"roi_{angle:.0f}deg")
        projector.show_idle(config)   # structured_capture leaves projector at 0; re-illuminate for ROI
        time.sleep(0.15)
        frame = capture_raw(sub_run, camera, name="roi_capture.png", flush_frames=1)
        roi_mask = run_roi_stage(sub_run, camera, projector, config, source=frame)

        _stage(f"capture_{angle:.0f}deg")
        run_structured_capture(sub_run, camera, projector, patterns, params, config)

        _stage(f"phase_{angle:.0f}deg")
        run_phase_stage(sub_run, params, config, roi_mask)

        _stage(f"unwrap_{angle:.0f}deg")
        run_unwrap_stage(sub_run, config, roi_mask)

        _stage(f"reconstruct_{angle:.0f}deg")
        run_reconstruct_stage(sub_run, calibration, config, roi_mask)

        write_json(sub_run.root / "run.json", {
            "run_id": sub_run.run_id,
            "type": "multi_scan_angle",
            "angle_deg": angle,
            "status": "completed",
        })
        scan_run_dirs.append((angle, sub_run.root))
        print(f"[multi_scan] angle={angle:.1f}° done  ({sub_run.run_id})")

    # ── return to home ────────────────────────────────────────────────────────
    _stage("turntable_home")
    turntable.home(settle_s=0.5)

    # ── merge into a NEW run created last → latest timestamp in runs/ ─────────
    _stage("merge")
    merged_run = create_pipeline_run(config)
    meta = merge_run_dirs(scan_run_dirs, config, merged_run.root / "merged")

    write_json(merged_run.root / "run.json", {
        "run_id": merged_run.run_id,
        "type": "multi_scan_merged",
        "status": "completed",
        "angles_deg": angles,
        "sub_run_dirs": [str(p) for _, p in scan_run_dirs],
        **meta,
    })

    _stage("completed")
    return {
        "run_id": merged_run.run_id,
        "run_dir": str(merged_run.root),
        "ply_path": str(merged_run.root / "merged" / "merged.ply"),
        **meta,
    }
