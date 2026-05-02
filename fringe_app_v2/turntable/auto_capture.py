"""
Automatic 360° turntable calibration capture.

Rotates the turntable in steps, captures a frame at each position,
runs ChArUco detection + pose estimation, and saves everything to a session.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from fringe_app_v2.core.turntable import TurntableClient, TurntableError
from fringe_app_v2.turntable.charuco_pose import process_frame
from fringe_app_v2.turntable.session import TurntableSession, FrameRecord, add_frame


def run_auto_capture(
    session: TurntableSession,
    turntable: TurntableClient,
    camera,
    charuco_cfg: dict[str, Any],
    K: np.ndarray | None,
    D: np.ndarray | None,
    step_deg: float = 15.0,
    total_deg: float = 360.0,
    settle_ms: int = 600,
    on_frame: Callable[[int, int, FrameRecord], None] | None = None,
) -> list[FrameRecord]:
    """
    Rotate the turntable and capture one frame at each step.

    Args:
        session:      Active TurntableSession (will be saved after each frame).
        turntable:    Connected TurntableClient.
        camera:       CameraService (must already be started).
        charuco_cfg:  ChArUco board config dict.
        K, D:         Camera intrinsics (None → skip pose estimation).
        step_deg:     Angle between captures (default 15°).
        total_deg:    Total arc to cover (default 360°).
        settle_ms:    Time to wait after each rotation before capturing.
        on_frame:     Optional callback(frame_index, total_frames, record) for progress.

    Returns:
        List of FrameRecord objects added to the session.
    """
    angles = [step_deg * i for i in range(int(round(total_deg / step_deg)))]
    n = len(angles)
    settle_s = settle_ms / 1000.0
    min_corners = int(charuco_cfg.get("min_corners", 6))
    records: list[FrameRecord] = []

    for idx, angle in enumerate(angles):
        # Rotate
        try:
            turntable.go_to(angle, settle_s=settle_s)
        except TurntableError as exc:
            print(f"[auto_capture] turntable error at {angle}°: {exc} — skipping")
            continue

        # Capture
        image = camera.capture(flush_frames=1)

        label = f"angle_{int(round(angle)):03d}"
        frame_dir = session.frame_dir(label)
        frame_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(frame_dir / "image.png"), image)

        # Detect + pose
        if K is not None:
            pose = process_frame(frame_dir, image, charuco_cfg, K, D, min_corners=min_corners)
        else:
            pose = {"ok": False, "n_corners": 0, "reprojection_error_px": None}

        rec = add_frame(session, angle, str(frame_dir / "image.png"))
        rec.charuco_ok = pose.get("ok", False) or (pose.get("n_corners", 0) > 0)
        rec.n_corners = int(pose.get("n_corners") or 0)
        rec.pose_ok = bool(pose.get("ok"))
        rec.reprojection_error_px = float(pose.get("reprojection_error_px") or 0.0)
        session.save()
        records.append(rec)

        badge = "✅" if rec.status == "good" else ("⚠️" if rec.status == "warning" else "❌")
        print(f"[auto_capture] {idx+1}/{n}  angle={angle:.0f}°  corners={rec.n_corners}  {badge}")

        if on_frame:
            on_frame(idx + 1, n, rec)

    # Return to home
    try:
        turntable.home(settle_s=0.5)
        print("[auto_capture] Returned to home (0°)")
    except TurntableError as exc:
        print(f"[auto_capture] Warning: could not return to home: {exc}")

    return records
