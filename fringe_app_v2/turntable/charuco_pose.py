"""
ChArUco detection + solvePnP pose estimation for turntable calibration frames.

Saves per-frame:
  charuco.json    CharucoDetection serialised
  overlay.png     Annotated image
  pose.json       {"ok", "rvec", "tvec", "reprojection_error_px", "reject_reason"}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fringe_app_v2.turntable.charuco_detect import (
    CharucoDetection,
    charuco_object_points,
    detect_charuco,
)


def _reprojection_error(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(
        obj_pts.reshape(-1, 1, 3).astype(np.float32),
        rvec, tvec, K, D,
    )
    return float(np.mean(np.linalg.norm(
        img_pts.reshape(-1, 2) - projected.reshape(-1, 2), axis=1
    )))


def detect_and_pose(
    image: np.ndarray,
    charuco_cfg: dict[str, Any],
    K: np.ndarray,
    D: np.ndarray,
    min_corners: int = 6,
) -> dict[str, Any]:
    """
    Run ChArUco detection then solvePnP.

    Returns dict with:
      ok, rvec, tvec, reprojection_error_px, n_corners,
      charuco (CharucoDetection), overlay (ndarray), reject_reason
    """
    det, overlay, board = detect_charuco(image, charuco_cfg)

    if not det.found or det.corner_count < min_corners:
        return {
            "ok": False,
            "rvec": None,
            "tvec": None,
            "reprojection_error_px": None,
            "n_corners": det.corner_count,
            "charuco": det,
            "overlay": overlay,
            "reject_reason": det.reject_reason or f"too_few_corners ({det.corner_count} < {min_corners})",
        }

    obj_pts = charuco_object_points(board, det.ids)
    img_pts = det.corners.reshape(-1, 1, 2).astype(np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts.reshape(-1, 1, 3),
        img_pts,
        K, D,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not ok:
        return {
            "ok": False,
            "rvec": None,
            "tvec": None,
            "reprojection_error_px": None,
            "n_corners": det.corner_count,
            "charuco": det,
            "overlay": overlay,
            "reject_reason": "solvePnP_failed",
        }

    reproj_err = _reprojection_error(obj_pts, det.corners, rvec, tvec, K, D)

    try:
        sq = charuco_cfg.get("square_length_m", 0.015)
        cv2.drawFrameAxes(overlay, K, D, rvec, tvec, sq * 2)
    except Exception:
        pass

    return {
        "ok": True,
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist(),
        "reprojection_error_px": reproj_err,
        "n_corners": det.corner_count,
        "charuco": det,
        "overlay": overlay,
        "reject_reason": None,
    }


def process_frame(
    frame_dir: Path,
    image: np.ndarray,
    charuco_cfg: dict[str, Any],
    K: np.ndarray,
    D: np.ndarray,
    min_corners: int = 6,
) -> dict[str, Any]:
    """
    Detect, pose-estimate, and write outputs for one frame directory.

    Writes: charuco.json, overlay.png, pose.json
    Returns the pose result dict (without numpy arrays).
    """
    result = detect_and_pose(image, charuco_cfg, K, D, min_corners=min_corners)

    (frame_dir / "charuco.json").write_text(
        json.dumps(result["charuco"].to_json_dict(), indent=2)
    )
    cv2.imwrite(str(frame_dir / "overlay.png"), result["overlay"])

    pose_data = {
        "ok": result["ok"],
        "rvec": result["rvec"],
        "tvec": result["tvec"],
        "reprojection_error_px": result["reprojection_error_px"],
        "n_corners": result["n_corners"],
        "reject_reason": result["reject_reason"],
    }
    (frame_dir / "pose.json").write_text(json.dumps(pose_data, indent=2))

    return pose_data
