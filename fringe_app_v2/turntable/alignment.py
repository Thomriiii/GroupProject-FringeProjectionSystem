"""
Pairwise 2D homography alignment for turntable inspection frames.

For each frame, compute homography H such that H @ src_corners ≈ ref_corners,
using matched ChArUco corner IDs between the source and reference frame.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _load_charuco(angle_dir: Path) -> dict[str, Any] | None:
    p = angle_dir / "charuco.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _matched_corners(
    det_a: dict[str, Any],
    det_b: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    ids_a = {int(i): (x, y) for i, (x, y) in zip(det_a["ids"], det_a["corners_px"])}
    ids_b = {int(i): (x, y) for i, (x, y) in zip(det_b["ids"], det_b["corners_px"])}
    common = sorted(set(ids_a.keys()) & set(ids_b.keys()))
    if not common:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pts_a = np.array([ids_a[i] for i in common], dtype=np.float32)
    pts_b = np.array([ids_b[i] for i in common], dtype=np.float32)
    return pts_a, pts_b


def compute_homography(pts_src: np.ndarray, pts_dst: np.ndarray) -> dict[str, Any]:
    n = len(pts_src)
    if n < 4:
        return {"ok": False, "n_matches": n, "H": None,
                "reprojection_error_px": None,
                "reject_reason": f"too_few_matches ({n} < 4)"}

    H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 3.0)
    if H is None:
        return {"ok": False, "n_matches": n, "H": None,
                "reprojection_error_px": None, "reject_reason": "findHomography_failed"}

    inliers = int(mask.sum()) if mask is not None else n
    projected = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2), H).reshape(-1, 2)
    reproj_err = float(np.mean(np.linalg.norm(pts_dst - projected, axis=1)))

    return {
        "ok": True,
        "n_matches": n,
        "n_inliers": inliers,
        "H": H.tolist(),
        "reprojection_error_px": reproj_err,
        "reject_reason": None,
    }


def run_alignment(
    session_root: Path,
    reference_angle: float = 0.0,
) -> dict[str, Any]:
    """Compute reference→frame homographies for all frames. Write alignment_report.json."""
    frames_dir = session_root / "frames"

    angle_dirs: list[tuple[float, Path]] = []
    for d in sorted(frames_dir.iterdir()):
        if not d.name.startswith("angle_"):
            continue
        name = d.name.replace("angle_", "").lstrip("0")
        try:
            angle_deg = float(name) if name else 0.0
        except ValueError:
            continue
        angle_dirs.append((angle_deg, d))
    angle_dirs.sort(key=lambda x: x[0])

    detections = {angle: _load_charuco(d) for angle, d in angle_dirs}

    # Find reference detection
    ref_det = detections.get(reference_angle)
    if ref_det is None or not ref_det.get("found"):
        for angle, _ in angle_dirs:
            if detections.get(angle) and detections[angle].get("found"):
                reference_angle = angle
                ref_det = detections[angle]
                break

    pairs: list[dict[str, Any]] = []
    for angle, _ in angle_dirs:
        if angle == reference_angle:
            continue
        src_det = detections.get(angle)
        if src_det is None or not src_det.get("found"):
            pairs.append({"src_angle": angle, "dst_angle": reference_angle,
                          "ok": False, "reject_reason": "no_charuco_detection"})
            continue
        pts_src, pts_dst = _matched_corners(src_det, ref_det)
        result = compute_homography(pts_src, pts_dst)
        result["src_angle"] = angle
        result["dst_angle"] = reference_angle
        pairs.append(result)

    n_ok = sum(1 for p in pairs if p.get("ok"))
    report = {
        "reference_angle_deg": reference_angle,
        "n_pairs": len(pairs),
        "n_ok": n_ok,
        "pairs": pairs,
    }

    cal_dir = session_root / "calibration"
    cal_dir.mkdir(exist_ok=True)
    (cal_dir / "alignment_report.json").write_text(json.dumps(report, indent=2))
    return report
