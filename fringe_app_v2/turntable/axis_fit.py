"""
Fit turntable rotation axis and step accuracy from a sequence of board poses.

For each consecutive pose pair, extract relative rotation angle from R_rel = R_j @ R_i^T.
Circle-fit the board centre translations (XY camera plane) to locate the rotation axis.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _rvec_to_R(rvec: list[float]) -> np.ndarray:
    r, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64).reshape(3, 1))
    return r


def _R_to_angle_deg(R: np.ndarray) -> float:
    rvec, _ = cv2.Rodrigues(R)
    return math.degrees(float(np.linalg.norm(rvec)))


def _R_to_axis_angle(R: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Return unit rotation axis and angle in degrees from a rotation matrix."""
    rvec, _ = cv2.Rodrigues(R)
    rv = np.asarray(rvec, dtype=np.float64).reshape(3)
    mag = float(np.linalg.norm(rv))
    if mag <= 1e-12:
        return None, 0.0
    axis = rv / mag
    # Keep a stable sign convention for reporting/config use. The same physical
    # line can be represented by +/-axis, but merge sign handling expects a
    # consistent direction across calibration runs.
    if axis[2] < 0:
        axis = -axis
    return axis, math.degrees(mag)


def _circle_fit(points: np.ndarray) -> tuple[float, float, float]:
    """Taubin algebraic circle fit. Returns (cx, cy, radius)."""
    if len(points) < 3:
        raise ValueError("Need ≥3 points")
    x, y = points[:, 0], points[:, 1]
    xm, ym = x.mean(), y.mean()
    u, v = x - xm, y - ym
    Suu = (u * u).sum(); Svv = (v * v).sum(); Suv = (u * v).sum()
    Suuu = (u**3).sum(); Svvv = (v**3).sum()
    Suvv = (u * v * v).sum(); Svuu = (v * u * u).sum()
    A = np.array([[Suu, Suv], [Suv, Svv]])
    b = np.array([0.5 * (Suuu + Suvv), 0.5 * (Svvv + Svuu)])
    try:
        uc, vc = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        uc, vc = 0.0, 0.0
    cx, cy = uc + xm, vc + ym
    r = float(np.sqrt(uc**2 + vc**2 + (Suu + Svv) / len(points)))
    return cx, cy, r


def fit_axis(
    poses: list[dict[str, Any]],
    nominal_step_deg: float = 15.0,
) -> dict[str, Any]:
    """
    Fit the turntable axis and step accuracy.

    Args:
        poses: List with keys "angle_deg", "rvec", "tvec", "ok".
        nominal_step_deg: Expected angular step between captures.

    Returns:
        Calibration result dict.
    """
    good = [p for p in poses if p.get("ok") and p.get("rvec") and p.get("tvec")]
    if len(good) < 2:
        return {"ok": False, "message": f"Need ≥2 valid poses, got {len(good)}"}

    good.sort(key=lambda p: float(p["angle_deg"]))
    Rs = [_rvec_to_R(p["rvec"]) for p in good]
    tvecs = [np.array(p["tvec"], dtype=np.float64) for p in good]

    # Step angles from consecutive relative rotations.
    # Skip any pair where the nominal angle gap is > 1.5× the step — this
    # means a frame was missed between them and the measured rotation would
    # be a multiple of the step, which would corrupt the statistics.
    max_gap = nominal_step_deg * 1.5
    step_angles = []
    step_axes = []
    for i in range(len(good) - 1):
        nominal_gap = abs(float(good[i + 1]["angle_deg"]) - float(good[i]["angle_deg"]))
        if nominal_gap > max_gap:
            continue
        axis, angle_deg = _R_to_axis_angle(Rs[i + 1] @ Rs[i].T)
        step_angles.append(angle_deg)
        if axis is not None:
            step_axes.append(axis)

    if not step_angles:
        return {"ok": False, "message": "No consecutive pose pairs found"}

    step_arr = np.array(step_angles)
    step_mean = float(step_arr.mean())
    step_std = float(step_arr.std())
    step_error = step_mean - nominal_step_deg

    axis_direction = None
    axis_direction_std = None
    axis_tilt_from_camera_z_deg = None
    if step_axes:
        axes = np.asarray(step_axes, dtype=np.float64)
        # Align all samples to the first axis before averaging; this protects
        # against any 180-degree sign ambiguity in Rodrigues output.
        ref = axes[0]
        for i in range(axes.shape[0]):
            if float(np.dot(axes[i], ref)) < 0:
                axes[i] = -axes[i]
        mean_axis = np.mean(axes, axis=0)
        norm = float(np.linalg.norm(mean_axis))
        if norm > 1e-12:
            mean_axis = mean_axis / norm
            if mean_axis[2] < 0:
                mean_axis = -mean_axis
            axis_direction = [float(v) for v in mean_axis]
            axis_direction_std = [float(v) for v in np.std(axes, axis=0)]
            axis_tilt_from_camera_z_deg = float(
                math.degrees(math.acos(float(np.clip(mean_axis[2], -1.0, 1.0))))
            )

    # Circle fit on board-centre XY projections
    centres_xy = np.array([[t[0], t[1]] for t in tvecs])
    axis_x = axis_y = radius = None
    if len(centres_xy) >= 3:
        try:
            axis_x, axis_y, radius = _circle_fit(centres_xy)
            axis_x = float(axis_x)
            axis_y = float(axis_y)
            radius = float(radius)
        except ValueError:
            pass

    if step_std < 0.5 and abs(step_error) < 0.5:
        quality = "good"
    elif step_std < 1.5 and abs(step_error) < 1.5:
        quality = "warning"
    else:
        quality = "bad"

    message = (
        f"step mean={step_mean:.2f}° std={step_std:.2f}° "
        f"error={step_error:+.2f}° quality={quality}"
    )

    return {
        "ok": True,
        "n_poses_used": len(good),
        "axis_x_camera_m": axis_x,
        "axis_y_camera_m": axis_y,
        "axis_direction_camera": axis_direction,
        "axis_direction_std": axis_direction_std,
        "axis_tilt_from_camera_z_deg": axis_tilt_from_camera_z_deg,
        "radius_m": radius,
        "step_angles_deg": [float(a) for a in step_angles],
        "step_mean_deg": step_mean,
        "step_std_deg": step_std,
        "step_error_deg": step_error,
        "nominal_step_deg": nominal_step_deg,
        "quality": quality,
        "message": message,
    }


def run_analysis(
    session_root: Path,
    nominal_step_deg: float = 15.0,
) -> dict[str, Any]:
    """Load all pose.json files from a session, run fit_axis, write report."""
    frames_dir = session_root / "frames"
    poses: list[dict[str, Any]] = []

    for angle_dir in sorted(frames_dir.iterdir()):
        pose_path = angle_dir / "pose.json"
        if not pose_path.exists():
            continue
        data = json.loads(pose_path.read_text())
        # Extract numeric angle from directory name e.g. "angle_045" → 45.0
        name = angle_dir.name.replace("angle_", "").lstrip("0")
        data["angle_deg"] = float(name) if name else 0.0
        poses.append(data)

    result = fit_axis(poses, nominal_step_deg=nominal_step_deg)

    cal_dir = session_root / "calibration"
    cal_dir.mkdir(exist_ok=True)
    (cal_dir / "axis_fit_report.json").write_text(json.dumps(result, indent=2))
    return result
