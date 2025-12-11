"""
projector_calibration.py

Projector intrinsic + stereo calibration using decoded GrayCode correspondences.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from calibration.camera_calibration import compute_reprojection_errors, load_calibration


ImageSize = Tuple[int, int]


@dataclass
class PoseData:
    name: str
    object_points: np.ndarray      # Nx3 float32
    cam_points: np.ndarray         # Nx2 float32
    proj_points: np.ndarray        # Nx2 float32
    valid_frac: float
    image_size_cam: ImageSize | None = None
    image_size_proj: ImageSize | None = None


@dataclass
class CalibrationOutputs:
    Kp: np.ndarray
    dist_p: np.ndarray
    image_size_proj: ImageSize
    rms_proj: float
    rms_stereo: float
    R_proj_to_cam: np.ndarray
    T_proj_to_cam: np.ndarray
    per_view_errors: List[float]
    pose_names: List[str]
    baseline_m: float


def _load_pose_file(path: Path) -> PoseData | None:
    data = np.load(path)
    required = ["object_points", "image_points_cam", "image_points_proj"]
    if not all(k in data for k in required):
        return None

    obj = data["object_points"].astype(np.float32)
    cam = data["image_points_cam"].astype(np.float32)
    proj = data["image_points_proj"].astype(np.float32)
    valid_frac = float(data["valid_fraction"]) if "valid_fraction" in data else 0.0
    image_size_cam = None
    image_size_proj = None
    if "image_size_cam" in data:
        size_cam = data["image_size_cam"]
        image_size_cam = (int(size_cam[0]), int(size_cam[1]))
    if "proj_size" in data:
        size_proj = data["proj_size"]
        image_size_proj = (int(size_proj[0]), int(size_proj[1]))

    return PoseData(
        name=path.parent.name,
        object_points=obj,
        cam_points=cam,
        proj_points=proj,
        valid_frac=valid_frac,
        image_size_cam=image_size_cam,
        image_size_proj=image_size_proj,
    )


def load_session_poses(session_dir: Path) -> Tuple[List[PoseData], ImageSize | None, ImageSize | None]:
    pose_files = sorted(session_dir.glob("pose_*/pose_data.npz"))
    poses: List[PoseData] = []
    image_size_cam = None
    image_size_proj = None

    for p in pose_files:
        pose = _load_pose_file(p)
        if pose is None:
            continue
        poses.append(pose)
        if pose.image_size_cam is not None:
            image_size_cam = pose.image_size_cam
        if pose.image_size_proj is not None:
            image_size_proj = pose.image_size_proj

    return poses, image_size_cam, image_size_proj


def calibrate_projector_intrinsics(
    poses: List[PoseData],
    image_size_proj: ImageSize,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], float, List[float], List[PoseData]]:
    obj_points = []
    proj_points = []
    used_poses: List[PoseData] = []
    for p in poses:
        mask = np.isfinite(p.proj_points).all(axis=1)
        obj_filtered = p.object_points[mask].astype(np.float32)
        proj_filtered = p.proj_points[mask].astype(np.float32)
        if len(obj_filtered) < 4:
            continue
        # Reject poses with extremely low spread (degenerate for calibration)
        if proj_filtered.std(axis=0).min() < 5.0:
            continue
        obj_points.append(obj_filtered)
        proj_points.append(proj_filtered)
        used_poses.append(p)

    ret, Kp, dist_p, rvecs_p, tvecs_p = cv2.calibrateCamera(
        obj_points,
        proj_points,
        image_size_proj,
        None,
        None,
    )

    per_view_err = compute_reprojection_errors(obj_points, proj_points, rvecs_p, tvecs_p, Kp, dist_p)
    return Kp, dist_p, rvecs_p, tvecs_p, float(ret), per_view_err, used_poses


def stereo_calibrate(
    poses: List[PoseData],
    Kc: np.ndarray,
    dist_c: np.ndarray,
    Kp: np.ndarray,
    dist_p: np.ndarray,
    image_size_cam: ImageSize,
) -> Tuple[np.ndarray, np.ndarray, float]:
    obj_points = []
    cam_points = []
    proj_points = []
    for p in poses:
        mask = np.isfinite(p.proj_points).all(axis=1)
        obj_filtered = p.object_points[mask].astype(np.float32)
        cam_filtered = p.cam_points[mask].astype(np.float32)
        proj_filtered = p.proj_points[mask].astype(np.float32)
        if len(obj_filtered) < 4:
            continue
        obj_points.append(obj_filtered)
        cam_points.append(cam_filtered)
        proj_points.append(proj_filtered)

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6)

    ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        obj_points,
        cam_points,
        proj_points,
        Kc,
        dist_c,
        Kp,
        dist_p,
        image_size_cam,
        flags=flags,
        criteria=criteria,
    )

    return R, T, float(ret)


def write_report(
    path: Path,
    calib: CalibrationOutputs,
) -> None:
    lines = []
    lines.append("Projector + Stereo Calibration Report\n")
    lines.append(f"Projector image size: {calib.image_size_proj[0]} x {calib.image_size_proj[1]}\n")
    lines.append(f"Views used: {len(calib.pose_names)}\n")
    lines.append(f"Projector RMS: {calib.rms_proj:.4f} px\n")
    lines.append(f"Stereo RMS: {calib.rms_stereo:.4f} px\n")
    lines.append(f"Baseline (|T|): {calib.baseline_m:.4f} m\n\n")
    lines.append("Projector Intrinsics (Kp):\n")
    lines.append(str(calib.Kp))
    lines.append("\n\nDistortion (k1,k2,p1,p2,k3...):\n")
    lines.append(str(calib.dist_p.ravel()))
    lines.append("\n\nPer-view RMS (projector):\n")
    for name, err in zip(calib.pose_names, calib.per_view_errors):
        lines.append(f"  {name}: {err:.4f}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"[PROJ-CALIB] Report written to {path}")


def run_from_session(
    session_dir: Path,
    camera_calib_path: Path = Path("camera_intrinsics.npz"),
    min_views: int = 6,
    max_proj_error: float | None = 2.0,
) -> CalibrationOutputs:
    session_dir = Path(session_dir)
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    poses, image_size_cam, image_size_proj = load_session_poses(session_dir)
    if len(poses) < min_views:
        raise ValueError(f"Need at least {min_views} valid poses, found {len(poses)}.")

    if image_size_proj is None:
        raise ValueError("Projector image size missing from session.")

    if image_size_cam is None:
        # Fallback: read from camera intrinsics
        _, _, img_size_cam, _ = load_calibration(camera_calib_path)
        image_size_cam = img_size_cam

    Kc, dist_c, _, _ = load_calibration(camera_calib_path)

    print(f"[PROJ-CALIB] Loaded {len(poses)} poses for calibration.")
    print(f"[PROJ-CALIB] Projector size: {image_size_proj}, Camera size: {image_size_cam}")

    Kp, dist_p, rvecs_p, tvecs_p, rms_proj, per_view_err, poses_used = calibrate_projector_intrinsics(poses, image_size_proj)

    # Optional rejection of high-error views
    if max_proj_error is not None:
        keep_mask = [err <= max_proj_error for err in per_view_err]
        if not all(keep_mask) and any(keep_mask):
            kept = [p for p, k in zip(poses_used, keep_mask) if k]
            print(f"[PROJ-CALIB] Dropping {len(poses_used) - len(kept)} views above {max_proj_error} px.")
            poses_used = kept
            Kp, dist_p, rvecs_p, tvecs_p, rms_proj, per_view_err, poses_used = calibrate_projector_intrinsics(poses_used, image_size_proj)

    if not poses_used:
        raise ValueError("No valid poses after filtering NaNs/coverage.")

    R_cam_to_proj, T_cam_to_proj, rms_stereo = stereo_calibrate(
        poses_used,
        Kc=Kc,
        dist_c=dist_c,
        Kp=Kp,
        dist_p=dist_p,
        image_size_cam=image_size_cam,
    )

    # Convert cam->proj to proj->cam (for reconstruction)
    R_proj_to_cam = R_cam_to_proj.T
    T_proj_to_cam = -R_proj_to_cam @ T_cam_to_proj
    baseline = float(np.linalg.norm(T_proj_to_cam))

    # Save intrinsics
    intr_path = session_dir / "projector_intrinsics.npz"
    np.savez_compressed(
        intr_path,
        Kp=Kp,
        dist_p=dist_p,
        image_size_proj=np.array(image_size_proj, dtype=np.int32),
        rms=np.array([rms_proj], dtype=np.float32),
    )
    print(f"[PROJ-CALIB] Saved projector intrinsics to {intr_path}")

    stereo_path = session_dir / "stereo_params.npz"
    np.savez_compressed(
        stereo_path,
        R=R_proj_to_cam,
        T=T_proj_to_cam,
        Kc=Kc,
        dist_c=dist_c,
        Kp=Kp,
        dist_p=dist_p,
        rms=np.array([rms_stereo], dtype=np.float32),
    )
    print(f"[STEREO] Saved stereo parameters to {stereo_path}")

    calib_out = CalibrationOutputs(
        Kp=Kp,
        dist_p=dist_p,
        image_size_proj=image_size_proj,
        rms_proj=rms_proj,
        rms_stereo=rms_stereo,
        R_proj_to_cam=R_proj_to_cam,
        T_proj_to_cam=T_proj_to_cam,
        per_view_errors=per_view_err,
        pose_names=[p.name for p in poses_used],
        baseline_m=baseline,
    )

    write_report(session_dir / "projector_calibration_report.txt", calib_out)
    return calib_out
