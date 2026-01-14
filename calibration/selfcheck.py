"""
selfcheck.py

Sanity-check stereo parameters using captured projector calibration poses.

For each pose:
  - Solve board pose in the camera using the checkerboard correspondences.
  - Transform board points into the projector frame using stereo extrinsics.
  - Reproject into projector pixels and measure RMS error against decoded UVs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from calibration.projector_calibration import load_session_poses
from calibration.camera_calibration import load_calibration


def load_stereo_params(path: Path):
    """
    Load stereo parameters from an NPZ file.
    """
    data = np.load(path)
    R = data["R"]
    T = data["T"].reshape(3)
    Kc = data["Kc"]
    dist_c = data["dist_c"]
    Kp = data["Kp"]
    dist_p = data["dist_p"]
    rms = float(data["rms"][0]) if "rms" in data else float("nan")
    return R, T, Kc, dist_c, Kp, dist_p, rms


def reproj_pose(
    obj: np.ndarray,
    img_cam: np.ndarray,
    img_proj: np.ndarray,
    Kc: np.ndarray,
    dist_c: np.ndarray,
    Kp: np.ndarray,
    dist_p: np.ndarray,
    R_proj_to_cam: np.ndarray,
    T_proj_to_cam: np.ndarray,
) -> Tuple[float, float]:
    """
    Reproject a single pose and return (rms_cam, rms_proj).
    """
    # Solve board pose in camera frame.
    ok, rvec, tvec = cv2.solvePnP(obj, img_cam, Kc, dist_c, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return np.nan, np.nan

    # Reproject to camera to check.
    proj_cam, _ = cv2.projectPoints(obj, rvec, tvec, Kc, dist_c)
    proj_cam = proj_cam.reshape(-1, 2)
    err_cam = np.linalg.norm(proj_cam - img_cam, axis=1)
    rms_cam = float(np.sqrt(np.mean(err_cam**2)))

    # Convert board pose to projector frame: X_p = R_pc * X_c + t_pc.
    R_cam_to_proj = R_proj_to_cam.T
    t_cam_to_proj = -R_cam_to_proj @ T_proj_to_cam
    R_board_in_proj = R_cam_to_proj @ cv2.Rodrigues(rvec)[0]
    t_board_in_proj = R_cam_to_proj @ tvec + t_cam_to_proj.reshape(3, 1)
    rvec_proj, _ = cv2.Rodrigues(R_board_in_proj)

    proj_proj, _ = cv2.projectPoints(obj, rvec_proj, t_board_in_proj, Kp, dist_p)
    proj_proj = proj_proj.reshape(-1, 2)
    err_proj = np.linalg.norm(proj_proj - img_proj, axis=1)
    rms_proj = float(np.sqrt(np.mean(err_proj**2)))
    return rms_cam, rms_proj


def run_selfcheck(session_dir: Path, stereo_path: Path, camera_path: Path):
    """
    Run reprojection checks across all poses in a session.
    """
    poses, image_size_cam, image_size_proj = load_session_poses(session_dir)
    if not poses:
        raise ValueError(f"No poses in session {session_dir}")

    R, T, Kc, dist_c, Kp, dist_p, rms_stereo = load_stereo_params(stereo_path)
    print(f"[CHECK] Loaded {len(poses)} poses. Stereo RMS in file: {rms_stereo:.4f} px")

    rms_cam_list: List[float] = []
    rms_proj_list: List[float] = []

    for pose in poses:
        mask = np.isfinite(pose.proj_points).all(axis=1)
        if mask.sum() < 4:
            print(f"[CHECK] Skipping {pose.name}: not enough valid corners.")
            continue
        obj = pose.object_points[mask].astype(np.float32)
        cam = pose.cam_points[mask].astype(np.float32)
        proj = pose.proj_points[mask].astype(np.float32)

        rms_cam, rms_proj = reproj_pose(obj, cam, proj, Kc, dist_c, Kp, dist_p, R, T)
        rms_cam_list.append(rms_cam)
        rms_proj_list.append(rms_proj)
        print(f"[CHECK] {pose.name}: cam RMS={rms_cam:.4f} px, proj RMS={rms_proj:.4f} px")

    if rms_cam_list:
        print(f"[CHECK] Camera reprojection RMS: mean={np.mean(rms_cam_list):.4f}, max={np.max(rms_cam_list):.4f}")
    if rms_proj_list:
        print(f"[CHECK] Projector reprojection RMS: mean={np.mean(rms_proj_list):.4f}, max={np.max(rms_proj_list):.4f}")


def main():
    """
    CLI entry point for stereo self-check.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, type=Path, help="Projector calibration session directory")
    ap.add_argument("--stereo", default=Path("stereo_params.npz"), type=Path, help="Stereo params file")
    ap.add_argument("--camera", default=Path("camera_intrinsics.npz"), type=Path, help="Camera intrinsics file")
    args = ap.parse_args()
    run_selfcheck(args.session, args.stereo, args.camera)


if __name__ == "__main__":
    main()
