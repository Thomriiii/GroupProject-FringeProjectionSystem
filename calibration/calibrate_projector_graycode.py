#!/usr/bin/env python3
"""
calibrate_projector_graycode.py

Calibrate projector intrinsics/extrinsics from Gray code dataset.
"""

from __future__ import annotations

import argparse
import numpy as np
import cv2

from calibration.projector_intrinsics import (
    solve_intrinsics_from_homographies,
    extrinsics_from_H,
)


def calibrate(dataset_path: str, cam_intr_path: str):
    print(f"[CALIB] Loading camera intrinsics: {cam_intr_path}")
    cam_intr = np.load(cam_intr_path)
    K_cam = cam_intr["K"]
    dist_cam = cam_intr["dist"]

    print(f"[CALIB] Loading dataset: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    cam_pts_list = data["cam_points_list"]
    proj_pts_list = data["proj_points_list"]

    H_list = []
    for idx, (cam_pts, proj_pts) in enumerate(zip(cam_pts_list, proj_pts_list)):
        cam_pts = np.asarray(cam_pts, dtype=np.float32)
        proj_pts = np.asarray(proj_pts, dtype=np.float32)

        # Optional subsample for stability/speed if very dense
        if cam_pts.shape[0] > 50000:
            cam_pts = cam_pts[::10]
            proj_pts = proj_pts[::10]

        if cam_pts.shape[0] < 4 or proj_pts.shape[0] < 4:
            print(f"[CALIB][WARN] Pose {idx}: not enough points, skipping.")
            continue
        cam_ud = cv2.undistortPoints(cam_pts.reshape(-1, 1, 2), K_cam, dist_cam, P=K_cam).reshape(-1, 2)

        # Use RANSAC for robustness
        H, _ = cv2.findHomography(cam_ud, proj_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is None:
            print(f"[CALIB][WARN] Pose {idx}: homography failed, skipping.")
            continue
        H_list.append(H)

    if not H_list:
        raise RuntimeError("No valid homographies.")

    print("[CALIB] Solving projector intrinsics...")
    K_proj = solve_intrinsics_from_homographies(H_list)
    print("Projector K:\n", K_proj)

    # Extrinsics from first pose
    H0 = H_list[0]
    R_cam, t_cam = extrinsics_from_H(K_cam, H0)
    H0_inv = np.linalg.inv(H0)
    R_proj, t_proj = extrinsics_from_H(K_proj, H0_inv)

    R_cam_proj = R_proj @ R_cam.T
    t_cam_proj = t_proj - R_cam_proj @ t_cam

    np.savez("projector_intrinsics_graycode.npz", K=K_proj)
    np.savez("stereo_params_graycode.npz", R=R_cam_proj, t=t_cam_proj)
    print("[CALIB] Saved projector_intrinsics_graycode.npz and stereo_params_graycode.npz")


def main():
    parser = argparse.ArgumentParser(description="Projector calibration from Gray code dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--camera", default="camera_intrinsics.npz")
    args = parser.parse_args()
    calibrate(args.dataset, args.camera)


if __name__ == "__main__":
    main()
