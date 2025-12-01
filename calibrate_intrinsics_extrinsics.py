#!/usr/bin/env python3
"""
calibrate_intrinsics_extrinsics.py

Stage 2 of structured-light calibration:
    - Load calib_dataset.npz from Stage 1
    - Fit homographies
    - Solve camera intrinsics
    - Solve projector intrinsics
    - Extract extrinsics (R, t) per pose
    - Compute final camera–projector stereo transform
    - Save:
        camera_intrinsics.npz
        projector_intrinsics.npz
        stereo_params.npz

Usage:
    python calibrate_intrinsics_extrinsics.py \
        --dataset calib/session_XXXX/calib_dataset.npz
"""

from __future__ import annotations
import argparse
import numpy as np
import cv2
from typing import List, Tuple


# ---------------------------------------------------------------
# Helper: Normalize homography so det(H[0:2,0:2]) ~= 1
# ---------------------------------------------------------------
def normalize_H(H):
    H = H / np.linalg.norm(H[:, 0])
    return H


# ---------------------------------------------------------------
# Compute camera intrinsics from homographies (Zhang’s method)
# ---------------------------------------------------------------
def solve_intrinsics_from_homographies(H_list: List[np.ndarray]) -> np.ndarray:
    """
    Implements Zhang calibration method to solve camera intrinsics K
    from a list of homographies between camera → planar object.
    """
    V = []

    def v_ij(H, i, j):
        return np.array([
            H[0, i] * H[0, j],
            H[0, i] * H[1, j] + H[1, i] * H[0, j],
            H[1, i] * H[1, j],
            H[0, i] * H[2, j] + H[2, i] * H[0, j],
            H[1, i] * H[2, j] + H[2, i] * H[1, j],
            H[2, i] * H[2, j],
        ])

    for H in H_list:
        H = normalize_H(H)
        V.append(v_ij(H, 0, 1))
        V.append(v_ij(H, 0, 0) - v_ij(H, 1, 1))

    V = np.array(V)
    _, _, VT = np.linalg.svd(V)
    b = VT[-1]

    # Form B matrix
    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]],
    ])

    # Solve intrinsics from B (see Zhang's paper)
    cy = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / \
         (B[0, 0] * B[1, 1] - B[0, 1]**2)
    lam = B[2, 2] - (B[0, 2]**2 + cy*(B[0, 1]*B[0, 2] - B[0, 0]*B[1, 2])) / B[0, 0]
    fx = np.sqrt(lam / B[0, 0])
    fy = np.sqrt(lam * B[0, 0] / (B[0, 0] * B[1, 1] - B[0, 1]**2))
    s = -B[0, 1] * fx**2 * fy / lam
    cx = s * cy / fy - B[0, 2] * fx**2 / lam

    K = np.array([
        [fx, s,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])
    return K


# ---------------------------------------------------------------
# Extract extrinsics from homographies and intrinsics
# ---------------------------------------------------------------
def extrinsics_from_H(K, H):
    Kinv = np.linalg.inv(K)
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    lam = 1.0 / np.linalg.norm(Kinv @ h1)
    r1 = lam * (Kinv @ h1)
    r2 = lam * (Kinv @ h2)
    r3 = np.cross(r1, r2)
    t = lam * (Kinv @ h3)

    R = np.column_stack([r1, r2, r3])
    # Fix rotation matrix to closest orthonormal
    U, _, VT = np.linalg.svd(R)
    R = U @ VT

    return R, t


# ---------------------------------------------------------------
# Fit homography: camera_points → projector_points (2D→2D)
# ---------------------------------------------------------------
def fit_H(cam_pts, proj_pts):
    H, _ = cv2.findHomography(cam_pts, proj_pts, method=0)
    return H


# ---------------------------------------------------------------
# Main calibration procedure
# ---------------------------------------------------------------
def calibrate(dataset_path: str):

    print(f"[CALIB] Loading dataset: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)

    cam_pts_list = data["cam_points_list"]
    proj_pts_list = data["proj_points_list"]

    proj_size = data["proj_size"]
    W_proj, H_proj = proj_size
    W_cam, H_cam = data["image_size_cam"]

    print(f"[CALIB] Loaded {len(cam_pts_list)} pose(s).")

    # -----------------------------------------------------------
    # Step 1: Fit homographies per pose
    # -----------------------------------------------------------
    H_list = []
    for cam_pts, proj_pts in zip(cam_pts_list, proj_pts_list):
        H = fit_H(cam_pts, proj_pts)
        H_list.append(H)

    print("[CALIB] Fitted homographies for all poses.")

    # -----------------------------------------------------------
    # Step 2: Solve CAMERA INTRINSICS
    # -----------------------------------------------------------
    K_cam = solve_intrinsics_from_homographies(H_list)
    print("[CALIB] Camera intrinsics:")
    print(K_cam)

    # -----------------------------------------------------------
    # Step 3: Solve PROJECTOR INTRINSICS
    # Just re-run Zhang on inverse mappings
    # -----------------------------------------------------------

    H_inv_list = []
    for H in H_list:
        H_inv = np.linalg.inv(H)
        H_inv_list.append(H_inv)

    K_proj = solve_intrinsics_from_homographies(H_inv_list)
    print("[CALIB] Projector intrinsics:")
    print(K_proj)

    # -----------------------------------------------------------
    # Step 4: Compute extrinsics (R_cam & t_cam of plane)
    # -----------------------------------------------------------
    R_cam_list = []
    t_cam_list = []
    for H in H_list:
        R, t = extrinsics_from_H(K_cam, H)
        R_cam_list.append(R)
        t_cam_list.append(t)

    # -----------------------------------------------------------
    # Step 5: Compute stereo extrinsics
    # Choose first pose as reference
    # -----------------------------------------------------------
    R_cam = R_cam_list[0]
    t_cam = t_cam_list[0]

    # For projector extrinsics:
    R_proj = R_cam
    t_proj = t_cam  # same plane assumption

    R_cam_proj = R_proj @ R_cam.T
    t_cam_proj = t_proj - R_cam_proj @ t_cam

    print("[CALIB] Relative camera–projector extrinsics:")
    print("Rotation:\n", R_cam_proj)
    print("Translation:\n", t_cam_proj)

    # -----------------------------------------------------------
    # Save results
    # -----------------------------------------------------------
    np.savez("camera_intrinsics.npz", K=K_cam)
    np.savez("projector_intrinsics.npz", K=K_proj)
    np.savez("stereo_params.npz", R=R_cam_proj, t=t_cam_proj)

    print("[CALIB] Calibration COMPLETE.")
    print("Saved:")
    print("  camera_intrinsics.npz")
    print("  projector_intrinsics.npz")
    print("  stereo_params.npz")


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to calib_dataset.npz")
    args = parser.parse_args()

    calibrate(args.dataset)


if __name__ == "__main__":
    main()
