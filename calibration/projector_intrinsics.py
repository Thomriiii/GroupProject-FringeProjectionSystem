#!/usr/bin/env python3
"""
calibration/projector_intrinsics.py

Calibrate projector intrinsics (and camera–projector extrinsics) using
correspondences from structured-light captures.

Inputs:
    camera_intrinsics.npz (K, dist, size)
    calib_dataset.npz     (cam_points_list, proj_points_list, proj_size)

Process:
    - Undistort camera points
    - Fit homographies between cam (undistorted) and projector points
    - Solve projector intrinsics via Zhang's method
    - Compute extrinsics (R, t) between camera and projector

Outputs:
    projector_intrinsics.npz  (K)
    stereo_params.npz         (R, t) projector relative to camera
"""

from __future__ import annotations

import argparse
import numpy as np
import cv2
from typing import List, Tuple


# -------------------------------------------------------------------------
# Helper: Normalize homography
# -------------------------------------------------------------------------
def normalize_H(H: np.ndarray) -> np.ndarray:
    H = H / np.linalg.norm(H[:, 0])
    return H


# -------------------------------------------------------------------------
# Zhang intrinsics from homographies
# -------------------------------------------------------------------------
def solve_intrinsics_from_homographies(H_list: List[np.ndarray]) -> np.ndarray:
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
        if H is None:
            continue
        H = normalize_H(H)
        V.append(v_ij(H, 0, 1))
        V.append(v_ij(H, 0, 0) - v_ij(H, 1, 1))

    if not V:
        raise RuntimeError("No valid homographies to solve intrinsics; check dataset.")

    V = np.array(V)
    _, _, VT = np.linalg.svd(V)
    b = VT[-1]

    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]],
    ])

    if B[0, 0] < 0:
        B = -B

    denom = (B[0, 0] * B[1, 1] - B[0, 1]**2)
    if abs(denom) < 1e-12 or abs(B[0, 0]) < 1e-12:
        raise RuntimeError("Degenerate homography set: unable to solve intrinsics.")

    cy = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / denom
    lam = B[2, 2] - (B[0, 2]**2 + cy*(B[0, 1]*B[0, 2] - B[0, 0]*B[1, 2])) / B[0, 0]

    lam = abs(lam)
    denom_abs = abs(denom)

    fx = np.sqrt(abs(lam / B[0, 0]))
    fy = np.sqrt(abs(lam * B[0, 0] / denom_abs))
    s = -B[0, 1] * fx**2 * fy / lam
    cx = s * cy / fy - B[0, 2] * fx**2 / lam

    K = np.array([
        [fx, s,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])
    return K


# -------------------------------------------------------------------------
# Extrinsics from homography
# -------------------------------------------------------------------------
def extrinsics_from_H(K: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    U, _, VT = np.linalg.svd(R)
    R = U @ VT
    return R, t


# -------------------------------------------------------------------------
# Calibration pipeline
# -------------------------------------------------------------------------
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

    cam_undist_list = []
    for cam_pts in cam_pts_list:
        cam_pts = np.asarray(cam_pts, dtype=np.float32)
        cam_pts_ud = cv2.undistortPoints(
            cam_pts.reshape(-1, 1, 2), K_cam, dist_cam, P=K_cam
        ).reshape(-1, 2)
        cam_undist_list.append(cam_pts_ud)

    for idx, (cam_ud, proj) in enumerate(zip(cam_undist_list, proj_pts_list)):
        proj = np.asarray(proj, dtype=np.float32)

        if cam_ud.shape[0] < 4 or proj.shape[0] < 4:
            print(f"[CALIB][WARN] Pose {idx}: not enough correspondences (have {cam_ud.shape[0]}), skipping.")
            continue

        H, _ = cv2.findHomography(cam_ud, proj, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is None:
            print(f"[CALIB][WARN] Pose {idx}: homography estimation failed, skipping.")
            continue

        H_list.append(H)

    if not H_list:
        raise RuntimeError("No valid homographies; check dataset and captures.")

    print("[CALIB] Solving projector intrinsics...")
    K_proj = solve_intrinsics_from_homographies(H_list)
    print("Projector K:\n", K_proj)

    # Extrinsics (use first pose as reference)
    H0 = H_list[0]
    R_cam, t_cam = extrinsics_from_H(K_cam, H0)
    H0_inv = np.linalg.inv(H0)
    R_proj, t_proj = extrinsics_from_H(K_proj, H0_inv)

    R_cam_proj = R_proj @ R_cam.T
    t_cam_proj = t_proj - R_cam_proj @ t_cam

    np.savez("projector_intrinsics.npz", K=K_proj)
    np.savez("stereo_params.npz", R=R_cam_proj, t=t_cam_proj)

    print("[CALIB] Saved projector_intrinsics.npz and stereo_params.npz")


def main():
    parser = argparse.ArgumentParser(description="Projector calibration using structured-light dataset.")
    parser.add_argument("--dataset", required=True, help="Path to calib_dataset.npz")
    parser.add_argument("--camera", default="camera_intrinsics.npz", help="Path to camera intrinsics (npz).")
    args = parser.parse_args()
    calibrate(args.dataset, args.camera)


if __name__ == "__main__":
    main()
