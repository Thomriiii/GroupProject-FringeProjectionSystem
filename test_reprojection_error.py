#!/usr/bin/env python3
"""
test_reprojection_error.py

Reprojection error evaluator for structured-light calibration.

This script:
  - Loads:
        camera_intrinsics.npz
        projector_intrinsics.npz
        stereo_params.npz
        calib_dataset.npz
  - For each calibration pose:
        camera pixel (u, v)
        projector pixel (x_proj_meas, y_proj_meas)
    → Project camera ray into projector using extrinsics
    → Compare predicted project pixel to measured projector pixel
  - Outputs:
        RMS reprojection error per-pose
        Overall RMS
        Per-pixel heatmap PNG for each pose
        (red = large error, blue = small)

Usage:
    python test_reprojection_error.py \
        --dataset calib/session_xxx/calib_dataset.npz
"""

from __future__ import annotations
import argparse
import numpy as np
import cv2
import os


# ---------------------------------------------------------------------
# 3D ray from camera pixel
# ---------------------------------------------------------------------
def cam_pixel_to_ray(u, v, K_inv):
    """
    Returns normalized camera ray direction in camera coordinates.
    """
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    d = K_inv @ uv1
    return d / np.linalg.norm(d)


# ---------------------------------------------------------------------
# Project 3D point into projector
# ---------------------------------------------------------------------
def project_point_to_proj(X_cam, K_proj, R, t):
    """
    Transform camera-space point X_cam into projector pixel coordinates.
    """
    X_proj = R @ X_cam + t
    x = X_proj[0] / X_proj[2]
    y = X_proj[1] / X_proj[2]
    uv = K_proj @ np.array([x, y, 1.0])
    return uv[0], uv[1]


# ---------------------------------------------------------------------
# Main reprojection test
# ---------------------------------------------------------------------
def test_reprojection_error(dataset_path):

    # ------------------------
    # Load calibration results
    # ------------------------
    print("[REPROJ] Loading calibration outputs...")
    K_cam = np.load("camera_intrinsics.npz")["K"]
    K_proj = np.load("projector_intrinsics.npz")["K"]
    stereo = np.load("stereo_params.npz")
    R = stereo["R"]
    t = stereo["t"]

    K_cam_inv = np.linalg.inv(K_cam)

    # ------------------------
    # Load dataset
    # ------------------------
    print(f"[REPROJ] Loading dataset: {dataset_path}")
    d = np.load(dataset_path, allow_pickle=True)

    cam_pts_list = d["cam_points_list"]
    proj_pts_list = d["proj_points_list"]
    proj_w, proj_h = d["proj_size"]

    # For making heatmaps
    image_w, image_h = d["image_size_cam"]

    pose_count = len(cam_pts_list)
    print(f"[REPROJ] Found {pose_count} calibration poses.\n")

    all_errors = []

    # --------------------------------------------------------
    # For each pose
    # --------------------------------------------------------
    for pose_idx, (cam_pts, proj_pts_meas) in enumerate(zip(cam_pts_list, proj_pts_list)):
        print(f"[REPROJ] Pose {pose_idx:02d}: computing errors...")

        N = cam_pts.shape[0]
        err_pix = np.zeros(N, dtype=np.float64)

        # Prealloc heatmap
        heatmap = np.zeros((image_h, image_w), dtype=np.float32)

        for i in range(N):
            u, v = cam_pts[i]
            xp_meas, yp_meas = proj_pts_meas[i]

            # 1. ray from camera pixel
            d_cam = cam_pixel_to_ray(u, v, K_cam_inv)

            # 2. Find intersection with projector imaging plane:
            # We assume the camera ray intersects a virtual plane at Z=1 in camera coords:
            # Scale ray so that Z_cam = 1
            if abs(d_cam[2]) < 1e-9:
                continue
            X_cam = d_cam / d_cam[2]

            # 3. Project into projector
            xp_pred, yp_pred = project_point_to_proj(X_cam, K_proj, R, t)

            # 4. Error
            err = np.sqrt((xp_pred - xp_meas)**2 + (yp_pred - yp_meas)**2)
            err_pix[i] = err

            ix = int(v)
            iy = int(u)
            if 0 <= ix < image_h and 0 <= iy < image_w:
                heatmap[ix, iy] = err

        # summary
        rms = np.sqrt(np.mean(err_pix**2))
        mean_err = np.mean(err_pix)
        max_err = np.max(err_pix)

        print(f"  Pose {pose_idx:02d} RMS error:  {rms:.3f} px")
        print(f"  Mean error: {mean_err:.3f} px")
        print(f"  Max error:  {max_err:.3f} px\n")

        all_errors.extend(err_pix.tolist())

        # write heatmap
        heatmap_norm = heatmap / (np.max(heatmap) + 1e-6)
        heatmap_u8 = (255 * heatmap_norm).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        out_path = f"reproj_pose_{pose_idx:02d}.png"
        cv2.imwrite(out_path, heatmap_color)
        print(f"  Saved heatmap: {out_path}")

    # Final overall error
    all_errors = np.array(all_errors)
    overall_rms = np.sqrt(np.mean(all_errors**2))
    print("\n==============================================")
    print(f"Overall RMS reprojection error: {overall_rms:.3f} px")
    print("==============================================\n")

    return overall_rms


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="calib_dataset.npz from calibration stage"
    )
    args = parser.parse_args()

    rms = test_reprojection_error(args.dataset)

    print("\n[REPROJ] Done.")
    print("If RMS < 2 px, calibration is good.")
    print("If RMS < 1 px, calibration is excellent.")


if __name__ == "__main__":
    main()
