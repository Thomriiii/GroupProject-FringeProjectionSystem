#!/usr/bin/env python3
"""
test_graycode_reprojection_error.py

Evaluate reprojection error for Gray code projector calibration.
"""

from __future__ import annotations

import argparse
import numpy as np
import cv2
import os


def project_point_to_cam(xp, yp, K_cam, K_proj, R, t):
    inv_Kp = np.linalg.inv(K_proj)
    xn, yn, _ = inv_Kp @ np.array([xp, yp, 1.0])
    ray_p = np.array([xn, yn, 1.0])
    ray_c = R.T @ (ray_p - t.reshape(3))
    if abs(ray_c[2]) < 1e-9:
        return None
    Xc = ray_c / ray_c[2]
    u, v, _ = K_cam @ Xc
    return u, v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--camera", default="camera_intrinsics.npz")
    parser.add_argument("--proj", default="projector_intrinsics_graycode.npz")
    parser.add_argument("--stereo", default="stereo_params_graycode.npz")
    args = parser.parse_args()

    K_cam = np.load(args.camera)["K"]
    K_proj = np.load(args.proj)["K"]
    stereo = np.load(args.stereo)
    R = stereo["R"]
    t = stereo["t"]

    data = np.load(args.dataset, allow_pickle=True)
    cam_pts_list = data["cam_points_list"]
    proj_pts_list = data["proj_points_list"]
    image_size = data["image_size_cam"]
    W_cam, H_cam = image_size

    all_err = []
    for idx, (cam_pts, proj_pts) in enumerate(zip(cam_pts_list, proj_pts_list)):
        cam_pts = np.asarray(cam_pts)
        proj_pts = np.asarray(proj_pts)
        errs = []
        for (u_gt, v_gt), (xp, yp) in zip(cam_pts, proj_pts):
            proj = project_point_to_cam(xp, yp, K_cam, K_proj, R, t)
            if proj is None:
                continue
            u_pred, v_pred = proj
            errs.append(((u_pred - u_gt) ** 2 + (v_pred - v_gt) ** 2) ** 0.5)
        if not errs:
            continue
        rms = (np.mean(np.array(errs) ** 2)) ** 0.5
        all_err.extend(errs)
        print(f"Pose {idx}: RMS {rms:.3f} px")

        heat = np.zeros((H_cam, W_cam), dtype=np.float32)
        for (u_gt, v_gt), e in zip(cam_pts, errs):
            ui, vi = int(u_gt), int(v_gt)
            if 0 <= ui < W_cam and 0 <= vi < H_cam:
                heat[vi, ui] = e
        heat_norm = heat / (heat.max() + 1e-6)
        heat_u8 = (heat_norm * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        out_path = os.path.join(os.path.dirname(args.dataset), f"reproj_pose_{idx:02d}.png")
        cv2.imwrite(out_path, heat_color)
        print(f"Saved {out_path}")

    if all_err:
        overall = (np.mean(np.array(all_err) ** 2)) ** 0.5
        print(f"Overall RMS: {overall:.3f} px")
    else:
        print("No errors computed.")


if __name__ == "__main__":
    main()
