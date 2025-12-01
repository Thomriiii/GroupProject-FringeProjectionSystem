#!/usr/bin/env python3
"""
calibration/build_projector_dataset.py

Builds calib_dataset.npz from structured-light calibration poses.
Each pose directory must contain:
    phi_final.npy   (unwrapped vertical phase)
    mask_final.npy  (valid pixels)

Conversion to projector coordinates:
    proj_x = (phi / (2π * f_max)) * proj_width
    proj_y = proj_height / 2  (no horizontal phase available)

Outputs:
    calib_dataset.npz with:
        cam_points_list : list of (N_i x 2) camera pixels
        proj_points_list: list of (N_i x 2) projector pixels
        proj_size       : (proj_width, proj_height)
        image_size_cam  : (W_cam, H_cam)
        f_max           : max frequency used
"""

from __future__ import annotations

import os
import argparse
import numpy as np


def load_pose(pose_dir: str, proj_width: int, proj_height: int, f_max: int):
    # Support both generic names (phi_final/mask_final) and scan.py calibration outputs
    phi_candidates = [
        os.path.join(pose_dir, "phi_final.npy"),
        os.path.join(pose_dir, "phase_vert_final.npy"),
    ]
    mask_candidates = [
        os.path.join(pose_dir, "mask_final.npy"),
        os.path.join(pose_dir, "mask_vert_final.npy"),
    ]

    phi_path = next((p for p in phi_candidates if os.path.isfile(p)), None)
    mask_path = next((p for p in mask_candidates if os.path.isfile(p)), None)

    if phi_path is None or mask_path is None:
        raise FileNotFoundError(
            f"Missing phase/mask files in {pose_dir}; expected phi_final.npy or phase_vert_final.npy "
            f"and corresponding mask."
        )

    Phi = np.load(phi_path)
    mask = np.load(mask_path)

    H, W = Phi.shape

    v_idx, u_idx = np.nonzero(mask)
    cam_pts = np.stack([u_idx.astype(np.float32), v_idx.astype(np.float32)], axis=1)

    proj_x = (Phi[v_idx, u_idx] / (2.0 * np.pi * float(f_max))) * proj_width
    proj_y = np.full_like(proj_x, proj_height * 0.5, dtype=np.float32)
    proj_pts = np.stack([proj_x.astype(np.float32), proj_y], axis=1)

    return cam_pts, proj_pts, (W, H)


def build_dataset(session_root: str, proj_width: int, proj_height: int, f_max: int, subsample: int):
    entries = sorted(os.listdir(session_root))
    pose_dirs = [
        os.path.join(session_root, d)
        for d in entries
        if d.startswith("pose_") and os.path.isdir(os.path.join(session_root, d))
    ]

    if not pose_dirs:
        raise RuntimeError("No pose_* directories found.")

    cam_points_list = []
    proj_points_list = []
    image_size_cam = None

    for pose_dir in pose_dirs:
        cam_pts, proj_pts, img_size = load_pose(pose_dir, proj_width, proj_height, f_max)

        if subsample > 1:
            cam_pts = cam_pts[::subsample]
            proj_pts = proj_pts[::subsample]

        if cam_pts.shape[0] < 4:
            print(f"[DATA][WARN] {pose_dir}: not enough points after masking/subsample; skipping pose.")
            continue

        cam_points_list.append(cam_pts)
        proj_points_list.append(proj_pts)
        image_size_cam = img_size

        print(f"[DATA] {pose_dir}: {cam_pts.shape[0]} points")

    if not cam_points_list:
        raise RuntimeError("No valid poses contained enough correspondences; dataset is empty.")

    out_path = os.path.join(session_root, "calib_dataset.npz")
    np.savez(
        out_path,
        cam_points_list=np.array(cam_points_list, dtype=object),
        proj_points_list=np.array(proj_points_list, dtype=object),
        proj_size=(proj_width, proj_height),
        image_size_cam=image_size_cam,
        f_max=f_max,
    )
    print(f"[DATA] Saved dataset to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build projector calibration dataset.")
    parser.add_argument("--session", required=True, help="Session root containing pose_* folders.")
    parser.add_argument("--proj-width", type=int, required=True, help="Projector width in pixels.")
    parser.add_argument("--proj-height", type=int, required=True, help="Projector height in pixels.")
    parser.add_argument("--f-max", type=int, required=True, help="Maximum PSP frequency used.")
    parser.add_argument("--subsample", type=int, default=1, help="Optional subsampling factor.")
    args = parser.parse_args()

    build_dataset(
        session_root=args.session,
        proj_width=args.proj_width,
        proj_height=args.proj_height,
        f_max=args.f_max,
        subsample=args.subsample,
    )


if __name__ == "__main__":
    main()
