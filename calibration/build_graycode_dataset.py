#!/usr/bin/env python3
"""
build_graycode_dataset.py

Build calibration dataset from Gray code captures.
For each pose_* directory under a session, expects captured PNGs named:
    x_bitXX_on.png/off.png, y_bitXX_on.png/off.png
Decodes to projector X/Y maps, builds camera–projector correspondences,
and saves calib_dataset_graycode.npz with:
    cam_points_list, proj_points_list, proj_size, image_size_cam
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import cv2

from calibration.patterns_graycode.graycode_decode import decode_graycode


def load_images_for_pose(pose_dir: str) -> dict[str, np.ndarray]:
    imgs = {}
    for fname in os.listdir(pose_dir):
        if not fname.endswith(".png"):
            continue
        label = os.path.splitext(fname)[0]
        path = os.path.join(pose_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        imgs[label] = img
    return imgs


def build_dataset(session_root: str, proj_width: int, proj_height: int, bits_x: int, bits_y: int, subsample: int):
    pose_dirs = [
        os.path.join(session_root, d)
        for d in sorted(os.listdir(session_root))
        if d.startswith("pose_") and os.path.isdir(os.path.join(session_root, d))
    ]
    if not pose_dirs:
        raise RuntimeError("No pose_* directories found.")

    cam_points_list = []
    proj_points_list = []
    image_size_cam = None

    for pose_dir in pose_dirs:
        imgs = load_images_for_pose(pose_dir)
        if not imgs:
            print(f"[DATA][WARN] {pose_dir}: no images, skipping.")
            continue

        # Decode
        proj_x, proj_y, mask = decode_graycode(imgs, proj_width, proj_height, bits_x, bits_y)

        v_idx, u_idx = np.nonzero(mask)
        if v_idx.size < 4:
            print(f"[DATA][WARN] {pose_dir}: too few valid points, skipping.")
            continue

        cam_pts = np.stack([u_idx.astype(np.float32), v_idx.astype(np.float32)], axis=1)
        proj_pts = np.stack([proj_x[v_idx, u_idx].astype(np.float32), proj_y[v_idx, u_idx].astype(np.float32)], axis=1)

        if subsample > 1:
            cam_pts = cam_pts[::subsample]
            proj_pts = proj_pts[::subsample]

        cam_points_list.append(cam_pts)
        proj_points_list.append(proj_pts)
        sample_img = imgs[next(iter(imgs))]
        image_size_cam = (sample_img.shape[1], sample_img.shape[0])

        print(f"[DATA] {pose_dir}: {cam_pts.shape[0]} points")

    out_path = os.path.join(session_root, "calib_dataset_graycode.npz")
    np.savez(
        out_path,
        cam_points_list=np.array(cam_points_list, dtype=object),
        proj_points_list=np.array(proj_points_list, dtype=object),
        proj_size=(proj_width, proj_height),
        image_size_cam=image_size_cam,
        bits_x=bits_x,
        bits_y=bits_y,
    )
    print(f"[DATA] Saved dataset to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build Gray code calibration dataset.")
    parser.add_argument("--session", required=True, help="Session root with pose_* folders.")
    parser.add_argument("--proj-width", type=int, required=True)
    parser.add_argument("--proj-height", type=int, required=True)
    parser.add_argument("--bits-x", type=int, required=True)
    parser.add_argument("--bits-y", type=int, required=True)
    parser.add_argument("--subsample", type=int, default=2)
    args = parser.parse_args()

    build_dataset(args.session, args.proj_width, args.proj_height, args.bits_x, args.bits_y, args.subsample)


if __name__ == "__main__":
    main()
