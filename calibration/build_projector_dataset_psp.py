"""
build_projector_dataset_psp.py

Builds a projector calibration dataset from PSP-unwrapped phase maps.

Each pose directory must contain:
    phi_vert.npy
    phi_horiz.npy
    mask_vert.npy
    mask_horiz.npy

Output dataset (Python dict):
    {
        "cam_points" : list of Nx2 float32 arrays  (camera pixels)
        "proj_points": list of Nx2 float32 arrays  (projector pixels)
        "proj_w"     : int
        "proj_h"     : int
    }
"""

from __future__ import annotations
import os
import numpy as np
import cv2


def phase_to_proj_coords(phi, f_max, proj_size):
    """
    Convert unwrapped phase to projector pixel coordinates.

    phi : float32 array
    f_max : highest frequency used (e.g. 32)
    proj_size : projector dimension (width or height)
    """
    return (phi / (2 * np.pi * f_max)) * proj_size


def load_pose(pose_dir, f_max, proj_w, proj_h):
    """
    Load PSP outputs for one pose and convert to 2D projector coords.

    Returns:
        cam_points  (Nx2)
        proj_points (Nx2)
    """
    phi_v = np.load(os.path.join(pose_dir, "phi_vert.npy")).astype(np.float32)
    phi_h = np.load(os.path.join(pose_dir, "phi_horiz.npy")).astype(np.float32)
    mask_v = np.load(os.path.join(pose_dir, "mask_vert.npy"))
    mask_h = np.load(os.path.join(pose_dir, "mask_horiz.npy"))

    # combined mask
    mask = mask_v & mask_h
    ys, xs = np.where(mask)

    if len(xs) < 2000:
        print(f"[DATASET] WARNING: very few valid points in {pose_dir}")
        return None, None

    # Camera pixels
    cam_pts = np.column_stack([xs, ys]).astype(np.float32)

    # Projector pixels
    proj_x = phase_to_proj_coords(phi_v, f_max, proj_w)
    proj_y = phase_to_proj_coords(phi_h, f_max, proj_h)

    proj_pts = np.column_stack([proj_x[ys, xs], proj_y[ys, xs]]).astype(np.float32)

    return cam_pts, proj_pts


def build_projector_dataset_psp(session_dir, proj_width, proj_height, freqs):
    """
    Build calibration dataset from all pose_XXX folders inside session_dir.

    Args:
        session_dir : root containing pose directories
        proj_width, proj_height : projector resolution
        freqs : list of PSP frequencies (e.g. [4,8,16,32])

    Returns:
        dataset dict
    """

    poses = [
        os.path.join(session_dir, d)
        for d in sorted(os.listdir(session_dir))
        if d.startswith("pose_")
    ]

    print(f"[DATASET] Found {len(poses)} pose(s) in {session_dir}")

    f_max = max(freqs)

    cam_all = []
    proj_all = []

    for p in poses:
        print(f"[DATASET] Loading pose: {p}")
        cam_pts, proj_pts = load_pose(p, f_max, proj_width, proj_height)
        if cam_pts is None:
            continue

        cam_all.append(cam_pts)
        proj_all.append(proj_pts)

    print(f"[DATASET] Using {len(cam_all)} valid pose(s).")

    return {
        "cam_points": cam_all,
        "proj_points": proj_all,
        "proj_w": proj_width,
        "proj_h": proj_height,
    }
