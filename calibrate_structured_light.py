#!/usr/bin/env python3
"""
calibrate_structured_light.py

Offline helper script for structured-light camera–projector calibration.

Stage 1: DATASET EXTRACTION

This script:
  - Scans a calibration session directory produced by the scanner's
    "Capture Calibration Pose" feature.
  - For each pose:
      * Loads final vertical & horizontal unwrapped phase maps
        (phase_vert_final.npy, phase_horiz_final.npy)
      * Loads their masks (mask_vert_final.npy, mask_horiz_final.npy)
      * Converts phases → projector pixel coordinates (x_proj, y_proj)
      * Extracts matched (u, v) camera pixel coordinates
        and corresponding (x_proj, y_proj) projector coordinates.
  - Subsamples correspondences to keep data manageable.
  - Saves a consolidated dataset:

      session_root/
        calib_dataset.npz

    containing:
      - cam_points_list : list of (N_i x 2) float32 arrays (per pose)
      - proj_points_list: list of (N_i x 2) float32 arrays (per pose)
      - image_size_cam  : (width, height)
      - proj_size       : (proj_width, proj_height)
      - freqs           : list of frequencies used
      - f_max           : highest frequency (used for phase→pixel mapping)

STAGE 2 (NOT IMPLEMENTED HERE):

  From this dataset we will later derive:
    - Camera intrinsics (K_cam, dist_cam)
    - Projector intrinsics (K_proj, dist_proj)
    - Camera–projector extrinsics (R, t)

Usage:
    python calibrate_structured_light.py \
        --session calib/session_YYYYMMDD_HHMMSS \
        --proj-width 1920 \
        --proj-height 1080 \
        --freqs 4 8 16 32

If --session is omitted, the script picks the latest session under 'calib/'.
"""

from __future__ import annotations

import argparse
import os
import glob
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------
# CONFIG DEFAULTS
# ---------------------------------------------------------------------

DEFAULT_CALIB_ROOT = "calib"
DEFAULT_PROJ_WIDTH = 1920
DEFAULT_PROJ_HEIGHT = 1080
DEFAULT_FREQS = [4, 8, 16, 32]
DEFAULT_SUBSAMPLE = 8  # pick every Nth pixel to reduce dataset size


# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------

def find_latest_session(calib_root: str) -> str | None:
    """
    Find the most recent calibration session directory under calib_root.

    Looks for directories matching 'session_*'.
    """
    pattern = os.path.join(calib_root, "session_*")
    candidates = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    if not candidates:
        return None
    # sort by name (timestamp in name) and take last
    candidates.sort()
    return candidates[-1]


def list_pose_dirs(session_root: str) -> List[str]:
    """
    List pose directories (pose_XXX) in a session.
    """
    pattern = os.path.join(session_root, "pose_*")
    pose_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    pose_dirs.sort()
    return pose_dirs


def phase_to_projector_coord(
    phi: np.ndarray,
    proj_size: Tuple[int, int],
    f_max: int,
    axis: str,
) -> np.ndarray:
    """
    Convert final unwrapped phase map at highest frequency into projector
    pixel coordinates along one axis.

    Assumes:
      - Patterns were generated with:

          phase = 2π * f * coord_norm + delta, coord_norm in [0, 1)

        so the final temporally unwrapped phase satisfies:

          phi_final ≈ 2π * f_max * coord_norm

      - Therefore:

          coord_norm ≈ phi_final / (2π * f_max)

      - Projector coordinate:

          x_proj = coord_norm * proj_width
          y_proj = coord_norm * proj_height

    Parameters
    ----------
    phi : HxW float
        Unwrapped phase map (final, highest frequency).
    proj_size : (width, height)
        Projector resolution.
    f_max : int
        Highest frequency used in PSP patterns.
    axis : {"x", "y"}
        Which projector axis to map to.

    Returns
    -------
    coord_proj : HxW float32
        Projector coordinate per pixel (either x or y).
    """
    proj_w, proj_h = proj_size
    phi = phi.astype(np.float64)

    # Map phase -> normalized coord
    coord_norm = phi / (2.0 * np.pi * float(f_max))

    # Optional clipping to [0, 1)
    coord_norm = np.mod(coord_norm, 1.0)  # wrap into [0,1)
    coord_norm = np.clip(coord_norm, 0.0, 0.999999)

    if axis == "x":
        coord_proj = coord_norm * float(proj_w - 1)
    elif axis == "y":
        coord_proj = coord_norm * float(proj_h - 1)
    else:
        raise ValueError("axis must be 'x' or 'y'")

    return coord_proj.astype(np.float32)


# ---------------------------------------------------------------------
# MAIN DATASET EXTRACTION
# ---------------------------------------------------------------------

def build_calib_dataset(
    session_root: str,
    proj_width: int,
    proj_height: int,
    freqs: List[int],
    subsample: int = DEFAULT_SUBSAMPLE,
) -> str:
    """
    Build a calibration dataset from the given session.

    Parameters
    ----------
    session_root : str
        Path to a session_... directory under calib/.
    proj_width, proj_height : int
        Projector resolution in pixels.
    freqs : list[int]
        Frequencies used in PSP patterns.
    subsample : int
        Subsampling factor for correspondences (e.g. 8 → every 8th pixel).

    Returns
    -------
    out_path : str
        Path to the saved calib_dataset.npz.
    """
    proj_size = (proj_width, proj_height)
    f_max = max(freqs)

    pose_dirs = list_pose_dirs(session_root)
    if not pose_dirs:
        raise RuntimeError(f"No pose_* directories found in session: {session_root}")

    print(f"[CALIB-DATA] Found {len(pose_dirs)} pose(s) in {session_root}")

    cam_points_list = []
    proj_points_list = []

    image_size_cam = None

    for pose_dir in pose_dirs:
        print(f"[CALIB-DATA] Processing pose: {pose_dir}")

        # Load vertical phase + mask
        phase_vert_path = os.path.join(pose_dir, "phase_vert_final.npy")
        mask_vert_path = os.path.join(pose_dir, "mask_vert_final.npy")

        # Load horizontal phase + mask
        phase_horiz_path = os.path.join(pose_dir, "phase_horiz_final.npy")
        mask_horiz_path = os.path.join(pose_dir, "mask_horiz_final.npy")

        if not (os.path.exists(phase_vert_path) and
                os.path.exists(mask_vert_path) and
                os.path.exists(phase_horiz_path) and
                os.path.exists(mask_horiz_path)):
            print(f"  [WARN] Missing phase/mask files in {pose_dir}, skipping.")
            continue

        Phi_vert = np.load(phase_vert_path)    # HxW
        mask_vert = np.load(mask_vert_path)    # HxW bool
        Phi_horiz = np.load(phase_horiz_path)  # HxW
        mask_horiz = np.load(mask_horiz_path)  # HxW bool

        if image_size_cam is None:
            H_cam, W_cam = Phi_vert.shape
            image_size_cam = (W_cam, H_cam)
            print(f"  [INFO] Camera image size: {image_size_cam[0]}x{image_size_cam[1]}")

        # Combined mask: valid where both vertical and horizontal are valid
        mask_both = mask_vert & mask_horiz

        if not np.any(mask_both):
            print("  [WARN] No valid pixels for this pose (mask empty), skipping.")
            continue

        # Compute projector coordinates from phase maps
        x_proj_map = phase_to_projector_coord(
            Phi_vert,
            proj_size=proj_size,
            f_max=f_max,
            axis="x",
        )
        y_proj_map = phase_to_projector_coord(
            Phi_horiz,
            proj_size=proj_size,
            f_max=f_max,
            axis="y",
        )

        # Subsample the mask for manageable correspondences
        mask_sub = np.zeros_like(mask_both)
        mask_sub[::subsample, ::subsample] = True
        mask_use = mask_both & mask_sub

        ys, xs = np.where(mask_use)
        if ys.size == 0:
            print("  [WARN] No valid pixels after subsampling, skipping.")
            continue

        # Camera points: (u, v) = (x, y)
        cam_pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)

        # Projector points: (x_proj, y_proj)
        proj_x = x_proj_map[ys, xs]
        proj_y = y_proj_map[ys, xs]
        proj_pts = np.stack([proj_x, proj_y], axis=1)

        print(f"  [INFO] Using {cam_pts.shape[0]} correspondences for this pose.")

        cam_points_list.append(cam_pts)
        proj_points_list.append(proj_pts)

    if not cam_points_list:
        raise RuntimeError("No valid correspondences found in any pose. "
                           "Check masks / phase maps / subsample factor.")

    # Save dataset
    out_path = os.path.join(session_root, "calib_dataset.npz")
    np.savez(
        out_path,
        cam_points_list=np.array(cam_points_list, dtype=object),
        proj_points_list=np.array(proj_points_list, dtype=object),
        image_size_cam=(W_cam, H_cam),
        proj_size=np.array([proj_width, proj_height], dtype=np.int32),
        freqs=np.array(freqs, dtype=np.int32),
        f_max=int(f_max),
    )

    print(f"[CALIB-DATA] Saved calibration dataset to: {out_path}")
    print(f"[CALIB-DATA] Poses used: {len(cam_points_list)}")

    return out_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Structured-light calibration dataset builder."
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Path to a specific calibration session (calib/session_...). "
             "If omitted, the latest session in 'calib/' is used.",
    )
    parser.add_argument(
        "--calib-root",
        type=str,
        default=DEFAULT_CALIB_ROOT,
        help=f"Calibration root directory (default: {DEFAULT_CALIB_ROOT})",
    )
    parser.add_argument(
        "--proj-width",
        type=int,
        default=DEFAULT_PROJ_WIDTH,
        help=f"Projector width in pixels (default: {DEFAULT_PROJ_WIDTH})",
    )
    parser.add_argument(
        "--proj-height",
        type=int,
        default=DEFAULT_PROJ_HEIGHT,
        help=f"Projector height in pixels (default: {DEFAULT_PROJ_HEIGHT})",
    )
    parser.add_argument(
        "--freqs",
        type=int,
        nargs="+",
        default=DEFAULT_FREQS,
        help=f"PSP frequencies used (default: {DEFAULT_FREQS})",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=DEFAULT_SUBSAMPLE,
        help=f"Subsampling factor for correspondences (default: {DEFAULT_SUBSAMPLE})",
    )

    args = parser.parse_args()

    # Resolve session path
    if args.session is not None:
        session_root = args.session
        if not os.path.isdir(session_root):
            raise RuntimeError(f"Specified session path does not exist: {session_root}")
    else:
        session_root = find_latest_session(args.calib_root)
        if session_root is None:
            raise RuntimeError(
                f"No calibration sessions found under calib root: {args.calib_root}"
            )
        print(f"[CALIB-DATA] Using latest session: {session_root}")

    dataset_path = build_calib_dataset(
        session_root=session_root,
        proj_width=args.proj_width,
        proj_height=args.proj_height,
        freqs=args.freqs,
        subsample=args.subsample,
    )

    print("[CALIB-DATA] Dataset ready. Next step is to implement the "
          "intrinsics/extrinsics solver using this file:")
    print(f"    {dataset_path}")


if __name__ == "__main__":
    main()
