"""
triangulation.py

Minimal 3D reconstruction utilities using camera-projector ray
intersection. This uses placeholder projector intrinsics/extrinsics
until a real calibration is available.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2


# =====================================================================
# Intrinsics / Extrinsics
# =====================================================================

def load_camera_intrinsics(path: str | Path = "camera_intrinsics.npz"):
    data = np.load(path)
    K = data["K"]
    dist = data["dist"]
    image_size = data["image_size"]
    return K, dist, image_size


def get_fake_projector_parameters(proj_size: Tuple[int, int]):
    """
    Approximate projector intrinsics estimated from throw geometry.
    """
    proj_w, proj_h = proj_size

    fx = 2515.2
    fy = 2319.3
    cx = proj_w / 2.0  # 960 for 1920x1080
    cy = proj_h / 2.0  # 540 for 1920x1080

    Kp = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    dp = np.zeros(5, dtype=np.float32)
    return Kp, dp


def get_fake_extrinsics():
    """
    Approximate projector pose in camera frame based on tape-measured offsets.
    """
    # Projector 18.5 cm LEFT, 1 cm above, 1.5 cm behind the camera
    T = np.array([-0.185, 0.010, -0.015], dtype=np.float32)  # camera frame

    # Toe-in yaw toward the scene centre (positive because projector sits left).
    # Point the projector z-axis roughly toward (0, 0, z_target) in camera frame.
    z_target = 0.60  # metres in front of camera; used only to derive yaw guess
    vec_to_target = np.array([abs(T[0]), 0.0, z_target - T[2]], dtype=np.float64)
    yaw = float(np.arctan2(vec_to_target[0], vec_to_target[2]))  # radians, ~17 deg
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [ c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,  c]
    ], dtype=np.float32)  # projector → camera

    return R, T


def load_projector_intrinsics(path: str | Path = "projector_intrinsics.npz"):
    path = Path(path)
    if not path.exists():
        return None
    data = np.load(path)
    Kp = data["Kp"]
    dist_p = data["dist_p"]
    image_size = tuple(int(x) for x in data["image_size_proj"])
    rms = float(data["rms"][0]) if "rms" in data else float("nan")
    return Kp, dist_p, image_size, rms


def load_stereo_params(path: str | Path = "stereo_params.npz"):
    path = Path(path)
    if not path.exists():
        return None
    data = np.load(path)
    R = data["R"]
    T = data["T"].reshape(3)
    Kc = data["Kc"]
    dist_c = data["dist_c"]
    Kp = data["Kp"]
    dist_p = data["dist_p"]
    rms = float(data["rms"][0]) if "rms" in data else float("nan")
    return R, T, Kc, dist_c, Kp, dist_p, rms


# =====================================================================
# Rays
# =====================================================================

def compute_camera_rays(cam_uv: np.ndarray, Kc: np.ndarray) -> np.ndarray:
    """
    cam_uv: Nx2 pixel coords (u,v). Returns Nx3 unit direction vectors in camera frame.
    """
    uv1 = np.hstack([cam_uv, np.ones((cam_uv.shape[0], 1), dtype=np.float64)])
    K_inv = np.linalg.inv(Kc.astype(np.float64))
    dirs = (K_inv @ uv1.T).T
    norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    return (dirs / norms).astype(np.float32)


def compute_projector_rays(proj_uv: np.ndarray, Kp: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    proj_uv: Nx2 projector pixel coords. Returns Nx3 unit direction vectors in camera frame.
    Assumes camera looks +Z, projector pose given by (R, T) mapping projector frame into camera frame.
    """
    uv1 = np.hstack([proj_uv, np.ones((proj_uv.shape[0], 1), dtype=np.float64)])
    Kp_inv = np.linalg.inv(Kp.astype(np.float64))
    dirs_proj = (Kp_inv @ uv1.T).T
    norms = np.linalg.norm(dirs_proj, axis=1, keepdims=True) + 1e-12
    dirs_proj = dirs_proj / norms

    # Rotate into camera frame (R maps projector → camera)
    dirs_cam = (R.astype(np.float64) @ dirs_proj.T).T
    norms_cam = np.linalg.norm(dirs_cam, axis=1, keepdims=True) + 1e-12
    return (dirs_cam / norms_cam).astype(np.float32)


# =====================================================================
# Triangulation
# =====================================================================

def triangulate_rays(rays_cam: np.ndarray, rays_proj_cam: np.ndarray, T: np.ndarray):
    """
    rays_cam: Nx3 camera-frame directions (camera origin at 0).
    rays_proj_cam: Nx3 directions of projector rays expressed in camera frame.
    T: projector origin in camera frame.

    Returns points_3d (Nx3) and per-point errors (distance between closest points on the two rays).
    """
    N = rays_cam.shape[0]
    points = np.full((N, 3), np.nan, dtype=np.float32)
    errors = np.full((N,), np.nan, dtype=np.float32)

    for i in range(N):
        a = rays_cam[i]
        b = rays_proj_cam[i]
        p0 = np.zeros(3, dtype=np.float32)
        p1 = T.astype(np.float32)

        a_dot_a = np.dot(a, a)
        b_dot_b = np.dot(b, b)
        a_dot_b = np.dot(a, b)
        denom = a_dot_a * b_dot_b - a_dot_b * a_dot_b
        if np.abs(denom) < 1e-9:
            continue  # nearly parallel

        r = p0 - p1
        a_dot_r = np.dot(a, r)
        b_dot_r = np.dot(b, r)

        s = (a_dot_b * b_dot_r - b_dot_b * a_dot_r) / denom
        t = (a_dot_b * a_dot_r - a_dot_a * b_dot_r) / denom

        p_cam = p0 + s * a
        p_proj = p1 + t * b
        midpoint = 0.5 * (p_cam + p_proj)
        err = np.linalg.norm(p_cam - p_proj)

        points[i] = midpoint.astype(np.float32)
        errors[i] = float(err)

    return points, errors


# =====================================================================
# Point cloud output
# =====================================================================

def save_point_cloud_ply(filename: str | Path, points: np.ndarray, colors: np.ndarray | None = None):
    """
    Save point cloud to PLY. Points shape (N,3); colors optional uint8 (N,3).
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    pts = points.astype(np.float32)
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    if colors is not None:
        colors = colors[valid]

    with open(filename, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, p in enumerate(pts):
            if colors is not None:
                c = colors[i]
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
            else:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")


# =====================================================================
# Reconstruction pipeline
# =====================================================================

def reconstruct_3d_from_scan(scan_dir: str, proj_size: Tuple[int, int], color_path: str | Path | None = None):
    """
    Build a 3D reconstruction using calibrated projector parameters if available,
    otherwise fall back to approximate fake parameters.
    """
    scan_dir = Path(scan_dir)
    u_map = np.load(scan_dir / "proj_u.npy")
    v_map = np.load(scan_dir / "proj_v.npy")
    mask = np.load(scan_dir / "mask_final.npy")

    H, W = u_map.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    valid = mask & np.isfinite(u_map) & np.isfinite(v_map)
    if valid.sum() == 0:
        raise ValueError("No valid projector UV samples after masking/finite check.")
    cam_uv = np.stack([xs[valid], ys[valid]], axis=1).astype(np.float32)
    proj_uv = np.stack([u_map[valid], v_map[valid]], axis=1).astype(np.float32)

    if cam_uv.size == 0:
        raise ValueError("No valid pixels for reconstruction.")

    # Load camera intrinsics
    Kc, dist, _ = load_camera_intrinsics()

    stereo = load_stereo_params()
    stereo_rms = None
    using_fake = False
    rms_proj = None

    if stereo is not None:
        R, T, Kc_s, dist_s, Kp, dp, stereo_rms = stereo
        Kc, dist = Kc_s, dist_s
        print(f"[RECON] Using calibrated projector + stereo. Stereo RMS={stereo_rms:.4f}px")
    else:
        intr = load_projector_intrinsics()
        if intr is not None:
            Kp, dp, _, rms_proj = intr
            print(f"[RECON] Using calibrated projector intrinsics only (fake extrinsics). RMS={rms_proj:.4f}px")
        else:
            Kp, dp = get_fake_projector_parameters(proj_size)
            print("[RECON] WARNING: Projector calibration not found. Using fake intrinsics.")
        R, T = get_fake_extrinsics()
        using_fake = True

    # Compute rays (undistort if calibration is available)
    if stereo is not None or rms_proj is not None:
        cam_norm = cv2.undistortPoints(cam_uv.reshape(-1, 1, 2), Kc, dist)
        proj_norm = cv2.undistortPoints(proj_uv.reshape(-1, 1, 2), Kp, dp)

        def _dirs_from_norm(norm_xy: np.ndarray) -> np.ndarray:
            dirs = np.hstack([norm_xy, np.ones((norm_xy.shape[0], 1), dtype=np.float64)])
            norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
            return (dirs / norms).astype(np.float32)

        rays_cam = _dirs_from_norm(cam_norm[:, 0, :])
        dirs_proj = _dirs_from_norm(proj_norm[:, 0, :])
        dirs_proj_cam = (R.astype(np.float64) @ dirs_proj.T).T
        norms_cam = np.linalg.norm(dirs_proj_cam, axis=1, keepdims=True) + 1e-12
        rays_proj_cam = (dirs_proj_cam / norms_cam).astype(np.float32)
    else:
        rays_cam = compute_camera_rays(cam_uv, Kc)
        rays_proj_cam = compute_projector_rays(proj_uv, Kp, R, T)

    points, errors = triangulate_rays(rays_cam, rays_proj_cam, T)

    # Basic validity diagnostics
    z_vals = points[:, 2]
    finite_pts = np.isfinite(points).all(axis=1)
    if finite_pts.any():
        z_valid = z_vals[finite_pts]
        err_valid = errors[finite_pts]
        print(f"[RECON] UV range u:{np.nanmin(u_map):.2f}-{np.nanmax(u_map):.2f} v:{np.nanmin(v_map):.2f}-{np.nanmax(v_map):.2f}")
        print(f"[RECON] Depth stats (m): min={np.nanmin(z_valid):.3f}, max={np.nanmax(z_valid):.3f}, mean={np.nanmean(z_valid):.3f}")
        print(f"[RECON] Ray intersection error (m): mean={np.nanmean(err_valid):.4f}, median={np.nanmedian(err_valid):.4f}")
        # Angle between camera and projector rays (should vary, nonzero)
        dots = np.einsum("ij,ij->i", rays_cam[finite_pts], rays_proj_cam[finite_pts])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.rad2deg(np.arccos(dots))
        print(f"[RECON] Ray angle stats (deg): min={np.nanmin(angles):.2f}, max={np.nanmax(angles):.2f}, mean={np.nanmean(angles):.2f}")
        if stereo_rms is not None:
            print(f"[RECON] Stereo RMS loaded: {stereo_rms:.4f} px")

    colors = None
    if color_path is not None and os.path.exists(color_path):
        import cv2
        color_img = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color_img is not None:
            colors = color_img.reshape(-1, 3)[valid.ravel()][np.isfinite(points[:, 0])]

    suffix = "calibrated" if not using_fake else "fake"
    np.savez_compressed(
        scan_dir / f"points_{suffix}.npz",
        points=points,
        errors=errors,
        cam_uv=cam_uv,
        proj_uv=proj_uv,
        Kc=Kc,
        dist=dist,
        Kp=Kp,
        dp=dp,
        R=R,
        T=T,
        stereo_rms=np.array([stereo_rms if stereo_rms is not None else np.nan], dtype=np.float32),
    )

    save_point_cloud_ply(scan_dir / f"points_{suffix}.ply", points, colors=colors)
    return points, errors
