"""
triangulation.py

Minimal 3D reconstruction utilities using camera-projector ray
intersection. This uses placeholder projector intrinsics/extrinsics
until a real calibration is available.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, List
import json

import numpy as np
try:
    import cv2
except Exception:
    cv2 = None


# =====================================================================
# Intrinsics / Extrinsics
# =====================================================================

def load_camera_intrinsics(path: str | Path = "camera_intrinsics.npz"):
    data = np.load(path)
    K = data["K"]
    dist = data["dist"]
    image_size = data["image_size"]
    return K, dist, image_size


FAKE_CONFIG_PATH = Path("config/fake_projector_config.json")


def _load_fake_config():
    if FAKE_CONFIG_PATH.exists():
        try:
            return json.loads(FAKE_CONFIG_PATH.read_text())
        except Exception:
            print(f"[RECON][WARN] Failed to parse {FAKE_CONFIG_PATH}, using defaults.")
    return {}


def get_fake_projector_parameters(proj_size: Tuple[int, int]):
    """
    Approximate projector intrinsics estimated from throw geometry.
    """
    cfg = _load_fake_config()
    proj_w, proj_h = proj_size

    fx = float(cfg.get("fx", 2515.2))
    fy = float(cfg.get("fy", 2319.3))
    cx = float(cfg.get("cx", proj_w / 2.0))
    cy = float(cfg.get("cy", proj_h / 2.0))

    Kp = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    dp = np.zeros(5, dtype=np.float32)
    if "distortion" in cfg and isinstance(cfg["distortion"], (list, tuple)):
        vals = list(cfg["distortion"])
        while len(vals) < 5:
            vals.append(0.0)
        dp = np.array(vals[:5], dtype=np.float32)
    elif FAKE_CONFIG_PATH.exists():
        print("[RECON][WARN] Fake projector distortion not provided; assuming zero.")
    return Kp, dp


def get_fake_extrinsics():
    """
    Approximate projector pose in camera frame based on tape-measured offsets.
    """
    cfg = _load_fake_config()
    # Projector offset (camera frame): LEFT/RIGHT, UP, FORWARD (negative means behind)
    T = np.array(cfg.get("T", [-0.185, 0.010, -0.015]), dtype=np.float32)

    # Toe-in yaw toward the scene centre (positive because projector sits left).
    # Point the projector z-axis roughly toward (0, 0, z_target) in camera frame.
    z_target = float(cfg.get("z_target", 0.60))  # metres in front of camera; used only to derive yaw guess
    yaw_override = cfg.get("yaw_rad", None)
    vec_to_target = np.array([abs(T[0]), 0.0, z_target - T[2]], dtype=np.float64)
    yaw = float(np.arctan2(vec_to_target[0], vec_to_target[2])) if yaw_override is None else float(yaw_override)
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
    uv_bbox = data["uv_bbox"] if "uv_bbox" in data else None
    return Kp, dist_p, image_size, rms, uv_bbox


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
    uv_bbox = data["uv_bbox"] if "uv_bbox" in data else None
    return R, T, Kc, dist_c, Kp, dist_p, rms, uv_bbox


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
    Points where the ray parameters are invalid (s<=0 or t<=0) or rays are near-parallel
    are marked as NaN.
    """
    N = rays_cam.shape[0]
    points = np.full((N, 3), np.nan, dtype=np.float32)
    errors = np.full((N,), np.nan, dtype=np.float32)
    s_params = np.full((N,), np.nan, dtype=np.float32)
    t_params = np.full((N,), np.nan, dtype=np.float32)

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
        t = (a_dot_a * b_dot_r - a_dot_b * a_dot_r) / denom
        s_params[i] = float(s)
        t_params[i] = float(t)

        if s <= 0.0 or t <= 0.0:
            continue

        p_cam = p0 + s * a
        p_proj = p1 + t * b
        midpoint = 0.5 * (p_cam + p_proj)
        err = np.linalg.norm(p_cam - p_proj)

        points[i] = midpoint.astype(np.float32)
        errors[i] = float(err)

    return points, errors, s_params, t_params


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


def save_mesh_ply(filename: str | Path, vertices: np.ndarray, faces: List[Tuple[int, int, int]]):
    """
    Save a triangle mesh (faces reference filtered vertices).
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def save_mesh_obj(filename: str | Path, vertices: np.ndarray, faces: List[Tuple[int, int, int]]):
    """
    Save OBJ mesh with only vertex/face records.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="ascii") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            # OBJ indices are 1-based
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


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
    mask_quality_path = scan_dir / "mask_quality.npy"
    mask_quality = np.load(mask_quality_path) if mask_quality_path.exists() else None
    mask_used = mask_quality if mask_quality is not None else mask
    diag_scan = {}
    diag_path = scan_dir / "scan_diag_scan.json"
    if diag_path.exists():
        try:
            diag_scan = json.loads(diag_path.read_text())
        except Exception:
            diag_scan = {}

    H, W = u_map.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    valid = mask_used & np.isfinite(u_map) & np.isfinite(v_map)
    if valid.sum() == 0:
        raise ValueError("No valid projector UV samples after masking/finite check.")
    cam_uv = np.stack([xs[valid], ys[valid]], axis=1).astype(np.float32)
    proj_uv = np.stack([u_map[valid], v_map[valid]], axis=1).astype(np.float32)
    xs_valid = xs[valid]
    ys_valid = ys[valid]

    if cam_uv.size == 0:
        raise ValueError("No valid pixels for reconstruction.")

    # Load camera intrinsics
    Kc, dist, _ = load_camera_intrinsics()

    stereo = load_stereo_params()
    stereo_rms = None
    using_fake = False

    if stereo is not None:
        R_st, T_st, Kc_s, dist_s, Kp_st, dp_st, stereo_rms, _ = stereo
        Kc, dist = Kc_s, dist_s
        Kp, dp = Kp_st, dp_st
        print(f"[RECON] Using calibrated projector + stereo. Stereo RMS={stereo_rms:.4f}px")
    else:
        if os.getenv("REQUIRE_PROJECTOR_CALIB", "0") == "1":
            raise RuntimeError(
                "stereo_params.npz not found. Please run projector calibration "
                "or unset REQUIRE_PROJECTOR_CALIB=1 to fall back to fake parameters."
            )
        intr = load_projector_intrinsics()
        if intr is not None:
            Kp, dp, _, rms_proj, _ = intr
            print(f"[RECON] Using calibrated projector intrinsics only (fake extrinsics). RMS={rms_proj:.4f}px")
        else:
            Kp, dp = get_fake_projector_parameters(proj_size)
            print("[RECON][WARN] Projector calibration not found. Using FAKE intrinsics/extrinsics; geometry will be approximate.")
            if np.allclose(dp, 0):
                print("[RECON][WARN] Fake projector distortion is zero; distortion is NOT corrected. "
                      f"Provide {FAKE_CONFIG_PATH} with 'distortion' coefficients if known.")
            if FAKE_CONFIG_PATH.exists():
                print(f"[RECON] Loaded fake projector parameters from {FAKE_CONFIG_PATH}.")
        R_st, T_st = get_fake_extrinsics()
        using_fake = True

    # stereo_params.npz already stores projector->camera extrinsics.
    # Do NOT invert or flip here.
    R = R_st
    T = T_st

    if cv2 is not None:
        cam_norm = cv2.undistortPoints(cam_uv.reshape(-1, 1, 2), Kc, dist)[:, 0, :]
        proj_norm = cv2.undistortPoints(proj_uv.reshape(-1, 1, 2), Kp, dp)[:, 0, :]
    else:
        cam_norm = cam_uv
        proj_norm = proj_uv

    def _dirs_from_norm(norm_xy: np.ndarray) -> np.ndarray:
        dirs = np.hstack([norm_xy, np.ones((norm_xy.shape[0], 1), dtype=np.float64)])
        norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        return (dirs / norms).astype(np.float32)

    rays_cam = _dirs_from_norm(cam_norm)
    dirs_proj = _dirs_from_norm(proj_norm)
    dirs_proj_cam = (R.astype(np.float64) @ dirs_proj.T).T
    dirs_proj_cam /= np.linalg.norm(dirs_proj_cam, axis=1, keepdims=True) + 1e-12
    rays_proj_cam = dirs_proj_cam.astype(np.float32)

    points, errors, s_params, t_params = triangulate_rays(rays_cam, rays_proj_cam, T)

    # Basic validity diagnostics
    z_vals = points[:, 2]
    finite_pts = np.isfinite(points).all(axis=1)
    def neighbor_jump_stats(arr: np.ndarray, mask_in: np.ndarray, thresh: float = 20.0):
        # right and down neighbors
        mask = mask_in & np.isfinite(arr)
        # right
        mr = mask[:, :-1] & mask[:, 1:]
        vr = np.abs(arr[:, :-1] - arr[:, 1:])[mr]
        md = mask[:-1, :] & mask[1:, :]
        vd = np.abs(arr[:-1, :] - arr[1:, :])[md]
        total = vr.size + vd.size
        if total == 0:
            return 0.0
        jumps = np.count_nonzero(vr > thresh) + np.count_nonzero(vd > thresh)
        return 100.0 * jumps / float(total)

    if finite_pts.any():
        z_valid = z_vals[finite_pts]
        err_valid = errors[finite_pts]
        pos_frac = np.mean(z_valid > 0)
        print(f"[RECON] UV range u:{np.nanmin(u_map):.2f}-{np.nanmax(u_map):.2f} v:{np.nanmin(v_map):.2f}-{np.nanmax(v_map):.2f}")
        print(f"[RECON] Depth stats (m): min={np.nanmin(z_valid):.3f}, max={np.nanmax(z_valid):.3f}, mean={np.nanmean(z_valid):.3f}, %Z>0={pos_frac*100:.1f}%")
        print(f"[RECON] Ray intersection error (m): mean={np.nanmean(err_valid):.4f}, median={np.nanmedian(err_valid):.4f}, max={np.nanmax(err_valid):.4f}")
        # Angle between camera and projector rays (should vary, nonzero)
        dots = np.einsum("ij,ij->i", rays_cam[finite_pts], rays_proj_cam[finite_pts])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.rad2deg(np.arccos(dots))
        print(f"[RECON] Ray angle stats (deg): min={np.nanmin(angles):.2f}, max={np.nanmax(angles):.2f}, mean={np.nanmean(angles):.2f}")
        jump_u = neighbor_jump_stats(u_map, mask_used)
        jump_v = neighbor_jump_stats(v_map, mask_used)
        print(f"[RECON] UV neighbor jumps >20px: u={jump_u:.2f}% v={jump_v:.2f}%")
        if stereo_rms is not None:
            print(f"[RECON] Stereo RMS loaded: {stereo_rms:.4f} px")
        print(
            f"[RECON] Z mean={np.nanmean(z_valid):.3f} m, "
            f"median ray err={np.nanmedian(err_valid):.4f} m"
        )

    colors = None
    if color_path is not None and os.path.exists(color_path) and cv2 is not None:
        color_img = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color_img is not None:
            colors = color_img.reshape(-1, 3)[valid.ravel()]

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

    # Filter points by triangulation error
    errors_finite = errors[np.isfinite(errors)]
    med_err = float(np.nanmedian(errors_finite)) if errors_finite.size > 0 else float("nan")
    err_cap = None
    ok = np.isfinite(errors)
    if not np.isnan(med_err):
        err_cap = min(0.02, 2.0 * med_err)
        ok &= errors <= err_cap
    keep = ok & finite_pts

    points_filtered = points[keep]
    errors_filtered = errors[keep]
    cam_uv_filtered = cam_uv[keep]
    proj_uv_filtered = proj_uv[keep]
    colors_filtered = colors[keep] if colors is not None else None

    med_err_after = float(np.nanmedian(errors_filtered)) if errors_filtered.size > 0 else float("nan")
    if not np.isnan(med_err):
        cap_val = err_cap if err_cap is not None else float("nan")
        print(f"[RECON] Error cap applied at {cap_val:.4f} m (median before={med_err:.4f}, after={med_err_after:.4f})")

    np.savez_compressed(
        scan_dir / "points_filtered.npz",
        points=points_filtered,
        errors=errors_filtered,
        cam_uv=cam_uv_filtered,
        proj_uv=proj_uv_filtered,
        Kc=Kc,
        dist=dist,
        Kp=Kp,
        dp=dp,
        R=R,
        T=T,
        stereo_rms=np.array([stereo_rms if stereo_rms is not None else np.nan], dtype=np.float32),
        mask_used=mask_used,
        valid_mask=valid,
        median_error=np.array([med_err], dtype=np.float32),
    )
    save_point_cloud_ply(scan_dir / "points_filtered.ply", points_filtered, colors=colors_filtered)

    # Build single-view mesh (2.5D) from filtered points
    mesh_stride = 1  # increase if Pi performance is constrained
    idx_map = -np.ones((H, W), dtype=np.int32)
    if points_filtered.size > 0:
        idx_map[ys_valid[keep].astype(int), xs_valid[keep].astype(int)] = np.arange(points_filtered.shape[0], dtype=np.int32)

    faces: List[Tuple[int, int, int]] = []
    for y in range(0, H - mesh_stride, mesh_stride):
        for x in range(0, W - mesh_stride, mesh_stride):
            i00 = idx_map[y, x]
            i10 = idx_map[y, x + mesh_stride]
            i01 = idx_map[y + mesh_stride, x]
            i11 = idx_map[y + mesh_stride, x + mesh_stride]
            if (i00 < 0) or (i10 < 0) or (i01 < 0) or (i11 < 0):
                continue
            faces.append((int(i00), int(i10), int(i01)))
            faces.append((int(i10), int(i11), int(i01)))

    save_mesh_ply(scan_dir / "mesh_single_view.ply", points_filtered, faces)
    save_mesh_obj(scan_dir / "mesh_single_view.obj", points_filtered, faces)

    # Diagnostics summary
    mask_quality_pct = float(100.0 * np.count_nonzero(mask_used) / mask_used.size)
    u_vals = u_map[valid]
    v_vals = v_map[valid]
    z_filtered = points_filtered[:, 2] if points_filtered.size > 0 else np.array([], dtype=np.float32)
    err_stats = {
        "median": float(np.nanmedian(errors_filtered)) if errors_filtered.size > 0 else None,
        "mean": float(np.nanmean(errors_filtered)) if errors_filtered.size > 0 else None,
        "p95": float(np.nanpercentile(errors_filtered, 95.0)) if errors_filtered.size > 0 else None,
    }
    jump_u = neighbor_jump_stats(u_map, mask_used)
    jump_v = neighbor_jump_stats(v_map, mask_used)
    summary = {
        "mask_quality_pct": mask_quality_pct,
        "proj_u_min": float(np.nanmin(u_vals)) if u_vals.size > 0 else None,
        "proj_u_max": float(np.nanmax(u_vals)) if u_vals.size > 0 else None,
        "proj_v_min": float(np.nanmin(v_vals)) if v_vals.size > 0 else None,
        "proj_v_max": float(np.nanmax(v_vals)) if v_vals.size > 0 else None,
        "z_min": float(np.nanmin(z_filtered)) if z_filtered.size > 0 else None,
        "z_max": float(np.nanmax(z_filtered)) if z_filtered.size > 0 else None,
        "z_mean": float(np.nanmean(z_filtered)) if z_filtered.size > 0 else None,
        "ray_error": err_stats,
        "ray_error_median_before_cap": med_err if not np.isnan(med_err) else None,
        "ray_error_median_after_cap": med_err_after if not np.isnan(med_err_after) else None,
        "ray_error_cap": err_cap,
        "neighbor_jumps_px": {
            "proj_u": jump_u,
            "proj_v": jump_v,
        },
    }
    summary.update(diag_scan)
    with open(scan_dir / "scan_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return points, errors
