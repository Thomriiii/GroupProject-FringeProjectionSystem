"""
triangulation.py

3D reconstruction utilities using camera-projector ray intersection.

The module supports calibrated projector parameters when available and
falls back to approximate intrinsics/extrinsics if calibration is missing.
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


# --- Intrinsics and extrinsics helpers ---

def load_camera_intrinsics(path: str | Path = "camera_intrinsics.npz"):
    """
    Load camera intrinsics, distortion, and image size from an NPZ file.
    """
    data = np.load(path)
    K = data["K"]
    dist = data["dist"]
    image_size = data["image_size"]
    # Stored as (W, H) int32 in the file.
    image_size_wh = (int(image_size[0]), int(image_size[1]))
    return K, dist, image_size_wh


def _format_K(K: np.ndarray) -> str:
    """
    Format a camera matrix into a compact string for logs.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}"


def _warn_if_principal_point_far(K: np.ndarray, image_size_wh: Tuple[int, int], frac: float = 0.10) -> None:
    """
    Print warnings or raise if the principal point looks implausible.
    """
    w, h = image_size_wh
    cx, cy = float(K[0, 2]), float(K[1, 2])
    # Hard validity checks catch swapped axes or wrong units early.
    if not np.isfinite([cx, cy, K[0, 0], K[1, 1]]).all():
        raise RuntimeError("[RECON][FATAL] Non-finite intrinsics detected.")
    if float(K[0, 0]) <= 0 or float(K[1, 1]) <= 0:
        raise RuntimeError("[RECON][FATAL] Non-positive focal length in intrinsics.")
    if (cx < -0.5 * w) or (cx > 1.5 * w) or (cy < -0.5 * h) or (cy > 1.5 * h):
        raise RuntimeError(
            "[RECON][FATAL] Principal point is far outside the image bounds. "
            "This strongly suggests a calibration/scan resolution mismatch or an unintended crop/rotation."
        )
    dx = abs(cx - (w / 2.0)) / float(w)
    dy = abs(cy - (h / 2.0)) / float(h)
    if (dx > frac) or (dy > frac):
        print(
            "[RECON][WARN] Principal point far from image center "
            f"(dx={dx*100:.1f}%, dy={dy*100:.1f}%). This can indicate cropping/rotation, "
            "sensor mode changes, or a weak calibration dataset."
        )


def _rescale_intrinsics(K: np.ndarray, calib_size_wh: Tuple[int, int], new_size_wh: Tuple[int, int]) -> np.ndarray:
    """
    Rescale camera intrinsics from calib_size_wh -> new_size_wh assuming the image was
    resized (no cropping/ROI). This preserves the normalized coordinates:
        x = (u - cx) / fx, y = (v - cy) / fy
    """
    calib_w, calib_h = calib_size_wh
    new_w, new_h = new_size_wh
    sx = float(new_w) / float(calib_w)
    sy = float(new_h) / float(calib_h)

    K2 = K.astype(np.float64).copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return K2.astype(K.dtype)


def load_camera_intrinsics_for_image(
    image_size_wh: Tuple[int, int],
    path: str | Path = "camera_intrinsics.npz",
    *,
    allow_rescale: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Load intrinsics and verify they match the given image size (W, H).

    If sizes differ only by a uniform resize, set allow_rescale=True to rescale
    fx/fy/cx/cy. This is logged explicitly; it is never silent.
    """
    K, dist, calib_size = load_camera_intrinsics(path)
    scan_w, scan_h = (int(image_size_wh[0]), int(image_size_wh[1]))
    if calib_size != (scan_w, scan_h):
        msg = f"calibration image_size={calib_size[0]}x{calib_size[1]} != scan image_size={scan_w}x{scan_h}"
        if not allow_rescale:
            raise RuntimeError(
                "[RECON][FATAL] Camera calibration resolution mismatch: " + msg + ". "
                "Either re-capture calibration at the scan resolution, or set ALLOW_INTRINSIC_RESCALE=1 "
                "to rescale intrinsics for pure image resizing (no cropping)."
            )

        # Only rescale if aspect ratio matches (resize without crop).
        calib_w, calib_h = calib_size
        if abs((scan_w / scan_h) - (calib_w / calib_h)) > 1e-6:
            raise RuntimeError(
                "[RECON][FATAL] Cannot rescale intrinsics: aspect ratio differs, indicating cropping/ROI/rotation. "
                + msg
            )

        K2 = _rescale_intrinsics(K, calib_size, (scan_w, scan_h))
        print(f"[RECON][WARN] Rescaling camera intrinsics due to size mismatch: {msg}")
        print(f"[RECON]  K (calib): {_format_K(K)}")
        print(f"[RECON]  K (scan) : {_format_K(K2)}")
        K = K2

    _warn_if_principal_point_far(K, (scan_w, scan_h))
    return K, dist, (scan_w, scan_h)


def prepare_intrinsics_for_image(
    K: np.ndarray,
    dist: np.ndarray,
    calib_size_wh: Tuple[int, int],
    image_size_wh: Tuple[int, int],
    *,
    allow_rescale: bool = False,
    label: str = "camera",
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Like load_camera_intrinsics_for_image, but operates on an already-loaded (K, dist)
    and an explicit calib_size_wh (W, H).

    This is used for stereo_params.npz where intrinsics are embedded but image size is not.
    """
    scan_w, scan_h = (int(image_size_wh[0]), int(image_size_wh[1]))
    if calib_size_wh != (scan_w, scan_h):
        msg = f"calibration image_size={calib_size_wh[0]}x{calib_size_wh[1]} != scan image_size={scan_w}x{scan_h}"
        if not allow_rescale:
            raise RuntimeError(
                f"[RECON][FATAL] {label} calibration resolution mismatch: " + msg + ". "
                "Either re-calibrate at the scan resolution, or set ALLOW_INTRINSIC_RESCALE=1 "
                "to rescale intrinsics for pure image resizing (no cropping)."
            )

        calib_w, calib_h = calib_size_wh
        if abs((scan_w / scan_h) - (calib_w / calib_h)) > 1e-6:
            raise RuntimeError(
                f"[RECON][FATAL] Cannot rescale {label} intrinsics: aspect ratio differs, indicating cropping/ROI/rotation. "
                + msg
            )

        K2 = _rescale_intrinsics(K, calib_size_wh, (scan_w, scan_h))
        print(f"[RECON][WARN] Rescaling {label} intrinsics due to size mismatch: {msg}")
        print(f"[RECON]  K (calib): {_format_K(K)}")
        print(f"[RECON]  K (scan) : {_format_K(K2)}")
        K = K2

    _warn_if_principal_point_far(K, (scan_w, scan_h))
    return K, dist, (scan_w, scan_h)


FAKE_CONFIG_PATH = Path("config/fake_projector_config.json")
UV_CONVENTION_PATH = Path("config/uv_convention.json")


def _load_fake_config():
    """
    Load optional fake projector parameters from config if present.
    """
    if FAKE_CONFIG_PATH.exists():
        try:
            return json.loads(FAKE_CONFIG_PATH.read_text())
        except Exception:
            print(f"[RECON][WARN] Failed to parse {FAKE_CONFIG_PATH}, using defaults.")
    return {}


def _load_uv_convention() -> dict:
    """
    Optional configuration for mapping (u,v) produced by the scan decoder into the
    projector calibration coordinate system.

    This is intentionally explicit: if your point cloud looks like a streak/fan but
    ray intersection errors are low, a swapped/flipped UV convention is a common cause.

    File format:
      config/uv_convention.json:
        { "transform": "identity" }

    Supported transforms: identity, swap, flip_u, flip_v, flip_u_flip_v, swap_flip_u, swap_flip_v
    """
    if UV_CONVENTION_PATH.exists():
        try:
            return json.loads(UV_CONVENTION_PATH.read_text())
        except Exception:
            print(f"[RECON][WARN] Failed to parse {UV_CONVENTION_PATH}, using default UV convention.")
    return {}


def _apply_proj_uv_transform(proj_uv: np.ndarray, proj_size: Tuple[int, int], name: str) -> np.ndarray:
    """
    Apply a *pure* coordinate convention transform to projector pixel coordinates.
    This does not change PSP decoding math; it only aligns axis conventions.
    """
    name = (name or "identity").strip().lower()
    if name == "identity":
        return proj_uv

    proj_w, proj_h = int(proj_size[0]), int(proj_size[1])
    u = proj_uv[:, 0]
    v = proj_uv[:, 1]

    if name == "swap":
        return np.stack([v, u], axis=1).astype(np.float32)
    if name == "flip_u":
        return np.stack([(proj_w - 1) - u, v], axis=1).astype(np.float32)
    if name == "flip_v":
        return np.stack([u, (proj_h - 1) - v], axis=1).astype(np.float32)
    if name == "flip_u_flip_v":
        return np.stack([(proj_w - 1) - u, (proj_h - 1) - v], axis=1).astype(np.float32)
    if name == "swap_flip_u":
        # Swap then flip new u (which was v).
        return np.stack([(proj_w - 1) - v, u], axis=1).astype(np.float32)
    if name == "swap_flip_v":
        # Swap then flip new v (which was u).
        return np.stack([v, (proj_h - 1) - u], axis=1).astype(np.float32)

    raise ValueError(f"Unknown PROJ_UV_TRANSFORM '{name}'")


def _diagnose_proj_uv_transform(
    cam_uv: np.ndarray,
    proj_uv: np.ndarray,
    proj_size: Tuple[int, int],
    Kc: np.ndarray,
    dist_c: np.ndarray,
    Kp: np.ndarray,
    dist_p: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    *,
    sample: int = 20000,
) -> None:
    """
    Evaluate candidate UV convention transforms and report ray error stats.
    """
    if cv2 is None:
        print("[RECON][WARN] UV transform diagnosis requires OpenCV (cv2).")
        return
    n = cam_uv.shape[0]
    if n == 0:
        return
    m = min(sample, n)
    idx = np.random.default_rng(0).choice(n, size=m, replace=False)
    cam_uv_s = cam_uv[idx].astype(np.float32)
    proj_uv_s = proj_uv[idx].astype(np.float32)

    cam_norm = cv2.undistortPoints(cam_uv_s.reshape(-1, 1, 2), Kc, dist_c)[:, 0, :]

    def _dirs(norm_xy: np.ndarray) -> np.ndarray:
        """Convert normalized image coordinates into unit ray directions."""
        dirs = np.hstack([norm_xy, np.ones((norm_xy.shape[0], 1), dtype=np.float64)])
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        return dirs.astype(np.float32)

    rays_cam = _dirs(cam_norm)

    candidates = ["identity", "swap", "flip_u", "flip_v", "flip_u_flip_v", "swap_flip_u", "swap_flip_v"]
    stats = []
    for name in candidates:
        uv_t = _apply_proj_uv_transform(proj_uv_s, proj_size, name)
        proj_norm = cv2.undistortPoints(uv_t.reshape(-1, 1, 2), Kp, dist_p)[:, 0, :]
        dirs_proj = _dirs(proj_norm)
        dirs_proj_cam = (R.astype(np.float64) @ dirs_proj.T).T
        dirs_proj_cam /= np.linalg.norm(dirs_proj_cam, axis=1, keepdims=True) + 1e-12
        _, errs, _, _ = triangulate_rays(rays_cam, dirs_proj_cam.astype(np.float32), T)
        ok = np.isfinite(errs)
        if not ok.any():
            continue
        stats.append((float(np.nanmedian(errs[ok])), float(np.nanmean(errs[ok])), name))

    if not stats:
        return
    stats.sort(key=lambda t: t[0])
    print("[RECON][DIAG] Projector UV convention candidates (lower is better):")
    for med, mean, name in stats:
        print(f"  - {name:12s} median_err={med:.4f} m mean_err={mean:.4f} m")
    best = stats[0][2]
    if best != "identity":
        print(
            f"[RECON][DIAG] Best candidate is '{best}'. If your point cloud looks wrong, set "
            f"PROJ_UV_TRANSFORM={best} (or write it into {UV_CONVENTION_PATH})."
        )


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
    # Projector offset in camera frame: left/right, up, forward (negative means behind).
    T = np.array(cfg.get("T", [-0.185, 0.010, -0.015]), dtype=np.float32)

    # Toe-in yaw toward the scene center (positive because projector sits left).
    # Point the projector z-axis roughly toward (0, 0, z_target) in camera frame.
    z_target = float(cfg.get("z_target", 0.60))  # Meters in front of camera; used only to derive yaw guess.
    yaw_override = cfg.get("yaw_rad", None)
    vec_to_target = np.array([abs(T[0]), 0.0, z_target - T[2]], dtype=np.float64)
    yaw = float(np.arctan2(vec_to_target[0], vec_to_target[2])) if yaw_override is None else float(yaw_override)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [ c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,  c]
    ], dtype=np.float32)  # projector -> camera

    return R, T


def load_projector_intrinsics(path: str | Path = "projector_intrinsics.npz"):
    """
    Load projector intrinsics from an NPZ file if it exists.
    """
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
    """
    Load stereo calibration parameters from an NPZ file if it exists.
    """
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


# --- Ray construction ---

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

    # Rotate into camera frame (R maps projector -> camera).
    dirs_cam = (R.astype(np.float64) @ dirs_proj.T).T
    norms_cam = np.linalg.norm(dirs_cam, axis=1, keepdims=True) + 1e-12
    return (dirs_cam / norms_cam).astype(np.float32)


# --- Triangulation ---

def triangulate_rays(rays_cam: np.ndarray, rays_proj_cam: np.ndarray, T: np.ndarray):
    """
    rays_cam: Nx3 camera-frame directions (camera origin at 0).
    rays_proj_cam: Nx3 directions of projector rays expressed in camera frame.
    T: projector origin in camera frame.

    Returns points_3d (Nx3) and per-point errors (distance between closest points on the two rays).
    Points where the ray parameters are invalid (s<=0 or t<=0) or rays are near-parallel
    are marked as NaN.
    """
    a = rays_cam.astype(np.float64)
    b = rays_proj_cam.astype(np.float64)
    T = T.astype(np.float64).reshape(3)
    N = a.shape[0]

    points = np.full((N, 3), np.nan, dtype=np.float32)
    errors = np.full((N,), np.nan, dtype=np.float32)
    s_params = np.full((N,), np.nan, dtype=np.float32)
    t_params = np.full((N,), np.nan, dtype=np.float32)

    a_dot_a = np.einsum("ij,ij->i", a, a)
    b_dot_b = np.einsum("ij,ij->i", b, b)
    a_dot_b = np.einsum("ij,ij->i", a, b)
    denom = a_dot_a * b_dot_b - a_dot_b * a_dot_b
    valid = np.abs(denom) >= 1e-9
    if not np.any(valid):
        return points, errors, s_params, t_params

    r = -T
    a_dot_r = a @ r
    b_dot_r = b @ r

    s = (a_dot_b * b_dot_r - b_dot_b * a_dot_r) / denom
    t = (a_dot_a * b_dot_r - a_dot_b * a_dot_r) / denom
    s_params[valid] = s[valid].astype(np.float32)
    t_params[valid] = t[valid].astype(np.float32)

    valid &= (s > 0.0) & (t > 0.0)
    if not np.any(valid):
        return points, errors, s_params, t_params

    p_cam = a[valid] * s[valid][:, None]
    p_proj = T[None, :] + b[valid] * t[valid][:, None]
    midpoint = 0.5 * (p_cam + p_proj)
    err = np.linalg.norm(p_cam - p_proj, axis=1)

    points[valid] = midpoint.astype(np.float32)
    errors[valid] = err.astype(np.float32)

    return points, errors, s_params, t_params


# --- Point cloud output ---

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
            # OBJ indices are 1-based.
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


# --- Reconstruction pipeline ---

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
    scan_size = (int(W), int(H))  # (W, H)
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

    # Load camera intrinsics and enforce resolution consistency.
    allow_rescale = os.getenv("ALLOW_INTRINSIC_RESCALE", "0") == "1"
    Kc_file, dist_file, calib_size = load_camera_intrinsics()
    Kc, dist, _ = prepare_intrinsics_for_image(
        Kc_file, dist_file, calib_size, scan_size, allow_rescale=allow_rescale, label="camera"
    )
    print(f"[RECON] Scan image size: {scan_size[0]}x{scan_size[1]} | Camera calib size: {calib_size[0]}x{calib_size[1]}")
    print(f"[RECON] Camera intrinsics: {_format_K(Kc)}")

    stereo = load_stereo_params()
    stereo_rms = None
    using_fake = False

    if stereo is not None:
        R_st, T_st, Kc_s, dist_s, Kp_st, dp_st, stereo_rms, _ = stereo
        # Stereo params store camera intrinsics too. The file does not store camera image_size,
        # so treat camera_intrinsics.npz's image_size as the reference calibration size.
        Kc, dist, _ = prepare_intrinsics_for_image(
            Kc_s, dist_s, calib_size, scan_size, allow_rescale=allow_rescale, label="stereo camera"
        )
        # Use projector intrinsics/distortion from stereo file (projector size is independent of camera image size).
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

    # stereo_params.npz already stores projector-to-camera extrinsics.
    # Do not invert or flip here.
    R = R_st
    T = T_st

    proj_uv_raw = proj_uv
    # Optional diagnostic: evaluate a few UV convention candidates on a random subset.
    if os.getenv("DIAGNOSE_PROJ_UV_TRANSFORM", "0") == "1":
        _diagnose_proj_uv_transform(cam_uv, proj_uv_raw, proj_size, Kc, dist, Kp, dp, R, T)

    # Align scan-derived projector (u,v) to calibration convention (optional but common).
    uv_cfg = _load_uv_convention()
    uv_transform = os.getenv("PROJ_UV_TRANSFORM", uv_cfg.get("transform", "identity"))
    if uv_transform and str(uv_transform).lower() != "identity":
        print(f"[RECON] Applying projector UV transform: {uv_transform}")
    proj_uv = _apply_proj_uv_transform(proj_uv_raw, proj_size=proj_size, name=str(uv_transform))

    if cv2 is not None:
        cam_norm = cv2.undistortPoints(cam_uv.reshape(-1, 1, 2), Kc, dist)[:, 0, :]
        proj_norm = cv2.undistortPoints(proj_uv.reshape(-1, 1, 2), Kp, dp)[:, 0, :]
    else:
        cam_norm = cam_uv
        proj_norm = proj_uv

    def _dirs_from_norm(norm_xy: np.ndarray) -> np.ndarray:
        """Convert normalized image coordinates into unit ray directions."""
        dirs = np.hstack([norm_xy, np.ones((norm_xy.shape[0], 1), dtype=np.float64)])
        norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        return (dirs / norms).astype(np.float32)

    rays_cam = _dirs_from_norm(cam_norm)
    dirs_proj = _dirs_from_norm(proj_norm)
    dirs_proj_cam = (R.astype(np.float64) @ dirs_proj.T).T
    dirs_proj_cam /= np.linalg.norm(dirs_proj_cam, axis=1, keepdims=True) + 1e-12
    rays_proj_cam = dirs_proj_cam.astype(np.float32)

    points, errors, s_params, t_params = triangulate_rays(rays_cam, rays_proj_cam, T)

    # Basic validity diagnostics.
    z_vals = points[:, 2]
    finite_pts = np.isfinite(points).all(axis=1)
    def neighbor_jump_stats(arr: np.ndarray, mask_in: np.ndarray, thresh: float = 20.0):
        """Compute the percentage of neighbor jumps above a threshold."""
        # Right and down neighbors.
        mask = mask_in & np.isfinite(arr)
        # Right.
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
        # Angle between camera and projector rays (should vary, nonzero).
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

    # Filter points by triangulation error.
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

    # Build single-view mesh (2.5D) from filtered points.
    mesh_stride = 1  # Increase if Pi performance is constrained.
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

    # Diagnostics summary.
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
