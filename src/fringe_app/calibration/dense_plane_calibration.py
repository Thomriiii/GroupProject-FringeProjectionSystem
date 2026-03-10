"""Dense board-plane projector calibration (calibration-only)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import json
import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    _cv2_import_error = exc
else:
    _cv2_import_error = None


def _require_cv2():
    if cv2 is None:
        raise RuntimeError(f"OpenCV is required for dense plane calibration: {_cv2_import_error}")
    return cv2


@dataclass(slots=True)
class DensePlaneConfig:
    enabled: bool = False
    pixel_stride: int = 4
    max_points_per_view: int = 8000
    min_points_per_view: int = 1200

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any] | None) -> "DensePlaneConfig":
        c = dict(cfg or {})
        return cls(
            enabled=bool(c.get("enabled", False)),
            pixel_stride=max(1, int(c.get("pixel_stride", 4))),
            max_points_per_view=max(100, int(c.get("max_points_per_view", 8000))),
            min_points_per_view=max(100, int(c.get("min_points_per_view", 1200))),
        )


def _load_camera_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    k = np.asarray(data.get("camera_matrix"), dtype=np.float64)
    d = np.asarray(data.get("dist_coeffs"), dtype=np.float64).reshape(-1, 1)
    if k.shape != (3, 3):
        raise ValueError("Invalid camera intrinsics matrix shape")
    return k, d


def _checkerboard_object_points(corners_x: int, corners_y: int, square_size_m: float) -> np.ndarray:
    obj = np.zeros((corners_x * corners_y, 3), dtype=np.float32)
    grid = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1, 2)
    obj[:, :2] = grid * float(square_size_m)
    return obj


def _board_mask(shape_hw: tuple[int, int], corners: np.ndarray) -> np.ndarray:
    cv = _require_cv2()
    h, w = int(shape_hw[0]), int(shape_hw[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    if corners.shape[0] < 3:
        return mask.astype(bool)
    hull = cv.convexHull(corners.reshape(-1, 1, 2).astype(np.float32))
    cv.fillConvexPoly(mask, hull.astype(np.int32), 255)
    return mask > 0


def _dense_points_from_view(
    *,
    view_dir: Path,
    corners_x: int,
    corners_y: int,
    square_size_m: float,
    cfg: DensePlaneConfig,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    cv = _require_cv2()
    corners_path = view_dir / "camera_corners.json"
    uv_dir = view_dir / "uv"
    u_path = uv_dir / "u.npy"
    v_path = uv_dir / "v.npy"
    mask_path = uv_dir / "mask_uv.npy"
    if not (corners_path.exists() and u_path.exists() and v_path.exists() and mask_path.exists()):
        return None, None, None, {"reason": "missing_files"}

    cam_data = json.loads(corners_path.read_text())
    cam_corners = np.asarray(cam_data.get("corners_px", []), dtype=np.float32).reshape(-1, 2)
    expected = int(corners_x * corners_y)
    if cam_corners.shape[0] != expected:
        return None, None, None, {"reason": "corner_count_mismatch", "found": int(cam_corners.shape[0]), "expected": expected}

    U = np.load(u_path).astype(np.float32)
    V = np.load(v_path).astype(np.float32)
    M = np.load(mask_path).astype(bool)
    if U.shape != V.shape or U.shape != M.shape:
        return None, None, None, {"reason": "uv_shape_mismatch"}
    h, w = U.shape

    obj_template = _checkerboard_object_points(corners_x, corners_y, square_size_m)
    obj_xy = obj_template[:, :2].astype(np.float32)
    H, inlier_mask = cv.findHomography(obj_xy, cam_corners.astype(np.float32), method=0)
    if H is None or not np.isfinite(H).all():
        return None, None, None, {"reason": "homography_failed"}
    if inlier_mask is not None:
        m = np.asarray(inlier_mask)
        if m.size == 0 or int(np.count_nonzero(m)) == 0:
            return None, None, None, {"reason": "homography_failed"}
    H_inv = np.linalg.inv(H)

    bmask = _board_mask((h, w), cam_corners)
    ys, xs = np.where(bmask)
    if ys.size == 0:
        return None, None, None, {"reason": "empty_board_mask"}
    stride = int(cfg.pixel_stride)
    if stride > 1:
        keep = ((ys % stride) == 0) & ((xs % stride) == 0)
        ys = ys[keep]
        xs = xs[keep]
    if ys.size == 0:
        return None, None, None, {"reason": "empty_after_stride"}

    uv_ok = M[ys, xs] & np.isfinite(U[ys, xs]) & np.isfinite(V[ys, xs])
    ys = ys[uv_ok]
    xs = xs[uv_ok]
    if ys.size < cfg.min_points_per_view:
        return None, None, None, {"reason": "insufficient_dense_uv", "valid_points": int(ys.size)}

    if ys.size > cfg.max_points_per_view:
        idx = np.linspace(0, ys.size - 1, cfg.max_points_per_view, dtype=np.int32)
        ys = ys[idx]
        xs = xs[idx]

    pix_h = np.column_stack([xs.astype(np.float64), ys.astype(np.float64), np.ones_like(xs, dtype=np.float64)])
    obj_h = (H_inv @ pix_h.T).T
    z = obj_h[:, 2]
    okz = np.abs(z) > 1e-9
    if not np.any(okz):
        return None, None, None, {"reason": "homography_denominator_zero"}
    obj_h = obj_h[okz]
    xs = xs[okz]
    ys = ys[okz]
    XY = obj_h[:, :2] / obj_h[:, 2:3]

    obj_pts = np.zeros((XY.shape[0], 3), dtype=np.float32)
    obj_pts[:, :2] = XY.astype(np.float32)
    cam_pts = np.column_stack([xs, ys]).astype(np.float32).reshape(-1, 1, 2)
    proj_pts = np.column_stack([U[ys, xs], V[ys, xs]]).astype(np.float32).reshape(-1, 1, 2)
    meta = {
        "dense_points": int(obj_pts.shape[0]),
        "stride": stride,
    }
    return obj_pts, cam_pts, proj_pts, meta


def calibrate_dense_plane_session(
    *,
    session_dir: Path,
    cfg: dict[str, Any],
    camera_intrinsics_path: Path,
    min_views_required: int,
    min_corner_valid_ratio: float,
    allowed_view_ids: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cv = _require_cv2()
    session = json.loads((session_dir / "session.json").read_text())
    pcal = cfg.get("projector_calibration", {}) or {}
    chk = pcal.get("checkerboard", {}) or {}
    proj_cfg = pcal.get("projector", {}) or {}
    dense_cfg = DensePlaneConfig.from_cfg((cfg.get("calibration", {}) or {}).get("dense_plane", {}))

    corners_x = int(chk.get("corners_x", 8))
    corners_y = int(chk.get("corners_y", 6))
    square_size_m = float(chk.get("square_size_m", 0.01))
    proj_size = (int(proj_cfg.get("width", 1024)), int(proj_cfg.get("height", 768)))

    object_points: list[np.ndarray] = []
    cam_points: list[np.ndarray] = []
    proj_points: list[np.ndarray] = []
    view_ids: list[str] = []
    dropped: list[dict[str, Any]] = []
    per_view_dense: list[dict[str, Any]] = []

    valid_views = [v for v in (session.get("views", []) or []) if str(v.get("status", "")).lower() == "valid"]
    for view in valid_views:
        view_id = str(view.get("view_id", "")).strip()
        if not view_id:
            continue
        if allowed_view_ids is not None and view_id not in allowed_view_ids:
            continue
        if float(view.get("valid_corner_ratio", 0.0)) < min_corner_valid_ratio:
            dropped.append({"view_id": view_id, "reason": "below_min_corner_ratio"})
            continue
        vdir = session_dir / "views" / view_id
        obj_pts, cam_pts, proj_pts, meta = _dense_points_from_view(
            view_dir=vdir,
            corners_x=corners_x,
            corners_y=corners_y,
            square_size_m=square_size_m,
            cfg=dense_cfg,
        )
        if obj_pts is None or cam_pts is None or proj_pts is None:
            dropped.append({"view_id": view_id, **meta})
            continue
        object_points.append(obj_pts.astype(np.float32))
        cam_points.append(cam_pts.astype(np.float32))
        proj_points.append(proj_pts.astype(np.float32))
        view_ids.append(view_id)
        per_view_dense.append({"view_id": view_id, **meta})

    if len(object_points) < int(min_views_required):
        raise ValueError(f"Need at least {min_views_required} valid dense-plane views.")

    k_cam, d_cam = _load_camera_intrinsics(camera_intrinsics_path)
    rms_proj, k_proj, d_proj, _, _ = cv.calibrateCamera(
        object_points,
        proj_points,
        proj_size,
        None,
        None,
    )
    flags = cv.CALIB_FIX_INTRINSIC
    rms_stereo, k1, d1, k2, d2, r, t, e, f = cv.stereoCalibrate(
        object_points,
        cam_points,
        proj_points,
        k_cam.copy(),
        d_cam.copy(),
        k_proj.copy(),
        d_proj.copy(),
        proj_size,
        flags=flags,
    )
    r1, r2, p1, p2, q, _, _ = cv.stereoRectify(k1, d1, k2, d2, proj_size, r, t)

    per_view_errors: list[dict[str, Any]] = []
    for obj, cam_obs, proj_obs, view_id in zip(object_points, cam_points, proj_points, view_ids):
        ok1, rv_cam, tv_cam = cv.solvePnP(obj, cam_obs, k1, d1)
        ok2, rv_proj, tv_proj = cv.solvePnP(obj, proj_obs, k2, d2)
        if not ok1 or not ok2:
            continue
        cam_reproj, _ = cv.projectPoints(obj, rv_cam, tv_cam, k1, d1)
        proj_reproj, _ = cv.projectPoints(obj, rv_proj, tv_proj, k2, d2)
        cam_err = float(np.linalg.norm(cam_obs.reshape(-1, 2) - cam_reproj.reshape(-1, 2), axis=1).mean())
        proj_err = float(np.linalg.norm(proj_obs.reshape(-1, 2) - proj_reproj.reshape(-1, 2), axis=1).mean())
        per_view_errors.append(
            {
                "view_id": view_id,
                "camera_reproj_error_px": cam_err,
                "projector_reproj_error_px": proj_err,
                "corners_used": int(obj.shape[0]),
            }
        )

    result = {
        "session_id": session.get("session_id"),
        "created_at": datetime.now().isoformat(),
        "views_used": int(len(object_points)),
        "view_ids": view_ids,
        "rms_projector_intrinsics": float(rms_proj),
        "rms_stereo": float(rms_stereo),
        "camera_matrix": k1.astype(float).tolist(),
        "camera_dist_coeffs": d1.reshape(-1).astype(float).tolist(),
        "projector_matrix": k2.astype(float).tolist(),
        "projector_dist_coeffs": d2.reshape(-1).astype(float).tolist(),
        "R": r.astype(float).tolist(),
        "T": t.reshape(-1).astype(float).tolist(),
        "E": e.astype(float).tolist(),
        "F": f.astype(float).tolist(),
        "rectification": {
            "R1": r1.astype(float).tolist(),
            "R2": r2.astype(float).tolist(),
            "P1": p1.astype(float).tolist(),
            "P2": p2.astype(float).tolist(),
            "Q": q.astype(float).tolist(),
        },
        "per_view_errors": per_view_errors,
        "checkerboard": {
            "corners_x": corners_x,
            "corners_y": corners_y,
            "square_size_m": square_size_m,
        },
        "projector": {"width": proj_size[0], "height": proj_size[1]},
        "points_source": "dense_plane",
        "dense_plane": {
            "config": {
                "pixel_stride": int(dense_cfg.pixel_stride),
                "max_points_per_view": int(dense_cfg.max_points_per_view),
                "min_points_per_view": int(dense_cfg.min_points_per_view),
            },
            "per_view_dense_points": per_view_dense,
            "dropped_views": dropped,
        },
    }
    mats = {
        "camera_matrix": k1,
        "camera_dist": d1,
        "projector_matrix": k2,
        "projector_dist": d2,
        "R": r,
        "T": t,
        "E": e,
        "F": f,
        "R1": r1,
        "R2": r2,
        "P1": p1,
        "P2": p2,
        "Q": q,
    }
    return result, mats
