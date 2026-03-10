"""Per-view gating rules for projector calibration v2."""

from __future__ import annotations

from typing import Any

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
        raise RuntimeError(f"OpenCV is required for projector calibration gating: {_cv2_import_error}")
    return cv2


def _board_area_ratio(corners: np.ndarray, image_shape_hw: tuple[int, int]) -> float:
    cv = _require_cv2()
    h = int(image_shape_hw[0])
    w = int(image_shape_hw[1])
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3 or h <= 0 or w <= 0:
        return 0.0
    hull = cv.convexHull(pts.reshape(-1, 1, 2))
    area = float(cv.contourArea(hull))
    return float(area / float(max(1, h * w)))


def _min_border_distance(corners: np.ndarray, image_shape_hw: tuple[int, int]) -> float:
    h = int(image_shape_hw[0])
    w = int(image_shape_hw[1])
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0 or h <= 0 or w <= 0:
        return float("nan")
    x = pts[:, 0]
    y = pts[:, 1]
    d_left = x
    d_right = (float(w - 1) - x)
    d_top = y
    d_bottom = (float(h - 1) - y)
    d_all = np.concatenate([d_left, d_right, d_top, d_bottom], axis=0)
    return float(np.min(d_all))


def _centroid_norm(corners: np.ndarray, image_shape_hw: tuple[int, int]) -> tuple[float, float] | None:
    h = int(image_shape_hw[0])
    w = int(image_shape_hw[1])
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0 or h <= 0 or w <= 0:
        return None
    cx = float(np.mean(pts[:, 0])) / float(w)
    cy = float(np.mean(pts[:, 1])) / float(h)
    return (float(np.clip(cx, 0.0, 1.0)), float(np.clip(cy, 0.0, 1.0)))


def estimate_tilt_deg(
    object_points: np.ndarray,
    corners_px: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float | None:
    cv = _require_cv2()
    obj = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    img = np.asarray(corners_px, dtype=np.float32).reshape(-1, 2)
    if obj.shape[0] < 4 or img.shape[0] != obj.shape[0]:
        return None
    ok, rvec, _ = cv.solvePnP(obj, img.reshape(-1, 1, 2), camera_matrix, dist_coeffs)
    if not ok:
        return None
    R, _ = cv.Rodrigues(rvec)
    normal = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    nz = float(np.clip(float(normal[2]), -1.0, 1.0))
    tilt = float(np.degrees(np.arccos(abs(nz))))
    return tilt


def evaluate_view(
    *,
    marker_count: int,
    corner_count: int,
    corners_px: np.ndarray,
    image_shape_hw: tuple[int, int],
    gating_cfg: dict[str, Any],
    object_points: np.ndarray | None = None,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
) -> dict[str, Any]:
    min_markers = int(gating_cfg.get("min_markers", 8))
    min_corners = int(gating_cfg.get("min_corners", 16))
    border_margin_px = float(gating_cfg.get("border_margin_px", 20))
    min_area_ratio = float(gating_cfg.get("min_area_ratio", 0.05))
    max_area_ratio = float(gating_cfg.get("max_area_ratio", 0.60))

    corners = np.asarray(corners_px, dtype=np.float32).reshape(-1, 2)
    area_ratio = _board_area_ratio(corners, image_shape_hw)
    border_min = _min_border_distance(corners, image_shape_hw)
    centroid = _centroid_norm(corners, image_shape_hw)

    tilt_deg: float | None = None
    if (
        object_points is not None
        and camera_matrix is not None
        and dist_coeffs is not None
    ):
        try:
            tilt_deg = estimate_tilt_deg(object_points, corners, camera_matrix, dist_coeffs)
        except Exception:
            tilt_deg = None

    reasons: list[str] = []
    hints: list[str] = []

    if int(marker_count) < min_markers:
        reasons.append(f"Detected {int(marker_count)} markers; need at least {min_markers}.")
        hints.append("Ensure more ArUco markers are visible on the board.")
    if int(corner_count) < min_corners:
        reasons.append(f"Detected {int(corner_count)} corners; need at least {min_corners}.")
        hints.append("Show more of the board and keep it in focus.")
    if not np.isfinite(border_min) or float(border_min) < border_margin_px:
        reasons.append(f"Board too close to image border (min margin {float(border_min):.1f}px, required {border_margin_px:.1f}px).")
        hints.append("Move board away from image edges.")
    if area_ratio < min_area_ratio:
        reasons.append(f"Board area ratio too small ({area_ratio:.3f} < {min_area_ratio:.3f}).")
        hints.append("Move the board closer to the camera.")
    if area_ratio > max_area_ratio:
        reasons.append(f"Board area ratio too large ({area_ratio:.3f} > {max_area_ratio:.3f}).")
        hints.append("Move the board farther from the camera.")

    accepted = len(reasons) == 0

    return {
        "accepted": bool(accepted),
        "reasons": reasons,
        "hints": sorted(set(hints)),
        "metrics": {
            "marker_count": int(marker_count),
            "corner_count": int(corner_count),
            "min_border_px": float(border_min) if np.isfinite(border_min) else None,
            "area_ratio": float(area_ratio),
            "centroid_norm": [float(centroid[0]), float(centroid[1])] if centroid is not None else None,
            "tilt_deg": float(tilt_deg) if tilt_deg is not None and np.isfinite(float(tilt_deg)) else None,
        },
        "thresholds": {
            "min_markers": int(min_markers),
            "min_corners": int(min_corners),
            "border_margin_px": float(border_margin_px),
            "min_area_ratio": float(min_area_ratio),
            "max_area_ratio": float(max_area_ratio),
        },
    }
