"""Per-view quality gating for projector calibration."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    _cv2_err = exc
else:
    _cv2_err = None


def _require_cv2():
    if cv2 is None:
        raise RuntimeError(f"OpenCV required: {_cv2_err}")
    return cv2


def _board_area_ratio(corners: np.ndarray, image_hw: tuple[int, int]) -> float:
    cv = _require_cv2()
    h, w = int(image_hw[0]), int(image_hw[1])
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3 or h <= 0 or w <= 0:
        return 0.0
    hull = cv.convexHull(pts.reshape(-1, 1, 2))
    area = float(cv.contourArea(hull))
    return area / float(max(1, h * w))


def _min_border_distance(corners: np.ndarray, image_hw: tuple[int, int]) -> float:
    h, w = int(image_hw[0]), int(image_hw[1])
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0 or h <= 0 or w <= 0:
        return float("nan")
    x, y = pts[:, 0], pts[:, 1]
    all_d = np.concatenate([x, float(w - 1) - x, y, float(h - 1) - y])
    return float(np.min(all_d))


def _centroid_norm(corners: np.ndarray, image_hw: tuple[int, int]) -> tuple[float, float] | None:
    h, w = int(image_hw[0]), int(image_hw[1])
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0 or h <= 0 or w <= 0:
        return None
    cx = float(np.clip(np.mean(pts[:, 0]) / float(w), 0.0, 1.0))
    cy = float(np.clip(np.mean(pts[:, 1]) / float(h), 0.0, 1.0))
    return (cx, cy)


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
    normal = R @ np.array([0.0, 0.0, 1.0])
    nz = float(np.clip(normal[2], -1.0, 1.0))
    return float(np.degrees(np.arccos(abs(nz))))


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
    """
    Gate a projector calibration view.

    Returns:
        {accepted, reasons, hints, metrics, thresholds}
    """
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
    if object_points is not None and camera_matrix is not None and dist_coeffs is not None:
        try:
            tilt_deg = estimate_tilt_deg(object_points, corners, camera_matrix, dist_coeffs)
        except Exception:
            pass

    reasons: list[str] = []
    hints: list[str] = []

    if marker_count < min_markers:
        reasons.append(f"Detected {marker_count} markers; need at least {min_markers}.")
        hints.append("Ensure more ArUco markers are visible on the board.")
    if corner_count < min_corners:
        reasons.append(f"Detected {corner_count} corners; need at least {min_corners}.")
        hints.append("Show more of the board and keep it in focus.")
    if not np.isfinite(border_min) or border_min < border_margin_px:
        reasons.append(f"Board too close to image border ({border_min:.1f}px margin, need {border_margin_px:.1f}px).")
        hints.append("Move board away from image edges.")
    if area_ratio < min_area_ratio:
        reasons.append(f"Board area ratio too small ({area_ratio:.3f} < {min_area_ratio:.3f}).")
        hints.append("Move the board closer to the camera.")
    if area_ratio > max_area_ratio:
        reasons.append(f"Board area ratio too large ({area_ratio:.3f} > {max_area_ratio:.3f}).")
        hints.append("Move the board farther from the camera.")

    return {
        "accepted": len(reasons) == 0,
        "reasons": reasons,
        "hints": sorted(set(hints)),
        "metrics": {
            "marker_count": int(marker_count),
            "corner_count": int(corner_count),
            "min_border_px": float(border_min) if np.isfinite(border_min) else None,
            "area_ratio": float(area_ratio),
            "centroid_norm": [float(centroid[0]), float(centroid[1])] if centroid else None,
            "tilt_deg": float(tilt_deg) if tilt_deg is not None and np.isfinite(tilt_deg) else None,
        },
        "thresholds": {
            "min_markers": min_markers,
            "min_corners": min_corners,
            "border_margin_px": border_margin_px,
            "min_area_ratio": min_area_ratio,
            "max_area_ratio": max_area_ratio,
        },
    }
