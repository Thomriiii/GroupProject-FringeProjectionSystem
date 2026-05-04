"""Calibration view-quality evaluation (capture-time auto rejection)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
        raise RuntimeError(f"OpenCV is required for view quality evaluation: {_cv2_import_error}")
    return cv2


@dataclass(slots=True)
class ViewQualityReport:
    n_camera_corners: int
    n_uv_valid: int
    uv_valid_ratio: float
    u_span_px: float
    v_span_px: float
    board_area_ratio: float
    tilt_angle_deg: float
    reproj_estimate_proxy: float
    conditioning_score: float
    accepted: bool
    rejection_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    c = dict(cfg or {})
    return {
        "auto_reject": bool(c.get("auto_reject", True)),
        "min_uv_valid_ratio": float(c.get("min_uv_valid_ratio", 0.85)),
        "min_u_span_px": float(c.get("min_u_span_px", 120.0)),
        "min_v_span_px": float(c.get("min_v_span_px", 80.0)),
        "min_board_area_ratio": float(c.get("min_board_area_ratio", 0.02)),
        "min_tilt_deg": float(c.get("min_tilt_deg", 8.0)),
        "max_tilt_deg": float(c.get("max_tilt_deg", 75.0)),
    }


def _board_area_ratio(camera_corners: np.ndarray, image_shape_hw: tuple[int, int]) -> float:
    cv = _require_cv2()
    h, w = int(image_shape_hw[0]), int(image_shape_hw[1])
    if h <= 0 or w <= 0 or camera_corners.shape[0] < 3:
        return 0.0
    hull = cv.convexHull(camera_corners.reshape(-1, 1, 2).astype(np.float32))
    area = float(cv.contourArea(hull))
    return float(area / float(h * w))


def _tilt_proxy_from_homography(camera_corners: np.ndarray) -> tuple[float, float]:
    """
    Return (tilt_proxy_deg, reproj_proxy_px) from board->image homography.
    This is intentionally approximate for fast per-view gating.
    """
    cv = _require_cv2()
    n = int(camera_corners.shape[0])
    if n < 4:
        return float("nan"), float("nan")
    side = int(round(np.sqrt(float(n))))
    if side * side == n:
        cols, rows = side, side
    else:
        cols = max(2, n // 6) if n >= 12 else n
        rows = max(2, n // max(cols, 1))
        if cols * rows != n:
            cols = n
            rows = 1
    obj_xy = np.zeros((n, 2), dtype=np.float32)
    if rows > 1:
        grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
        if grid.shape[0] == n:
            obj_xy = grid
        else:
            obj_xy[:, 0] = np.arange(n, dtype=np.float32)
    else:
        obj_xy[:, 0] = np.arange(n, dtype=np.float32)

    H, inlier_mask = cv.findHomography(obj_xy, camera_corners.astype(np.float32), method=0)
    if H is None or not np.isfinite(H).all():
        return float("nan"), float("nan")
    if inlier_mask is not None:
        m = np.asarray(inlier_mask)
        if m.size == 0 or int(np.count_nonzero(m)) == 0:
            return float("nan"), float("nan")

    h1 = H[:, 0]
    h2 = H[:, 1]
    n1 = float(np.linalg.norm(h1))
    n2 = float(np.linalg.norm(h2))
    if n1 <= 1e-9 or n2 <= 1e-9:
        tilt_deg = float("nan")
    else:
        ratio = float(min(n1, n2) / max(n1, n2))
        ratio = float(np.clip(ratio, 0.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(ratio)))

    pts_h = np.column_stack([obj_xy, np.ones((n,), dtype=np.float32)]).astype(np.float64)
    proj = (H @ pts_h.T).T
    z = proj[:, 2:3]
    valid = np.abs(z[:, 0]) > 1e-12
    reproj = np.full((n, 2), np.nan, dtype=np.float64)
    reproj[valid] = proj[valid, :2] / z[valid]
    finite = np.isfinite(reproj).all(axis=1) & np.isfinite(camera_corners).all(axis=1)
    if np.any(finite):
        reproj_proxy = float(np.mean(np.linalg.norm(reproj[finite] - camera_corners[finite], axis=1)))
    else:
        reproj_proxy = float("nan")
    return tilt_deg, reproj_proxy


def evaluate_view(
    camera_corners: np.ndarray,
    uv_corners: np.ndarray,
    mask_uv: np.ndarray,
    projector_size: tuple[int, int],
    config: dict[str, Any] | None = None,
    *,
    image_shape_hw: tuple[int, int] | None = None,
) -> ViewQualityReport:
    """
    Evaluate calibration-view quality for automatic acceptance/rejection.
    """
    cfg = _as_cfg(config)
    cam = np.asarray(camera_corners, dtype=np.float32).reshape(-1, 2)
    uv = np.asarray(uv_corners, dtype=np.float32).reshape(-1, 2)
    mask = np.asarray(mask_uv, dtype=bool)
    h, w = mask.shape
    ph, pw = int(projector_size[1]), int(projector_size[0])

    finite = np.isfinite(uv).all(axis=1)
    in_bounds = (
        (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(max(1, pw)))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(max(1, ph)))
    )
    xi = np.clip(np.rint(cam[:, 0]).astype(np.int32), 0, w - 1)
    yi = np.clip(np.rint(cam[:, 1]).astype(np.int32), 0, h - 1)
    on_mask = mask[yi, xi]
    uv_valid = finite & in_bounds & on_mask
    n_valid = int(np.count_nonzero(uv_valid))
    total = int(cam.shape[0])
    uv_valid_ratio = float(n_valid / max(total, 1))

    if np.any(uv_valid):
        u_span = float(np.max(uv[uv_valid, 0]) - np.min(uv[uv_valid, 0]))
        v_span = float(np.max(uv[uv_valid, 1]) - np.min(uv[uv_valid, 1]))
    else:
        u_span = 0.0
        v_span = 0.0

    img_shape = image_shape_hw if image_shape_hw is not None else (h, w)
    board_area_ratio = _board_area_ratio(cam, img_shape) if cam.shape[0] >= 3 else 0.0
    tilt_deg, reproj_proxy = _tilt_proxy_from_homography(cam)

    u_norm = float(np.clip(u_span / max(float(pw), 1.0), 0.0, 1.0))
    v_norm = float(np.clip(v_span / max(float(ph), 1.0), 0.0, 1.0))
    area_norm = float(np.clip(board_area_ratio / max(cfg["min_board_area_ratio"], 1e-6), 0.0, 1.5))
    if np.isfinite(tilt_deg):
        if tilt_deg < cfg["min_tilt_deg"]:
            tilt_score = float(np.clip(tilt_deg / max(cfg["min_tilt_deg"], 1e-6), 0.0, 1.0))
        elif tilt_deg > cfg["max_tilt_deg"]:
            tilt_score = 0.0
        else:
            tilt_score = 1.0
    else:
        tilt_score = 0.0
    conditioning_score = float(
        0.35 * np.clip(uv_valid_ratio / max(cfg["min_uv_valid_ratio"], 1e-6), 0.0, 1.5)
        + 0.20 * u_norm
        + 0.15 * v_norm
        + 0.15 * np.clip(area_norm, 0.0, 1.5)
        + 0.15 * tilt_score
    )

    rejected: list[str] = []
    if uv_valid_ratio < cfg["min_uv_valid_ratio"]:
        rejected.append("uv_valid_ratio_low")
    if u_span < cfg["min_u_span_px"]:
        rejected.append("u_span_too_small")
    if v_span < cfg["min_v_span_px"]:
        rejected.append("v_span_too_small")
    if board_area_ratio < cfg["min_board_area_ratio"]:
        rejected.append("board_area_too_small")
    if not np.isfinite(tilt_deg) or tilt_deg < cfg["min_tilt_deg"]:
        rejected.append("tilt_too_low")
    if np.isfinite(tilt_deg) and tilt_deg > cfg["max_tilt_deg"]:
        rejected.append("tilt_too_high")

    auto_reject = bool(cfg["auto_reject"])
    accepted = (len(rejected) == 0) or (not auto_reject)
    rejection_reason = rejected[0] if (rejected and auto_reject) else None
    return ViewQualityReport(
        n_camera_corners=total,
        n_uv_valid=n_valid,
        uv_valid_ratio=uv_valid_ratio,
        u_span_px=u_span,
        v_span_px=v_span,
        board_area_ratio=board_area_ratio,
        tilt_angle_deg=float(tilt_deg) if np.isfinite(tilt_deg) else float("nan"),
        reproj_estimate_proxy=float(reproj_proxy) if np.isfinite(reproj_proxy) else float("nan"),
        conditioning_score=conditioning_score,
        accepted=bool(accepted),
        rejection_reason=rejection_reason,
    )
