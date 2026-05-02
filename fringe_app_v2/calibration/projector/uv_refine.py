"""UV corner sampling with optional local plane refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class UvRefineConfig:
    enabled: bool = True
    patch_radius_px: int = 12
    plane_fit: bool = True
    max_delta_px: float = 3.0

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "UvRefineConfig":
        c = dict(cfg or {})
        return cls(
            enabled=bool(c.get("enabled", True)),
            patch_radius_px=max(1, int(c.get("patch_radius_px", 12))),
            plane_fit=bool(c.get("plane_fit", True)),
            max_delta_px=float(c.get("max_delta_px", 3.0)),
        )


def _fit_plane(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray | None:
    """Fit z = a*x + b*y + c by least squares. Returns [a, b, c] or None."""
    if xs.size < 6:
        return None
    A = np.column_stack([xs, ys, np.ones(xs.shape[0], dtype=np.float64)])
    try:
        coeff, *_ = np.linalg.lstsq(A, zs, rcond=None)
    except Exception:
        return None
    return coeff.astype(np.float64)


def sample_and_refine_uv(
    u_map: np.ndarray,
    v_map: np.ndarray,
    mask: np.ndarray,
    corners_px: np.ndarray,
    projector_size: tuple[int, int],
    cfg: UvRefineConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Sample projector UV coordinates at ChArUco corner positions.

    For each corner, extracts a local patch from the UV maps. If enough valid
    points exist and cfg.plane_fit is True, fits a local plane u = a*x + b*y + c
    and evaluates it at the corner position. If the plane prediction deviates
    from the patch median by more than cfg.max_delta_px, falls back to the median.

    Args:
        u_map:          HxW projector X coordinates (NaN = invalid)
        v_map:          HxW projector Y coordinates (NaN = invalid)
        mask:           HxW boolean valid mask
        corners_px:     Nx2 ChArUco corner pixel positions
        projector_size: (width, height) of projector in pixels
        cfg:            Refinement configuration

    Returns:
        uv_out:   Nx2 float32 array of sampled (u, v); NaN rows are invalid
        valid:    N bool array, True where uv_out is usable
        diag:     Diagnostics dict
    """
    u = np.asarray(u_map, dtype=np.float64)
    v = np.asarray(v_map, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    corners = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)

    if u.shape != v.shape or u.shape != m.shape:
        raise ValueError("u_map, v_map, and mask must have identical shapes")

    h, w = u.shape
    proj_w, proj_h = int(projector_size[0]), int(projector_size[1])
    r = int(cfg.patch_radius_px)

    uv_out = np.full((corners.shape[0], 2), np.nan, dtype=np.float32)
    valid = np.zeros(corners.shape[0], dtype=bool)
    per_corner: list[dict[str, Any]] = []

    for idx, (cx, cy) in enumerate(corners):
        xi, yi = int(round(cx)), int(round(cy))
        entry: dict[str, Any] = {
            "index": idx,
            "x": float(cx),
            "y": float(cy),
            "valid": False,
            "reason": "no_valid_uv",
            "patch_points": 0,
            "raw_u": None,
            "raw_v": None,
            "refined_u": None,
            "refined_v": None,
            "delta_px": None,
            "plane_fit_used": False,
        }

        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            entry["reason"] = "corner_oob"
            per_corner.append(entry)
            continue

        x0, x1 = max(0, xi - r), min(w, xi + r + 1)
        y0, y1 = max(0, yi - r), min(h, yi + r + 1)

        pm = m[y0:y1, x0:x1]
        pu = u[y0:y1, x0:x1]
        pv = v[y0:y1, x0:x1]
        good = pm & np.isfinite(pu) & np.isfinite(pv)
        n_good = int(np.count_nonzero(good))
        entry["patch_points"] = n_good

        if n_good == 0:
            per_corner.append(entry)
            continue

        vals_u = pu[good].astype(np.float64)
        vals_v = pv[good].astype(np.float64)
        raw_u = float(np.median(vals_u))
        raw_v = float(np.median(vals_v))
        entry["raw_u"] = raw_u if np.isfinite(raw_u) else None
        entry["raw_v"] = raw_v if np.isfinite(raw_v) else None

        final_u, final_v = raw_u, raw_v

        if cfg.plane_fit and n_good >= 6:
            ys_local, xs_local = np.where(good)
            xs_g = xs_local.astype(np.float64) + float(x0)
            ys_g = ys_local.astype(np.float64) + float(y0)
            coeff_u = _fit_plane(xs_g, ys_g, vals_u)
            coeff_v = _fit_plane(xs_g, ys_g, vals_v)
            if coeff_u is not None and coeff_v is not None:
                pred_u = float(coeff_u[0] * cx + coeff_u[1] * cy + coeff_u[2])
                pred_v = float(coeff_v[0] * cx + coeff_v[1] * cy + coeff_v[2])
                if np.isfinite(pred_u) and np.isfinite(pred_v):
                    delta = float(np.hypot(pred_u - raw_u, pred_v - raw_v))
                    entry["delta_px"] = delta
                    if delta <= cfg.max_delta_px:
                        final_u, final_v = pred_u, pred_v
                        entry["plane_fit_used"] = True

        if not (np.isfinite(final_u) and np.isfinite(final_v)):
            entry["reason"] = "nan_uv"
            per_corner.append(entry)
            continue

        if final_u < 0.0 or final_u >= proj_w or final_v < 0.0 or final_v >= proj_h:
            entry["reason"] = "uv_oob"
            per_corner.append(entry)
            continue

        uv_out[idx, 0] = np.float32(final_u)
        uv_out[idx, 1] = np.float32(final_v)
        valid[idx] = True
        entry["valid"] = True
        entry["reason"] = "ok"
        entry["refined_u"] = float(final_u)
        entry["refined_v"] = float(final_v)
        per_corner.append(entry)

    n_total = valid.size
    n_valid = int(np.count_nonzero(valid))
    diag = {
        "config": {
            "enabled": cfg.enabled,
            "patch_radius_px": cfg.patch_radius_px,
            "plane_fit": cfg.plane_fit,
            "max_delta_px": cfg.max_delta_px,
        },
        "summary": {
            "corners_total": n_total,
            "corners_valid": n_valid,
            "valid_ratio": float(n_valid / max(1, n_total)),
        },
        "per_corner": per_corner,
    }
    return uv_out, valid, diag
