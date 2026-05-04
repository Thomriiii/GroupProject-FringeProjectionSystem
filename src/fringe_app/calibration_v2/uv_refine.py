"""Minimal UV corner sampling and optional local plane refinement."""

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
    if xs.size < 6 or ys.size != xs.size or zs.size != xs.size:
        return None
    A = np.column_stack([xs, ys, np.ones(xs.shape[0], dtype=np.float64)])
    try:
        coeff, *_ = np.linalg.lstsq(A, zs, rcond=None)
    except Exception:
        return None
    return coeff.astype(np.float64)


def _safe_median(vals: np.ndarray) -> float:
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals.astype(np.float64)))


def sample_and_refine_uv(
    u_map: np.ndarray,
    v_map: np.ndarray,
    mask: np.ndarray,
    corners_px: np.ndarray,
    projector_size: tuple[int, int],
    cfg: UvRefineConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    u = np.asarray(u_map, dtype=np.float64)
    v = np.asarray(v_map, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    corners = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)

    if u.shape != v.shape or u.shape != m.shape:
        raise ValueError("u_map, v_map, and mask must have identical shapes")

    h, w = u.shape
    proj_w = int(projector_size[0])
    proj_h = int(projector_size[1])

    uv_out = np.full((corners.shape[0], 2), np.nan, dtype=np.float32)
    valid = np.zeros((corners.shape[0],), dtype=bool)
    per_corner: list[dict[str, Any]] = []

    r = int(cfg.patch_radius_px)
    for idx, (x, y) in enumerate(corners):
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        entry: dict[str, Any] = {
            "index": int(idx),
            "x": float(x),
            "y": float(y),
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

        x0 = max(0, xi - r)
        x1 = min(w, xi + r + 1)
        y0 = max(0, yi - r)
        y1 = min(h, yi + r + 1)

        pm = m[y0:y1, x0:x1]
        pu = u[y0:y1, x0:x1]
        pv = v[y0:y1, x0:x1]
        good = pm & np.isfinite(pu) & np.isfinite(pv)

        n_good = int(np.count_nonzero(good))
        entry["patch_points"] = n_good
        if n_good == 0:
            per_corner.append(entry)
            continue

        vals_u = pu[good]
        vals_v = pv[good]
        raw_u = _safe_median(vals_u)
        raw_v = _safe_median(vals_v)
        entry["raw_u"] = float(raw_u) if np.isfinite(raw_u) else None
        entry["raw_v"] = float(raw_v) if np.isfinite(raw_v) else None

        final_u = raw_u
        final_v = raw_v

        if bool(cfg.enabled) and bool(cfg.plane_fit) and n_good >= 6:
            ys_local, xs_local = np.where(good)
            xs = xs_local.astype(np.float64) + float(x0)
            ys = ys_local.astype(np.float64) + float(y0)
            coeff_u = _fit_plane(xs, ys, vals_u.astype(np.float64))
            coeff_v = _fit_plane(xs, ys, vals_v.astype(np.float64))
            if coeff_u is not None and coeff_v is not None:
                pred_u = float(coeff_u[0] * float(x) + coeff_u[1] * float(y) + coeff_u[2])
                pred_v = float(coeff_v[0] * float(x) + coeff_v[1] * float(y) + coeff_v[2])
                if np.isfinite(pred_u) and np.isfinite(pred_v):
                    delta = float(np.hypot(pred_u - raw_u, pred_v - raw_v))
                    entry["delta_px"] = delta
                    if delta <= float(cfg.max_delta_px):
                        final_u = pred_u
                        final_v = pred_v
                        entry["plane_fit_used"] = True

        if not (np.isfinite(final_u) and np.isfinite(final_v)):
            entry["reason"] = "nan_uv"
            per_corner.append(entry)
            continue

        if final_u < 0.0 or final_u >= float(proj_w) or final_v < 0.0 or final_v >= float(proj_h):
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

    n_total = int(valid.size)
    n_valid = int(np.count_nonzero(valid))
    diag = {
        "config": {
            "enabled": bool(cfg.enabled),
            "patch_radius_px": int(cfg.patch_radius_px),
            "plane_fit": bool(cfg.plane_fit),
            "max_delta_px": float(cfg.max_delta_px),
        },
        "summary": {
            "corners_total": n_total,
            "corners_valid": n_valid,
            "valid_ratio": float(n_valid / max(1, n_total)),
        },
        "per_corner": per_corner,
    }
    return uv_out, valid, diag
