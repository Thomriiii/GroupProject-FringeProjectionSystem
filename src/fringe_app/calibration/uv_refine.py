"""Offline UV corner refinement utilities for projector calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class UvRefineConfig:
    enabled: bool = True
    patch_radius: int = 4
    min_valid_points: int = 15
    robust_refit: bool = True
    robust_reject_frac: float = 0.10
    max_cond: float = 1e4

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "UvRefineConfig":
        c = dict(cfg or {})
        return cls(
            enabled=bool(c.get("enabled", True)),
            patch_radius=max(1, int(c.get("patch_radius", 4))),
            min_valid_points=max(3, int(c.get("min_valid_points", 15))),
            robust_refit=bool(c.get("robust_refit", True)),
            robust_reject_frac=float(c.get("robust_reject_frac", 0.10)),
            max_cond=float(c.get("max_cond", 1e4)),
        )


def _fit_plane(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    cfg: UvRefineConfig,
) -> tuple[np.ndarray | None, float, float, float, float]:
    """Fit z = a*(x-x0) + b*(y-y0) + c; returns (coeff, mean_abs_resid, cond, x0, y0)."""
    if xs.size < cfg.min_valid_points:
        return None, float("nan"), float("inf"), float("nan"), float("nan")
    x0 = float(np.mean(xs))
    y0 = float(np.mean(ys))
    xc = xs - x0
    yc = ys - y0
    A = np.column_stack([xc, yc, np.ones(xs.shape[0], dtype=np.float64)])
    try:
        cond = float(np.linalg.cond(A))
    except Exception:
        cond = float("inf")
    if not np.isfinite(cond) or cond > cfg.max_cond:
        return None, float("nan"), cond, x0, y0

    def _solve(A_: np.ndarray, z_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coeff, *_ = np.linalg.lstsq(A_, z_, rcond=None)
        pred = A_ @ coeff
        return coeff, np.abs(z_ - pred)

    try:
        coeff, resid = _solve(A, zs)
    except Exception:
        return None, float("nan"), cond, x0, y0

    if cfg.robust_refit and zs.size >= (cfg.min_valid_points + 3):
        frac = float(np.clip(cfg.robust_reject_frac, 0.0, 0.45))
        k_reject = int(np.floor(resid.size * frac))
        k_keep = resid.size - k_reject
        if k_reject > 0 and k_keep >= cfg.min_valid_points:
            keep_idx = np.argsort(resid)[:k_keep]
            A2 = A[keep_idx]
            z2 = zs[keep_idx]
            try:
                cond2 = float(np.linalg.cond(A2))
            except Exception:
                cond2 = float("inf")
            if np.isfinite(cond2) and cond2 <= cfg.max_cond:
                coeff, resid = _solve(A2, z2)
                cond = cond2

    return coeff.astype(np.float64), float(np.mean(resid)), cond, x0, y0


def refine_uv_at_corners(
    U: np.ndarray,
    V: np.ndarray,
    mask_uv: np.ndarray,
    corners_xy: np.ndarray,
    cfg: UvRefineConfig | dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Refine projector UV at checkerboard corners using local plane fitting.

    Returns:
      uv_refined: (N,2) float32 (NaN where invalid)
      corner_valid_mask: (N,) bool
      diag: per-corner and summary diagnostics
    """
    c = cfg if isinstance(cfg, UvRefineConfig) else UvRefineConfig.from_dict(cfg)
    corners = np.asarray(corners_xy, dtype=np.float64).reshape(-1, 2)
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    mask = np.asarray(mask_uv, dtype=bool)

    if U.shape != V.shape or U.shape != mask.shape:
        raise ValueError("U, V, and mask_uv must have identical shapes")

    h, w = U.shape
    n = corners.shape[0]
    uv_refined = np.full((n, 2), np.nan, dtype=np.float32)
    valid = np.zeros((n,), dtype=bool)
    per_corner: list[dict[str, Any]] = []

    r = int(c.patch_radius)
    for i, (x, y) in enumerate(corners):
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        entry: dict[str, Any] = {
            "index": int(i),
            "x": float(x),
            "y": float(y),
            "n_points": 0,
            "fit_resid_u": float("nan"),
            "fit_resid_v": float("nan"),
            "cond_u": float("inf"),
            "cond_v": float("inf"),
            "valid": False,
            "reason": "insufficient_points",
        }
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            entry["reason"] = "corner_oob"
            per_corner.append(entry)
            continue

        x0 = max(0, xi - r)
        x1 = min(w, xi + r + 1)
        y0 = max(0, yi - r)
        y1 = min(h, yi + r + 1)
        m = mask[y0:y1, x0:x1]
        u_patch = U[y0:y1, x0:x1]
        v_patch = V[y0:y1, x0:x1]
        good = m & np.isfinite(u_patch) & np.isfinite(v_patch)
        if not np.any(good):
            per_corner.append(entry)
            continue

        ys_local, xs_local = np.where(good)
        xs = xs_local.astype(np.float64) + float(x0)
        ys = ys_local.astype(np.float64) + float(y0)
        us = u_patch[good].astype(np.float64)
        vs = v_patch[good].astype(np.float64)
        n_pts = int(min(us.size, vs.size))
        entry["n_points"] = n_pts
        if n_pts < c.min_valid_points:
            per_corner.append(entry)
            continue

        coeff_u, resid_u, cond_u, x0_u, y0_u = _fit_plane(xs, ys, us, c)
        coeff_v, resid_v, cond_v, x0_v, y0_v = _fit_plane(xs, ys, vs, c)
        entry["fit_resid_u"] = float(resid_u)
        entry["fit_resid_v"] = float(resid_v)
        entry["cond_u"] = float(cond_u)
        entry["cond_v"] = float(cond_v)
        if coeff_u is None or coeff_v is None:
            entry["reason"] = "ill_conditioned"
            per_corner.append(entry)
            continue

        u_hat = float(coeff_u[0] * (x - x0_u) + coeff_u[1] * (y - y0_u) + coeff_u[2])
        v_hat = float(coeff_v[0] * (x - x0_v) + coeff_v[1] * (y - y0_v) + coeff_v[2])
        if not (np.isfinite(u_hat) and np.isfinite(v_hat)):
            entry["reason"] = "nan_prediction"
            per_corner.append(entry)
            continue

        uv_refined[i, 0] = np.float32(u_hat)
        uv_refined[i, 1] = np.float32(v_hat)
        valid[i] = True
        entry["valid"] = True
        entry["reason"] = "ok"
        per_corner.append(entry)

    n_valid = int(np.count_nonzero(valid))
    deltas = uv_refined[valid]
    resid_u_vals = [float(p["fit_resid_u"]) for p in per_corner if np.isfinite(float(p.get("fit_resid_u", np.nan)))]
    resid_v_vals = [float(p["fit_resid_v"]) for p in per_corner if np.isfinite(float(p.get("fit_resid_v", np.nan)))]
    summary = {
        "enabled": bool(c.enabled),
        "patch_radius": int(c.patch_radius),
        "min_valid_points": int(c.min_valid_points),
        "robust_refit": bool(c.robust_refit),
        "robust_reject_frac": float(c.robust_reject_frac),
        "max_cond": float(c.max_cond),
        "corners_total": int(n),
        "corners_valid": n_valid,
        "valid_ratio": float(n_valid / max(1, n)),
        "mean_patch_size": float(np.mean([p["n_points"] for p in per_corner])) if per_corner else 0.0,
        "mean_fit_resid_u": float(np.mean(resid_u_vals)) if resid_u_vals else float("nan"),
        "mean_fit_resid_v": float(np.mean(resid_v_vals)) if resid_v_vals else float("nan"),
        "mean_uv_abs": float(np.nanmean(np.abs(deltas))) if deltas.size else float("nan"),
    }
    diag = {
        "summary": summary,
        "per_corner": per_corner,
    }
    return uv_refined, valid, diag
