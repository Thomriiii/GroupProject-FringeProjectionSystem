"""Experimental reconstruction-quality filtering utilities.

This module is intentionally post-UV: it does not modify phase, unwrap, UV, or
triangulation math. It only controls which UV pixels are forwarded to
triangulation and optionally smooths depth afterward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fringe_app.recon import reconstruct_uv_run
from fringe_app.recon.io import save_ply
from fringe_app.recon.triangulate import ReconstructionResult, StereoModel


@dataclass(slots=True)
class ReconstructionQualityParams:
    enable_confidence_filter: bool = True
    max_reproj_error_px: float = 1.5
    min_modulation_B: float = 25.0
    enable_smoothing: bool = False
    enable_sweep: bool = False

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "ReconstructionQualityParams":
        c = dict(cfg or {})
        return cls(
            enable_confidence_filter=bool(c.get("enable_confidence_filter", c.get("enabled", True))),
            max_reproj_error_px=float(c.get("max_reproj_error_px", 1.5)),
            min_modulation_B=float(c.get("min_modulation_B", 25.0)),
            enable_smoothing=bool(c.get("enable_smoothing", False)),
            enable_sweep=bool(c.get("enable_sweep", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_depth_png(depth: np.ndarray, mask: np.ndarray, path: Path) -> None:
    d = np.asarray(depth, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    valid = m & np.isfinite(d)
    out = np.zeros(d.shape, dtype=np.uint8)
    if np.any(valid):
        vals = d[valid]
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.0))
        if not np.isfinite(lo):
            lo = float(np.min(vals))
        if not np.isfinite(hi):
            hi = float(np.max(vals))
        if hi <= lo:
            hi = lo + 1e-6
        out[valid] = np.clip(((d[valid] - lo) / (hi - lo)) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(path)


def _to_gray_png(values: np.ndarray, mask: np.ndarray, path: Path) -> None:
    v = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    valid = m & np.isfinite(v)
    out = np.zeros(v.shape, dtype=np.uint8)
    if np.any(valid):
        vals = v[valid]
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.0))
        if not np.isfinite(lo):
            lo = float(np.min(vals))
        if not np.isfinite(hi):
            hi = float(np.max(vals))
        if hi <= lo:
            hi = lo + 1e-6
        out[valid] = np.clip(((v[valid] - lo) / (hi - lo)) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(path)


def _to_mask_png(mask: np.ndarray, path: Path) -> None:
    m = np.asarray(mask, dtype=bool)
    out = np.where(m, 255, 0).astype(np.uint8)
    Image.fromarray(out).save(path)


def filter_points_by_confidence(
    uv,
    B_vertical,
    B_horizontal,
    reproj_err_cam,
    reproj_err_proj,
    params,
) -> np.ndarray:
    """Build quality mask from reprojection and modulation constraints."""
    if isinstance(params, ReconstructionQualityParams):
        p = params
    elif isinstance(params, dict):
        p = ReconstructionQualityParams.from_dict(params)
    else:
        p = ReconstructionQualityParams.from_dict(None)

    reproj_cam = np.asarray(reproj_err_cam, dtype=np.float64)
    reproj_proj = np.asarray(reproj_err_proj, dtype=np.float64)
    Bv = np.asarray(B_vertical, dtype=np.float64)
    Bh = np.asarray(B_horizontal, dtype=np.float64)

    if reproj_cam.shape != reproj_proj.shape:
        raise ValueError("reproj_err_cam and reproj_err_proj shapes must match")
    if Bv.shape != reproj_cam.shape or Bh.shape != reproj_cam.shape:
        raise ValueError("Modulation maps must match reprojection map shape")

    if not bool(p.enable_confidence_filter):
        return np.ones(reproj_cam.shape, dtype=bool)

    max_reproj = float(p.max_reproj_error_px)
    min_B = float(p.min_modulation_B)

    mask_quality = np.isfinite(reproj_cam) & np.isfinite(reproj_proj)
    mask_quality &= np.isfinite(Bv) & np.isfinite(Bh)
    mask_quality &= reproj_cam < max_reproj
    mask_quality &= reproj_proj < max_reproj
    mask_quality &= Bv > min_B
    mask_quality &= Bh > min_B

    if isinstance(uv, (tuple, list)) and len(uv) == 2:
        u = np.asarray(uv[0], dtype=np.float64)
        v = np.asarray(uv[1], dtype=np.float64)
        if u.shape == mask_quality.shape and v.shape == mask_quality.shape:
            mask_quality &= np.isfinite(u) & np.isfinite(v)
    elif isinstance(uv, np.ndarray) and uv.ndim == 3 and uv.shape[2] >= 2:
        if uv.shape[:2] == mask_quality.shape:
            mask_quality &= np.isfinite(uv[:, :, 0]) & np.isfinite(uv[:, :, 1])

    return mask_quality


def smooth_depth_bilateral(depth: np.ndarray) -> np.ndarray:
    """3x3 edge-preserving smoothing restricted to valid (finite) pixels."""
    d = np.asarray(depth, dtype=np.float64)
    out = d.copy()
    valid = np.isfinite(d)
    if not np.any(valid):
        return out.astype(np.float32)

    vals = d[valid]
    sigma_r = float(np.std(vals) * 0.10)
    sigma_r = max(sigma_r, 1e-4)
    sigma_s = 1.0

    coords = np.argwhere(valid)
    for yx in coords:
        y = int(yx[0])
        x = int(yx[1])
        y0 = max(0, y - 1)
        y1 = min(d.shape[0], y + 2)
        x0 = max(0, x - 1)
        x1 = min(d.shape[1], x + 2)

        patch = d[y0:y1, x0:x1]
        patch_valid = valid[y0:y1, x0:x1]
        if int(np.count_nonzero(patch_valid)) <= 1:
            continue

        yy, xx = np.indices(patch.shape)
        dy = yy.astype(np.float64) + float(y0 - y)
        dx = xx.astype(np.float64) + float(x0 - x)
        spatial = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_s * sigma_s))
        center = float(d[y, x])
        rng = np.exp(-((patch - center) ** 2) / (2.0 * sigma_r * sigma_r))
        w = spatial * rng * patch_valid.astype(np.float64)
        denom = float(np.sum(w))
        if denom <= 0.0:
            continue
        out[y, x] = float(np.sum(w * patch) / denom)

    return out.astype(np.float32)


def _fit_plane_residual_metrics(result: ReconstructionResult) -> dict[str, float | None]:
    valid = np.asarray(result.mask_recon, dtype=bool)
    xyz = np.asarray(result.xyz, dtype=np.float64)
    finite = valid & np.isfinite(xyz[:, :, 0]) & np.isfinite(xyz[:, :, 1]) & np.isfinite(xyz[:, :, 2])
    pts = xyz[finite]
    if pts.shape[0] < 3:
        return {
            "plane_residual_rms_m": None,
            "plane_residual_p95_m": None,
            "outlier_ratio": None,
        }

    A = np.column_stack([pts[:, 0], pts[:, 1], np.ones((pts.shape[0],), dtype=np.float64)])
    coeff, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
    pred = A @ coeff
    residual = pts[:, 2] - pred

    rms = float(np.sqrt(np.mean(np.square(residual))))
    p95 = float(np.percentile(np.abs(residual), 95.0))
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    sigma = 1.4826 * mad
    if sigma <= 1e-12:
        outlier_ratio = 0.0
    else:
        outlier_ratio = float(np.count_nonzero(np.abs(residual - med) > (3.0 * sigma)) / residual.size)

    return {
        "plane_residual_rms_m": rms,
        "plane_residual_p95_m": p95,
        "outlier_ratio": outlier_ratio,
    }


def summarize_quality_metrics(
    result: ReconstructionResult,
    quality_mask: np.ndarray,
) -> dict[str, Any]:
    metrics = {
        "point_count": int(np.count_nonzero(np.asarray(result.mask_recon, dtype=bool))),
        "quality_mask_ratio": float(
            np.count_nonzero(np.asarray(quality_mask, dtype=bool)) / max(1, np.asarray(quality_mask).size)
        ),
    }
    metrics.update(_fit_plane_residual_metrics(result))
    return metrics


def _load_B_from_phase_root(phase_root: Path) -> np.ndarray:
    direct = phase_root / "B.npy"
    if direct.exists():
        return np.load(direct).astype(np.float32)

    subdirs = [p for p in phase_root.glob("f_*") if p.is_dir()]
    if len(subdirs) == 0:
        raise FileNotFoundError(f"No phase frequency folders under: {phase_root}")

    def _freq_value(name: str) -> float:
        txt = name.replace("f_", "").replace("p", ".")
        try:
            return float(txt)
        except Exception:
            return 0.0

    subdirs.sort(key=lambda p: _freq_value(p.name))
    for d in reversed(subdirs):
        b = d / "B.npy"
        if b.exists():
            return np.load(b).astype(np.float32)
    raise FileNotFoundError(f"No B.npy found under: {phase_root}")


def load_modulation_maps(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    v_root = run_dir / "vertical" / "phase"
    h_root = run_dir / "horizontal" / "phase"

    if v_root.exists() and h_root.exists():
        Bv = _load_B_from_phase_root(v_root)
        Bh = _load_B_from_phase_root(h_root)
    else:
        p_root = run_dir / "phase"
        if not p_root.exists():
            raise FileNotFoundError("Missing phase folders for modulation maps")
        B = _load_B_from_phase_root(p_root)
        Bv = B.copy()
        Bh = B.copy()

    if Bv.shape != Bh.shape:
        raise ValueError(f"B map shape mismatch: {Bv.shape} vs {Bh.shape}")
    return Bv, Bh


def save_quality_diagnostics(
    out_dir: Path,
    quality_mask: np.ndarray,
    reproj_err_cam: np.ndarray,
    reproj_err_proj: np.ndarray,
    B_vertical: np.ndarray,
    B_horizontal: np.ndarray,
    depth_before: np.ndarray,
    depth_after: np.ndarray,
    mask_recon: np.ndarray,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = out_dir / "mask_quality.png"
    reproj_path = out_dir / "reprojection_error_map.png"
    mod_path = out_dir / "modulation_map.png"
    depth_before_path = out_dir / "depth_before.png"
    depth_after_path = out_dir / "depth_after.png"

    reproj_map = np.maximum(
        np.asarray(reproj_err_cam, dtype=np.float64),
        np.asarray(reproj_err_proj, dtype=np.float64),
    )
    modulation_map = np.minimum(
        np.asarray(B_vertical, dtype=np.float64),
        np.asarray(B_horizontal, dtype=np.float64),
    )

    reproj_valid = np.isfinite(reproj_map)
    modulation_valid = np.isfinite(modulation_map)

    _to_mask_png(quality_mask, mask_path)
    _to_gray_png(reproj_map, reproj_valid, reproj_path)
    _to_gray_png(modulation_map, modulation_valid, mod_path)
    _to_depth_png(depth_before, np.asarray(mask_recon, dtype=bool), depth_before_path)
    _to_depth_png(depth_after, np.asarray(mask_recon, dtype=bool), depth_after_path)

    return {
        "mask_quality": str(mask_path),
        "reprojection_error_map": str(reproj_path),
        "modulation_map": str(mod_path),
        "depth_before": str(depth_before_path),
        "depth_after": str(depth_after_path),
    }


def _export_result_ply(result: ReconstructionResult, out_path: Path, max_points: int = 800000) -> int:
    mask = np.asarray(result.mask_recon, dtype=bool)
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("No valid points to export")

    if ys.size > int(max_points) > 0:
        step = int(np.ceil(ys.size / float(max_points)))
        ys = ys[::step]
        xs = xs[::step]

    pts = np.asarray(result.xyz[ys, xs, :], dtype=np.float32)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    ys = ys[finite]
    xs = xs[finite]
    if pts.shape[0] == 0:
        raise ValueError("All exported points are non-finite")

    colors = None
    if result.rgb is not None:
        colors = np.asarray(result.rgb[ys, xs, :3], dtype=np.float32)

    save_ply(pts, colors, out_path)
    return int(pts.shape[0])


def sweep_reconstruction_quality(
    *,
    run_dir: Path,
    model: StereoModel,
    recon_cfg: dict[str, Any],
    uv,
    B_vertical: np.ndarray,
    B_horizontal: np.ndarray,
    reproj_err_cam: np.ndarray,
    reproj_err_proj: np.ndarray,
    out_dir: Path,
) -> dict[str, Any]:
    """Run experimental threshold sweep and persist summary artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    reproj_thresholds = [0.5, 1.0, 1.5, 2.0]
    modulation_thresholds = [15, 20, 25, 30]

    rows: list[dict[str, Any]] = []
    for rthr in reproj_thresholds:
        for bthr in modulation_thresholds:
            params = ReconstructionQualityParams(
                enable_confidence_filter=True,
                max_reproj_error_px=float(rthr),
                min_modulation_B=float(bthr),
                enable_smoothing=False,
                enable_sweep=True,
            )
            mask_quality = filter_points_by_confidence(
                uv=uv,
                B_vertical=B_vertical,
                B_horizontal=B_horizontal,
                reproj_err_cam=reproj_err_cam,
                reproj_err_proj=reproj_err_proj,
                params=params,
            )

            row: dict[str, Any] = {
                "max_reproj_error_px": float(rthr),
                "min_modulation_B": int(bthr),
                "quality_mask_ratio": float(np.count_nonzero(mask_quality) / max(1, mask_quality.size)),
            }
            try:
                result = reconstruct_uv_run(
                    run_dir,
                    model,
                    recon_cfg=recon_cfg,
                    extra_valid_mask=mask_quality,
                )
                metrics = summarize_quality_metrics(result, mask_quality)
                row.update(metrics)
                out_name = f"recon_r{rthr:.1f}_B{int(bthr)}.ply"
                out_path = out_dir / out_name
                exported = _export_result_ply(result, out_path, max_points=int((recon_cfg or {}).get("max_points", 800000)))
                row["exported_points"] = int(exported)
                row["output_file"] = out_name
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)

    report = {
        "sweep": rows,
        "reprojection_thresholds_px": reproj_thresholds,
        "modulation_thresholds_B": modulation_thresholds,
    }
    (out_dir / "quality_report.json").write_text(json.dumps(report, indent=2))
    return report
