#!/usr/bin/env python3
"""Offline reconstruction reprojection drift diagnostics.

This module is diagnostics-only and does not modify pipeline behavior.

Usage:
  PYTHONPATH=src .venv/bin/python -m fringe_app.tools.diagnose_reproj_drift --run 20260226_103308_uv
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        fv = float(value)
        return fv if math.isfinite(fv) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_run_dir(run_arg: str) -> Path:
    p = Path(run_arg)
    if p.exists():
        return p
    p2 = Path("data/runs") / run_arg
    if p2.exists():
        return p2
    raise SystemExit(f"Run not found: {run_arg}")


def _stats(values: np.ndarray, percentiles: tuple[int, ...] = (50, 90, 95, 99)) -> dict[str, float | None]:
    vals = values[np.isfinite(values)]
    out: dict[str, float | None] = {}
    if vals.size == 0:
        for p in percentiles:
            out[f"p{p}"] = None
        out["max"] = None
        out["mean"] = None
        return out
    for p in percentiles:
        out[f"p{p}"] = float(np.percentile(vals, p))
    out["max"] = float(np.max(vals))
    out["mean"] = float(np.mean(vals))
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or b.size < 3:
        return None
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 1e-12 or sb <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3 or y.size < 3:
        return None
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    sx = float(np.std(x))
    if sx <= 1e-12:
        return None
    coeff = np.polyfit(x, y, 1)
    return float(coeff[0])


def _maybe_load_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _plot_heatmap(arr: np.ndarray, title: str, out_path: Path, mask: np.ndarray | None = None, cmap: str = "magma") -> None:
    data = arr.astype(np.float64).copy()
    if mask is not None:
        data[~mask] = np.nan
    finite = np.isfinite(data)
    if np.any(finite):
        lo, hi = np.percentile(data[finite], [1, 99])
        if hi <= lo:
            hi = lo + 1e-6
    else:
        lo, hi = 0.0, 1.0
    plt.figure(figsize=(8, 5))
    plt.imshow(data, cmap=cmap, vmin=lo, vmax=hi)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_binary(mask: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.imshow(mask.astype(np.uint8), cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_curve(y: np.ndarray, title: str, ylabel: str, out_path: Path) -> None:
    x = np.arange(y.size)
    plt.figure(figsize=(9, 4))
    plt.plot(x, y, linewidth=1.2)
    plt.title(title)
    plt.xlabel("index")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    max_points: int = 60000,
) -> None:
    n = x.size
    if n == 0:
        plt.figure(figsize=(7, 5))
        plt.title(f"{title} (no data)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=140)
        plt.close()
        return
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        xs = x[idx]
        ys = y[idx]
    else:
        xs = x
        ys = y
    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, s=2, alpha=0.12)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_hist_bands(err: np.ndarray, mask: np.ndarray, out_path: Path) -> dict[str, Any]:
    vals = err[mask & np.isfinite(err)]
    bands = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, float("inf"))]
    counts: list[int] = []
    labels: list[str] = []
    total = int(vals.size)
    for lo, hi in bands:
        labels.append(f">{lo:g}" if math.isinf(hi) else f"{lo:g}-{hi:g}")
        if math.isinf(hi):
            c = int(np.count_nonzero(vals >= lo))
        else:
            c = int(np.count_nonzero((vals >= lo) & (vals < hi)))
        counts.append(c)

    fracs = [float(c / total) if total else 0.0 for c in counts]
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(labels)), fracs)
    plt.xticks(np.arange(len(labels)), labels)
    plt.ylabel("fraction of UV-valid")
    plt.title("Camera reprojection error bands")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    return {
        "bands_px": labels,
        "counts": counts,
        "fractions": fracs,
    }


def _profile_mean(values: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    if axis == 1:
        out = np.full(values.shape[0], np.nan, dtype=np.float64)
        for i in range(values.shape[0]):
            m = mask[i] & np.isfinite(values[i])
            if np.any(m):
                out[i] = float(np.mean(values[i][m]))
        return out
    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    for i in range(values.shape[1]):
        m = mask[:, i] & np.isfinite(values[:, i])
        if np.any(m):
            out[i] = float(np.mean(values[:, i][m]))
    return out


def _rejection_rate(mask_uv: np.ndarray, rejected: np.ndarray, axis: int) -> np.ndarray:
    if axis == 1:
        denom = mask_uv.sum(axis=1).astype(np.float64)
        numer = rejected.sum(axis=1).astype(np.float64)
    else:
        denom = mask_uv.sum(axis=0).astype(np.float64)
        numer = rejected.sum(axis=0).astype(np.float64)
    out = np.full(denom.shape, np.nan, dtype=np.float64)
    good = denom > 0
    out[good] = numer[good] / denom[good]
    return out


def _interp_nans(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float64).copy()
    n = y.size
    idx = np.arange(n)
    good = np.isfinite(y)
    if not np.any(good):
        return np.zeros_like(y)
    if np.count_nonzero(good) == 1:
        y[~good] = y[good][0]
        return y
    y[~good] = np.interp(idx[~good], idx[good], y[good])
    return y


def _fft_peak(profile: np.ndarray, out_path: Path, title: str) -> dict[str, Any]:
    y = _interp_nans(profile)
    y = y - float(np.mean(y))
    n = y.size
    if n < 4:
        return {"peak_bin": None, "peak_freq": None, "peak_period_px": None, "peak_mag": None}
    mag = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(n, d=1.0)
    if mag.size > 0:
        mag[0] = 0.0
    # Skip very low bins to avoid near-DC leakage.
    low_cut = max(1, int(round(0.002 * n)))
    mag_search = mag.copy()
    mag_search[:low_cut] = 0.0
    peak_bin = int(np.argmax(mag_search)) if np.any(mag_search > 0) else int(np.argmax(mag))
    peak_freq = float(freqs[peak_bin]) if peak_bin > 0 else None
    peak_period = float(n / peak_bin) if peak_bin > 0 else None
    peak_mag = float(mag[peak_bin]) if peak_bin >= 0 else None

    plt.figure(figsize=(9, 4))
    plt.plot(freqs, mag, linewidth=1.0)
    if peak_freq is not None:
        plt.axvline(peak_freq, color="red", linestyle="--", alpha=0.6)
    plt.title(title)
    plt.xlabel("frequency [cycles/pixel]")
    plt.ylabel("FFT magnitude")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    return {
        "peak_bin": peak_bin,
        "peak_freq": peak_freq,
        "peak_period_px": peak_period,
        "peak_mag": peak_mag,
        "low_bin_cut": low_cut,
    }


def _overlay_mask(base_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    out[mask, 0] = np.clip(0.25 * out[mask, 0] + 190.0, 0, 255)
    out[mask, 1] = np.clip(0.20 * out[mask, 1], 0, 255)
    out[mask, 2] = np.clip(0.20 * out[mask, 2], 0, 255)
    return out.astype(np.uint8)


def _load_background_camera(run_dir: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    candidates: list[Path] = []
    for base in [run_dir / "vertical" / "captures", run_dir / "captures"]:
        if not base.exists():
            continue
        for fd in sorted(base.glob("f_*")):
            candidates.extend(sorted(fd.glob("step_*.png")))
        candidates.extend(sorted(base.glob("frame_*.png")))
    for p in candidates:
        try:
            arr = np.array(Image.open(p))
        except Exception:
            continue
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=2)
        if arr.ndim == 3 and arr.shape[:2] == (h, w):
            return arr[:, :, :3].astype(np.uint8)
    return np.zeros((h, w, 3), dtype=np.uint8)


def _load_optional_b_map(run_dir: Path) -> tuple[np.ndarray | None, Path | None]:
    phase_root = run_dir / "vertical" / "phase"
    if not phase_root.exists():
        phase_root = run_dir / "phase"
    if not phase_root.exists():
        return None, None
    freq_dirs = sorted([p for p in phase_root.glob("f_*") if p.is_dir()])
    if not freq_dirs:
        return None, None
    def _key(p: Path) -> float:
        try:
            return float(p.name.split("_", 1)[1])
        except Exception:
            return -1.0
    high = max(freq_dirs, key=_key)
    b_path = high / "B.npy"
    if b_path.exists():
        return np.load(b_path), b_path
    return None, None


def _load_stereo() -> tuple[dict[str, Any] | None, Path | None]:
    path = Path("data/calibration/projector/stereo_latest.json")
    if not path.exists():
        return None, None
    try:
        return _load_json(path), path
    except Exception:
        return None, path


@dataclass
class CoverageInfo:
    available: bool
    session_id: str | None
    coverage_ratio: float | None
    per_view_coverage_ratio: dict[str, float]
    row_coverage: list[float] | None
    col_coverage: list[float] | None
    used_views_missing_correspondences: list[str]
    notes: list[str]


def _coverage_proxy_from_session(
    stereo: dict[str, Any] | None,
    grid_x: int = 16,
    grid_y: int = 16,
) -> CoverageInfo:
    if not stereo:
        return CoverageInfo(False, None, None, {}, None, None, [], ["stereo json missing"])
    sid = str(stereo.get("session_id", "")).strip()
    if not sid:
        return CoverageInfo(False, None, None, {}, None, None, [], ["session_id missing in stereo json"])
    session_dir = Path("data/calibration/projector/sessions") / sid
    if not session_dir.exists():
        return CoverageInfo(False, sid, None, {}, None, None, [], [f"session folder missing: {session_dir}"])

    proj = stereo.get("projector", {}) or {}
    pw = float(proj.get("width", 0))
    ph = float(proj.get("height", 0))
    if pw <= 0 or ph <= 0:
        return CoverageInfo(False, sid, None, {}, None, None, [], ["invalid projector size in stereo json"])

    per_view = stereo.get("per_view_errors", []) or []
    used_views = [str(v.get("view_id")) for v in per_view if str(v.get("view_id", "")).strip()]
    if not used_views:
        return CoverageInfo(False, sid, None, {}, None, None, [], ["no per_view_errors in stereo json"])

    grid = np.zeros((grid_y, grid_x), dtype=bool)
    per_view_cov: dict[str, float] = {}
    missing: list[str] = []

    for vid in used_views:
        cpath = session_dir / "views" / vid / "correspondences.json"
        if not cpath.exists():
            missing.append(vid)
            continue
        try:
            cjson = _load_json(cpath)
            pts = np.asarray(cjson.get("projector_corners_px", []), dtype=np.float64)
            vmask = np.asarray(cjson.get("valid_mask", []), dtype=bool)
            if pts.ndim != 2 or pts.shape[1] != 2:
                missing.append(vid)
                continue
            if vmask.size == pts.shape[0]:
                pts = pts[vmask]
            finite = np.isfinite(pts).all(axis=1)
            pts = pts[finite]
            if pts.size == 0:
                per_view_cov[vid] = 0.0
                continue
            bx = np.floor((pts[:, 0] / pw) * grid_x).astype(int)
            by = np.floor((pts[:, 1] / ph) * grid_y).astype(int)
            bx = np.clip(bx, 0, grid_x - 1)
            by = np.clip(by, 0, grid_y - 1)
            view_grid = np.zeros_like(grid)
            view_grid[by, bx] = True
            per_view_cov[vid] = float(np.mean(view_grid))
            grid[by, bx] = True
        except Exception:
            missing.append(vid)
            continue

    coverage_ratio = float(np.mean(grid))
    row_cov = grid.mean(axis=1).astype(float).tolist()
    col_cov = grid.mean(axis=0).astype(float).tolist()
    notes = []
    if missing:
        notes.append(f"missing correspondences for {len(missing)} used views")
    return CoverageInfo(
        available=True,
        session_id=sid,
        coverage_ratio=coverage_ratio,
        per_view_coverage_ratio=per_view_cov,
        row_coverage=row_cov,
        col_coverage=col_cov,
        used_views_missing_correspondences=missing,
        notes=notes,
    )


def _plot_coverage_grid(stereo: dict[str, Any] | None, cov: CoverageInfo, out_path: Path) -> None:
    if not stereo or not cov.available:
        return
    sid = cov.session_id
    if not sid:
        return
    session_dir = Path("data/calibration/projector/sessions") / sid
    per_view = stereo.get("per_view_errors", []) or []
    used_views = [str(v.get("view_id")) for v in per_view if str(v.get("view_id", "")).strip()]
    proj = stereo.get("projector", {}) or {}
    pw = float(proj.get("width", 0))
    ph = float(proj.get("height", 0))
    if pw <= 0 or ph <= 0:
        return

    gx = 16
    gy = 16
    grid = np.zeros((gy, gx), dtype=float)
    for vid in used_views:
        cpath = session_dir / "views" / vid / "correspondences.json"
        if not cpath.exists():
            continue
        try:
            cjson = _load_json(cpath)
            pts = np.asarray(cjson.get("projector_corners_px", []), dtype=np.float64)
            vmask = np.asarray(cjson.get("valid_mask", []), dtype=bool)
            if pts.ndim != 2 or pts.shape[1] != 2:
                continue
            if vmask.size == pts.shape[0]:
                pts = pts[vmask]
            finite = np.isfinite(pts).all(axis=1)
            pts = pts[finite]
            if pts.size == 0:
                continue
            bx = np.floor((pts[:, 0] / pw) * gx).astype(int)
            by = np.floor((pts[:, 1] / ph) * gy).astype(int)
            bx = np.clip(bx, 0, gx - 1)
            by = np.clip(by, 0, gy - 1)
            for x, y in zip(bx, by):
                grid[y, x] += 1.0
        except Exception:
            continue

    plt.figure(figsize=(7, 4.5))
    plt.imshow(grid, cmap="viridis")
    plt.colorbar(fraction=0.046, pad=0.04, label="corner hits")
    plt.title("Projector corner coverage (used calibration views)")
    plt.xlabel("projector x bin")
    plt.ylabel("projector y bin")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline reprojection drift diagnostics.")
    parser.add_argument("--run", required=True, help="Run id or path (e.g., 20260226_103308_uv)")
    parser.add_argument("--out", default=None, help="Output directory (default: <run>/diagnostics_step1b)")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run)
    out_dir = Path(args.out) if args.out else (run_dir / "diagnostics_step1b")
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "missing_files": [],
    }

    required_paths = {
        "u": run_dir / "projector_uv" / "u.npy",
        "v": run_dir / "projector_uv" / "v.npy",
        "mask_uv": run_dir / "projector_uv" / "mask_uv.npy",
        "reconstruction_meta": run_dir / "reconstruction" / "reconstruction_meta.json",
        "mask_recon": run_dir / "reconstruction" / "masks" / "mask_recon.npy",
        "reproj_err_cam": run_dir / "reconstruction" / "reproj_err_cam.npy",
        "reproj_err_proj": run_dir / "reconstruction" / "reproj_err_proj.npy",
    }
    for key, path in required_paths.items():
        if not path.exists():
            report["missing_files"].append({key: str(path)})

    if report["missing_files"]:
        report["status"] = "missing_required_files"
        (out_dir / "report.json").write_text(json.dumps(_json_safe(report), indent=2))
        (out_dir / "conclusions.txt").write_text(
            "Missing required files; see report.json for exact paths.\n"
        )
        print(json.dumps(_json_safe(report), indent=2))
        return 2

    u = np.load(required_paths["u"]).astype(np.float32)
    v = np.load(required_paths["v"]).astype(np.float32)
    mask_uv = np.load(required_paths["mask_uv"]).astype(bool)
    mask_recon = np.load(required_paths["mask_recon"]).astype(bool)
    err_cam = np.load(required_paths["reproj_err_cam"]).astype(np.float32)
    err_proj = np.load(required_paths["reproj_err_proj"]).astype(np.float32)
    recon_meta = _load_json(required_paths["reconstruction_meta"])

    uv_meta_path = run_dir / "projector_uv" / "uv_meta.json"
    uv_meta = _load_json(uv_meta_path) if uv_meta_path.exists() else {}
    summary_path = run_dir / "diagnostics" / "recon_reproj_spaces" / "summary.json"
    prev_summary = _load_json(summary_path) if summary_path.exists() else {}

    depth = _first_existing([
        run_dir / "reconstruction" / "depth.npy",
        run_dir / "reconstruction" / "depth_map.npy",
    ])
    depth_map = np.load(depth).astype(np.float32) if depth else None

    b_map, b_map_path = _load_optional_b_map(run_dir)

    residual_v = _maybe_load_npy(run_dir / "vertical" / "unwrap" / "residual.npy")
    residual_h = _maybe_load_npy(run_dir / "horizontal" / "unwrap" / "residual.npy")

    stereo_json, stereo_path = _load_stereo()
    coverage = _coverage_proxy_from_session(stereo_json, grid_x=16, grid_y=16)

    shapes = {
        "u": list(u.shape),
        "v": list(v.shape),
        "mask_uv": list(mask_uv.shape),
        "mask_recon": list(mask_recon.shape),
        "reproj_err_cam": list(err_cam.shape),
        "reproj_err_proj": list(err_proj.shape),
        "depth": list(depth_map.shape) if depth_map is not None else None,
        "b_map": list(b_map.shape) if b_map is not None else None,
    }
    report["shapes"] = shapes

    h, w = u.shape
    align_ok = (
        v.shape == (h, w)
        and mask_uv.shape == (h, w)
        and mask_recon.shape == (h, w)
        and err_cam.shape == (h, w)
        and err_proj.shape == (h, w)
        and (depth_map is None or depth_map.shape == (h, w))
        and (b_map is None or b_map.shape == (h, w))
    )
    report["shape_alignment_ok"] = bool(align_ok)

    uv_valid = mask_uv & np.isfinite(u) & np.isfinite(v)
    err_valid = uv_valid & np.isfinite(err_cam)
    recon_valid = uv_valid & mask_recon
    rejected = uv_valid & (~mask_recon)
    rej_gt2 = err_valid & (err_cam > 2.0)
    rej_gt3 = err_valid & (err_cam > 3.0)

    _plot_heatmap(err_cam, "Camera reprojection error (UV-valid region)", plots_dir / "reproj_err_cam_heatmap.png", mask=uv_valid)
    _plot_binary(rej_gt2, "Rejected by camera reproj (err > 2px)", plots_dir / "rejected_cam_gt2.png")
    _plot_binary(rej_gt3, "Rejected by camera reproj (err > 3px)", plots_dir / "rejected_cam_gt3.png")

    row_err = _profile_mean(err_cam, err_valid, axis=1)
    col_err = _profile_mean(err_cam, err_valid, axis=0)
    _plot_curve(row_err, "Row mean camera reprojection error", "error [px]", plots_dir / "row_mean_reproj_err.png")
    _plot_curve(col_err, "Column mean camera reprojection error", "error [px]", plots_dir / "col_mean_reproj_err.png")

    ys, xs = np.where(err_valid)
    ev = err_cam[ys, xs].astype(np.float64)
    uu = u[ys, xs].astype(np.float64)
    vv = v[ys, xs].astype(np.float64)
    _plot_scatter(uu, ev, "Camera reproj error vs U", "U [projector px]", "error [px]", plots_dir / "scatter_err_vs_u.png")
    _plot_scatter(vv, ev, "Camera reproj error vs V", "V [projector px]", "error [px]", plots_dir / "scatter_err_vs_v.png")

    if depth_map is not None and depth_map.shape == (h, w):
        dvals = depth_map[ys, xs].astype(np.float64)
        fin = np.isfinite(dvals)
        _plot_scatter(dvals[fin], ev[fin], "Camera reproj error vs depth", "depth [m]", "error [px]", plots_dir / "scatter_err_vs_depth.png")

    if b_map is not None and b_map.shape == (h, w):
        bvals = b_map[ys, xs].astype(np.float64)
        fin = np.isfinite(bvals)
        _plot_scatter(bvals[fin], ev[fin], "Camera reproj error vs B", "B modulation", "error [px]", plots_dir / "scatter_err_vs_B.png")

    row_rej = _rejection_rate(uv_valid, rejected, axis=1)
    col_rej = _rejection_rate(uv_valid, rejected, axis=0)
    _plot_curve(row_rej, "Row rejection rate (within UV-valid)", "rate", plots_dir / "row_rejection_rate.png")
    _plot_curve(col_rej, "Column rejection rate (within UV-valid)", "rate", plots_dir / "col_rejection_rate.png")

    fft_row = _fft_peak(row_rej, plots_dir / "fft_rejection_rows.png", "FFT of row rejection rate")
    fft_col = _fft_peak(col_rej, plots_dir / "fft_rejection_cols.png", "FFT of column rejection rate")

    background = _load_background_camera(run_dir, (h, w))
    overlay_rejected = _overlay_mask(background, rejected)
    Image.fromarray(overlay_rejected).save(plots_dir / "reject_overlay_on_camera.png")

    uv_overlay_path = run_dir / "projector_uv" / "uv_overlay.png"
    if uv_overlay_path.exists():
        uv_img = np.array(Image.open(uv_overlay_path))
        if uv_img.ndim == 2:
            uv_img = np.stack([uv_img, uv_img, uv_img], axis=2)
        uv_img = uv_img[:, :, :3]
        if uv_img.shape[:2] == (h, w):
            Image.fromarray(_overlay_mask(uv_img, rejected)).save(plots_dir / "reject_overlay_on_uv.png")

    err_bands = _plot_hist_bands(err_cam, err_valid, plots_dir / "reproj_err_band_fractions.png")

    # Drift/bias tests.
    proj_w = float((uv_meta.get("projector_size") or [w, h])[0] if isinstance(uv_meta.get("projector_size"), list) else (stereo_json or {}).get("projector", {}).get("width", w))
    proj_h = float((uv_meta.get("projector_size") or [w, h])[1] if isinstance(uv_meta.get("projector_size"), list) else (stereo_json or {}).get("projector", {}).get("height", h))
    if proj_w <= 0:
        proj_w = float((stereo_json or {}).get("projector", {}).get("width", w))
    if proj_h <= 0:
        proj_h = float((stereo_json or {}).get("projector", {}).get("height", h))
    if proj_w <= 0:
        proj_w = float(w)
    if proj_h <= 0:
        proj_h = float(h)

    edge_dist = np.minimum.reduce([uu, vv, proj_w - 1.0 - uu, proj_h - 1.0 - vv])
    cam_x = xs.astype(np.float64)
    cam_y = ys.astype(np.float64)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    cam_radius = np.sqrt((cam_x - cx) ** 2 + (cam_y - cy) ** 2)
    cam_radius /= max(1e-6, math.sqrt(cx**2 + cy**2))

    depth_corr = None
    depth_slope = None
    if depth_map is not None and depth_map.shape == (h, w):
        d = depth_map[ys, xs].astype(np.float64)
        finite = np.isfinite(d)
        depth_corr = _corr(ev[finite], d[finite])
        depth_slope = _linear_slope(d[finite], ev[finite])

    b_corr = None
    if b_map is not None and b_map.shape == (h, w):
        bvals = b_map[ys, xs].astype(np.float64)
        fin = np.isfinite(bvals)
        b_corr = _corr(ev[fin], bvals[fin])

    drift_tests = {
        "depth_bias": {
            "corr_err_depth": depth_corr,
            "slope_err_per_meter": depth_slope,
        },
        "uv_bias": {
            "corr_err_u": _corr(ev, uu),
            "corr_err_v": _corr(ev, vv),
            "corr_err_edge_distance": _corr(ev, edge_dist),
        },
        "camera_pos_bias": {
            "corr_err_cam_x": _corr(ev, cam_x),
            "corr_err_cam_y": _corr(ev, cam_y),
            "corr_err_cam_radius": _corr(ev, cam_radius),
        },
        "banding": {
            "row_err_std": float(np.nanstd(row_err)),
            "col_err_std": float(np.nanstd(col_err)),
            "row_reject_std": float(np.nanstd(row_rej)),
            "col_reject_std": float(np.nanstd(col_rej)),
            "fft_row": fft_row,
            "fft_col": fft_col,
            "strongest_axis": "rows" if float(np.nanstd(row_rej)) >= float(np.nanstd(col_rej)) else "cols",
        },
        "corr_err_B": b_corr,
    }

    gate = float(recon_meta.get("max_reproj_err_px", np.nan))
    stereo_rms = float((stereo_json or {}).get("rms_stereo", np.nan))
    stereo_compare = {
        "stereo_rms_px": stereo_rms if math.isfinite(stereo_rms) else None,
        "recon_gate_px": gate if math.isfinite(gate) else None,
        "gate_minus_stereo_rms_px": (gate - stereo_rms) if math.isfinite(gate) and math.isfinite(stereo_rms) else None,
        "gate_over_stereo_rms": (gate / stereo_rms) if math.isfinite(gate) and math.isfinite(stereo_rms) and abs(stereo_rms) > 1e-12 else None,
    }

    # Calibration-side diagnosis.
    per_view = (stereo_json or {}).get("per_view_errors", []) or []
    per_view_sorted_proj = sorted(
        per_view,
        key=lambda r: float(r.get("projector_reproj_error_px", float("inf"))),
        reverse=True,
    )
    top_projector_views = per_view_sorted_proj[:5]
    top_stereo_views = sorted(
        per_view,
        key=lambda r: float(r.get("camera_reproj_error_px", 0.0) + r.get("projector_reproj_error_px", 0.0)),
        reverse=True,
    )[:5]

    subset_info: dict[str, Any] = {"status": "unknown"}
    if stereo_json and coverage.session_id:
        sess_results = Path("data/calibration/projector/sessions") / coverage.session_id / "results"
        best_match = None
        for name in ["prune_report.json", "prune_refined_report.json"]:
            p = sess_results / name
            if p.exists():
                try:
                    j = _load_json(p)
                    view_key = "best_view_ids" if "best_view_ids" in j else ("final_view_ids" if "final_view_ids" in j else None)
                    if view_key:
                        best_match = {
                            "report": name,
                            "view_ids": sorted([str(x) for x in j.get(view_key, [])]),
                            "final_rms": j.get("final_rms"),
                            "initial_rms": j.get("initial_rms"),
                        }
                        break
                except Exception:
                    continue
        current_ids = sorted([str(v.get("view_id")) for v in per_view if str(v.get("view_id", "")).strip()])
        if best_match:
            subset_info = {
                "status": "compared_with_prune_report",
                "current_view_ids": current_ids,
                "best_view_ids_from_report": best_match["view_ids"],
                "matches_best_subset": current_ids == best_match["view_ids"],
                "report_file": best_match["report"],
                "report_initial_rms": best_match["initial_rms"],
                "report_final_rms": best_match["final_rms"],
            }
        else:
            subset_info = {
                "status": "no_prune_report_found",
                "current_view_ids": current_ids,
            }

    _plot_coverage_grid(stereo_json, coverage, plots_dir / "calibration_projector_coverage_grid.png")
    if coverage.row_coverage is not None:
        _plot_curve(np.asarray(coverage.row_coverage, dtype=np.float64), "Projector coverage by Y-bin", "coverage ratio", plots_dir / "coverage_rows.png")
    if coverage.col_coverage is not None:
        _plot_curve(np.asarray(coverage.col_coverage, dtype=np.float64), "Projector coverage by X-bin", "coverage ratio", plots_dir / "coverage_cols.png")

    report["inputs"] = {
        "reconstruction_meta": str(required_paths["reconstruction_meta"]),
        "uv_meta": str(uv_meta_path) if uv_meta_path.exists() else None,
        "stereo_latest": str(stereo_path) if stereo_path else None,
        "depth_map": str(depth) if depth else None,
        "b_map": str(b_map_path) if b_map_path else None,
        "prev_summary": str(summary_path) if summary_path.exists() else None,
    }
    report["uv_recon"] = {
        "uv_valid_ratio": float(np.count_nonzero(uv_valid) / uv_valid.size),
        "recon_valid_ratio": float(np.count_nonzero(recon_valid) / uv_valid.size),
        "uv_valid_px": int(np.count_nonzero(uv_valid)),
        "recon_valid_px": int(np.count_nonzero(recon_valid)),
        "rejected_px": int(np.count_nonzero(rejected)),
        "meta_rejected_total_px": int(recon_meta.get("rejected_total_px", -1)),
        "meta_rejected_by_cam_reproj_px": int(recon_meta.get("rejected_by_cam_reproj_px", -1)),
        "meta_rejected_by_proj_reproj_px": int(recon_meta.get("rejected_by_proj_reproj_px", -1)),
        "meta_rejected_by_depth_range_px": int(recon_meta.get("rejected_by_depth_range_px", -1)),
        "max_reproj_err_px": gate if math.isfinite(gate) else None,
    }
    report["error_stats"] = {
        "cam_uv": _stats(err_cam[err_valid]),
        "cam_recon": _stats(err_cam[recon_valid & np.isfinite(err_cam)]),
        "cam_rejected": _stats(err_cam[rejected & np.isfinite(err_cam)]),
        "proj_uv": _stats(err_proj[uv_valid & np.isfinite(err_proj)]),
    }
    report["stereo_vs_gate"] = stereo_compare
    report["error_bands"] = err_bands
    report["drift_tests"] = drift_tests
    report["calibration_side"] = {
        "session_id": coverage.session_id,
        "coverage": {
            "available": coverage.available,
            "coverage_ratio_16x16": coverage.coverage_ratio,
            "row_coverage_min": float(np.min(coverage.row_coverage)) if coverage.row_coverage is not None else None,
            "row_coverage_mean": float(np.mean(coverage.row_coverage)) if coverage.row_coverage is not None else None,
            "col_coverage_min": float(np.min(coverage.col_coverage)) if coverage.col_coverage is not None else None,
            "col_coverage_mean": float(np.mean(coverage.col_coverage)) if coverage.col_coverage is not None else None,
            "used_views_missing_correspondences": coverage.used_views_missing_correspondences,
            "notes": coverage.notes,
        },
        "worst_projector_views": top_projector_views,
        "worst_combined_views": top_stereo_views,
        "subset_check": subset_info,
    }
    report["previous_diagnostics_summary"] = prev_summary

    # Conclusion logic.
    reasons: list[str] = []
    strongest_axis = drift_tests["banding"]["strongest_axis"]
    row_std = float(drift_tests["banding"]["row_reject_std"])
    col_std = float(drift_tests["banding"]["col_reject_std"])
    depth_corr_abs = abs(drift_tests["depth_bias"]["corr_err_depth"]) if drift_tests["depth_bias"]["corr_err_depth"] is not None else 0.0
    uv_corr_abs = max(
        abs(drift_tests["uv_bias"]["corr_err_u"] or 0.0),
        abs(drift_tests["uv_bias"]["corr_err_v"] or 0.0),
        abs(drift_tests["uv_bias"]["corr_err_edge_distance"] or 0.0),
    )
    cam_corr_abs = max(
        abs(drift_tests["camera_pos_bias"]["corr_err_cam_x"] or 0.0),
        abs(drift_tests["camera_pos_bias"]["corr_err_cam_y"] or 0.0),
        abs(drift_tests["camera_pos_bias"]["corr_err_cam_radius"] or 0.0),
    )

    primary = "mixed"
    if row_std > 0.20 or col_std > 0.20:
        primary = "structured_row_col_drift"
        reasons.append(
            f"Strong banding in rejection map (row_std={row_std:.3f}, col_std={col_std:.3f}, strongest_axis={strongest_axis})."
        )
    if uv_corr_abs >= 0.30:
        reasons.append(f"Error has projector-coordinate dependence (max |corr(err,u/v/edge)|={uv_corr_abs:.3f}).")
    if cam_corr_abs >= 0.30:
        reasons.append(f"Error has camera-position dependence (max |corr(err,x/y/radius)|={cam_corr_abs:.3f}).")
    if depth_corr_abs >= 0.30:
        reasons.append(f"Depth dependence is material (|corr(err,depth)|={depth_corr_abs:.3f}).")
    if float(recon_meta.get("rejected_by_cam_reproj_px", 0)) > 0 and float(recon_meta.get("rejected_by_proj_reproj_px", 0)) == 0:
        reasons.append("Rejection is camera-reprojection dominated; projector reprojection is not the active rejector.")

    # Coverage-informed note.
    if coverage.available and coverage.coverage_ratio is not None:
        cov_ratio = float(coverage.coverage_ratio)
        if cov_ratio < 0.35:
            reasons.append(f"Calibration projector-corner coverage is sparse (16x16 coverage={cov_ratio:.3f}).")
        if strongest_axis == "rows" and coverage.row_coverage is not None:
            reasons.append(
                f"Strongest drift axis is rows; projector Y-bin coverage min/mean={float(np.min(coverage.row_coverage)):.3f}/{float(np.mean(coverage.row_coverage)):.3f}."
            )
        if strongest_axis == "cols" and coverage.col_coverage is not None:
            reasons.append(
                f"Strongest drift axis is cols; projector X-bin coverage min/mean={float(np.min(coverage.col_coverage)):.3f}/{float(np.mean(coverage.col_coverage)):.3f}."
            )

    report["conclusion"] = {
        "primary_hypothesis": primary,
        "evidence": reasons,
        "next_capture_strategy": [
            "Capture more projector calibration views targeting under-covered projector bins (especially along the strongest drift axis).",
            "Include board poses that sweep edges/corners and multiple tilts/depths while keeping full-corner validity high.",
            "Prioritize views that lower worst per-view projector reprojection error before recalibration.",
        ],
    }

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2))

    lines: list[str] = []
    lines.append(f"Run: {run_dir.name}")
    lines.append(f"Primary hypothesis: {primary}")
    lines.append("")
    lines.append("Evidence:")
    if reasons:
        for r in reasons:
            lines.append(f"- {r}")
    else:
        lines.append("- No single dominant signal; inspect report.json details.")
    lines.append("")
    lines.append("Next calibration capture strategy:")
    for s in report["conclusion"]["next_capture_strategy"]:
        lines.append(f"- {s}")
    (out_dir / "conclusions.txt").write_text("\n".join(lines) + "\n")

    print("=== Reprojection drift diagnostics ===")
    print(f"run: {run_dir}")
    print(f"out: {out_dir}")
    print(
        f"uv_valid_ratio={report['uv_recon']['uv_valid_ratio']:.4f} "
        f"recon_valid_ratio={report['uv_recon']['recon_valid_ratio']:.4f}"
    )
    print(
        "rejections(meta): "
        f"cam={report['uv_recon']['meta_rejected_by_cam_reproj_px']} "
        f"proj={report['uv_recon']['meta_rejected_by_proj_reproj_px']} "
        f"depth={report['uv_recon']['meta_rejected_by_depth_range_px']}"
    )
    print(f"primary_hypothesis={primary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

