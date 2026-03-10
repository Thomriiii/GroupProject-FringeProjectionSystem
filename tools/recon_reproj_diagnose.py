#!/usr/bin/env python3
"""Offline reconstruction rejection diagnostics (step 1).

Diagnostics-only script. It does not modify pipeline logic.

Usage:
  PYTHONPATH=src .venv/bin/python tools/recon_reproj_diagnose.py --run data/runs/20260226_103308_uv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return _json_safe(v.tolist())
    return v


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _maybe_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except Exception:
        return None


def _resolve_run_dir(arg: str) -> Path:
    p = Path(arg)
    if p.exists():
        return p
    p2 = Path("data/runs") / arg
    if p2.exists():
        return p2
    raise SystemExit(f"Run not found: {arg}")


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _load_npy_required(paths: list[Path], label: str) -> tuple[np.ndarray, Path]:
    p = _first_existing(paths)
    if p is None:
        tried = "\n- ".join(str(x) for x in paths)
        raise SystemExit(f"Missing required {label}; tried:\n- {tried}")
    return np.load(p), p


def _load_image_rgb(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=2)
    if arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _load_background(run_dir: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    candidates: list[Path] = [
        run_dir / "projector_uv" / "uv_overlay.png",
        run_dir / "reconstruction" / "depth_debug_autoscale.png",
    ]
    for base in [run_dir / "vertical" / "captures", run_dir / "captures"]:
        if base.exists():
            for fd in sorted(base.glob("f_*")):
                candidates.extend(sorted(fd.glob("step_*.png")))
            candidates.extend(sorted(base.glob("frame_*.png")))

    for p in candidates:
        img = _load_image_rgb(p)
        if img is not None and img.shape[:2] == (h, w):
            return img.astype(np.uint8)

    return np.zeros((h, w, 3), dtype=np.uint8)


def _stats(vals: np.ndarray, ps=(50, 90, 95, 99)) -> dict[str, float | None]:
    a = vals[np.isfinite(vals)]
    out: dict[str, float | None] = {}
    if a.size == 0:
        for p in ps:
            out[f"p{p}"] = None
        out["max"] = None
        out["mean"] = None
        return out
    for p in ps:
        out[f"p{p}"] = float(np.percentile(a, p))
    out["max"] = float(np.max(a))
    out["mean"] = float(np.mean(a))
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or b.size < 3:
        return None
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 1e-12 or sb <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _map_from_vec(shape_hw: tuple[int, int], ys: np.ndarray, xs: np.ndarray, vals: np.ndarray) -> np.ndarray:
    out = np.full(shape_hw, np.nan, dtype=np.float32)
    out[ys, xs] = vals.astype(np.float32)
    return out


def _save_heatmap(data: np.ndarray, title: str, out_path: Path, cmap: str = "viridis") -> None:
    plt.figure(figsize=(8, 5))
    finite = data[np.isfinite(data)]
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
        if hi <= lo:
            hi = lo + 1e-6
    else:
        lo, hi = 0.0, 1.0
    plt.imshow(data, cmap=cmap, vmin=lo, vmax=hi)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _normalize_to_u8(arr: np.ndarray) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.uint8)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out
    vals = arr[finite].astype(np.float64)
    lo, hi = np.percentile(vals, [1, 99])
    if not np.isfinite(lo):
        lo = float(np.min(vals))
    if not np.isfinite(hi):
        hi = float(np.max(vals))
    if hi <= lo:
        hi = lo + 1e-6
    scaled = np.clip((arr[finite] - lo) / (hi - lo), 0.0, 1.0)
    out[finite] = np.rint(scaled * 255.0).astype(np.uint8)
    return out


def _save_binary(mask: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.imshow(mask.astype(np.uint8), cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _overlay_red(base_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    if np.any(mask):
        out[mask, 0] = np.clip(0.35 * out[mask, 0] + 165.0, 0, 255)
        out[mask, 1] = np.clip(0.35 * out[mask, 1], 0, 255)
        out[mask, 2] = np.clip(0.35 * out[mask, 2], 0, 255)
    return out.astype(np.uint8)


def _rate_by_axis(mask_uv: np.ndarray, rejected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uv_r = mask_uv.sum(axis=1).astype(np.float64)
    uv_c = mask_uv.sum(axis=0).astype(np.float64)
    rej_r = rejected.sum(axis=1).astype(np.float64)
    rej_c = rejected.sum(axis=0).astype(np.float64)

    rr = np.full(mask_uv.shape[0], np.nan, dtype=np.float64)
    cc = np.full(mask_uv.shape[1], np.nan, dtype=np.float64)
    good_r = uv_r > 0
    good_c = uv_c > 0
    rr[good_r] = rej_r[good_r] / uv_r[good_r]
    cc[good_c] = rej_c[good_c] / uv_c[good_c]
    return rr, cc


def _masked_mean_profile(values: np.ndarray, mask: np.ndarray, axis: str) -> np.ndarray:
    if axis == "rows":
        out = np.full(values.shape[0], np.nan, dtype=np.float64)
        for i in range(values.shape[0]):
            m = mask[i] & np.isfinite(values[i])
            if np.any(m):
                out[i] = float(np.mean(values[i][m]))
        return out
    if axis == "cols":
        out = np.full(values.shape[1], np.nan, dtype=np.float64)
        for j in range(values.shape[1]):
            m = mask[:, j] & np.isfinite(values[:, j])
            if np.any(m):
                out[j] = float(np.mean(values[:, j][m]))
        return out
    raise ValueError("axis must be 'rows' or 'cols'")


def _contiguous_bands(rate: np.ndarray) -> list[dict[str, Any]]:
    finite = np.isfinite(rate)
    if not np.any(finite):
        return []
    vals = rate[finite]
    thr = float(np.nanmean(vals) + np.nanstd(vals))
    hi = finite & (rate > thr)
    out: list[dict[str, Any]] = []
    i = 0
    n = hi.size
    while i < n:
        if not hi[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and hi[j + 1]:
            j += 1
        out.append({
            "start": int(i),
            "end": int(j),
            "length": int(j - i + 1),
            "mean_rate": float(np.nanmean(rate[i : j + 1])),
        })
        i = j + 1
    return out


def _fft_peak_info(rate: np.ndarray, min_bin: int = 2) -> dict[str, Any]:
    finite = np.isfinite(rate)
    if not np.any(finite):
        return {
            "n": int(rate.size),
            "raw_peak_bin": None,
            "raw_peak_freq_cpp": None,
            "raw_peak_period_px": None,
            "raw_is_near_dc": None,
            "peak_bin_excluding_low": None,
            "peak_freq_cpp_excluding_low": None,
            "peak_period_px_excluding_low": None,
            "freqs": [],
            "power": [],
        }

    x = rate.astype(np.float64).copy()
    x[~finite] = float(np.nanmean(x[finite]))
    x -= np.mean(x)
    n = x.size

    spec = np.fft.rfft(x)
    power = (np.abs(spec) ** 2).astype(np.float64)
    freqs = np.fft.rfftfreq(n, d=1.0)
    if power.size:
        power[0] = 0.0

    raw_bin = int(np.argmax(power)) if power.size else None
    raw_freq = float(freqs[raw_bin]) if raw_bin is not None and raw_bin < freqs.size else None
    raw_period = float(1.0 / raw_freq) if raw_freq is not None and raw_freq > 1e-12 else None
    raw_is_near_dc = bool(raw_bin is not None and raw_bin <= max(1, min_bin - 1))

    p2 = power.copy()
    if p2.size:
        p2[:min_bin] = 0.0
    pk_bin = int(np.argmax(p2)) if p2.size else None
    pk_freq = float(freqs[pk_bin]) if pk_bin is not None and pk_bin < freqs.size else None
    pk_period = float(1.0 / pk_freq) if pk_freq is not None and pk_freq > 1e-12 else None

    return {
        "n": int(n),
        "raw_peak_bin": raw_bin,
        "raw_peak_freq_cpp": raw_freq,
        "raw_peak_period_px": raw_period,
        "raw_is_near_dc": raw_is_near_dc,
        "peak_bin_excluding_low": pk_bin,
        "peak_freq_cpp_excluding_low": pk_freq,
        "peak_period_px_excluding_low": pk_period,
        "freqs": freqs.tolist(),
        "power": power.tolist(),
    }


def _plot_rate(rate: np.ndarray, axis_name: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(rate.size), rate, linewidth=1.2)
    plt.ylim(-0.02, 1.02)
    plt.xlabel(f"{axis_name} index")
    plt.ylabel("rejection rate")
    plt.title(f"Rejection Rate by {axis_name.capitalize()}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_fft(fft_info: dict[str, Any], axis_name: str, out_path: Path) -> None:
    freqs = np.asarray(fft_info.get("freqs", []), dtype=np.float64)
    power = np.asarray(fft_info.get("power", []), dtype=np.float64)
    plt.figure(figsize=(10, 4))
    if freqs.size and power.size:
        plt.plot(freqs, power, linewidth=1.2)
    plt.xlabel("cycles / pixel")
    plt.ylabel("power")
    plt.title(f"FFT of {axis_name} rejection-rate")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _triangulate_all(
    u: np.ndarray,
    v: np.ndarray,
    mask_uv: np.ndarray,
    Kc: np.ndarray,
    distc: np.ndarray,
    Kp: np.ndarray,
    distp: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
) -> dict[str, Any]:
    valid = mask_uv & np.isfinite(u) & np.isfinite(v)
    ys, xs = np.where(valid)
    if ys.size == 0:
        raise SystemExit("No UV valid pixels for triangulation")

    cam_px = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    proj_px = np.stack([u[ys, xs].astype(np.float64), v[ys, xs].astype(np.float64)], axis=1)

    cam_pts = cam_px.reshape(-1, 1, 2)
    proj_pts = proj_px.reshape(-1, 1, 2)

    cam_norm = cv2.undistortPoints(cam_pts, Kc, distc).reshape(-1, 2)
    proj_norm = cv2.undistortPoints(proj_pts, Kp, distp).reshape(-1, 2)

    P1 = np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)])
    P2 = np.hstack([R.astype(np.float64), T.astype(np.float64)])

    Xh = cv2.triangulatePoints(P1, P2, cam_norm.T, proj_norm.T)
    X = (Xh[:3] / Xh[3]).T.astype(np.float64)

    return {
        "valid": valid,
        "ys": ys,
        "xs": xs,
        "cam_px": cam_px,
        "proj_px": proj_px,
        "cam_norm": cam_norm,
        "X": X,
    }


def _variant_maps(
    shape_hw: tuple[int, int],
    tri: dict[str, Any],
    K_st: np.ndarray,
    dist_st: np.ndarray,
    K_ci: np.ndarray,
    dist_ci: np.ndarray,
    rect: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    X = tri["X"]
    ys = tri["ys"]
    xs = tri["xs"]
    cam_px = tri["cam_px"]
    cam_norm = tri["cam_norm"]

    n = X.shape[0]
    finite_xyz = np.isfinite(X).all(axis=1)

    variants: dict[str, np.ndarray] = {}

    def proj_err(K: np.ndarray, dist: np.ndarray, name: str) -> None:
        err = np.full(n, np.nan, dtype=np.float64)
        if np.any(finite_xyz):
            rep, _ = cv2.projectPoints(
                X[finite_xyz].reshape(-1, 1, 3),
                np.zeros((3, 1), dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                K,
                dist,
            )
            err[finite_xyz] = np.linalg.norm(rep.reshape(-1, 2) - cam_px[finite_xyz], axis=1)
        variants[name] = _map_from_vec(shape_hw, ys, xs, err)

    # A: Distorted pixel with stereo camera model.
    proj_err(K_st, dist_st, "A_distorted_pixel_stereo")
    # Extra check: distorted pixel with camera-intrinsics model used by reconstruction code.
    proj_err(K_ci, dist_ci, "A_distorted_pixel_camera_intr")

    # B: Undistorted normalized.
    err_b = np.full(n, np.nan, dtype=np.float64)
    okz = finite_xyz & (np.abs(X[:, 2]) > 1e-12)
    if np.any(okz):
        pred_norm = X[okz, :2] / X[okz, 2:3]
        err_b[okz] = np.linalg.norm(pred_norm - cam_norm[okz], axis=1)
    variants["B_undistorted_normalized"] = _map_from_vec(shape_hw, ys, xs, err_b)

    if rect is not None:
        R1 = rect["R1"]
        P1 = rect["P1"]
        Krect = P1[:, :3]

        cam_rect_obs = cv2.undistortPoints(
            cam_px.reshape(-1, 1, 2),
            K_st,
            dist_st,
            R=R1,
            P=Krect,
        ).reshape(-1, 2)

        # C: Proper rectified reproj vs rectified obs.
        err_c = np.full(n, np.nan, dtype=np.float64)
        # D1: rectified reproj vs unrectified obs.
        err_d1 = np.full(n, np.nan, dtype=np.float64)
        # D2: distorted reproj vs undistorted-pixel obs.
        err_d2 = np.full(n, np.nan, dtype=np.float64)
        # D3: use P1 as if K in unrectified camera space.
        err_d3 = np.full(n, np.nan, dtype=np.float64)

        if np.any(okz):
            Xr = (R1 @ X[okz].T).T
            okz2 = np.abs(Xr[:, 2]) > 1e-12
            if np.any(okz2):
                idx = np.where(okz)[0][okz2]
                xr = Xr[okz2]
                uv = (Krect @ xr.T).T
                uv = uv[:, :2] / uv[:, 2:3]
                err_c[idx] = np.linalg.norm(uv - cam_rect_obs[idx], axis=1)
                err_d1[idx] = np.linalg.norm(uv - cam_px[idx], axis=1)

                uv_bad = (Krect @ X[idx].T).T
                uv_bad = uv_bad[:, :2] / uv_bad[:, 2:3]
                err_d3[idx] = np.linalg.norm(uv_bad - cam_px[idx], axis=1)

            rep_dist, _ = cv2.projectPoints(
                X[okz].reshape(-1, 1, 3),
                np.zeros((3, 1), dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                K_st,
                dist_st,
            )
            und_px = cv2.undistortPoints(
                cam_px[okz].reshape(-1, 1, 2),
                K_st,
                dist_st,
                P=K_st,
            ).reshape(-1, 2)
            err_d2[okz] = np.linalg.norm(rep_dist.reshape(-1, 2) - und_px, axis=1)

        variants["C_rectified_pixel"] = _map_from_vec(shape_hw, ys, xs, err_c)
        variants["D_mixed_rectified_vs_unrectified_obs"] = _map_from_vec(shape_hw, ys, xs, err_d1)
        variants["D_mixed_distorted_vs_undistorted_obs"] = _map_from_vec(shape_hw, ys, xs, err_d2)
        variants["D_mixed_P1_as_K_unrectified"] = _map_from_vec(shape_hw, ys, xs, err_d3)

    return variants


def _match_scores(stored: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    m = mask & np.isfinite(stored) & np.isfinite(candidate)
    if not np.any(m):
        return {"corr": None, "abs_diff_p50": None, "abs_diff_p90": None, "abs_diff_p95": None, "n": 0}
    a = stored[m].astype(np.float64)
    b = candidate[m].astype(np.float64)
    d = np.abs(a - b)
    return {
        "corr": _corr(a, b),
        "abs_diff_p50": float(np.percentile(d, 50)),
        "abs_diff_p90": float(np.percentile(d, 90)),
        "abs_diff_p95": float(np.percentile(d, 95)),
        "n": int(a.size),
    }


def _best_variant(scores: dict[str, dict[str, float | int | None]]) -> str | None:
    ranking: list[tuple[float, float, str]] = []
    for name, s in scores.items():
        c = s.get("corr")
        p95 = s.get("abs_diff_p95")
        if c is None or p95 is None:
            continue
        ranking.append((float(c), -float(p95), name))
    if not ranking:
        return None
    ranking.sort(reverse=True)
    return ranking[0][2]


def _load_b_map(run_dir: Path, shape_hw: tuple[int, int]) -> tuple[np.ndarray | None, str | None]:
    candidates = [
        run_dir / "vertical" / "phase" / "f_016" / "B.npy",
        run_dir / "horizontal" / "phase" / "f_016" / "B.npy",
        run_dir / "phase" / "f_016" / "B.npy",
        run_dir / "phase" / "B.npy",
    ]
    for p in candidates:
        if p.exists():
            b = np.load(p).astype(np.float32)
            if b.shape == shape_hw:
                return b, str(p)
    return None, None


def _write_report_md(out_dir: Path, report: dict[str, Any]) -> None:
    ev = report["evidence"]
    hypotheses = report["root_cause_shortlist"]

    lines: list[str] = []
    lines.append("# Reconstruction Rejection Diagnostic (Step 1)")
    lines.append("")
    lines.append(f"Run: `{report['run']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- uv_valid_ratio: `{ev['uv_valid_ratio']:.4f}`")
    lines.append(f"- recon_valid_ratio: `{ev['recon_valid_ratio']:.4f}`")
    lines.append(f"- rejected_by_cam_reproj_px: `{ev['rejected_by_cam_reproj_px']}`")
    lines.append(f"- rejected_by_proj_reproj_px: `{ev['rejected_by_proj_reproj_px']}`")
    lines.append(f"- rejected_by_depth_range_px: `{ev['rejected_by_depth_range_px']}`")
    lines.append(f"- max_reproj_err_px: `{ev['max_reproj_err_px']}`")
    lines.append("")
    lines.append("## Reprojection-Space Check")
    lines.append(f"- Best-matching variant: `{ev['best_variant']}`")
    lines.append(f"- Best variant corr: `{ev['best_variant_corr']}`")
    lines.append(f"- Best variant abs diff p95: `{ev['best_variant_abs_diff_p95']}`")
    lines.append("")
    lines.append("## Banding")
    lines.append(f"- strongest_axis: `{ev['strongest_axis']}`")
    lines.append(f"- strongest_axis_peak_period_px_excluding_low: `{ev['strongest_axis_peak_period_px_excluding_low']}`")
    lines.append(f"- strongest_axis_raw_peak_near_dc: `{ev['strongest_axis_raw_peak_near_dc']}`")
    lines.append("")
    lines.append("## Root-Cause Shortlist (ranked)")
    for i, h in enumerate(hypotheses, start=1):
        lines.append(f"{i}. **{h['title']}**")
        for b in h.get("evidence", []):
            lines.append(f"- {b}")
    lines.append("")
    lines.append("## What This Is NOT")
    for w in report.get("what_this_is_not", []):
        lines.append(f"- {w}")
    lines.append("")
    lines.append("## Diagnostic Follow-ups (no pipeline edits)")
    for n in report.get("next_steps_diagnostics_only", []):
        lines.append(f"- {n}")

    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run id or run directory path")
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run)
    out_dir = run_dir / "diagnostics_step1"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    recon_dir = run_dir / "reconstruction"
    recon_meta_path = recon_dir / "reconstruction_meta.json"
    if not recon_meta_path.exists():
        raise SystemExit(f"Missing reconstruction meta: {recon_meta_path}")
    recon_meta = _load_json(recon_meta_path)

    mask_uv, mask_uv_path = _load_npy_required(
        [recon_dir / "masks" / "mask_uv.npy", run_dir / "projector_uv" / "mask_uv.npy"],
        "mask_uv",
    )
    mask_recon, mask_recon_path = _load_npy_required(
        [recon_dir / "masks" / "mask_recon.npy", recon_dir / "mask_recon.npy"],
        "mask_recon",
    )
    err_cam, err_cam_path = _load_npy_required(
        [recon_dir / "reproj_err_cam.npy"],
        "reproj_err_cam",
    )
    err_proj, err_proj_path = _load_npy_required(
        [recon_dir / "reproj_err_proj.npy"],
        "reproj_err_proj",
    )
    u, u_path = _load_npy_required([run_dir / "projector_uv" / "u.npy"], "u")
    v, v_path = _load_npy_required([run_dir / "projector_uv" / "v.npy"], "v")

    depth = None
    depth_path = _first_existing([recon_dir / "depth.npy", recon_dir / "depth_map.npy"])
    if depth_path is not None:
        depth = np.load(depth_path)

    uv_meta = _maybe_json(run_dir / "projector_uv" / "uv_meta.json") or {}
    run_meta = _maybe_json(run_dir / "meta.json") or {}
    v_phase_meta = _maybe_json(run_dir / "vertical" / "phase" / "f_016" / "phase_meta.json") or {}
    h_phase_meta = _maybe_json(run_dir / "horizontal" / "phase" / "f_016" / "phase_meta.json") or {}
    v_unwrap_meta = _maybe_json(run_dir / "vertical" / "unwrap" / "unwrap_meta.json") or {}
    h_unwrap_meta = _maybe_json(run_dir / "horizontal" / "unwrap" / "unwrap_meta.json") or {}

    shape_table: dict[str, Any] = {
        "u": {"shape": list(u.shape), "dtype": str(u.dtype), "path": str(u_path)},
        "v": {"shape": list(v.shape), "dtype": str(v.dtype), "path": str(v_path)},
        "mask_uv": {"shape": list(mask_uv.shape), "dtype": str(mask_uv.dtype), "path": str(mask_uv_path)},
        "mask_recon": {"shape": list(mask_recon.shape), "dtype": str(mask_recon.dtype), "path": str(mask_recon_path)},
        "reproj_err_cam": {"shape": list(err_cam.shape), "dtype": str(err_cam.dtype), "path": str(err_cam_path)},
        "reproj_err_proj": {"shape": list(err_proj.shape), "dtype": str(err_proj.dtype), "path": str(err_proj_path)},
    }
    if depth is not None and depth_path is not None:
        shape_table["depth"] = {"shape": list(depth.shape), "dtype": str(depth.dtype), "path": str(depth_path)}

    hw = mask_uv.shape
    align_ok = all(tuple(x["shape"]) == tuple(hw) for x in shape_table.values())

    # Calibration sources (prefer explicit paths if present).
    repo_root = Path(__file__).resolve().parents[1]
    stereo_path_from_meta = recon_meta.get("projector_stereo") or recon_meta.get("stereo_path")
    cam_intr_path_from_meta = recon_meta.get("camera_intrinsics") or recon_meta.get("camera_intrinsics_path")

    stereo_path = None
    stereo_path_source = None
    if stereo_path_from_meta:
        sp = Path(str(stereo_path_from_meta))
        if sp.exists():
            stereo_path = sp
            stereo_path_source = "reconstruction_meta"
    if stereo_path is None:
        sp = repo_root / "data" / "calibration" / "projector" / "stereo_latest.json"
        if sp.exists():
            stereo_path = sp
            stereo_path_source = "fallback_default"
    if stereo_path is None:
        raise SystemExit("Could not find stereo calibration JSON")

    cam_intr_path = None
    cam_intr_path_source = None
    if cam_intr_path_from_meta:
        cp = Path(str(cam_intr_path_from_meta))
        if cp.exists():
            cam_intr_path = cp
            cam_intr_path_source = "reconstruction_meta"
    if cam_intr_path is None:
        cp = repo_root / "data" / "calibration" / "camera" / "intrinsics_latest.json"
        if cp.exists():
            cam_intr_path = cp
            cam_intr_path_source = "fallback_default"
    if cam_intr_path is None:
        raise SystemExit("Could not find camera intrinsics JSON")

    stereo = _load_json(stereo_path)
    cam_intr = _load_json(cam_intr_path)

    K_st = np.asarray(stereo.get("camera_matrix"), dtype=np.float64)
    dist_st = np.asarray(stereo.get("camera_dist_coeffs"), dtype=np.float64).reshape(-1, 1)
    K_ci = np.asarray(cam_intr.get("camera_matrix"), dtype=np.float64)
    dist_ci = np.asarray(cam_intr.get("dist_coeffs"), dtype=np.float64).reshape(-1, 1)
    Kp = np.asarray(stereo.get("projector_matrix"), dtype=np.float64)
    distp = np.asarray(stereo.get("projector_dist_coeffs"), dtype=np.float64).reshape(-1, 1)
    R = np.asarray(stereo.get("R"), dtype=np.float64)
    T = np.asarray(stereo.get("T"), dtype=np.float64).reshape(3, 1)

    rect = None
    rect_s = stereo.get("rectification") or {}
    if isinstance(rect_s, dict) and "R1" in rect_s and "P1" in rect_s:
        R1 = np.asarray(rect_s["R1"], dtype=np.float64)
        P1 = np.asarray(rect_s["P1"], dtype=np.float64)
        if R1.shape == (3, 3) and P1.shape == (3, 4):
            rect = {"R1": R1, "P1": P1}

    # Consistency table.
    run_res = (run_meta.get("params") or {}).get("resolution")
    uv_proj = [uv_meta.get("projector_width"), uv_meta.get("projector_height")]
    stereo_proj = [
        (stereo.get("projector") or {}).get("width"),
        (stereo.get("projector") or {}).get("height"),
    ]
    stereo_img_size = stereo.get("image_size") if "image_size" in stereo else "NOT FOUND"
    cam_intr_img_size = cam_intr.get("image_size") if "image_size" in cam_intr else "NOT FOUND"

    consistency = {
        "run_resolution": run_res,
        "mask_hw": [int(hw[1]), int(hw[0])],
        "uv_projector_size": uv_proj,
        "stereo_projector_size": stereo_proj,
        "stereo_image_size": stereo_img_size,
        "camera_intrinsics_image_size": cam_intr_img_size,
        "camera_matrix_stereo_cx_cy": [float(K_st[0, 2]), float(K_st[1, 2])] if K_st.shape == (3, 3) else "NOT FOUND",
        "camera_matrix_intrinsics_cx_cy": [float(K_ci[0, 2]), float(K_ci[1, 2])] if K_ci.shape == (3, 3) else "NOT FOUND",
        "camera_matrix_delta_abs": (
            np.abs(K_st - K_ci).tolist() if K_st.shape == (3, 3) and K_ci.shape == (3, 3) else "NOT FOUND"
        ),
        "dist_coeffs_stereo": dist_st.reshape(-1).tolist(),
        "dist_coeffs_intrinsics": dist_ci.reshape(-1).tolist(),
        "calibration_paths": {
            "stereo_path": str(stereo_path),
            "stereo_path_source": stereo_path_source,
            "camera_intrinsics_path": str(cam_intr_path),
            "camera_intrinsics_path_source": cam_intr_path_source,
        },
    }

    # Reproduce rejection stats.
    valid_uv = mask_uv.astype(bool)
    rejected = valid_uv & (~mask_recon.astype(bool))
    accepted = mask_recon.astype(bool)
    thr = float(recon_meta.get("max_reproj_err_px", np.nan))

    cam_acc = err_cam[accepted & np.isfinite(err_cam)]
    cam_rej = err_cam[rejected & np.isfinite(err_cam)]

    rej_cam_by_map = int(np.count_nonzero(rejected & np.isfinite(err_cam) & (err_cam > thr))) if np.isfinite(thr) else None
    rej_proj_by_map = int(np.count_nonzero(rejected & np.isfinite(err_proj) & (err_proj > thr))) if np.isfinite(thr) else None

    # Recompute reprojection variants.
    tri = _triangulate_all(u.astype(np.float64), v.astype(np.float64), valid_uv, K_ci, dist_ci, Kp, distp, R, T)
    variant_maps = _variant_maps(hw, tri, K_st, dist_st, K_ci, dist_ci, rect)

    # Match each variant to stored map.
    variant_scores = {name: _match_scores(err_cam, vm, valid_uv) for name, vm in variant_maps.items()}
    best_variant = _best_variant(variant_scores)

    # Band analysis.
    row_rate, col_rate = _rate_by_axis(valid_uv, rejected)
    row_bands = _contiguous_bands(row_rate)
    col_bands = _contiguous_bands(col_rate)
    row_fft = _fft_peak_info(row_rate, min_bin=2)
    col_fft = _fft_peak_info(col_rate, min_bin=2)

    strongest_axis = "rows" if float(np.nanstd(row_rate)) >= float(np.nanstd(col_rate)) else "cols"
    strongest_fft = row_fft if strongest_axis == "rows" else col_fft

    # UV profile banding diagnostics.
    row_u_mean = _masked_mean_profile(u.astype(np.float64), valid_uv, axis="rows")
    row_v_mean = _masked_mean_profile(v.astype(np.float64), valid_uv, axis="rows")
    col_u_mean = _masked_mean_profile(u.astype(np.float64), valid_uv, axis="cols")
    col_v_mean = _masked_mean_profile(v.astype(np.float64), valid_uv, axis="cols")

    m_row_u = np.isfinite(row_rate) & np.isfinite(row_u_mean)
    m_row_v = np.isfinite(row_rate) & np.isfinite(row_v_mean)
    m_col_u = np.isfinite(col_rate) & np.isfinite(col_u_mean)
    m_col_v = np.isfinite(col_rate) & np.isfinite(col_v_mean)

    corr_row_rej_u_mean = _corr(row_rate[m_row_u], row_u_mean[m_row_u]) if np.any(m_row_u) else None
    corr_row_rej_v_mean = _corr(row_rate[m_row_v], row_v_mean[m_row_v]) if np.any(m_row_v) else None
    corr_col_rej_u_mean = _corr(col_rate[m_col_u], col_u_mean[m_col_u]) if np.any(m_col_u) else None
    corr_col_rej_v_mean = _corr(col_rate[m_col_v], col_v_mean[m_col_v]) if np.any(m_col_v) else None

    row_u_fft = _fft_peak_info(row_u_mean, min_bin=2)
    row_v_fft = _fft_peak_info(row_v_mean, min_bin=2)
    col_u_fft = _fft_peak_info(col_u_mean, min_bin=2)
    col_v_fft = _fft_peak_info(col_v_mean, min_bin=2)

    # Gradient and residual correlations.
    du_y, du_x = np.gradient(u.astype(np.float64))
    dv_y, dv_x = np.gradient(v.astype(np.float64))
    u_grad = np.hypot(du_x, du_y).astype(np.float32)
    v_grad = np.hypot(dv_x, dv_y).astype(np.float32)

    m_corr = valid_uv & np.isfinite(u_grad) & np.isfinite(v_grad)
    reject_f = rejected[m_corr].astype(np.float64)

    corr_reject_u_grad = _corr(reject_f, u_grad[m_corr].astype(np.float64)) if np.any(m_corr) else None
    corr_reject_v_grad = _corr(reject_f, v_grad[m_corr].astype(np.float64)) if np.any(m_corr) else None

    b_map, b_map_path = _load_b_map(run_dir, hw)
    corr_reject_b = None
    if b_map is not None:
        m_b = valid_uv & np.isfinite(b_map)
        corr_reject_b = _corr(rejected[m_b].astype(np.float64), b_map[m_b].astype(np.float64)) if np.any(m_b) else None

    residual_info: dict[str, Any] = {}
    v_res_path = run_dir / "vertical" / "unwrap" / "residual.npy"
    h_res_path = run_dir / "horizontal" / "unwrap" / "residual.npy"
    if v_res_path.exists() and h_res_path.exists():
        rv = np.load(v_res_path).astype(np.float32)
        rh = np.load(h_res_path).astype(np.float32)
        if rv.shape == hw and rh.shape == hw:
            rmag = np.maximum(np.abs(rv), np.abs(rh))
            mr = valid_uv & np.isfinite(rmag)
            residual_info["corr_reject_residual_mag"] = _corr(rejected[mr].astype(np.float64), rmag[mr].astype(np.float64)) if np.any(mr) else None
            mgt = rejected & np.isfinite(rmag)
            if np.any(mgt):
                residual_info["rejected_with_residual_gt_1rad_pct"] = float(np.count_nonzero((rmag > 1.0) & mgt) / np.count_nonzero(mgt))
            residual_info["source"] = "residual_maps"
        else:
            residual_info["source"] = "residual_maps_shape_mismatch"
    else:
        residual_info = {
            "source": "unwrap_meta",
            "vertical_residual_gt_1rad_pct": v_unwrap_meta.get("residual_gt_1rad_pct"),
            "horizontal_residual_gt_1rad_pct": h_unwrap_meta.get("residual_gt_1rad_pct"),
        }

    # Save plots.
    bg = _load_background(run_dir, hw)
    reject_overlay_uv = _overlay_red(bg, rejected)
    plt.imsave(plots_dir / "reject_overlay_on_uv.png", reject_overlay_uv)

    # Camera overlay from first vertical capture if available.
    cam_bg = None
    for p in [
        run_dir / "vertical" / "captures" / "f_016" / "step_000.png",
        run_dir / "vertical" / "captures" / "f_004" / "step_000.png",
        run_dir / "vertical" / "captures" / "f_001" / "step_000.png",
    ]:
        img = _load_image_rgb(p)
        if img is not None and img.shape[:2] == hw:
            cam_bg = img
            break
    if cam_bg is None:
        cam_bg = bg
    plt.imsave(plots_dir / "reject_overlay_on_camera.png", _overlay_red(cam_bg, rejected))

    _save_binary(rejected, plots_dir / "reject_mask.png", "Rejected = mask_uv & ~mask_recon")
    _save_heatmap(err_cam, "stored reproj_err_cam", plots_dir / "reproj_err_cam_clipped.png")
    _save_heatmap(err_proj, "stored reproj_err_proj", plots_dir / "reproj_err_proj_clipped.png")
    _save_heatmap(u_grad, "|grad(u)|", plots_dir / "u_grad.png")
    _save_heatmap(v_grad, "|grad(v)|", plots_dir / "v_grad.png")
    u8_u = _normalize_to_u8(u_grad)
    u8_v = _normalize_to_u8(v_grad)
    plt.imsave(plots_dir / "u_grad_reject_overlay.png", _overlay_red(np.repeat(u8_u[..., None], 3, axis=2), rejected))
    plt.imsave(plots_dir / "v_grad_reject_overlay.png", _overlay_red(np.repeat(u8_v[..., None], 3, axis=2), rejected))

    # Stored vs best variant side-by-side.
    if best_variant is not None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        vm = variant_maps[best_variant]
        vals = np.concatenate([err_cam[np.isfinite(err_cam)], vm[np.isfinite(vm)]])
        if vals.size:
            lo, hi = np.percentile(vals, [1, 99])
            if hi <= lo:
                hi = lo + 1e-6
        else:
            lo, hi = 0.0, 1.0
        ax[0].imshow(err_cam, cmap="viridis", vmin=lo, vmax=hi)
        ax[0].set_title("stored reproj_err_cam")
        ax[1].imshow(vm, cmap="viridis", vmin=lo, vmax=hi)
        ax[1].set_title(f"best variant: {best_variant}")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(plots_dir / "reproj_err_cam_variant_compare.png", dpi=140)
        plt.close(fig)

    _plot_rate(row_rate, "rows", plots_dir / "rejection_rate_rows.png")
    _plot_rate(col_rate, "cols", plots_dir / "rejection_rate_cols.png")
    _plot_fft(row_fft, "rows", plots_dir / "fft_rejection_rows.png")
    _plot_fft(col_fft, "cols", plots_dir / "fft_rejection_cols.png")

    # Root-cause shortlist generation.
    shortlist: list[dict[str, Any]] = []

    # H3 likely if reprojection-space is consistent and gradients/residual correlate.
    if best_variant in {"A_distorted_pixel_stereo", "A_distorted_pixel_camera_intr"}:
        evidence = [
            f"Stored reproj_err_cam best matches {best_variant} (corr={variant_scores[best_variant]['corr']}, diff_p95={variant_scores[best_variant]['abs_diff_p95']}).",
            f"Rejection dominated by camera reproj: cam={int(recon_meta.get('rejected_by_cam_reproj_px', 0))}, proj={int(recon_meta.get('rejected_by_proj_reproj_px', 0))}, depth={int(recon_meta.get('rejected_by_depth_range_px', 0))}.",
            f"corr(reject, |grad(u)|)={corr_reject_u_grad}, corr(reject, |grad(v)|)={corr_reject_v_grad}",
            f"corr(row_rejection_rate, row_u_mean)={corr_row_rej_u_mean}, corr(row_rejection_rate, row_v_mean)={corr_row_rej_v_mean}",
            f"row rejection peak period={row_fft.get('peak_period_px_excluding_low')} px; row_u_mean peak={row_u_fft.get('peak_period_px_excluding_low')} px; row_v_mean peak={row_v_fft.get('peak_period_px_excluding_low')} px",
        ]
        if residual_info.get("corr_reject_residual_mag") is not None:
            evidence.append(f"corr(reject, residual_mag)={residual_info.get('corr_reject_residual_mag')}")
        shortlist.append({
            "title": "UV/phase ripple likely drives stripey reprojection errors",
            "evidence": evidence,
        })

    # H1 mismatch if stereo/intrinsics disagree strongly.
    k_delta = None
    if isinstance(consistency.get("camera_matrix_delta_abs"), list):
        arr = np.asarray(consistency["camera_matrix_delta_abs"], dtype=np.float64)
        k_delta = float(np.max(arr))
    if (consistency.get("camera_intrinsics_image_size") not in (None, "NOT FOUND") and run_res is not None
            and list(run_res) != list(consistency.get("camera_intrinsics_image_size"))):
        shortlist.append({
            "title": "Camera model / resolution mismatch",
            "evidence": [
                f"run resolution={run_res} vs camera intrinsics image_size={consistency.get('camera_intrinsics_image_size')}",
            ],
        })
    elif k_delta is not None and k_delta > 3.0:
        shortlist.append({
            "title": "Camera intrinsics model mismatch between stereo and camera intrinsics",
            "evidence": [f"max |K_stereo - K_intrinsics| = {k_delta}"]
        })

    # H2 mismatch if best variant is not canonical distorted pixel.
    if best_variant is not None and best_variant not in {"A_distorted_pixel_stereo", "A_distorted_pixel_camera_intr"}:
        shortlist.append({
            "title": "Coordinate-space mismatch in reprojection check",
            "evidence": [
                f"best variant is {best_variant}, not distorted-pixel space",
                f"scores={variant_scores.get(best_variant)}",
            ],
        })

    # H4 if shape misalignment detected.
    if not align_ok:
        shortlist.append({
            "title": "Mask/grid misalignment",
            "evidence": ["Not all arrays share identical (H,W).", f"shapes={shape_table}"],
        })

    if not shortlist:
        shortlist.append({
            "title": "Inconclusive from available artifacts",
            "evidence": ["No single hypothesis strongly dominates."],
        })

    what_not = [
        "Projector reprojection threshold is not the reject driver (rejected_by_proj_reproj_px is near zero).",
        "Depth range gate is not the reject driver (rejected_by_depth_range_px is near zero).",
    ]

    next_diag = [
        "Compare this run to another run with lower rejection using the same script and diff variant scores/correlations.",
        "Inspect `plots/u_grad_reject_overlay.png` and `plots/v_grad_reject_overlay.png` against fringe direction for periodic ripple alignment.",
        "Inspect `vertical/unwrap/residual.npy` and `horizontal/unwrap/residual.npy` slices at detected high-rejection row/col bands.",
    ]

    report = {
        "run": str(run_dir),
        "inputs": {
            "reconstruction_meta": str(recon_meta_path),
            "mask_uv": str(mask_uv_path),
            "mask_recon": str(mask_recon_path),
            "reproj_err_cam": str(err_cam_path),
            "reproj_err_proj": str(err_proj_path),
            "u": str(u_path),
            "v": str(v_path),
            "depth": str(depth_path) if depth_path else None,
            "b_map": b_map_path,
        },
        "shape_table": shape_table,
        "shape_alignment_ok": align_ok,
        "meta_consistency": consistency,
        "phase_meta": {
            "vertical_phase_meta_path": str(run_dir / "vertical" / "phase" / "f_016" / "phase_meta.json"),
            "horizontal_phase_meta_path": str(run_dir / "horizontal" / "phase" / "f_016" / "phase_meta.json"),
            "vertical_roi_valid_ratio_core": v_phase_meta.get("roi_valid_ratio_core"),
            "horizontal_roi_valid_ratio_core": h_phase_meta.get("roi_valid_ratio_core"),
        },
        "unwrap_meta": {
            "vertical_unwrap_meta_path": str(run_dir / "vertical" / "unwrap" / "unwrap_meta.json"),
            "horizontal_unwrap_meta_path": str(run_dir / "horizontal" / "unwrap" / "unwrap_meta.json"),
            "vertical_residual_p95": v_unwrap_meta.get("residual_p95"),
            "horizontal_residual_p95": h_unwrap_meta.get("residual_p95"),
        },
        "recomputed_rejection": {
            "uv_valid_px": int(np.count_nonzero(valid_uv)),
            "recon_valid_px": int(np.count_nonzero(accepted)),
            "rejected_px": int(np.count_nonzero(rejected)),
            "threshold": thr,
            "rejected_cam_gt_threshold_by_map": rej_cam_by_map,
            "rejected_proj_gt_threshold_by_map": rej_proj_by_map,
            "cam_err_accepted": _stats(cam_acc, ps=(50, 90, 95, 99)),
            "cam_err_rejected": _stats(cam_rej, ps=(50, 90, 95, 99)),
        },
        "variant_scores_vs_stored_reproj_cam": variant_scores,
        "best_variant": best_variant,
        "band_analysis": {
            "row_rate_bands": row_bands,
            "col_rate_bands": col_bands,
            "row_fft": {
                k: v for k, v in row_fft.items() if k not in {"freqs", "power"}
            },
            "col_fft": {
                k: v for k, v in col_fft.items() if k not in {"freqs", "power"}
            },
            "strongest_axis": strongest_axis,
            "strongest_axis_peak_period_px_excluding_low": strongest_fft.get("peak_period_px_excluding_low"),
            "strongest_axis_raw_peak_near_dc": strongest_fft.get("raw_is_near_dc"),
        },
        "uv_profile_banding": {
            "corr_row_rej_u_mean": corr_row_rej_u_mean,
            "corr_row_rej_v_mean": corr_row_rej_v_mean,
            "corr_col_rej_u_mean": corr_col_rej_u_mean,
            "corr_col_rej_v_mean": corr_col_rej_v_mean,
            "row_u_fft": {k: v for k, v in row_u_fft.items() if k not in {"freqs", "power"}},
            "row_v_fft": {k: v for k, v in row_v_fft.items() if k not in {"freqs", "power"}},
            "col_u_fft": {k: v for k, v in col_u_fft.items() if k not in {"freqs", "power"}},
            "col_v_fft": {k: v for k, v in col_v_fft.items() if k not in {"freqs", "power"}},
        },
        "correlations": {
            "corr_reject_u_grad": corr_reject_u_grad,
            "corr_reject_v_grad": corr_reject_v_grad,
            "corr_reject_B": corr_reject_b,
            "residual": residual_info,
        },
        "root_cause_shortlist": shortlist,
        "what_this_is_not": what_not,
        "next_steps_diagnostics_only": next_diag,
        "plots_dir": str(plots_dir),
        "evidence": {
            "uv_valid_ratio": float(np.count_nonzero(valid_uv) / valid_uv.size),
            "recon_valid_ratio": float(np.count_nonzero(accepted) / valid_uv.size),
            "rejected_by_cam_reproj_px": int(recon_meta.get("rejected_by_cam_reproj_px", 0)),
            "rejected_by_proj_reproj_px": int(recon_meta.get("rejected_by_proj_reproj_px", 0)),
            "rejected_by_depth_range_px": int(recon_meta.get("rejected_by_depth_range_px", 0)),
            "max_reproj_err_px": thr,
            "best_variant": best_variant,
            "best_variant_corr": (
                variant_scores.get(best_variant, {}).get("corr") if best_variant else None
            ),
            "best_variant_abs_diff_p95": (
                variant_scores.get(best_variant, {}).get("abs_diff_p95") if best_variant else None
            ),
            "strongest_axis": strongest_axis,
            "strongest_axis_peak_period_px_excluding_low": strongest_fft.get("peak_period_px_excluding_low"),
            "strongest_axis_raw_peak_near_dc": strongest_fft.get("raw_is_near_dc"),
        },
    }

    (out_dir / "report.json").write_text(json.dumps(_json_safe(report), indent=2))
    _write_report_md(out_dir, report)

    print("=== Step1 reconstruction rejection diagnostic ===")
    print(f"run: {run_dir}")
    print(f"out: {out_dir}")
    print(f"uv_valid_ratio={report['evidence']['uv_valid_ratio']:.4f} recon_valid_ratio={report['evidence']['recon_valid_ratio']:.4f}")
    print(
        "rejections (meta): "
        f"cam={report['evidence']['rejected_by_cam_reproj_px']} "
        f"proj={report['evidence']['rejected_by_proj_reproj_px']} "
        f"depth={report['evidence']['rejected_by_depth_range_px']}"
    )
    print(
        "best reproj-space variant: "
        f"{report['evidence']['best_variant']} "
        f"(corr={report['evidence']['best_variant_corr']}, diff_p95={report['evidence']['best_variant_abs_diff_p95']})"
    )
    print(
        "band strongest axis: "
        f"{report['evidence']['strongest_axis']} "
        f"period_excluding_low={report['evidence']['strongest_axis_peak_period_px_excluding_low']} "
        f"raw_peak_near_dc={report['evidence']['strongest_axis_raw_peak_near_dc']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
