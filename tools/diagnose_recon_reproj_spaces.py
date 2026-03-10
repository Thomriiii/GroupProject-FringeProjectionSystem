#!/usr/bin/env python3
"""Offline diagnostics for reconstruction reprojection-space mismatches and striping.

Usage:
  PYTHONPATH=src python tools/diagnose_recon_reproj_spaces.py \
      --run-dir data/runs/20260226_103308_uv \
      --out data/runs/20260226_103308_uv/diagnostics/recon_reproj_spaces
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def _json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, (np.floating, float)):
        fv = float(v)
        return fv if math.isfinite(fv) else None
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return _json_safe(v.tolist())
    return v


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _percentiles(vals: np.ndarray, ps=(50, 90, 95, 99)) -> dict[str, float | None]:
    arr = vals[np.isfinite(vals)]
    out: dict[str, float | None] = {}
    if arr.size == 0:
        for p in ps:
            out[f"p{p}"] = None
        out["max"] = None
        out["mean"] = None
        return out
    for p in ps:
        out[f"p{p}"] = float(np.percentile(arr, p))
    out["max"] = float(np.max(arr))
    out["mean"] = float(np.mean(arr))
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


def _resolve_run_dir(arg: str) -> Path:
    p = Path(arg)
    if p.exists():
        return p
    p2 = Path("data/runs") / arg
    if p2.exists():
        return p2
    raise SystemExit(f"Run directory not found: {arg}")


def _find_camera_background(run_dir: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    candidates: list[Path] = []
    for base in [run_dir / "vertical" / "captures", run_dir / "captures"]:
        if base.exists():
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
        if arr.ndim == 3 and arr.shape[0] == h and arr.shape[1] == w:
            return arr[:, :, :3].astype(np.uint8)

    dimg = run_dir / "reconstruction" / "depth_debug_autoscale.png"
    if dimg.exists():
        arr = np.array(Image.open(dimg))
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=2)
        return arr[:, :, :3].astype(np.uint8)

    return np.zeros((h, w, 3), dtype=np.uint8)


def _load_calibration(run_dir: Path) -> tuple[dict[str, Any], Path]:
    repo_root = Path(__file__).resolve().parents[1]

    candidates = [
        repo_root / "data" / "calibration" / "projector" / "stereo_latest.json",
        run_dir / "calibration" / "projector" / "stereo_latest.json",
    ]
    for c in candidates:
        if c.exists():
            return _load_json(c), c

    raise SystemExit(
        "Missing stereo calibration JSON. Tried:\n- " + "\n- ".join(str(c) for c in candidates)
    )


def _camera_intrinsics_from_stereo(stereo: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "camera_matrix" in stereo and "camera_dist_coeffs" in stereo:
        K = np.asarray(stereo["camera_matrix"], dtype=np.float64)
        dist = np.asarray(stereo["camera_dist_coeffs"], dtype=np.float64).reshape(-1, 1)
        return K, dist
    raise SystemExit("Stereo JSON missing camera_matrix/camera_dist_coeffs")


def _projector_intrinsics_from_stereo(stereo: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = ["projector_matrix", "projector_dist_coeffs", "R", "T"]
    missing = [k for k in required if k not in stereo]
    if missing:
        raise SystemExit("Stereo JSON missing keys: " + ", ".join(missing))
    Kp = np.asarray(stereo["projector_matrix"], dtype=np.float64)
    distp = np.asarray(stereo["projector_dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    R = np.asarray(stereo["R"], dtype=np.float64)
    T = np.asarray(stereo["T"], dtype=np.float64).reshape(3, 1)
    return Kp, distp, R, T


def _triangulate_all_uv(
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
        raise SystemExit("No valid UV points")

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
        "proj_norm": proj_norm,
        "X": X,
    }


def _error_map_from_vector(shape_hw: tuple[int, int], ys: np.ndarray, xs: np.ndarray, err: np.ndarray) -> np.ndarray:
    out = np.full(shape_hw, np.nan, dtype=np.float32)
    out[ys, xs] = err.astype(np.float32)
    return out


def _variant_errors(
    X: np.ndarray,
    cam_px: np.ndarray,
    cam_norm_obs: np.ndarray,
    Kc: np.ndarray,
    distc: np.ndarray,
    rect: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    n = X.shape[0]
    finite_xyz = np.isfinite(X).all(axis=1) & (np.abs(X[:, 2]) > 1e-12)

    errs: dict[str, np.ndarray] = {}

    # A) Distorted pixel reprojection (canonical pixel-space)
    err_a = np.full(n, np.nan, dtype=np.float64)
    if np.any(finite_xyz):
        Xf = X[finite_xyz].reshape(-1, 1, 3)
        rep, _ = cv2.projectPoints(
            Xf,
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            Kc,
            distc,
        )
        rep = rep.reshape(-1, 2)
        err_a[finite_xyz] = np.linalg.norm(rep - cam_px[finite_xyz], axis=1)
    errs["A_distorted_pixel"] = err_a

    # B) Undistorted normalized reprojection
    err_b = np.full(n, np.nan, dtype=np.float64)
    if np.any(finite_xyz):
        proj_norm = X[finite_xyz, :2] / X[finite_xyz, 2:3]
        err_b[finite_xyz] = np.linalg.norm(proj_norm - cam_norm_obs[finite_xyz], axis=1)
    errs["B_undist_normalized"] = err_b

    # C) Rectified pixel reprojection using R1/P1
    if rect is not None:
        R1 = rect["R1"]
        Krect = rect["Krect"]
        err_c = np.full(n, np.nan, dtype=np.float64)
        err_d1 = np.full(n, np.nan, dtype=np.float64)
        err_d2 = np.full(n, np.nan, dtype=np.float64)
        err_d3 = np.full(n, np.nan, dtype=np.float64)

        cam_rect_obs = cv2.undistortPoints(
            cam_px.reshape(-1, 1, 2),
            Kc,
            distc,
            R=R1,
            P=Krect,
        ).reshape(-1, 2)

        if np.any(finite_xyz):
            Xr = (R1 @ X[finite_xyz].T).T
            z = Xr[:, 2:3]
            okz = np.abs(z[:, 0]) > 1e-12
            if np.any(okz):
                Xr_ok = Xr[okz]
                uv_rect = (Krect @ Xr_ok.T).T
                uv_rect = uv_rect[:, :2] / uv_rect[:, 2:3]
                idx = np.where(finite_xyz)[0][okz]

                # C: rectified vs rectified obs
                err_c[idx] = np.linalg.norm(uv_rect - cam_rect_obs[idx], axis=1)
                # D1: rectified reprojection vs unrectified obs
                err_d1[idx] = np.linalg.norm(uv_rect - cam_px[idx], axis=1)

                # D3: misuse P1 K as if no rectification rotation
                uv_d3 = (Krect @ X[idx].T).T
                uv_d3 = uv_d3[:, :2] / uv_d3[:, 2:3]
                err_d3[idx] = np.linalg.norm(uv_d3 - cam_px[idx], axis=1)

            # D2: distorted reprojection vs undistorted pixel obs
            rep_dist, _ = cv2.projectPoints(
                X[finite_xyz].reshape(-1, 1, 3),
                np.zeros((3, 1), dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                Kc,
                distc,
            )
            rep_dist = rep_dist.reshape(-1, 2)
            cam_undist_px = cv2.undistortPoints(
                cam_px[finite_xyz].reshape(-1, 1, 2),
                Kc,
                distc,
                P=Kc,
            ).reshape(-1, 2)
            err_d2[finite_xyz] = np.linalg.norm(rep_dist - cam_undist_px, axis=1)

        errs["C_rectified_pixel"] = err_c
        errs["D_mixed_rect_vs_unrect_obs"] = err_d1
        errs["D_mixed_dist_vs_undist_obs"] = err_d2
        errs["D_mixed_use_P1_as_K"] = err_d3

    return errs


def _compare_to_stored(
    stored_map: np.ndarray,
    variant_map: np.ndarray,
    mask_uv: np.ndarray,
) -> dict[str, float | None]:
    valid = mask_uv & np.isfinite(stored_map) & np.isfinite(variant_map)
    if not np.any(valid):
        return {
            "corr": None,
            "diff_p50": None,
            "diff_p90": None,
            "diff_p95": None,
            "n": 0,
        }
    a = stored_map[valid].astype(np.float64)
    b = variant_map[valid].astype(np.float64)
    diff = np.abs(a - b)
    return {
        "corr": _corr(a, b),
        "diff_p50": float(np.percentile(diff, 50)),
        "diff_p90": float(np.percentile(diff, 90)),
        "diff_p95": float(np.percentile(diff, 95)),
        "n": int(a.size),
    }


def _plot_imshow(arr: np.ndarray, title: str, out_path: Path, cmap: str = "viridis") -> None:
    plt.figure(figsize=(8, 5))
    v = arr[np.isfinite(arr)]
    if v.size:
        lo, hi = np.percentile(v, [1, 99])
        if hi <= lo:
            hi = lo + 1e-6
    else:
        lo, hi = 0.0, 1.0
    plt.imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _rejection_rates(mask_uv: np.ndarray, rejected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uv_row = mask_uv.sum(axis=1).astype(np.float64)
    uv_col = mask_uv.sum(axis=0).astype(np.float64)
    rej_row = rejected.sum(axis=1).astype(np.float64)
    rej_col = rejected.sum(axis=0).astype(np.float64)

    row_rate = np.full(mask_uv.shape[0], np.nan, dtype=np.float64)
    col_rate = np.full(mask_uv.shape[1], np.nan, dtype=np.float64)
    good_r = uv_row > 0
    good_c = uv_col > 0
    row_rate[good_r] = rej_row[good_r] / uv_row[good_r]
    col_rate[good_c] = rej_col[good_c] / uv_col[good_c]
    return row_rate, col_rate


def _find_bands(rate: np.ndarray) -> list[tuple[int, int, float]]:
    finite = np.isfinite(rate)
    if not np.any(finite):
        return []
    vals = rate[finite]
    thr = float(np.nanmean(vals) + np.nanstd(vals))
    above = finite & (rate > thr)
    bands: list[tuple[int, int, float]] = []
    i = 0
    n = above.size
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and above[j + 1]:
            j += 1
        band_mean = float(np.nanmean(rate[i : j + 1]))
        bands.append((i, j, band_mean))
        i = j + 1
    return bands


def _fft_1d(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, float | None]:
    finite = np.isfinite(signal)
    if not np.any(finite):
        return np.array([]), np.array([]), None
    x = signal.copy().astype(np.float64)
    fill = float(np.nanmean(x[finite]))
    x[~finite] = fill
    x = x - np.mean(x)
    n = x.size
    if n < 4:
        return np.array([]), np.array([]), None
    spec = np.fft.rfft(x)
    power = (np.abs(spec) ** 2).astype(np.float64)
    freqs = np.fft.rfftfreq(n, d=1.0)
    if power.size <= 1:
        return freqs, power, None
    power[0] = 0.0
    k = int(np.argmax(power))
    if k <= 0 or freqs[k] <= 1e-12:
        return freqs, power, None
    period = float(1.0 / freqs[k])
    return freqs, power, period


def _save_rate_plot(rate: np.ndarray, axis_name: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    x = np.arange(rate.size)
    plt.plot(x, rate, linewidth=1.2)
    plt.ylim(-0.02, 1.02)
    plt.xlabel(f"{axis_name} index")
    plt.ylabel("rejection rate")
    plt.title(f"Rejection Rate by {axis_name.capitalize()}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _save_fft_plot(freqs: np.ndarray, power: np.ndarray, axis_name: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    if freqs.size and power.size:
        plt.plot(freqs, power, linewidth=1.2)
    plt.xlabel("cycles / pixel")
    plt.ylabel("power")
    plt.title(f"FFT of {axis_name} rejection-rate signal")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _save_rejected_overlay(bg: np.ndarray, rejected: np.ndarray, out_path: Path) -> None:
    img = bg.astype(np.float32).copy()
    if np.any(rejected):
        img[rejected, 0] = np.clip(0.35 * img[rejected, 0] + 165.0, 0, 255)
        img[rejected, 1] = np.clip(0.35 * img[rejected, 1], 0, 255)
        img[rejected, 2] = np.clip(0.35 * img[rejected, 2], 0, 255)
    Image.fromarray(img.astype(np.uint8)).save(out_path)


def _save_side_by_side(stored: np.ndarray, recomputed: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    valid = np.isfinite(stored) | np.isfinite(recomputed)
    vals = np.concatenate([
        stored[np.isfinite(stored)].ravel(),
        recomputed[np.isfinite(recomputed)].ravel(),
    ])
    if vals.size:
        lo, hi = np.percentile(vals, [1, 99])
        if hi <= lo:
            hi = lo + 1e-6
    else:
        lo, hi = 0.0, 1.0

    axes[0].imshow(stored, vmin=lo, vmax=hi, cmap="viridis")
    axes[0].set_title("stored reproj_err_cam")
    axes[1].imshow(recomputed, vmin=lo, vmax=hi, cmap="viridis")
    axes[1].set_title("best-match recomputed")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _best_match_variant(scores: dict[str, dict[str, float | None]]) -> str | None:
    ranked: list[tuple[float, float, str]] = []
    for name, s in scores.items():
        corr = s.get("corr")
        p95 = s.get("diff_p95")
        if corr is None or p95 is None:
            continue
        ranked.append((float(corr), -float(p95), name))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][2]


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        f = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(f):
        return "NA"
    return f"{f:.{nd}f}"


def _write_report(
    out_dir: Path,
    summary: dict[str, Any],
    recon_meta: dict[str, Any],
    match_scores: dict[str, dict[str, float | None]],
    best_name: str | None,
    stripe: dict[str, Any],
) -> None:
    rej = summary["rejection"]
    uv = summary["uv_recon"]

    mismatch_msg = "No clear reprojection-space mismatch detected."
    if best_name is not None and best_name != "A_distorted_pixel":
        a = match_scores.get("A_distorted_pixel", {})
        b = match_scores.get(best_name, {})
        a_corr = a.get("corr")
        b_corr = b.get("corr")
        a_p95 = a.get("diff_p95")
        b_p95 = b.get("diff_p95")
        if (
            a_corr is not None and b_corr is not None and b_corr > a_corr + 0.05
            and a_p95 is not None and b_p95 is not None and b_p95 < 0.8 * a_p95
        ):
            mismatch_msg = (
                "Potential reprojection-space mismatch: "
                f"{best_name} matches stored map better than A_distorted_pixel."
            )

    lines: list[str] = []
    lines.append("# Reconstruction Reprojection-Space Diagnostic")
    lines.append("")
    lines.append(f"Run: `{summary['run_dir_name']}`")
    lines.append("")
    lines.append("## 1) Reconstruction Summary")
    lines.append(f"- uv_valid_ratio: `{_fmt(uv['uv_valid_ratio'])}`")
    lines.append(f"- recon_valid_ratio: `{_fmt(uv['recon_valid_ratio'])}`")
    lines.append(f"- accepted_px: `{uv['accepted_px']}`")
    lines.append(f"- rejected_total_px: `{uv['rejected_total_px']}`")
    lines.append(f"- rejected_by_cam_reproj_px: `{rej['rejected_by_cam_reproj_px']}`")
    lines.append(f"- rejected_by_proj_reproj_px: `{rej['rejected_by_proj_reproj_px']}`")
    lines.append(f"- rejected_by_depth_range_px: `{rej['rejected_by_depth_range_px']}`")
    lines.append(f"- max_reproj_err_px: `{_fmt(uv['max_reproj_err_px'])}`")
    lines.append("")
    lines.append("## 2) Rejection Cause Confirmation")
    lines.append(f"- rejected fraction in UV mask: `{_fmt(rej['rejected_fraction_in_uv'])}`")
    lines.append(
        f"- reproj_err_cam accepted p50/p95: `{_fmt(rej['cam_err_accepted']['p50'])}` / `{_fmt(rej['cam_err_accepted']['p95'])}`"
    )
    lines.append(
        f"- reproj_err_cam rejected p50/p95: `{_fmt(rej['cam_err_rejected']['p50'])}` / `{_fmt(rej['cam_err_rejected']['p95'])}`"
    )
    lines.append(f"- cam-dominant rejection (from counts): `{rej['cam_dominant']}`")
    lines.append("")
    lines.append("## 3) Reprojection Convention Matching")
    lines.append(f"- Best-matching variant: `{best_name}`")
    for name, s in match_scores.items():
        lines.append(
            f"- `{name}`: corr={_fmt(s.get('corr'))}, diff_p50={_fmt(s.get('diff_p50'))}, "
            f"diff_p95={_fmt(s.get('diff_p95'))}, n={s.get('n')}"
        )
    lines.append(f"- Interpretation: {mismatch_msg}")
    lines.append("")
    lines.append("## 4) Stripe Analysis")
    lines.append(f"- strongest_axis: `{stripe.get('strongest_axis')}`")
    lines.append(f"- strongest_axis_peak_period_px: `{_fmt(stripe.get('strongest_axis_peak_period_px'))}`")
    lines.append(f"- high-rejection row bands: `{stripe.get('row_bands')}`")
    lines.append(f"- high-rejection col bands: `{stripe.get('col_bands')}`")
    lines.append("")

    if mismatch_msg.startswith("Potential"):
        lines.append("## Conclusion")
        lines.append("Stored reprojection appears to be computed in a different coordinate convention than expected.")
        lines.append("Inspect camera reprojection space consistency (distorted/undistorted/rectified) before further phase-side tuning.")
    else:
        lines.append("## Conclusion")
        lines.append("No reprojection-space mismatch was found; the stored map is consistent with the canonical distorted-pixel reprojection.")
        lines.append("Likely source is UV/phase ripple (stripey error structure), not reconstruction-space convention mismatch.")
        lines.append("Next experiments:")
        lines.append("- Gamma prewarp experiment for projector nonlinearity")
        lines.append("- Recon-only stronger B-threshold mask")
        lines.append("- Tighten unwrap residual gating for recon-only acceptance")

    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    recon_dir = run_dir / "reconstruction"
    req = [
        recon_dir / "reconstruction_meta.json",
        recon_dir / "masks" / "mask_uv.npy",
        recon_dir / "masks" / "mask_recon.npy",
        recon_dir / "reproj_err_cam.npy",
        recon_dir / "reproj_err_proj.npy",
        run_dir / "projector_uv" / "u.npy",
        run_dir / "projector_uv" / "v.npy",
    ]
    missing = [str(p) for p in req if not p.exists()]
    if missing:
        raise SystemExit("Missing required inputs:\n- " + "\n- ".join(missing))

    recon_meta = _load_json(recon_dir / "reconstruction_meta.json")
    mask_uv = np.load(recon_dir / "masks" / "mask_uv.npy").astype(bool)
    mask_recon = np.load(recon_dir / "masks" / "mask_recon.npy").astype(bool)
    stored_err_cam = np.load(recon_dir / "reproj_err_cam.npy").astype(np.float32)
    stored_err_proj = np.load(recon_dir / "reproj_err_proj.npy").astype(np.float32)

    if stored_err_cam.shape != mask_uv.shape or stored_err_proj.shape != mask_uv.shape:
        raise SystemExit("Shape mismatch between reproj maps and mask_uv")

    u = np.load(run_dir / "projector_uv" / "u.npy").astype(np.float32)
    v = np.load(run_dir / "projector_uv" / "v.npy").astype(np.float32)

    depth = None
    for cand in [recon_dir / "depth.npy", recon_dir / "depth_map.npy"]:
        if cand.exists():
            depth = np.load(cand).astype(np.float32)
            break

    b_map = None
    for cand in [
        run_dir / "vertical" / "phase" / "f_016" / "B.npy",
        run_dir / "vertical" / "phase" / "B.npy",
    ]:
        if cand.exists():
            b_map = np.load(cand).astype(np.float32)
            if b_map.shape != mask_uv.shape:
                b_map = None
            break

    stereo, stereo_path = _load_calibration(run_dir)
    Kc, distc = _camera_intrinsics_from_stereo(stereo)
    Kp, distp, R, T = _projector_intrinsics_from_stereo(stereo)

    rect = None
    rect_s = stereo.get("rectification", {}) or {}
    if "R1" in rect_s and "P1" in rect_s:
        R1 = np.asarray(rect_s["R1"], dtype=np.float64)
        P1 = np.asarray(rect_s["P1"], dtype=np.float64)
        if R1.shape == (3, 3) and P1.shape == (3, 4):
            rect = {"R1": R1, "Krect": P1[:, :3], "P1": P1}

    tri = _triangulate_all_uv(u, v, mask_uv, Kc, distc, Kp, distp, R, T)
    ys = tri["ys"]
    xs = tri["xs"]
    shape_hw = mask_uv.shape

    variants = _variant_errors(
        X=tri["X"],
        cam_px=tri["cam_px"],
        cam_norm_obs=tri["cam_norm"],
        Kc=Kc,
        distc=distc,
        rect=rect,
    )

    variant_maps: dict[str, np.ndarray] = {
        name: _error_map_from_vector(shape_hw, ys, xs, err)
        for name, err in variants.items()
    }

    match_scores: dict[str, dict[str, float | None]] = {}
    for name, vm in variant_maps.items():
        match_scores[name] = _compare_to_stored(stored_err_cam, vm, mask_uv)

    best_name = _best_match_variant(match_scores)
    best_map = variant_maps.get(best_name, np.full(shape_hw, np.nan, dtype=np.float32))

    rejected = mask_uv & (~mask_recon)
    accepted = mask_recon.copy()

    cam_acc = stored_err_cam[accepted & np.isfinite(stored_err_cam)]
    cam_rej = stored_err_cam[rejected & np.isfinite(stored_err_cam)]

    uv_count = int(np.count_nonzero(mask_uv))
    recon_count = int(np.count_nonzero(mask_recon))
    rejected_total = int(np.count_nonzero(rejected))

    rejected_by_cam = int(recon_meta.get("rejected_by_cam_reproj_px", 0))
    rejected_by_proj = int(recon_meta.get("rejected_by_proj_reproj_px", 0))
    rejected_by_depth = int(recon_meta.get("rejected_by_depth_range_px", 0))
    if rejected_total > 0 and (rejected_by_cam + rejected_by_proj + rejected_by_depth) == 0:
        thr = float(recon_meta.get("max_reproj_err_px", 2.0))
        uv_finite = mask_uv & np.isfinite(stored_err_cam) & np.isfinite(stored_err_proj)
        cam_fail = uv_finite & (stored_err_cam > thr)
        proj_fail = uv_finite & (~cam_fail) & (stored_err_proj > thr)
        rejected_by_cam = int(np.count_nonzero(cam_fail))
        rejected_by_proj = int(np.count_nonzero(proj_fail))
        rejected_by_depth = max(0, rejected_total - rejected_by_cam - rejected_by_proj)

    row_rate, col_rate = _rejection_rates(mask_uv, rejected)
    row_bands = _find_bands(row_rate)
    col_bands = _find_bands(col_rate)

    row_freqs, row_pow, row_period = _fft_1d(np.nan_to_num(row_rate, nan=0.0))
    col_freqs, col_pow, col_period = _fft_1d(np.nan_to_num(col_rate, nan=0.0))

    row_std = float(np.nanstd(row_rate)) if np.any(np.isfinite(row_rate)) else 0.0
    col_std = float(np.nanstd(col_rate)) if np.any(np.isfinite(col_rate)) else 0.0
    strongest_axis = "rows" if row_std >= col_std else "cols"
    strongest_period = row_period if strongest_axis == "rows" else col_period

    bg = _find_camera_background(run_dir, shape_hw)

    _save_rejected_overlay(bg, rejected, out_dir / "rejected_overlay.png")
    _save_side_by_side(stored_err_cam, best_map, out_dir / "reproj_err_cam.png")
    _save_rate_plot(row_rate, "rows", out_dir / "rejection_rate_rows.png")
    _save_rate_plot(col_rate, "cols", out_dir / "rejection_rate_cols.png")
    _save_fft_plot(row_freqs, row_pow, "rows", out_dir / "fft_rejection_rows.png")
    _save_fft_plot(col_freqs, col_pow, "cols", out_dir / "fft_rejection_cols.png")
    _plot_imshow(stored_err_proj, "stored reproj_err_proj", out_dir / "reproj_err_proj.png")

    summary = {
        "run_dir": str(run_dir),
        "run_dir_name": run_dir.name,
        "stereo_path": str(stereo_path),
        "uv_recon": {
            "uv_valid_ratio": float(uv_count / float(mask_uv.size)),
            "recon_valid_ratio": float(recon_count / float(mask_uv.size)),
            "uv_valid_px": uv_count,
            "recon_valid_px": recon_count,
            "accepted_px": int(recon_meta.get("accepted_px", recon_count)),
            "rejected_total_px": int(recon_meta.get("rejected_total_px", rejected_total)),
            "max_reproj_err_px": float(recon_meta.get("max_reproj_err_px", np.nan)),
        },
        "rejection": {
            "rejected_fraction_in_uv": float(rejected_total / max(1, uv_count)),
            "rejected_by_cam_reproj_px": rejected_by_cam,
            "rejected_by_proj_reproj_px": rejected_by_proj,
            "rejected_by_depth_range_px": rejected_by_depth,
            "cam_err_accepted": _percentiles(cam_acc, ps=(50, 95)),
            "cam_err_rejected": _percentiles(cam_rej, ps=(50, 95)),
            "cam_dominant": bool(rejected_by_cam >= max(rejected_by_proj, rejected_by_depth)),
        },
        "stored_maps": {
            "reproj_err_cam_uv": _percentiles(stored_err_cam[mask_uv], ps=(50, 90, 95, 99)),
            "reproj_err_cam_recon": _percentiles(stored_err_cam[mask_recon], ps=(50, 90, 95, 99)),
            "reproj_err_proj_uv": _percentiles(stored_err_proj[mask_uv], ps=(50, 90, 95, 99)),
            "reproj_err_proj_recon": _percentiles(stored_err_proj[mask_recon], ps=(50, 90, 95, 99)),
        },
        "variant_match": match_scores,
        "best_variant": best_name,
        "stripe": {
            "row_rate_std": row_std,
            "col_rate_std": col_std,
            "strongest_axis": strongest_axis,
            "strongest_axis_peak_period_px": strongest_period,
            "row_bands": row_bands,
            "col_bands": col_bands,
        },
    }

    if b_map is not None:
        finite = rejected & np.isfinite(b_map)
        summary["optional_b_map"] = {
            "available": True,
            "rejected_b_stats": _percentiles(b_map[finite], ps=(50, 90, 95, 99)) if np.any(finite) else None,
        }
    else:
        summary["optional_b_map"] = {"available": False}

    # residual overlap
    residual = {}
    v_res = run_dir / "vertical" / "unwrap" / "residual.npy"
    h_res = run_dir / "horizontal" / "unwrap" / "residual.npy"
    if v_res.exists() and h_res.exists():
        rv = np.load(v_res).astype(np.float32)
        rh = np.load(h_res).astype(np.float32)
        if rv.shape == mask_uv.shape and rh.shape == mask_uv.shape:
            m = rejected & np.isfinite(rv) & np.isfinite(rh)
            if np.any(m):
                gt = (np.abs(rv) > 1.0) | (np.abs(rh) > 1.0)
                residual = {
                    "source": "residual_maps",
                    "rejected_with_residual_gt_1rad_pct": float(np.count_nonzero(gt & m) / np.count_nonzero(m)),
                }
    if not residual:
        for orient in ["vertical", "horizontal"]:
            p = run_dir / orient / "unwrap" / "unwrap_meta.json"
            if p.exists():
                md = _load_json(p)
                residual[f"{orient}_residual_gt_1rad_pct"] = md.get("residual_gt_1rad_pct")
        residual["source"] = "unwrap_meta"
    summary["residual"] = residual

    (out_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2))
    _write_report(out_dir, summary, recon_meta, match_scores, best_name, summary["stripe"])

    print("=== Reconstruction Diagnostic Summary ===")
    print(f"run_dir: {run_dir}")
    print(f"out_dir: {out_dir}")
    print(
        f"uv_valid_ratio={summary['uv_recon']['uv_valid_ratio']:.4f} "
        f"recon_valid_ratio={summary['uv_recon']['recon_valid_ratio']:.4f}"
    )
    print(
        "rejections: "
        f"cam={summary['rejection']['rejected_by_cam_reproj_px']} "
        f"proj={summary['rejection']['rejected_by_proj_reproj_px']} "
        f"depth={summary['rejection']['rejected_by_depth_range_px']}"
    )
    print(f"best_variant_match={best_name}")
    if best_name in match_scores:
        s = match_scores[best_name]
        print(
            f"best_variant corr={_fmt(s.get('corr'))} "
            f"diff_p95={_fmt(s.get('diff_p95'))}"
        )
    print(
        f"stripe strongest_axis={summary['stripe']['strongest_axis']} "
        f"period_px={_fmt(summary['stripe']['strongest_axis_peak_period_px'])}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
