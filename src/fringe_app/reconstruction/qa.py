"""Offline QA metrics for saved reconstruction depth maps."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _fit_plane(depth_map: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = mask.astype(bool) & np.isfinite(depth_map)
    ys, xs = np.where(valid)
    if ys.size < 3:
        raise ValueError("Need at least 3 valid depth points to fit a plane")
    z = depth_map[ys, xs].astype(np.float64)
    A = np.column_stack(
        [
            xs.astype(np.float64),
            ys.astype(np.float64),
            np.ones(xs.shape[0], dtype=np.float64),
        ]
    )
    coeff, *_ = np.linalg.lstsq(A, z, rcond=None)
    pred = (A @ coeff).astype(np.float64)
    residual = (z - pred).astype(np.float64)
    residual_map = np.full(depth_map.shape, np.nan, dtype=np.float32)
    residual_map[ys, xs] = residual.astype(np.float32)
    return coeff.astype(np.float64), residual, residual_map


def _plane_metrics(residual: np.ndarray) -> dict[str, float]:
    if residual.size == 0:
        return {
            "plane_fit_rms": float("nan"),
            "plane_fit_p95": float("nan"),
            "plane_fit_peak_to_peak": float("nan"),
        }
    abs_r = np.abs(residual)
    return {
        "plane_fit_rms": float(np.sqrt(np.mean(residual ** 2))),
        "plane_fit_p95": float(np.percentile(abs_r, 95)),
        "plane_fit_peak_to_peak": float(np.max(residual) - np.min(residual)),
    }


def _fft_metrics(residual_map: np.ndarray, mask: np.ndarray) -> tuple[dict[str, float], np.ndarray, tuple[int, int] | None]:
    valid = mask.astype(bool) & np.isfinite(residual_map)
    h, w = residual_map.shape
    if not np.any(valid):
        return (
            {
                "depth_fft_peak_period_px": float("nan"),
                "depth_fft_peak_energy_ratio": float("nan"),
                "stripe_direction_deg": float("nan"),
            },
            np.zeros((h, w), dtype=np.float32),
            None,
        )

    centered = np.zeros((h, w), dtype=np.float64)
    vals = residual_map[valid].astype(np.float64)
    centered[valid] = vals - float(np.mean(vals))
    win_y = np.hanning(h).astype(np.float64)
    win_x = np.hanning(w).astype(np.float64)
    window = np.outer(win_y, win_x)
    spectrum = np.fft.fftshift(np.fft.fft2(centered * window))
    power = (np.abs(spectrum) ** 2).astype(np.float64)

    cy, cx = h // 2, w // 2
    dc_r = 2
    y0 = max(0, cy - dc_r)
    y1 = min(h, cy + dc_r + 1)
    x0 = max(0, cx - dc_r)
    x1 = min(w, cx + dc_r + 1)
    power[y0:y1, x0:x1] = 0.0

    total_energy = float(np.sum(power))
    if not np.isfinite(total_energy) or total_energy <= 0.0:
        return (
            {
                "depth_fft_peak_period_px": float("nan"),
                "depth_fft_peak_energy_ratio": float("nan"),
                "stripe_direction_deg": float("nan"),
            },
            power.astype(np.float32),
            None,
        )

    peak_idx = np.unravel_index(int(np.argmax(power)), power.shape)
    peak_y, peak_x = int(peak_idx[0]), int(peak_idx[1])
    peak_energy = float(power[peak_y, peak_x])
    fy = np.fft.fftshift(np.fft.fftfreq(h, d=1.0))[peak_y]
    fx = np.fft.fftshift(np.fft.fftfreq(w, d=1.0))[peak_x]
    freq_norm = float(np.hypot(fx, fy))
    period = float(1.0 / freq_norm) if freq_norm > 1e-9 else float("nan")
    # Stripe direction is orthogonal to the FFT frequency vector.
    stripe_direction = float((np.degrees(np.arctan2(fy, fx)) + 90.0) % 180.0)

    metrics = {
        "depth_fft_peak_period_px": period,
        "depth_fft_peak_energy_ratio": float(peak_energy / total_energy),
        "stripe_direction_deg": stripe_direction,
    }
    return metrics, power.astype(np.float32), (peak_x, peak_y)


def _corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return float("nan")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _snr_metrics(residual_map: np.ndarray, mask: np.ndarray, b_map: np.ndarray | None) -> dict[str, float | str]:
    if b_map is None:
        return {
            "corr_abs_residual_vs_invB": "NOT_AVAILABLE",
            "corr_abs_residual_vs_B": "NOT_AVAILABLE",
        }
    valid = (
        mask.astype(bool)
        & np.isfinite(residual_map)
        & np.isfinite(b_map)
        & (b_map > 0)
    )
    if not np.any(valid):
        return {
            "corr_abs_residual_vs_invB": float("nan"),
            "corr_abs_residual_vs_B": float("nan"),
        }
    abs_res = np.abs(residual_map[valid].astype(np.float64))
    b_vals = b_map[valid].astype(np.float64)
    inv_b = 1.0 / b_vals
    return {
        "corr_abs_residual_vs_invB": _corrcoef_safe(abs_res, inv_b),
        "corr_abs_residual_vs_B": _corrcoef_safe(abs_res, b_vals),
    }


def _residual_heatmap(residual_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask.astype(bool) & np.isfinite(residual_map)
    h, w = residual_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    if not np.any(valid):
        return rgb
    vals = residual_map[valid].astype(np.float64)
    lim = float(np.percentile(np.abs(vals), 95))
    lim = max(lim, 1e-9)
    norm = np.zeros((h, w), dtype=np.float64)
    norm_vals = np.clip((vals / lim + 1.0) * 0.5, 0.0, 1.0)
    norm[valid] = norm_vals
    rgb[:, :, 0] = np.clip(255.0 * norm, 0.0, 255.0).astype(np.uint8)
    rgb[:, :, 2] = np.clip(255.0 * (1.0 - norm), 0.0, 255.0).astype(np.uint8)
    rgb[:, :, 1] = np.clip(255.0 * (1.0 - np.abs(norm - 0.5) * 2.0), 0.0, 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def _fft_image(power: np.ndarray) -> np.ndarray:
    p = np.maximum(power.astype(np.float64), 0.0)
    logp = np.log1p(p)
    vmax = float(np.max(logp)) if logp.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    norm = np.clip(logp / vmax, 0.0, 1.0)
    g = (norm * 255.0).astype(np.uint8)
    return np.stack([g, g, g], axis=2)


def _draw_direction_overlay(base_rgb: np.ndarray, peak_xy: tuple[int, int] | None, stripe_direction_deg: float) -> np.ndarray:
    img = Image.fromarray(base_rgb)
    draw = ImageDraw.Draw(img)
    h, w = base_rgb.shape[:2]
    cx, cy = w // 2, h // 2
    draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), outline=(255, 255, 64), width=1)
    if peak_xy is not None:
        px, py = peak_xy
        draw.line((cx, cy, px, py), fill=(255, 120, 64), width=1)
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), outline=(255, 120, 64), width=1)
    if math.isfinite(stripe_direction_deg):
        theta = math.radians(float(stripe_direction_deg))
        r = int(min(h, w) * 0.35)
        dx = int(round(math.cos(theta) * r))
        dy = int(round(math.sin(theta) * r))
        draw.line((cx - dx, cy + dy, cx + dx, cy - dy), fill=(64, 220, 120), width=2)
    draw.rectangle((8, 8, 360, 32), fill=(0, 0, 0))
    label = (
        "stripe_direction_deg="
        + (f"{stripe_direction_deg:.2f}" if math.isfinite(stripe_direction_deg) else "nan")
    )
    draw.text((12, 12), label, fill=(255, 255, 255))
    return np.asarray(img, dtype=np.uint8)


def _scatter_plot_b_vs_residual(
    b_vals: np.ndarray,
    abs_residual: np.ndarray,
    width: int = 960,
    height: int = 640,
) -> np.ndarray:
    img = Image.new("RGB", (width, height), (20, 22, 26))
    draw = ImageDraw.Draw(img)
    m_l, m_r, m_t, m_b = 70, 30, 40, 55
    draw.rectangle((m_l, m_t, width - m_r, height - m_b), outline=(120, 120, 120), width=1)
    if b_vals.size == 0 or abs_residual.size == 0:
        draw.text((16, 12), "No valid B/residual data", fill=(255, 255, 255))
        return np.asarray(img, dtype=np.uint8)
    b_min, b_max = float(np.min(b_vals)), float(np.max(b_vals))
    r_min, r_max = float(np.min(abs_residual)), float(np.max(abs_residual))
    if b_max <= b_min:
        b_max = b_min + 1.0
    if r_max <= r_min:
        r_max = r_min + 1.0
    n = min(5000, b_vals.size)
    if b_vals.size > n:
        idx = np.linspace(0, b_vals.size - 1, n, dtype=np.int64)
        b_draw = b_vals[idx]
        r_draw = abs_residual[idx]
    else:
        b_draw = b_vals
        r_draw = abs_residual
    x = m_l + (b_draw - b_min) / (b_max - b_min) * (width - m_l - m_r)
    y = (height - m_b) - (r_draw - r_min) / (r_max - r_min) * (height - m_t - m_b)
    for px, py in zip(x, y, strict=False):
        xi = int(round(float(px)))
        yi = int(round(float(py)))
        if m_l <= xi < (width - m_r) and m_t <= yi < (height - m_b):
            img.putpixel((xi, yi), (80, 180, 255))
    draw.text((16, 12), "B vs |plane residual|", fill=(235, 235, 235))
    draw.text((m_l, height - 36), f"B [{b_min:.3f}, {b_max:.3f}]", fill=(220, 220, 220))
    draw.text((m_l, 14), f"|residual| [{r_min:.6f}, {r_max:.6f}]", fill=(220, 220, 220))
    return np.asarray(img, dtype=np.uint8)


def compute_recon_qa(
    depth_map: np.ndarray,
    mask: np.ndarray,
    B_map: np.ndarray | None = None,
    depth_units: str = "unknown",
) -> dict[str, Any]:
    """
    Compute offline reconstruction QA metrics from depth and mask only.
    """
    if depth_map.shape != mask.shape:
        raise ValueError("depth_map and mask must have identical shape")
    if B_map is not None and B_map.shape != depth_map.shape:
        raise ValueError("B_map shape must match depth_map shape")
    mask_b = mask.astype(bool)
    coeff, residual, residual_map = _fit_plane(depth_map.astype(np.float64), mask_b)
    plane_fit = _plane_metrics(residual)
    stripe_diag, _, _ = _fft_metrics(residual_map, mask_b)
    snr = _snr_metrics(residual_map, mask_b, B_map)
    return {
        "depth_units": str(depth_units),
        "plane_fit": {**plane_fit, "units": str(depth_units)},
        "stripe_diagnostics": stripe_diag,
        "snr_relation": snr,
        "plane_model_z_eq_ax_by_c": {
            "a": float(coeff[0]),
            "b": float(coeff[1]),
            "c": float(coeff[2]),
        },
    }


def save_recon_qa_outputs(
    out_dir: Path,
    depth_map: np.ndarray,
    mask: np.ndarray,
    B_map: np.ndarray | None = None,
    depth_units: str = "unknown",
) -> dict[str, Any]:
    """
    Save qa_report.json and QA plots under reconstruction outputs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "qa_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if depth_map.shape != mask.shape:
        raise ValueError("depth_map and mask shape mismatch")
    if B_map is not None and B_map.shape != depth_map.shape:
        raise ValueError("B_map shape mismatch")

    mask_b = mask.astype(bool)
    coeff, residual, residual_map = _fit_plane(depth_map.astype(np.float64), mask_b)
    plane_fit = _plane_metrics(residual)
    stripe_diag, fft_power, peak_xy = _fft_metrics(residual_map, mask_b)
    snr = _snr_metrics(residual_map, mask_b, B_map)

    heatmap = _residual_heatmap(residual_map, mask_b)
    Image.fromarray(heatmap).save(plots_dir / "plane_fit_residuals.png")

    fft_rgb = _fft_image(fft_power)
    Image.fromarray(fft_rgb).save(plots_dir / "depth_fft.png")

    direction = float(stripe_diag.get("stripe_direction_deg", float("nan")))
    stripe_overlay = _draw_direction_overlay(fft_rgb, peak_xy, direction)
    Image.fromarray(stripe_overlay).save(plots_dir / "stripe_direction_estimate.png")

    b_plot_written = False
    if B_map is not None:
        valid_b = (
            mask_b
            & np.isfinite(residual_map)
            & np.isfinite(B_map)
            & (B_map > 0)
        )
        b_vals = B_map[valid_b].astype(np.float64)
        abs_res = np.abs(residual_map[valid_b].astype(np.float64))
        scatter = _scatter_plot_b_vs_residual(b_vals, abs_res)
        Image.fromarray(scatter).save(plots_dir / "b_vs_depth_residual.png")
        b_plot_written = True

    report = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(),
        "depth_units": str(depth_units),
        "plane_fit": {**plane_fit, "units": str(depth_units)},
        "stripe_diagnostics": stripe_diag,
        "snr_relation": snr,
        "plane_model_z_eq_ax_by_c": {
            "a": float(coeff[0]),
            "b": float(coeff[1]),
            "c": float(coeff[2]),
        },
        "inputs": {
            "shape_hw": [int(depth_map.shape[0]), int(depth_map.shape[1])],
            "valid_depth_pixels": int(np.count_nonzero(mask_b & np.isfinite(depth_map))),
            "b_map_used": bool(B_map is not None),
        },
        "plots": {
            "plane_fit_residuals": "qa_plots/plane_fit_residuals.png",
            "depth_fft": "qa_plots/depth_fft.png",
            "stripe_direction_estimate": "qa_plots/stripe_direction_estimate.png",
            "b_vs_depth_residual": "qa_plots/b_vs_depth_residual.png" if b_plot_written else None,
        },
    }
    (out_dir / "qa_report.json").write_text(json.dumps(_json_safe(report), indent=2))
    return report
