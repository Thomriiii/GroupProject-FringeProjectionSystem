"""Multi-frequency temporal phase unwrapping."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def unwrap_multi_frequency(
    phases: list[np.ndarray],
    masks: list[np.ndarray],
    freqs: list[float],
    roi_mask: np.ndarray | None = None,
    use_roi: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    Temporal unwrapping coarse->fine.

    Uses rounding: k = round((r*Phi_low - phi_high) / (2*pi))
    """
    if len(phases) < 2:
        raise ValueError("Need at least two frequencies for temporal unwrapping")

    order = np.argsort(freqs)
    phases = [phases[i] for i in order]
    masks = [masks[i] for i in order]
    freqs = [freqs[i] for i in order]

    two_pi = 2.0 * np.pi
    phi_low = phases[0].astype(np.float32)
    mask_low = masks[0].astype(bool)
    if use_roi and roi_mask is not None:
        mask_low = mask_low & roi_mask
    # Lowest frequency is assumed wrap-free in temporal PU; shift to [0, 2*pi)
    # so higher-stage unwrapped phases have a stable positive anchor.
    phi_low = np.mod(phi_low, two_pi).astype(np.float32)
    phi_low[~mask_low] = np.nan

    stage_info = []
    residual_final = None
    for i in range(1, len(freqs)):
        phi_high = phases[i].astype(np.float32)
        mask_high = masks[i].astype(bool)
        mask_pair = mask_low & mask_high
        if use_roi and roi_mask is not None:
            mask_pair = mask_pair & roi_mask

        r = freqs[i] / freqs[i - 1]
        Phi_high = np.full_like(phi_high, np.nan, dtype=np.float32)
        if np.any(mask_pair):
            k = np.round((r * phi_low[mask_pair] - phi_high[mask_pair]) / two_pi).astype(np.float32)
            Phi_high[mask_pair] = phi_high[mask_pair] + two_pi * k
        Phi_high[~mask_pair] = np.nan

        residual = np.full_like(phi_high, np.nan, dtype=np.float32)
        if np.any(mask_pair):
            residual[mask_pair] = wrap_to_pi(r * phi_low[mask_pair] - Phi_high[mask_pair]).astype(np.float32)
        residual_final = residual

        phi_low = Phi_high
        mask_low = np.isfinite(Phi_high)
        if use_roi and roi_mask is not None:
            mask_low &= roi_mask

        valid_ratio = float(np.count_nonzero(mask_pair) / mask_pair.size)
        stage_info.append({
            "f_low": freqs[i - 1],
            "f_high": freqs[i],
            "ratio": r,
            "pair_valid_ratio": valid_ratio,
            "pair_valid_px": int(np.count_nonzero(mask_pair)),
        })

    phi_abs = phi_low.astype(np.float32)
    mask_unwrap = np.isfinite(phi_abs)
    if use_roi and roi_mask is not None:
        mask_unwrap &= roi_mask
        phi_abs[~mask_unwrap] = np.nan

    if residual_final is None:
        residual_final = np.full_like(phi_abs, np.nan, dtype=np.float32)
    residual_final[~mask_unwrap] = np.nan

    finite = phi_abs[np.isfinite(phi_abs)]
    residual_vals = np.abs(residual_final[np.isfinite(residual_final)])
    residual_p95_thresh = 0.2
    meta = {
        "frequencies": freqs,
        "stages": stage_info,
        "phi_min": float(np.min(finite)) if finite.size else None,
        "phi_max": float(np.max(finite)) if finite.size else None,
        "valid_ratio": float(np.count_nonzero(mask_unwrap) / mask_unwrap.size),
        "unwrap_valid_px": int(np.count_nonzero(mask_unwrap)),
        "residual_rms": float(np.sqrt(np.mean(np.square(residual_vals)))) if residual_vals.size else None,
        "residual_p50": float(np.percentile(residual_vals, 50)) if residual_vals.size else None,
        "residual_p75": float(np.percentile(residual_vals, 75)) if residual_vals.size else None,
        "residual_p95": float(np.percentile(residual_vals, 95)) if residual_vals.size else None,
        "residual_max_abs": float(np.max(residual_vals)) if residual_vals.size else None,
        "residual_gt_1rad_pct": float(np.mean(residual_vals > 1.0) * 100.0) if residual_vals.size else None,
        "residual_p95_threshold": residual_p95_thresh,
        "residual_ok": bool(np.percentile(residual_vals, 95) < residual_p95_thresh) if residual_vals.size else False,
    }
    return phi_abs, mask_unwrap, meta, residual_final


def _save_debug_images(phi_abs: np.ndarray, mask: np.ndarray, out_dir: Path, f_max: float) -> None:
    out_dir.mkdir(exist_ok=True, parents=True)
    mask = mask.astype(bool)
    valid = mask & np.isfinite(phi_abs)

    # Fixed scaling: map modulo 2*pi*f_max to 0..255
    two_pi = 2.0 * np.pi
    span = two_pi * float(f_max)
    if span <= 0:
        span = two_pi
    phi_mod = np.mod(phi_abs, span)
    img_fixed = np.zeros_like(phi_abs, dtype=np.uint8)
    img_fixed[valid] = np.clip((phi_mod[valid] / span) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img_fixed).save(out_dir / "phi_abs_debug_fixed.png")

    # Autoscale based on percentiles
    if np.any(valid):
        vals = phi_abs[valid]
        lo, hi = np.percentile(vals, [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        img_auto = np.zeros_like(phi_abs, dtype=np.uint8)
        img_auto[valid] = np.clip(((phi_abs[valid] - lo) / (hi - lo)) * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(img_auto).save(out_dir / "phi_abs_debug_autoscale.png")


def _save_residual_debug_images(residual: np.ndarray, out_dir: Path) -> None:
    valid = np.isfinite(residual)
    # Fixed scaling in [-pi, pi]
    clipped = np.clip(residual, -np.pi, np.pi)
    fixed = np.zeros_like(residual, dtype=np.uint8)
    fixed[valid] = np.clip(((clipped[valid] + np.pi) / (2.0 * np.pi)) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(fixed).save(out_dir / "residual_debug_fixed.png")

    if np.any(valid):
        vals = residual[valid]
        lo, hi = np.percentile(vals, [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        auto = np.zeros_like(residual, dtype=np.uint8)
        auto[valid] = np.clip(((residual[valid] - lo) / (hi - lo)) * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(auto).save(out_dir / "residual_debug_autoscale.png")


def save_unwrap_outputs(
    run_dir: Path,
    phi_abs: np.ndarray,
    mask_unwrap: np.ndarray,
    meta: dict,
    f_max: float,
    residual: np.ndarray | None = None,
) -> None:
    unwrap_dir = run_dir / "unwrap"
    unwrap_dir.mkdir(exist_ok=True)
    np.save(unwrap_dir / "phi_abs.npy", phi_abs.astype(np.float32))
    np.save(unwrap_dir / "mask_unwrap.npy", mask_unwrap.astype(np.bool_))
    if residual is not None:
        np.save(unwrap_dir / "residual.npy", residual.astype(np.float32))
    (unwrap_dir / "unwrap_meta.json").write_text(json.dumps(meta, indent=2))
    _save_debug_images(phi_abs, mask_unwrap, unwrap_dir, f_max)
    if residual is not None:
        _save_residual_debug_images(residual, unwrap_dir)
