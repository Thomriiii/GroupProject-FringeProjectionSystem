"""Phase visualization utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


def _save_png(array_u8: np.ndarray, path: str) -> None:
    if Image is None:
        raise RuntimeError("PIL not available for PNG output")
    Image.fromarray(array_u8).save(path)


def save_phase_png_autoscale(phi: np.ndarray, mask: np.ndarray, percentiles: Tuple[float, float], path: str) -> None:
    valid = mask & np.isfinite(phi)
    vals = phi[valid]
    if vals.size:
        p_low, p_high = np.percentile(vals, percentiles)
    else:
        p_low, p_high = -np.pi, np.pi
    phi_clip = np.clip(phi, p_low, p_high)
    norm = (phi_clip - p_low) / (p_high - p_low + 1e-9)
    norm[~valid] = 0.0
    img_u8 = (norm * 255).astype(np.uint8)
    _save_png(img_u8, path)


def save_phase_png_fixed(phi: np.ndarray, mask: np.ndarray, path: str) -> None:
    phi_clip = np.clip(phi, -np.pi, np.pi)
    norm = (phi_clip + np.pi) / (2.0 * np.pi)
    valid = mask & np.isfinite(phi_clip)
    norm[~valid] = 0.0
    img_u8 = (norm * 255).astype(np.uint8)
    _save_png(img_u8, path)


def save_mask_png(mask: np.ndarray, path: str) -> None:
    img_u8 = (mask.astype(np.uint8) * 255)
    _save_png(img_u8, path)


def save_modulation_png(B: np.ndarray, mask: np.ndarray, path: str) -> None:
    valid = mask & np.isfinite(B)
    vals = B[valid]
    if vals.size:
        lo, hi = np.percentile(vals, (1, 99))
    else:
        lo, hi = 0.0, 1.0
    B_clip = np.clip(B, lo, hi)
    norm = (B_clip - lo) / (hi - lo + 1e-9)
    norm[~valid] = 0.0
    img_u8 = (norm * 255).astype(np.uint8)
    _save_png(img_u8, path)
