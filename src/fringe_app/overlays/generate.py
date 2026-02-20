"""Overlay generation for ROI and masks."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    h, w = mask.shape
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    neighborhoods = [
        p[0:h, 0:w], p[0:h, 1:w + 1], p[0:h, 2:w + 2],
        p[1:h + 1, 0:w], p[1:h + 1, 1:w + 1], p[1:h + 1, 2:w + 2],
        p[2:h + 2, 0:w], p[2:h + 2, 1:w + 1], p[2:h + 2, 2:w + 2],
    ]
    eroded = np.logical_and.reduce(neighborhoods)
    return mask & (~eroded)


def overlay_masks(base: np.ndarray, roi: np.ndarray, valid: np.ndarray, out_path: Path) -> None:
    base_rgb = base
    if base_rgb.ndim == 2:
        base_rgb = np.repeat(base_rgb[:, :, None], 3, axis=2)
    out = base_rgb.copy().astype(np.uint8)

    roi_edge = _boundary(roi)
    valid_edge = _boundary(valid)
    both_edge = _boundary(roi & valid)

    out[roi_edge] = [0, 255, 0]
    out[valid_edge] = [0, 0, 255]
    out[both_edge] = [255, 0, 0]

    Image.fromarray(out).save(out_path)


def overlay_clipping(base: np.ndarray, roi: np.ndarray, clipped: np.ndarray, out_path: Path) -> None:
    """
    Overlay clipping diagnostics:
    - ROI boundary in green
    - Clipped-any pixels in red tint
    """
    base_rgb = base
    if base_rgb.ndim == 2:
        base_rgb = np.repeat(base_rgb[:, :, None], 3, axis=2)
    out = base_rgb.copy().astype(np.uint8)

    roi_edge = _boundary(roi)
    out[roi_edge] = [0, 255, 0]

    clipped = clipped.astype(bool)
    if np.any(clipped):
        out_clip = out[clipped].astype(np.int16)
        # Red tint while preserving luminance detail.
        out_clip[:, 0] = np.clip(out_clip[:, 0] + 120, 0, 255)
        out_clip[:, 1] = np.clip(out_clip[:, 1] * 0.5, 0, 255)
        out_clip[:, 2] = np.clip(out_clip[:, 2] * 0.5, 0, 255)
        out[clipped] = out_clip.astype(np.uint8)

    Image.fromarray(out).save(out_path)
