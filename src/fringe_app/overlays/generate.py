"""Overlay generation for ROI and masks."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    h, w = mask.shape
    eroded = np.zeros_like(mask)
    for y in range(h):
        for x in range(w):
            y0 = max(0, y - 1)
            y1 = min(h, y + 2)
            x0 = max(0, x - 1)
            x1 = min(w, x + 2)
            eroded[y, x] = np.all(mask[y0:y1, x0:x1])
    return mask ^ eroded


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
