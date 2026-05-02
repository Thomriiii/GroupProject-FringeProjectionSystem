"""Shared utilities for height-based defect segmentation and features."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from fringe_app_v2.utils.io import save_image


@dataclass(frozen=True, slots=True)
class Region:
    label: int
    pixel_count: int
    bbox_xywh: tuple[int, int, int, int]
    rows: np.ndarray
    cols: np.ndarray


def apply_mask_nan(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    out[~np.asarray(mask, dtype=bool)] = np.nan
    return out


def gaussian_smooth_nan(values: np.ndarray, sigma: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if sigma <= 0:
        return arr.copy()

    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0).astype(np.float32)
    weights = valid.astype(np.float32)
    try:
        from scipy.ndimage import gaussian_filter

        numerator = gaussian_filter(filled, sigma=float(sigma), mode="nearest")
        denominator = gaussian_filter(weights, sigma=float(sigma), mode="nearest")
    except Exception:
        kernel = _gaussian_kernel(float(sigma))
        numerator = _convolve_separable(filled, kernel)
        denominator = _convolve_separable(weights, kernel)

    smoothed = np.full(arr.shape, np.nan, dtype=np.float32)
    np.divide(numerator, denominator, out=smoothed, where=denominator > 1e-6)
    return smoothed


def cleanup_mask(mask: np.ndarray, valid_mask: np.ndarray, min_area: int, open_radius: int, close_radius: int) -> np.ndarray:
    cleaned = np.asarray(mask, dtype=bool) & np.asarray(valid_mask, dtype=bool)
    if open_radius > 0:
        cleaned = binary_open(cleaned, open_radius)
    if close_radius > 0:
        cleaned = binary_close(cleaned, close_radius)
    cleaned &= np.asarray(valid_mask, dtype=bool)
    return remove_small_components(cleaned, min_area)


def connected_components(mask: np.ndarray) -> list[Region]:
    src = np.asarray(mask, dtype=bool)
    try:
        from scipy import ndimage

        labels, count = ndimage.label(src, structure=np.ones((3, 3), dtype=np.uint8))
        slices = ndimage.find_objects(labels)
        regions: list[Region] = []
        for label in range(1, int(count) + 1):
            component_slice = slices[label - 1]
            if component_slice is None:
                continue
            local = labels[component_slice] == label
            rows_local, cols_local = np.where(local)
            y0 = int(component_slice[0].start)
            x0 = int(component_slice[1].start)
            rows = rows_local.astype(np.int32) + y0
            cols = cols_local.astype(np.int32) + x0
            regions.append(_region_from_pixels(label, rows, cols))
        return regions
    except Exception:
        return _connected_components_python(src)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    if min_area <= 1:
        return src.copy()
    out = np.zeros_like(src, dtype=bool)
    for region in connected_components(src):
        if region.pixel_count >= min_area:
            out[region.rows, region.cols] = True
    return out


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return src.copy()
    out = np.zeros_like(src, dtype=bool)
    for dy, dx in _disk_offsets(radius):
        out |= _shift_bool(src, dy, dx, fill=False)
    return out


def binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return src.copy()
    out = np.ones_like(src, dtype=bool)
    for dy, dx in _disk_offsets(radius):
        out &= _shift_bool(src, dy, dx, fill=False)
    return out


def binary_open(mask: np.ndarray, radius: int) -> np.ndarray:
    return binary_dilate(binary_erode(mask, radius), radius)


def binary_close(mask: np.ndarray, radius: int) -> np.ndarray:
    return binary_erode(binary_dilate(mask, radius), radius)


def normalise_to_u8(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    valid = finite if mask is None else finite & np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(arr[valid], [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(arr[valid]))
        hi = float(np.nanmax(arr[valid]))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    scaled[~finite] = 0.0
    return np.rint(scaled * 255.0).astype(np.uint8)


def save_overlay(path: Path, base: np.ndarray, defect_mask: np.ndarray, valid_mask: np.ndarray) -> None:
    base_u8 = normalise_to_u8(base, valid_mask)
    rgb = np.repeat(base_u8[:, :, None], 3, axis=2).astype(np.float32)
    rgb[~np.asarray(valid_mask, dtype=bool)] *= 0.25
    defects = np.asarray(defect_mask, dtype=bool)
    red = np.array([255.0, 32.0, 32.0], dtype=np.float32)
    rgb[defects] = 0.35 * rgb[defects] + 0.65 * red
    save_image(path, np.clip(np.rint(rgb), 0, 255).astype(np.uint8))


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return (kernel / np.sum(kernel)).astype(np.float32)


def _convolve_separable(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return _convolve_axis(_convolve_axis(values, kernel, axis=1), kernel, axis=0)


def _convolve_axis(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    pad_width = [(0, 0)] * values.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(values, pad_width, mode="edge")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), axis, padded).astype(np.float32)


def _disk_offsets(radius: int) -> Iterable[tuple[int, int]]:
    r2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= r2:
                yield dy, dx


def _shift_bool(mask: np.ndarray, dy: int, dx: int, fill: bool) -> np.ndarray:
    h, w = mask.shape
    out = np.full((h, w), fill, dtype=bool)
    y_src0 = max(0, -dy)
    y_src1 = h - max(0, dy)
    x_src0 = max(0, -dx)
    x_src1 = w - max(0, dx)
    if y_src1 <= y_src0 or x_src1 <= x_src0:
        return out
    y_dst0 = max(0, dy)
    y_dst1 = y_dst0 + (y_src1 - y_src0)
    x_dst0 = max(0, dx)
    x_dst1 = x_dst0 + (x_src1 - x_src0)
    out[y_dst0:y_dst1, x_dst0:x_dst1] = mask[y_src0:y_src1, x_src0:x_src1]
    return out


def _connected_components_python(mask: np.ndarray) -> list[Region]:
    visited = np.zeros_like(mask, dtype=bool)
    regions: list[Region] = []
    label = 0
    height, width = mask.shape
    for row0, col0 in zip(*np.where(mask & ~visited), strict=False):
        if visited[row0, col0]:
            continue
        label += 1
        queue: deque[tuple[int, int]] = deque([(int(row0), int(col0))])
        visited[row0, col0] = True
        rows: list[int] = []
        cols: list[int] = []
        while queue:
            row, col = queue.popleft()
            rows.append(row)
            cols.append(col)
            for next_row in range(max(0, row - 1), min(height, row + 2)):
                for next_col in range(max(0, col - 1), min(width, col + 2)):
                    if mask[next_row, next_col] and not visited[next_row, next_col]:
                        visited[next_row, next_col] = True
                        queue.append((next_row, next_col))
        regions.append(_region_from_pixels(label, np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)))
    return regions


def _region_from_pixels(label: int, rows: np.ndarray, cols: np.ndarray) -> Region:
    x_min = int(np.min(cols))
    x_max = int(np.max(cols))
    y_min = int(np.min(rows))
    y_max = int(np.max(rows))
    return Region(
        label=int(label),
        pixel_count=int(rows.size),
        bbox_xywh=(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1),
        rows=rows.astype(np.int32, copy=False),
        cols=cols.astype(np.int32, copy=False),
    )
