"""Object ROI detection on black background."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np


@dataclass(slots=True)
class ObjectRoiConfig:
    downscale_max_w: int = 640
    blur_ksize: int = 7
    black_bg_percentile: float = 70.0
    threshold_offset: float = 10.0
    min_area_ratio: float = 0.01
    max_area_ratio: float = 0.95
    close_iters: int = 2
    open_iters: int = 1
    fill_holes: bool = True


@dataclass(slots=True)
class ObjectRoiResult:
    roi_mask: np.ndarray
    bbox: Optional[Tuple[int, int, int, int]]
    debug: Dict[str, Any]


def detect_object_roi(image: np.ndarray, cfg: ObjectRoiConfig) -> ObjectRoiResult:
    gray = _to_gray(image)
    h, w = gray.shape
    scale = 1.0
    if w > cfg.downscale_max_w:
        scale = cfg.downscale_max_w / float(w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray_ds = _resize_nearest(gray, new_w, new_h)
    else:
        gray_ds = gray

    if cfg.blur_ksize >= 3:
        gray_ds = _box_blur(gray_ds, cfg.blur_ksize)

    bg = float(np.percentile(gray_ds, cfg.black_bg_percentile))
    thresh = bg + cfg.threshold_offset
    binary = gray_ds > thresh

    mask = binary
    for _ in range(cfg.close_iters):
        mask = _erode(_dilate(mask))
    for _ in range(cfg.open_iters):
        mask = _dilate(_erode(mask))

    cc_mask, bbox, area_ratio = _largest_component(mask)
    roi_fallback = False
    if area_ratio < cfg.min_area_ratio or area_ratio > cfg.max_area_ratio or cc_mask is None:
        roi_fallback = True
        roi_mask = np.ones_like(gray, dtype=bool)
        bbox_full = None
    else:
        if cfg.fill_holes:
            cc_mask = _fill_holes(cc_mask)
        roi_mask = _resize_nearest(cc_mask.astype(np.uint8), w, h).astype(bool)
        if bbox is not None:
            x, y, bw, bh = bbox
            if scale != 1.0:
                x = int(x / scale)
                y = int(y / scale)
                bw = int(bw / scale)
                bh = int(bh / scale)
            bbox_full = (x, y, bw, bh)
        else:
            bbox_full = None

    debug = {
        "bg_percentile": cfg.black_bg_percentile,
        "bg_value": bg,
        "threshold_offset": cfg.threshold_offset,
        "threshold": thresh,
        "area_ratio": area_ratio,
        "roi_fallback": roi_fallback,
    }

    return ObjectRoiResult(roi_mask=roi_mask, bbox=bbox_full, debug=debug)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim == 3 and image.shape[2] == 3:
        img = image.astype(np.float32)
        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    raise ValueError("Unsupported image format")


def _resize_nearest(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    y_idx = (np.linspace(0, img.shape[0] - 1, new_h)).astype(np.int32)
    x_idx = (np.linspace(0, img.shape[1] - 1, new_w)).astype(np.int32)
    return img[np.ix_(y_idx, x_idx)]


def _box_blur(img: np.ndarray, k: int) -> np.ndarray:
    k = max(3, k | 1)
    pad = k // 2
    padded = np.pad(img, pad, mode="edge")
    out = np.zeros_like(img)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            patch = padded[y:y + k, x:x + k]
            out[y, x] = patch.mean()
    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask)
    for y in range(h):
        for x in range(w):
            y0 = max(0, y - 1)
            y1 = min(h, y + 2)
            x0 = max(0, x - 1)
            x1 = min(w, x + 2)
            out[y, x] = np.any(mask[y0:y1, x0:x1])
    return out


def _erode(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask)
    for y in range(h):
        for x in range(w):
            y0 = max(0, y - 1)
            y1 = min(h, y + 2)
            x0 = max(0, x - 1)
            x1 = min(w, x + 2)
            out[y, x] = np.all(mask[y0:y1, x0:x1])
    return out


def _largest_component(mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], float]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_size = 0
    best_mask = None
    best_bbox = None
    total = float(h * w)

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            q = [(y, x)]
            visited[y, x] = True
            coords = []
            while q:
                cy, cx = q.pop()
                coords.append((cy, cx))
                if cy > 0 and mask[cy - 1, cx] and not visited[cy - 1, cx]:
                    visited[cy - 1, cx] = True
                    q.append((cy - 1, cx))
                if cy + 1 < h and mask[cy + 1, cx] and not visited[cy + 1, cx]:
                    visited[cy + 1, cx] = True
                    q.append((cy + 1, cx))
                if cx > 0 and mask[cy, cx - 1] and not visited[cy, cx - 1]:
                    visited[cy, cx - 1] = True
                    q.append((cy, cx - 1))
                if cx + 1 < w and mask[cy, cx + 1] and not visited[cy, cx + 1]:
                    visited[cy, cx + 1] = True
                    q.append((cy, cx + 1))
            size = len(coords)
            if size > best_size:
                best_size = size
                comp = np.zeros_like(mask)
                ys = [c[0] for c in coords]
                xs = [c[1] for c in coords]
                comp[ys, xs] = True
                best_mask = comp
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                best_bbox = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

    area_ratio = best_size / total if total > 0 else 0.0
    return best_mask, best_bbox, area_ratio


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    inv = ~mask
    visited = np.zeros_like(mask, dtype=bool)
    stack = []
    for x in range(w):
        if inv[0, x]:
            stack.append((0, x))
        if inv[h - 1, x]:
            stack.append((h - 1, x))
    for y in range(h):
        if inv[y, 0]:
            stack.append((y, 0))
        if inv[y, w - 1]:
            stack.append((y, w - 1))
    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w:
            continue
        if visited[y, x] or not inv[y, x]:
            continue
        visited[y, x] = True
        stack.extend([(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)])
    holes = inv & (~visited)
    return mask | holes
