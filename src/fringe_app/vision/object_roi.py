"""Object ROI detection on black background."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, Literal

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
    ref_method: Literal["frame0", "mean_over_frames", "median_over_frames", "max_over_frames"] = "median_over_frames"
    post_enabled: bool = True
    post_keep_largest_component: bool = True
    post_fill_small_holes: bool = True
    post_max_hole_area: int = 2000
    post_dilate_radius_px: int = 10


@dataclass(slots=True)
class ObjectRoiResult:
    roi_mask: np.ndarray
    bbox: Optional[Tuple[int, int, int, int]]
    raw_mask: np.ndarray
    post_mask: np.ndarray
    debug: Dict[str, Any]


def build_reference_from_stack(
    stack_u8: np.ndarray,
    ref_method: Literal["frame0", "mean_over_frames", "median_over_frames", "max_over_frames"] = "median_over_frames",
) -> np.ndarray:
    """
    Build a deterministic uint8 reference image from a grayscale frame stack.
    stack_u8 must be shape (N, H, W).
    """
    if stack_u8.ndim != 3:
        raise ValueError("stack_u8 must be (N,H,W)")
    if stack_u8.shape[0] == 0:
        raise ValueError("stack_u8 must contain at least one frame")

    if ref_method == "frame0":
        ref = stack_u8[0].astype(np.float32)
    elif ref_method == "mean_over_frames":
        ref = np.mean(stack_u8.astype(np.float32), axis=0)
    elif ref_method == "median_over_frames":
        ref = np.median(stack_u8.astype(np.float32), axis=0)
    elif ref_method == "max_over_frames":
        ref = np.max(stack_u8.astype(np.float32), axis=0)
    else:
        raise ValueError(f"Unknown ref_method: {ref_method}")

    return np.clip(np.rint(ref), 0, 255).astype(np.uint8)


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
        raw_mask_full = roi_mask.copy()
        post_mask_full = roi_mask.copy()
        bbox_full = None
    else:
        if cfg.fill_holes:
            cc_mask = _fill_holes(cc_mask)
        raw_ds = cc_mask.astype(bool).copy()
        if cfg.post_enabled:
            post_ds = raw_ds.copy()
            if cfg.post_keep_largest_component:
                lc, _, _ = _largest_component(post_ds)
                if lc is not None:
                    post_ds = lc
            if cfg.post_fill_small_holes:
                post_ds = _fill_small_holes(post_ds, int(max(1, cfg.post_max_hole_area)))
            if int(cfg.post_dilate_radius_px) > 0:
                for _ in range(int(cfg.post_dilate_radius_px)):
                    post_ds = _dilate(post_ds)
            post_ds, _, _ = _largest_component(post_ds)
            if post_ds is None:
                post_ds = raw_ds
        else:
            post_ds = raw_ds
        raw_mask_full = _resize_nearest(raw_ds.astype(np.uint8), w, h).astype(bool)
        post_mask_full = _resize_nearest(post_ds.astype(np.uint8), w, h).astype(bool)
        roi_mask = post_mask_full
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
        "post": {
            "enabled": bool(cfg.post_enabled),
            "keep_largest_component": bool(cfg.post_keep_largest_component),
            "fill_small_holes": bool(cfg.post_fill_small_holes),
            "max_hole_area": int(cfg.post_max_hole_area),
            "dilate_radius_px": int(cfg.post_dilate_radius_px),
        },
    }

    return ObjectRoiResult(
        roi_mask=roi_mask,
        bbox=bbox_full,
        raw_mask=raw_mask_full,
        post_mask=post_mask_full,
        debug=debug,
    )


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
    # Integral image for fast box blur.
    integ = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    h, w = img.shape
    y0 = np.arange(h)
    x0 = np.arange(w)
    y1 = y0 + k
    x1 = x0 + k
    sum_box = (
        integ[y1[:, None], x1]
        - integ[y0[:, None], x1]
        - integ[y1[:, None], x0]
        + integ[y0[:, None], x0]
    )
    return (sum_box / float(k * k)).astype(img.dtype)


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    neighborhoods = [
        p[0:h, 0:w], p[0:h, 1:w + 1], p[0:h, 2:w + 2],
        p[1:h + 1, 0:w], p[1:h + 1, 1:w + 1], p[1:h + 1, 2:w + 2],
        p[2:h + 2, 0:w], p[2:h + 2, 1:w + 1], p[2:h + 2, 2:w + 2],
    ]
    return np.logical_or.reduce(neighborhoods)


def _erode(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    neighborhoods = [
        p[0:h, 0:w], p[0:h, 1:w + 1], p[0:h, 2:w + 2],
        p[1:h + 1, 0:w], p[1:h + 1, 1:w + 1], p[1:h + 1, 2:w + 2],
        p[2:h + 2, 0:w], p[2:h + 2, 1:w + 1], p[2:h + 2, 2:w + 2],
    ]
    return np.logical_and.reduce(neighborhoods)


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


def _fill_small_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    h, w = mask.shape
    inv = ~mask
    visited = np.zeros_like(mask, dtype=bool)
    out = mask.copy()
    for y in range(h):
        for x in range(w):
            if not inv[y, x] or visited[y, x]:
                continue
            q = [(y, x)]
            visited[y, x] = True
            comp = []
            touches_border = False
            while q:
                cy, cx = q.pop()
                comp.append((cy, cx))
                if cy == 0 or cy == h - 1 or cx == 0 or cx == w - 1:
                    touches_border = True
                if cy > 0 and inv[cy - 1, cx] and not visited[cy - 1, cx]:
                    visited[cy - 1, cx] = True
                    q.append((cy - 1, cx))
                if cy + 1 < h and inv[cy + 1, cx] and not visited[cy + 1, cx]:
                    visited[cy + 1, cx] = True
                    q.append((cy + 1, cx))
                if cx > 0 and inv[cy, cx - 1] and not visited[cy, cx - 1]:
                    visited[cy, cx - 1] = True
                    q.append((cy, cx - 1))
                if cx + 1 < w and inv[cy, cx + 1] and not visited[cy, cx + 1]:
                    visited[cy, cx + 1] = True
                    q.append((cy, cx + 1))
            if (not touches_border) and len(comp) <= max_hole_area:
                ys = [p[0] for p in comp]
                xs = [p[1] for p in comp]
                out[ys, xs] = True
    return out
