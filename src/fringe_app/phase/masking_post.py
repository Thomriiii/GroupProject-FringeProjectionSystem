"""Post-processing utilities for phase masks."""

from __future__ import annotations

from collections import deque

import numpy as np


def _component_from(seed_y: int, seed_x: int, mask: np.ndarray, visited: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    q: deque[tuple[int, int]] = deque([(seed_y, seed_x)])
    visited[seed_y, seed_x] = True
    comp: list[tuple[int, int]] = []
    while q:
        y, x = q.popleft()
        comp.append((y, x))
        if y > 0 and mask[y - 1, x] and not visited[y - 1, x]:
            visited[y - 1, x] = True
            q.append((y - 1, x))
        if y + 1 < h and mask[y + 1, x] and not visited[y + 1, x]:
            visited[y + 1, x] = True
            q.append((y + 1, x))
        if x > 0 and mask[y, x - 1] and not visited[y, x - 1]:
            visited[y, x - 1] = True
            q.append((y, x - 1))
        if x + 1 < w and mask[y, x + 1] and not visited[y, x + 1]:
            visited[y, x + 1] = True
            q.append((y, x + 1))
    return comp


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    out = mask.copy()
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if not out[y, x] or visited[y, x]:
                continue
            comp = _component_from(y, x, out, visited)
            if len(comp) < min_area:
                for cy, cx in comp:
                    out[cy, cx] = False
    return out


def _fill_small_holes(mask: np.ndarray, domain: np.ndarray, max_hole_area: int) -> np.ndarray:
    """
    Fill enclosed invalid regions bounded by valid pixels, constrained to domain.
    """
    out = mask.copy()
    holes = domain & (~mask)
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if not holes[y, x] or visited[y, x]:
                continue
            comp = _component_from(y, x, holes, visited)
            touches_border = False
            for cy, cx in comp:
                if cy == 0 or cy == h - 1 or cx == 0 or cx == w - 1:
                    touches_border = True
                    break
                if not domain[cy, cx]:
                    touches_border = True
                    break
            if not touches_border and len(comp) <= max_hole_area:
                for cy, cx in comp:
                    out[cy, cx] = True
    return out


def cleanup_mask(
    mask: np.ndarray,
    roi_mask: np.ndarray | None,
    min_component_area: int = 200,
    fill_small_holes: bool = True,
    max_hole_area: int = 200,
) -> np.ndarray:
    """
    Remove small connected components and optionally fill tiny enclosed holes.

    Cleanup is confined to ROI if provided and never expands outside ROI.
    """
    base = mask.astype(bool)
    if roi_mask is not None:
        roi = roi_mask.astype(bool)
        domain = roi
        work = base & roi
    else:
        roi = None
        domain = np.ones_like(base, dtype=bool)
        work = base.copy()

    cleaned = _remove_small_components(work, int(max(min_component_area, 1)))
    if fill_small_holes:
        cleaned = _fill_small_holes(cleaned, domain, int(max(max_hole_area, 1)))

    if roi is not None:
        out = base.copy()
        out[roi] = cleaned[roi]
        out[~roi] = False
        return out
    return cleaned


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    best: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            comp = _component_from(y, x, mask, visited)
            if len(comp) > len(best):
                best = comp
    out = np.zeros_like(mask, dtype=bool)
    if best:
        ys = [p[0] for p in best]
        xs = [p[1] for p in best]
        out[ys, xs] = True
    return out


def _neighbors_or(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    n = [
        p[0:h, 0:w], p[0:h, 1:w + 1], p[0:h, 2:w + 2],
        p[1:h + 1, 0:w], p[1:h + 1, 1:w + 1], p[1:h + 1, 2:w + 2],
        p[2:h + 2, 0:w], p[2:h + 2, 1:w + 1], p[2:h + 2, 2:w + 2],
    ]
    return np.logical_or.reduce(n)


def binary_dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(max(0, int(r))):
        out = _neighbors_or(out)
    return out


def binary_erode(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(max(0, int(r))):
        out = ~_neighbors_or(~out)
    return out


def binary_close(mask: np.ndarray, r: int) -> np.ndarray:
    if int(r) <= 0:
        return mask.astype(bool)
    return binary_erode(binary_dilate(mask, r), r)


def build_unwrap_mask(
    mask_clean: np.ndarray,
    roi_mask: np.ndarray | None,
    clipped_any: np.ndarray | None,
    cfg: dict,
) -> np.ndarray:
    """
    Conservative mask for unwrapping:
    - start from clean mask
    - constrain to ROI if available
    - exclude clipped pixels
    - optionally close small bites
    - keep largest component
    - erode unreliable boundaries
    """
    m = mask_clean.astype(bool).copy()
    if roi_mask is not None:
        m &= roi_mask.astype(bool)
    if clipped_any is not None:
        m &= ~clipped_any.astype(bool)

    if int(cfg.get("closing_radius_px", 0)) > 0:
        m = binary_close(m, int(cfg.get("closing_radius_px", 0)))
        if roi_mask is not None:
            m &= roi_mask.astype(bool)
        if clipped_any is not None:
            m &= ~clipped_any.astype(bool)

    if bool(cfg.get("keep_largest_component", True)):
        m = largest_component(m)

    er = int(cfg.get("erosion_radius_px", 0))
    if er > 0:
        m2 = binary_erode(m, er)
        if int(np.count_nonzero(m2)) >= int(cfg.get("min_area_px", 5000)):
            m = m2

    if roi_mask is not None:
        m &= roi_mask.astype(bool)
    if clipped_any is not None:
        m &= ~clipped_any.astype(bool)
    return m


def build_mask_for_defects(
    mask_raw: np.ndarray,
    roi_mask: np.ndarray | None,
    clipped_any: np.ndarray | None,
    cfg: dict,
) -> np.ndarray:
    """
    Stable/inclusive mask for defect analysis.
    - no hole fill (preserve defect voids)
    - keep largest connected object
    - optional slight dilation to reduce boundary bites
    - never includes clipped pixels
    """
    m = mask_raw.astype(bool).copy()
    if roi_mask is not None:
        m &= roi_mask.astype(bool)
    if clipped_any is not None:
        m &= ~clipped_any.astype(bool)
    if bool(cfg.get("keep_largest_component", True)):
        m = largest_component(m)
    dr = int(cfg.get("dilate_radius_px", 3))
    if dr > 0:
        m = binary_dilate(m, dr)
    er = int(cfg.get("erosion_radius_px", 0))
    if er > 0:
        m = binary_erode(m, er)
    if roi_mask is not None:
        m &= roi_mask.astype(bool)
    if clipped_any is not None:
        m &= ~clipped_any.astype(bool)
    return m


def _box_blur(img: np.ndarray, k: int) -> np.ndarray:
    k = max(3, int(k) | 1)
    pad = k // 2
    p = np.pad(img, pad, mode="edge")
    integ = np.pad(p, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    h, w = img.shape
    y0 = np.arange(h)
    x0 = np.arange(w)
    y1 = y0 + k
    x1 = x0 + k
    s = integ[y1[:, None], x1] - integ[y0[:, None], x1] - integ[y1[:, None], x0] + integ[y0[:, None], x0]
    return (s / float(k * k)).astype(np.float32)


def build_mask_for_display(
    B: np.ndarray,
    mask_raw: np.ndarray,
    roi_mask: np.ndarray | None,
    clipped_any: np.ndarray | None,
    cfg: dict,
) -> np.ndarray:
    """
    Visual-stability mask for UI display.
    Not for unwrap logic.
    """
    if bool(cfg.get("enabled", True)) and bool(cfg.get("use_b_smooth", True)):
        ks = int(cfg.get("b_smooth_ksize", 5))
        b_thresh = float(cfg.get("b_thresh_display", 5.0))
        b_s = _box_blur(B.astype(np.float32), ks)
        m = b_s >= b_thresh
    else:
        m = mask_raw.astype(bool).copy()
    if bool(cfg.get("keep_largest_component", True)):
        m = largest_component(m)
    if roi_mask is not None:
        m &= roi_mask.astype(bool)
    if clipped_any is not None:
        m &= ~clipped_any.astype(bool)
    return m
