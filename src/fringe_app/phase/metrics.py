"""Mask scoring metrics for phase quality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(slots=True)
class MaskScore:
    valid_ratio: float
    largest_component_ratio: float
    edge_noise_ratio: float
    b_median: float
    b_p90: float
    score: float
    roi_valid_ratio: float
    roi_largest_component_ratio: float
    roi_edge_noise_ratio: float
    roi_b_median: float
    roi_score: float


def _largest_component_size(mask: np.ndarray) -> int:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            q = [(y, x)]
            visited[y, x] = True
            size = 0
            while q:
                cy, cx = q.pop()
                size += 1
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
            if size > best:
                best = size
    return best


def _edge_noise_ratio(mask: np.ndarray) -> float:
    h, w = mask.shape
    if mask.sum() == 0:
        return 0.0
    edge = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            if (y == 0 or not mask[y - 1, x] or
                y + 1 == h or not mask[y + 1, x] or
                x == 0 or not mask[y, x - 1] or
                x + 1 == w or not mask[y, x + 1]):
                edge += 1
    return edge / float(mask.sum())


def score_mask(
    mask: np.ndarray,
    B: np.ndarray,
    roi_mask: np.ndarray | None = None,
    target_valid_ratio: float = 0.30,
    target_b_median: float = 15.0,
) -> MaskScore:
    valid_ratio = float(mask.mean())
    total_valid = int(mask.sum())
    largest = _largest_component_size(mask)
    largest_ratio = float(largest / max(total_valid, 1))
    edge_ratio = _edge_noise_ratio(mask)

    if total_valid > 0:
        b_vals = B[mask]
        b_median = float(np.median(b_vals))
        b_p90 = float(np.percentile(b_vals, 90))
    else:
        b_median = 0.0
        b_p90 = 0.0

    def clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    score = (
        2.0 * clamp(valid_ratio / target_valid_ratio, 0.0, 1.5) +
        2.0 * largest_ratio +
        1.0 * clamp(b_median / target_b_median, 0.0, 1.5) -
        1.5 * edge_ratio
    )

    roi_valid_ratio = valid_ratio
    roi_largest_ratio = largest_ratio
    roi_edge_ratio = edge_ratio
    roi_b_median = b_median
    roi_score = score
    use_roi_score = False

    if roi_mask is not None:
        roi_mask = roi_mask.astype(bool)
        roi_valid = mask & roi_mask
        roi_area = roi_mask.sum()
        if roi_area > 0 and roi_area / float(mask.size) >= 0.01:
            roi_valid_ratio = float(roi_valid.mean())
            roi_largest_ratio = float(_largest_component_size(roi_valid) / max(int(roi_valid.sum()), 1))
            roi_edge_ratio = _edge_noise_ratio(roi_valid)
            if roi_valid.sum() > 0:
                roi_vals = B[roi_valid]
                roi_b_median = float(np.median(roi_vals))
            roi_score = (
                2.0 * clamp(roi_valid_ratio / target_valid_ratio, 0.0, 1.5) +
                2.0 * roi_largest_ratio +
                1.0 * clamp(roi_b_median / target_b_median, 0.0, 1.5) -
                1.5 * roi_edge_ratio
            )
            use_roi_score = True
        else:
            roi_score = score

    return MaskScore(
        valid_ratio=valid_ratio,
        largest_component_ratio=largest_ratio,
        edge_noise_ratio=edge_ratio,
        b_median=b_median,
        b_p90=b_p90,
        score=float(roi_score if use_roi_score else score),
        roi_valid_ratio=roi_valid_ratio,
        roi_largest_component_ratio=roi_largest_ratio,
        roi_edge_noise_ratio=roi_edge_ratio,
        roi_b_median=roi_b_median,
        roi_score=float(roi_score),
    )
