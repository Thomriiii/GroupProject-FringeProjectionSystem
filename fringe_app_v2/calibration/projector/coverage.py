"""Coverage tracking for projector calibration."""

from __future__ import annotations

from typing import Any

import numpy as np


def init_coverage(coverage_cfg: dict[str, Any]) -> dict[str, Any]:
    grid_x = max(1, int(coverage_cfg.get("grid_x", 3)))
    grid_y = max(1, int(coverage_cfg.get("grid_y", 3)))
    min_bins_filled = max(1, int(coverage_cfg.get("min_bins_filled", 5)))
    tilt_buckets = [float(v) for v in (coverage_cfg.get("tilt_buckets_deg") or [0, 10, 20, 35])]
    if len(tilt_buckets) < 2:
        tilt_buckets = [0.0, 10.0, 20.0, 35.0]
    min_bucket_counts = [int(v) for v in (coverage_cfg.get("min_bucket_counts") or [2, 2, 2])]
    n_buckets = len(tilt_buckets) - 1
    while len(min_bucket_counts) < n_buckets:
        min_bucket_counts.append(min_bucket_counts[-1] if min_bucket_counts else 1)
    min_bucket_counts = min_bucket_counts[:n_buckets]

    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "visited_bins": [[0] * grid_x for _ in range(grid_y)],
        "bins_filled": 0,
        "min_bins_filled": min_bins_filled,
        "tilt_buckets_deg": tilt_buckets,
        "bucket_counts": [0] * n_buckets,
        "min_bucket_counts": min_bucket_counts,
        "sufficient": False,
        "indicator": "Need more variety",
        "guidance": ["Capture board in more image regions.", "Capture low, medium, and high tilt views."],
    }


def _bucket_index(tilt_deg: float | None, tilt_buckets: list[float]) -> int | None:
    if tilt_deg is None or not np.isfinite(float(tilt_deg)):
        return None
    t = float(tilt_deg)
    last = len(tilt_buckets) - 2
    for i in range(len(tilt_buckets) - 1):
        lo, hi = float(tilt_buckets[i]), float(tilt_buckets[i + 1])
        if i == last:
            if t >= lo:
                return i
        else:
            if lo <= t < hi:
                return i
    return None


def _compute_indicator(coverage: dict[str, Any]) -> tuple[bool, str, list[str]]:
    bins_filled = int(coverage.get("bins_filled", 0))
    min_bins = int(coverage.get("min_bins_filled", 1))
    bucket_counts = [int(v) for v in (coverage.get("bucket_counts") or [])]
    min_bucket_counts = [int(v) for v in (coverage.get("min_bucket_counts") or [])]

    bins_ok = bins_filled >= min_bins
    tilt_ok = True
    if bucket_counts and len(bucket_counts) == len(min_bucket_counts):
        tilt_ok = all(bucket_counts[i] >= min_bucket_counts[i] for i in range(len(bucket_counts)))

    sufficient = bins_ok and tilt_ok
    guidance: list[str] = []
    if not bins_ok:
        guidance.append("Capture board in more image regions.")
    if not tilt_ok:
        guidance.append("Capture low, medium, and high tilt views.")
    if not guidance:
        guidance.append("Coverage requirements met.")

    return sufficient, ("Sufficient coverage" if sufficient else "Need more variety"), guidance


def update_coverage(
    coverage: dict[str, Any],
    centroid_norm: tuple[float, float] | None,
    tilt_deg: float | None,
) -> dict[str, Any]:
    grid_x = max(1, int(coverage.get("grid_x", 3)))
    grid_y = max(1, int(coverage.get("grid_y", 3)))
    visited = [list(map(int, row)) for row in (coverage.get("visited_bins") or [])]
    if len(visited) != grid_y or any(len(r) != grid_x for r in visited):
        visited = [[0] * grid_x for _ in range(grid_y)]

    bucket_counts = list(map(int, coverage.get("bucket_counts") or []))
    tilt_buckets = [float(v) for v in (coverage.get("tilt_buckets_deg") or [0, 10, 20, 35])]

    if centroid_norm is not None:
        cx = float(np.clip(centroid_norm[0], 0.0, 1.0))
        cy = float(np.clip(centroid_norm[1], 0.0, 1.0))
        bx = min(grid_x - 1, int(np.floor(cx * grid_x)))
        by = min(grid_y - 1, int(np.floor(cy * grid_y)))
        visited[by][bx] = 1

    bins_filled = sum(1 for row in visited for v in row if int(v) > 0)

    expected_buckets = max(1, len(tilt_buckets) - 1)
    if len(bucket_counts) != expected_buckets:
        bucket_counts = [0] * expected_buckets

    bidx = _bucket_index(tilt_deg, tilt_buckets)
    if bidx is not None and 0 <= bidx < len(bucket_counts):
        bucket_counts[bidx] += 1

    out = {
        **coverage,
        "visited_bins": visited,
        "bins_filled": bins_filled,
        "bucket_counts": bucket_counts,
    }
    sufficient, indicator, guidance = _compute_indicator(out)
    out["sufficient"] = sufficient
    out["indicator"] = indicator
    out["guidance"] = guidance
    return out


def recompute_coverage(
    coverage_cfg: dict[str, Any],
    accepted_views: list[dict[str, Any]],
) -> dict[str, Any]:
    cov = init_coverage(coverage_cfg)
    for view in accepted_views:
        metrics = (view.get("metrics") or {}) if isinstance(view, dict) else {}
        centroid = metrics.get("centroid_norm")
        centroid_t: tuple[float, float] | None = None
        if isinstance(centroid, (list, tuple)) and len(centroid) == 2:
            try:
                centroid_t = (float(centroid[0]), float(centroid[1]))
            except Exception:
                pass
        tilt_raw = metrics.get("tilt_deg")
        tilt_deg: float | None = None
        try:
            tilt_deg = float(tilt_raw) if tilt_raw is not None else None
        except Exception:
            pass
        cov = update_coverage(cov, centroid_t, tilt_deg)
    return cov
