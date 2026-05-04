"""Coverage tracking for projector calibration v2."""

from __future__ import annotations

from typing import Any

import numpy as np


def init_coverage(coverage_cfg: dict[str, Any]) -> dict[str, Any]:
    grid_x = max(1, int(coverage_cfg.get("grid_x", 3)))
    grid_y = max(1, int(coverage_cfg.get("grid_y", 3)))
    min_bins_filled = max(1, int(coverage_cfg.get("min_bins_filled", 5)))
    tilt_buckets = [float(v) for v in (coverage_cfg.get("tilt_buckets_deg", [0, 10, 20, 35]) or [0, 10, 20, 35])]
    if len(tilt_buckets) < 2:
        tilt_buckets = [0.0, 10.0, 20.0, 35.0]
    min_bucket_counts = [int(v) for v in (coverage_cfg.get("min_bucket_counts", [2, 2, 2]) or [2, 2, 2])]
    bucket_count = max(1, len(tilt_buckets) - 1)
    if len(min_bucket_counts) != bucket_count:
        if len(min_bucket_counts) < bucket_count:
            min_bucket_counts = min_bucket_counts + [min_bucket_counts[-1] if min_bucket_counts else 1] * (bucket_count - len(min_bucket_counts))
        else:
            min_bucket_counts = min_bucket_counts[:bucket_count]

    coverage = {
        "grid_x": int(grid_x),
        "grid_y": int(grid_y),
        "visited_bins": [[0 for _ in range(grid_x)] for _ in range(grid_y)],
        "bins_filled": 0,
        "min_bins_filled": int(min_bins_filled),
        "tilt_buckets_deg": [float(v) for v in tilt_buckets],
        "bucket_counts": [0 for _ in range(bucket_count)],
        "min_bucket_counts": [int(v) for v in min_bucket_counts],
        "sufficient": False,
        "indicator": "Need more variety",
        "guidance": ["Capture board in more image regions.", "Capture low, medium, and high tilt views."],
    }
    return coverage


def _bucket_index(tilt_deg: float | None, tilt_buckets: list[float]) -> int | None:
    if tilt_deg is None:
        return None
    if not np.isfinite(float(tilt_deg)):
        return None
    t = float(tilt_deg)
    if len(tilt_buckets) < 2:
        return None

    last = len(tilt_buckets) - 2
    for i in range(len(tilt_buckets) - 1):
        lo = float(tilt_buckets[i])
        hi = float(tilt_buckets[i + 1])
        if i == last:
            if t >= lo:
                return i
        else:
            if t >= lo and t < hi:
                return i
    return None


def _compute_indicator(coverage: dict[str, Any]) -> tuple[bool, str, list[str]]:
    bins_filled = int(coverage.get("bins_filled", 0))
    min_bins = int(coverage.get("min_bins_filled", 1))
    bucket_counts = [int(v) for v in (coverage.get("bucket_counts", []) or [])]
    min_bucket_counts = [int(v) for v in (coverage.get("min_bucket_counts", []) or [])]

    bins_ok = bins_filled >= min_bins
    tilt_ok = True
    if len(bucket_counts) == len(min_bucket_counts) and len(bucket_counts) > 0:
        tilt_ok = all(bucket_counts[i] >= min_bucket_counts[i] for i in range(len(bucket_counts)))

    sufficient = bool(bins_ok and tilt_ok)
    guidance: list[str] = []
    if not bins_ok:
        guidance.append("Capture board in more image regions.")
    if not tilt_ok:
        guidance.append("Capture low, medium, and high tilt views.")
    if len(guidance) == 0:
        guidance.append("Coverage requirements met.")

    indicator = "Sufficient coverage" if sufficient else "Need more variety"
    return sufficient, indicator, guidance


def update_coverage(
    coverage: dict[str, Any],
    centroid_norm: tuple[float, float] | None,
    tilt_deg: float | None,
) -> dict[str, Any]:
    out = {
        "grid_x": int(coverage.get("grid_x", 3)),
        "grid_y": int(coverage.get("grid_y", 3)),
        "visited_bins": [list(map(int, row)) for row in (coverage.get("visited_bins", []) or [])],
        "bins_filled": int(coverage.get("bins_filled", 0)),
        "min_bins_filled": int(coverage.get("min_bins_filled", 5)),
        "tilt_buckets_deg": [float(v) for v in (coverage.get("tilt_buckets_deg", [0, 10, 20, 35]) or [0, 10, 20, 35])],
        "bucket_counts": [int(v) for v in (coverage.get("bucket_counts", []) or [])],
        "min_bucket_counts": [int(v) for v in (coverage.get("min_bucket_counts", []) or [])],
        "sufficient": bool(coverage.get("sufficient", False)),
        "indicator": str(coverage.get("indicator", "Need more variety")),
        "guidance": [str(v) for v in (coverage.get("guidance", []) or [])],
    }

    grid_x = max(1, out["grid_x"])
    grid_y = max(1, out["grid_y"])
    visited = out["visited_bins"]
    if len(visited) != grid_y or any(len(row) != grid_x for row in visited):
        visited = [[0 for _ in range(grid_x)] for _ in range(grid_y)]
        out["visited_bins"] = visited

    if centroid_norm is not None:
        cx = float(np.clip(float(centroid_norm[0]), 0.0, 1.0))
        cy = float(np.clip(float(centroid_norm[1]), 0.0, 1.0))
        bx = min(grid_x - 1, int(np.floor(cx * grid_x)))
        by = min(grid_y - 1, int(np.floor(cy * grid_y)))
        visited[by][bx] = 1

    bins_filled = int(sum(1 for row in visited for v in row if int(v) > 0))
    out["bins_filled"] = bins_filled

    bucket_counts = out["bucket_counts"]
    tilt_buckets = out["tilt_buckets_deg"]
    expected_bucket_count = max(1, len(tilt_buckets) - 1)
    if len(bucket_counts) != expected_bucket_count:
        bucket_counts = [0 for _ in range(expected_bucket_count)]
        out["bucket_counts"] = bucket_counts

    bidx = _bucket_index(tilt_deg, tilt_buckets)
    if bidx is not None and bidx >= 0 and bidx < len(bucket_counts):
        bucket_counts[bidx] = int(bucket_counts[bidx]) + 1

    sufficient, indicator, guidance = _compute_indicator(out)
    out["sufficient"] = bool(sufficient)
    out["indicator"] = indicator
    out["guidance"] = guidance
    return out


def recompute_coverage(
    coverage_cfg: dict[str, Any],
    accepted_views: list[dict[str, Any]],
) -> dict[str, Any]:
    cov = init_coverage(coverage_cfg)
    for view in accepted_views:
        metrics = (view.get("metrics", {}) or {}) if isinstance(view, dict) else {}
        centroid = metrics.get("centroid_norm")
        centroid_tuple: tuple[float, float] | None = None
        if isinstance(centroid, (list, tuple)) and len(centroid) == 2:
            try:
                centroid_tuple = (float(centroid[0]), float(centroid[1]))
            except Exception:
                centroid_tuple = None
        tilt = metrics.get("tilt_deg")
        tilt_deg: float | None
        try:
            tilt_deg = float(tilt) if tilt is not None else None
        except Exception:
            tilt_deg = None
        cov = update_coverage(cov, centroid_tuple, tilt_deg)
    return cov
