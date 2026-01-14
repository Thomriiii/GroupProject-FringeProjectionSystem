#!/usr/bin/env python3
"""
plane_fit.py

Geometry validation: fit a plane to a reconstructed point cloud and report planarity.

This catches common calibration, undistortion, or resolution issues where the point
cloud collapses into a streak or fan instead of a coherent surface.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class PlaneFit:
    """Plane model fit results and inlier mask."""
    normal: np.ndarray  # (3,)
    d: float            # plane: n.x + d = 0
    inlier_mask: np.ndarray
    rms: float


def _fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit plane to points via SVD.

    Returns (normal, d) for plane n.x + d = 0 with ||n||=1.
    """
    centroid = points.mean(axis=0)
    X = points - centroid
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    normal = vh[-1, :]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    d = -float(np.dot(normal, centroid))
    return normal.astype(np.float64), d


def _plane_distances(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """
    Compute signed distances from points to a plane.
    """
    return (points @ normal + d).astype(np.float64)


def ransac_plane(points: np.ndarray, *, iters: int = 600, thresh: float = 0.003, seed: int = 0) -> PlaneFit:
    """
    Robustly fit a plane to 3D points using RANSAC.
    """
    rng = np.random.default_rng(seed)
    pts = points[np.isfinite(points).all(axis=1)]
    if pts.shape[0] < 50:
        raise ValueError(f"Need at least 50 finite points, got {pts.shape[0]}")

    best_inliers = None
    best_count = -1
    best_model = None

    n = pts.shape[0]
    for _ in range(iters):
        i = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = pts[i[0]], pts[i[1]], pts[i[2]]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -float(np.dot(normal, p0))
        dist = np.abs(_plane_distances(pts, normal, d))
        inliers = dist <= thresh
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_model = (normal, d)

    if best_inliers is None or best_model is None:
        raise RuntimeError("RANSAC failed to find a plane model.")

    # Refine using SVD on inliers.
    inlier_pts = pts[best_inliers]
    normal_ref, d_ref = _fit_plane_svd(inlier_pts)
    dist_ref = np.abs(_plane_distances(inlier_pts, normal_ref, d_ref))
    rms = float(np.sqrt(np.mean(dist_ref**2)))

    # Return mask in the original `points` indexing (finite points only were used).
    finite_mask = np.isfinite(points).all(axis=1)
    full_inliers = np.zeros(points.shape[0], dtype=bool)
    full_inliers[np.where(finite_mask)[0]] = best_inliers

    return PlaneFit(normal=normal_ref, d=d_ref, inlier_mask=full_inliers, rms=rms)


def _load_points_npz(path: Path) -> np.ndarray:
    """
    Load a points array from an NPZ file.
    """
    data = np.load(path)
    if "points" not in data:
        raise ValueError(f"{path} does not contain 'points'")
    return data["points"].astype(np.float64)


def main() -> int:
    """
    CLI entry point for plane fitting.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=Path, default=None, help="points npz containing 'points' (e.g. points_filtered.npz)")
    ap.add_argument("--scan-dir", type=Path, default=None, help="scan directory to reconstruct then fit")
    ap.add_argument("--proj-w", type=int, default=None, help="projector width (required with --scan-dir)")
    ap.add_argument("--proj-h", type=int, default=None, help="projector height (required with --scan-dir)")
    ap.add_argument("--iters", type=int, default=600, help="RANSAC iterations")
    ap.add_argument("--thresh", type=float, default=0.003, help="inlier threshold in metres")
    args = ap.parse_args()

    if (args.points is None) == (args.scan_dir is None):
        raise SystemExit("Provide exactly one of --points or --scan-dir")

    if args.scan_dir is not None:
        if args.proj_w is None or args.proj_h is None:
            raise SystemExit("--proj-w/--proj-h required with --scan-dir")
        from core.triangulation import reconstruct_3d_from_scan
        reconstruct_3d_from_scan(str(args.scan_dir), proj_size=(int(args.proj_w), int(args.proj_h)))
        # Prefer filtered points if present.
        p = args.scan_dir / "points_filtered.npz"
        if not p.exists():
            raise SystemExit(f"Reconstruction did not create {p}")
        pts = _load_points_npz(p)
        src = p
    else:
        if not args.points.exists():
            raise SystemExit(f"Missing points file: {args.points}")
        pts = _load_points_npz(args.points)
        src = args.points

    pts_finite = pts[np.isfinite(pts).all(axis=1)]
    print(f"[LOAD] {src}: {pts_finite.shape[0]} finite points")

    fit = ransac_plane(pts, iters=int(args.iters), thresh=float(args.thresh), seed=0)
    inliers = fit.inlier_mask
    inlier_pts = pts[inliers & np.isfinite(pts).all(axis=1)]
    dist = np.abs(_plane_distances(inlier_pts, fit.normal, fit.d))
    p50 = float(np.percentile(dist, 50.0)) if dist.size else float("nan")
    p95 = float(np.percentile(dist, 95.0)) if dist.size else float("nan")

    n = fit.normal
    print("[PLANE] n = [%.6f, %.6f, %.6f], d = %.6f" % (n[0], n[1], n[2], fit.d))
    print(f"[PLANE] inliers={inlier_pts.shape[0]}/{pts_finite.shape[0]} thresh={args.thresh:.4f} m")
    print(f"[PLANE] RMS distance to plane (inliers): {fit.rms:.6f} m")
    print(f"[PLANE] abs distance: median={p50:.6f} m, p95={p95:.6f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
