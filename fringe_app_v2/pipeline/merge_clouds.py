"""
Merge point clouds captured at multiple turntable angles into a single cloud.

COORDINATE CONVENTION
---------------------
Camera frame: X right, Y down, Z depth (standard OpenCV).

The turntable rotates the object around a vertical axis.  The correct rotation
matrix to use depends on how the camera is oriented:

  rotation_axis = "y"  → turntable axis follows the viewer/camera Y axis.

  rotation_axis = "z"  → turntable axis follows camera Z / optical depth.

  rotation_axis_vector = [x, y, z] in config can be used for a measured tilted
                          turntable axis.

  rotation_sign = +1  → positive turntable angle = CW from above  (default)
  rotation_sign = -1  → positive turntable angle = CCW from above

If the merged cloud looks "mirrored", flip rotation_sign in the config.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np


# ── rotation helpers ─────────────────────────────────────────────────────────

def _Ry(angle_deg: float) -> np.ndarray:
    """Rotation around camera Y axis."""
    θ = np.radians(angle_deg)
    c, s = np.cos(θ), np.sin(θ)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=np.float64)


def _Rz(angle_deg: float) -> np.ndarray:
    """Rotation around camera Z axis (optical axis / depth axis)."""
    θ = np.radians(angle_deg)
    c, s = np.cos(θ), np.sin(θ)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _axis_rotation(angle_deg: float, axis: np.ndarray) -> np.ndarray:
    """Rotation matrix for a right-handed rotation around an arbitrary axis."""
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        raise ValueError("rotation axis vector must be non-zero")
    x, y, z = axis / norm
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _rotation_matrix(angle_deg: float, rotation_axis: str | np.ndarray) -> np.ndarray:
    if isinstance(rotation_axis, str):
        axis = rotation_axis.lower().strip()
        if axis == "x":
            return _axis_rotation(angle_deg, np.array([1.0, 0.0, 0.0]))
        if axis == "y":
            return _Ry(angle_deg)
        if axis == "z":
            return _Rz(angle_deg)
        raise ValueError(f"Unsupported rotation_axis {rotation_axis!r}; use x, y, z, or a 3-vector")
    return _axis_rotation(angle_deg, rotation_axis)


def _axis_vector(rotation_axis: str | np.ndarray) -> np.ndarray:
    """Return the normalized 3D direction vector used as the rotation axis."""
    if isinstance(rotation_axis, str):
        axis = rotation_axis.lower().strip()
        if axis == "x":
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if axis == "y":
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if axis == "z":
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        raise ValueError(f"Unsupported rotation_axis {rotation_axis!r}; use x, y, z, or a 3-vector")
    axis_vec = np.asarray(rotation_axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis_vec))
    if norm <= 0:
        raise ValueError("rotation axis vector must be non-zero")
    return axis_vec / norm


def transform_cloud(
    points: np.ndarray,
    angle_deg: float,
    axis: np.ndarray,
    rotation_sign: float = 1.0,
    rotation_axis: str | np.ndarray = "z",
) -> np.ndarray:
    """
    Rotate a point cloud from scan angle θ back to the reference frame (θ=0).

    Args:
        points:        Nx3 float array of XYZ positions in camera coordinates.
        angle_deg:     Turntable angle at which this cloud was captured.
        axis:          3-element (x, y, z) position of the rotation axis centre.
        rotation_sign: +1 if positive angle = CW from above, -1 if CCW.
        rotation_axis: "z" for top-down camera (default), "y" for side camera.

    Returns:
        Nx3 transformed point cloud.
    """
    if len(points) == 0:
        return points
    undo_angle = -rotation_sign * angle_deg
    R = _rotation_matrix(undo_angle, rotation_axis)
    rel = points.astype(np.float64) - axis[np.newaxis, :]
    return ((R @ rel.T).T + axis[np.newaxis, :]).astype(np.float32)


def top_anchor(
    points: np.ndarray,
    axis_dir: np.ndarray,
    *,
    percentile: float = 90.0,
    side: str = "max",
    min_points: int = 200,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """
    Estimate a repeatable top anchor from the high/low end of a cloud along the
    turntable axis direction.

    This is intended for rotational scans of objects with a shared top feature
    visible in every angle. It deliberately returns a centroid of a top band
    rather than a single extremal point, which is too noisy in structured-light
    clouds.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if pts.shape[0] < max(3, min_points):
        return None, {
            "ok": False,
            "reason": "too_few_points",
            "points": int(pts.shape[0]),
        }

    axis = np.asarray(axis_dir, dtype=np.float64).reshape(3)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    proj = pts @ axis
    pct = float(np.clip(percentile, 0.0, 100.0))
    use_max = str(side).lower().strip() != "min"
    threshold = float(np.percentile(proj, pct if use_max else 100.0 - pct))
    keep = proj >= threshold if use_max else proj <= threshold
    n_keep = int(np.count_nonzero(keep))
    if n_keep < max(3, min_points):
        order = np.argsort(proj)
        if use_max:
            idx = order[-max(3, min_points):]
        else:
            idx = order[:max(3, min_points)]
        keep = np.zeros(pts.shape[0], dtype=bool)
        keep[idx] = True
        n_keep = int(np.count_nonzero(keep))

    anchor = np.mean(pts[keep], axis=0)
    return anchor, {
        "ok": True,
        "side": "max" if use_max else "min",
        "percentile": pct,
        "threshold_m": threshold,
        "input_points": int(pts.shape[0]),
        "anchor_points": n_keep,
        "anchor_m": anchor.astype(float).tolist(),
        "anchor_axis_m": float(anchor @ axis),
    }


def apply_top_anchor_refinement(
    source: np.ndarray,
    reference_anchor: np.ndarray,
    axis_dir: np.ndarray,
    config: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Translate a source cloud so its detected top anchor matches reference."""
    cfg = config or {}
    meta: dict[str, Any] = {
        "enabled": bool(cfg.get("enabled", False)),
        "applied": False,
    }
    if not meta["enabled"]:
        return source, meta

    percentile = float(cfg.get("percentile", 90.0))
    side = str(cfg.get("side", "max"))
    min_points = int(cfg.get("min_points", 200))
    align_axis_component = bool(cfg.get("align_axis_component", True))
    max_translation_m = float(cfg.get("max_translation_m", 0.08))

    src_anchor, src_meta = top_anchor(
        source,
        axis_dir,
        percentile=percentile,
        side=side,
        min_points=min_points,
    )
    meta["source_anchor"] = src_meta
    meta["reference_anchor_m"] = np.asarray(reference_anchor, dtype=np.float64).astype(float).tolist()
    if src_anchor is None:
        meta["reason"] = "source_anchor_failed"
        return source, meta

    translation = np.asarray(reference_anchor, dtype=np.float64).reshape(3) - src_anchor.reshape(3)
    axis = np.asarray(axis_dir, dtype=np.float64).reshape(3)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    if not align_axis_component:
        translation = translation - float(translation @ axis) * axis

    translation_norm = float(np.linalg.norm(translation))
    meta["requested_translation_m"] = translation.astype(float).tolist()
    meta["requested_translation_norm_m"] = translation_norm
    meta["align_axis_component"] = align_axis_component
    meta["max_translation_m"] = max_translation_m
    if translation_norm > max_translation_m:
        meta["reason"] = "translation_limit"
        return source, meta

    refined = (source.astype(np.float64) + translation.reshape(1, 3)).astype(np.float32)
    meta["applied"] = True
    meta["reason"] = "ok"
    meta["translation_m"] = translation.astype(float).tolist()
    meta["translation_norm_m"] = translation_norm
    return refined, meta


# ── axis point optimisation from top anchors ─────────────────────────────────

def _perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors spanning the plane perpendicular to axis."""
    ax = np.asarray(axis, dtype=np.float64).reshape(3)
    ax = ax / max(float(np.linalg.norm(ax)), 1e-12)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(ax, tmp))) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u1 = np.cross(ax, tmp)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(ax, u1)
    u2 /= np.linalg.norm(u2)
    return u1, u2


def optimize_axis_from_top(
    clouds_with_angles: list[tuple[float, np.ndarray]],
    axis_point: np.ndarray,
    rotation_sign: float,
    rotation_axis: str | np.ndarray,
    *,
    percentile: float = 95.0,
    side: str = "max",
    min_points: int = 500,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Refine the axis point (the XY-perpendicular position of the rotation axis)
    by solving a least-squares problem over the top anchors of all scans.

    The calibrated axis_x_m / axis_y_m are found at the calibration board's
    depth. For tall objects whose apex is significantly closer to the camera,
    the tilted rotation axis projects to a different XY position.  This function
    finds the correct perpendicular offset directly from the scan data:

      For each scan i: R_i @ (raw_anchor_i - P) + P  should equal raw_anchor_0
      Minimise  Σ_i ||(I - R_i) @ P - (raw_anchor_0 - R_i @ raw_anchor_i)||²

    Only the two components of P perpendicular to the rotation axis are solved;
    the component along the axis is left at its initial value (it doesn't affect
    the rotation output).

    Returns (refined_axis_point, meta_dict).
    """
    axis_dir = _axis_vector(rotation_axis)
    P0 = np.asarray(axis_point, dtype=np.float64).reshape(3)
    u1, u2 = _perpendicular_basis(axis_dir)

    raw_anchors: list[np.ndarray] = []
    per_scan_raw: list[dict[str, Any]] = []
    angles_used: list[float] = []

    for angle_deg, pts in clouds_with_angles:
        anc, ameta = top_anchor(
            pts, axis_dir,
            percentile=percentile, side=side, min_points=min_points,
        )
        entry: dict[str, Any] = {"angle_deg": float(angle_deg), **ameta}
        per_scan_raw.append(entry)
        if anc is not None:
            raw_anchors.append(anc)
            angles_used.append(float(angle_deg))

    if len(raw_anchors) < 2:
        return P0, {
            "ok": False,
            "reason": "too_few_anchors",
            "n_anchors": len(raw_anchors),
            "per_scan_raw": per_scan_raw,
        }

    Rs = [_rotation_matrix(-rotation_sign * a, rotation_axis) for a in angles_used]
    ref_anchor = raw_anchors[0]

    # Build 2-D perpendicular linear system
    A_rows: list[np.ndarray] = []
    b_rows: list[np.ndarray] = []
    for R, raw in zip(Rs, raw_anchors):
        B = np.eye(3) - R
        rhs = ref_anchor - R @ raw - B @ P0
        A_rows.append(np.column_stack([B @ u1, B @ u2]))
        b_rows.append(rhs)

    A_mat = np.vstack(A_rows)
    b_vec = np.concatenate(b_rows)
    s_opt, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

    P_opt = P0 + s_opt[0] * u1 + s_opt[1] * u2

    # Compute residual spread of rotated anchors with P_opt
    rotated = np.array([Rs[i] @ (raw_anchors[i] - P_opt) + P_opt for i in range(len(Rs))])
    spread = rotated - rotated.mean(axis=0)
    rms = float(np.sqrt(np.mean(np.sum(spread**2, axis=1))))
    max_err = float(np.max(np.sqrt(np.sum(spread**2, axis=1))))

    per_scan_out = []
    for i, (a, anc) in enumerate(zip(angles_used, rotated)):
        per_scan_out.append({
            "angle_deg": float(a),
            "anchor_m": anc.tolist(),
            "dist_from_ref_m": float(np.linalg.norm(anc - rotated[0])),
        })

    return P_opt, {
        "ok": True,
        "enabled": True,
        "applied": True,
        "n_anchors": len(raw_anchors),
        "percentile": percentile,
        "side": side,
        "initial_axis_point_m": P0.tolist(),
        "axis_point_m": P_opt.tolist(),
        "perpendicular_shift_m": (P_opt - P0).tolist(),
        "anchor_residual_rms_m": rms,
        "anchor_residual_max_m": max_err,
        "per_scan": per_scan_out,
    }


def optimize_axis_from_edges(
    clouds_with_angles: list[tuple[float, np.ndarray]],
    axis_point: np.ndarray,
    rotation_sign: float,
    rotation_axis: str | np.ndarray,
    *,
    max_shift_m: float = 0.015,
    coarse_steps: int = 7,
    sample_points: int = 3000,
    seed: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Refine the axis point by minimizing adjacent-scan side/edge disagreement.

    Unlike top-anchor alignment, this tests the merged result itself. It samples
    each scan, tries small offsets in the plane perpendicular to the turntable
    axis, and scores adjacent scans in lower/mid/upper height bands. This keeps
    the correction rigid while avoiding view-dependent top centroids.
    """
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        return np.asarray(axis_point, dtype=np.float64), {
            "enabled": True,
            "applied": False,
            "reason": f"scipy_unavailable: {exc}",
        }

    valid = [(float(a), np.asarray(p, dtype=np.float64).reshape(-1, 3)) for a, p in clouds_with_angles if len(p) > 0]
    if len(valid) < 2:
        return np.asarray(axis_point, dtype=np.float64), {
            "enabled": True,
            "applied": False,
            "reason": "too_few_clouds",
        }

    rng = random.Random(seed)
    sampled: list[tuple[float, np.ndarray]] = []
    for angle_deg, pts in valid:
        finite = pts[np.all(np.isfinite(pts), axis=1)]
        if len(finite) > sample_points:
            idx = rng.sample(range(len(finite)), sample_points)
            finite = finite[np.asarray(idx, dtype=np.int64)]
        sampled.append((angle_deg, finite))

    axis_dir = _axis_vector(rotation_axis)
    u1, u2 = _perpendicular_basis(axis_dir)
    p0 = np.asarray(axis_point, dtype=np.float64).reshape(3)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref, axis_dir))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    b1 = ref - float(ref @ axis_dir) * axis_dir
    b1 /= max(float(np.linalg.norm(b1)), 1e-12)
    b2 = np.cross(axis_dir, b1)
    basis = np.vstack([b1, b2, axis_dir]).T
    bands = [((5.0, 30.0), 1.5), ((30.0, 70.0), 1.0), ((70.0, 95.0), 1.2)]

    def transform_parts(axis: np.ndarray) -> list[np.ndarray]:
        return [
            pts if angle_deg == 0.0 else transform_cloud(pts, angle_deg, axis, rotation_sign, rotation_axis).astype(np.float64)
            for angle_deg, pts in sampled
        ]

    def score(axis: np.ndarray) -> tuple[float, list[float]]:
        parts = transform_parts(axis)
        band_scores: list[float] = []
        for (lo_pct, hi_pct), weight in bands:
            adjacent: list[float] = []
            for i in range(len(parts)):
                a = parts[i]
                b = parts[(i + 1) % len(parts)]
                qa = (a - axis.reshape(1, 3)) @ basis
                qb = (b - axis.reshape(1, 3)) @ basis
                lo_a, hi_a = np.percentile(qa[:, 2], [lo_pct, hi_pct])
                lo_b, hi_b = np.percentile(qb[:, 2], [lo_pct, hi_pct])
                aa = a[(qa[:, 2] >= lo_a) & (qa[:, 2] <= hi_a)]
                bb = b[(qb[:, 2] >= lo_b) & (qb[:, 2] <= hi_b)]
                if len(aa) < 20 or len(bb) < 20:
                    continue
                dists, _ = cKDTree(bb).query(aa, k=1)
                adjacent.append(float(np.median(dists)))
            band_scores.append(float(np.mean(adjacent)) if adjacent else 0.05)
        total = sum(weight * band_score for (_, weight), band_score in zip(bands, band_scores))
        return float(total), band_scores

    best_axis = p0
    best_score, best_bands = score(best_axis)
    max_shift = float(max_shift_m)
    steps = max(3, int(coarse_steps))
    if steps % 2 == 0:
        steps += 1

    centre_a = 0.0
    centre_b = 0.0
    for radius, grid_steps in [(max_shift, steps), (max_shift / 3.0, steps), (max_shift / 9.0, steps)]:
        for a in np.linspace(centre_a - radius, centre_a + radius, grid_steps):
            for b in np.linspace(centre_b - radius, centre_b + radius, grid_steps):
                if math.hypot(float(a), float(b)) > max_shift:
                    continue
                axis = p0 + float(a) * u1 + float(b) * u2
                total, band_scores = score(axis)
                if total < best_score:
                    best_score = total
                    best_bands = band_scores
                    best_axis = axis
                    centre_a = float(a)
                    centre_b = float(b)

    initial_score, initial_bands = score(p0)
    if not np.all(np.isfinite(best_axis)):
        return p0, {"enabled": True, "applied": False, "reason": "non_finite_axis"}

    return best_axis, {
        "enabled": True,
        "applied": True,
        "initial_axis_point_m": p0.astype(float).tolist(),
        "axis_point_m": best_axis.astype(float).tolist(),
        "shift_m": (best_axis - p0).astype(float).tolist(),
        "initial_score_m": float(initial_score),
        "score_m": float(best_score),
        "initial_band_scores_m": [float(v) for v in initial_bands],
        "band_scores_m": [float(v) for v in best_bands],
        "sample_points": int(sample_points),
        "max_shift_m": max_shift,
    }


# ── fiducial-based alignment ─────────────────────────────────────────────────

def load_fiducial_positions(
    run_dir: Path,
    pixel_coords: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extract 3D positions of fiducial marks from a reconstruction run.

    Reads from reconstruct/fiducials.json (produced by _reconstruct_fiducial_pixels,
    which bypasses the ROI mask). Falls back to xyz.npy for older runs.

    Returns (positions, valid_mask) where:
      positions:   (N, 3) float64 — NaN rows for failed marks
      valid_mask:  (N,) bool — True where the mark was successfully reconstructed
    Returns None if fewer than 3 marks are valid.
    """
    import json as _json

    fid_json = Path(run_dir) / "reconstruct" / "fiducials.json"
    if fid_json.exists():
        data = _json.loads(fid_json.read_text())
        n = len(data)
        positions = np.full((n, 3), np.nan, dtype=np.float64)
        valid = np.zeros(n, dtype=bool)
        for i, entry in enumerate(data):
            if entry.get("ok"):
                positions[i] = entry["xyz_m"]
                valid[i] = True
        if int(np.count_nonzero(valid)) < 3:
            return None
        return positions, valid

    # Fallback: xyz.npy (marks may be NaN if ROI excludes them)
    xyz_path = Path(run_dir) / "reconstruct" / "xyz.npy"
    if not xyz_path.exists():
        return None
    xyz = np.load(xyz_path)
    h, w = xyz.shape[:2]
    n = len(pixel_coords)
    positions = np.full((n, 3), np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for i, (u, v) in enumerate(pixel_coords):
        if 0 <= int(v) < h and 0 <= int(u) < w:
            pt = xyz[int(v), int(u)].astype(np.float64)
            if np.all(np.isfinite(pt)):
                positions[i] = pt
                valid[i] = True
    if int(np.count_nonzero(valid)) < 3:
        return None
    return positions, valid


def _best_fiducial_transform(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Find the rigid transform from src_pts to ref_pts using exhaustive permutation
    matching. Tries all N! orderings of src_pts and returns the (R, t, perm, residual)
    with the lowest mean per-mark error.

    This is needed because the marks rotate with the turntable: at each scan angle
    the marks appear at different pixel positions, so their order in fiducials.json
    (sorted top-to-bottom in the image) changes. We cannot assume mark index i in
    scan A corresponds to mark index i in scan B.
    """
    from itertools import combinations, permutations

    best_R: np.ndarray | None = None
    best_t: np.ndarray | None = None
    best_src_idx: tuple[int, ...] | None = None
    best_ref_idx: tuple[int, ...] | None = None
    best_perm: tuple[int, ...] | None = None
    best_residual = float("inf")

    n_use = min(len(src_pts), len(ref_pts))
    if n_use < 3:
        raise ValueError("at least 3 fiducial points are required")

    for src_idx in combinations(range(len(src_pts)), n_use):
        src_subset = src_pts[list(src_idx)]
        for ref_idx in combinations(range(len(ref_pts)), n_use):
            ref_subset = ref_pts[list(ref_idx)]
            for perm in permutations(range(n_use)):
                src_ordered = src_subset[list(perm)]
                R, t = _rigid_transform(src_ordered, ref_subset)
                aligned = (R @ src_ordered.T).T + t
                residual = float(np.mean(np.linalg.norm(aligned - ref_subset, axis=1)))
                if residual < best_residual:
                    best_residual = residual
                    best_R = R
                    best_t = t
                    best_src_idx = src_idx
                    best_ref_idx = ref_idx
                    best_perm = perm

    if best_R is None or best_t is None or best_src_idx is None or best_ref_idx is None or best_perm is None:
        raise ValueError("fiducial permutation search failed")

    # Return source indexes in the order corresponding to ref indexes.
    src_match = np.array([best_src_idx[j] for j in best_perm], dtype=int)
    ref_match = np.array(best_ref_idx, dtype=int)
    return best_R, best_t, np.stack([src_match, ref_match], axis=1), best_residual


def align_clouds_from_fiducials(
    clouds_with_angles: list[tuple[float, np.ndarray]],
    run_dirs: list[Path],
    pixel_coords: list[tuple[int, int]],
) -> tuple[list[tuple[float, np.ndarray]], list[dict[str, Any]], bool]:
    """
    Compute per-scan rigid transforms from 3D fiducial mark positions.

    Uses exhaustive permutation matching so mark ordering does not need to be
    consistent between scans — the marks rotate with the turntable and will
    appear in different image positions (and thus different sorted order) at
    each angle.

    Returns (aligned_clouds_with_angles, per_scan_meta, all_applied).
    all_applied is True only when every scan was successfully fiducial-aligned.
    """
    fid_data = [load_fiducial_positions(rd, pixel_coords) for rd in run_dirs]

    ref_idx = next((i for i, d in enumerate(fid_data) if d is not None), None)
    if ref_idx is None:
        return (
            clouds_with_angles,
            [{"enabled": True, "applied": False, "reason": "no_fiducials_reconstructed"} for _ in clouds_with_angles],
            False,
        )

    ref_positions, ref_valid = fid_data[ref_idx]
    ref_pts = ref_positions[ref_valid]
    aligned: list[tuple[float, np.ndarray]] = []
    meta_list: list[dict[str, Any]] = []
    n_applied = 0

    for i, ((angle_deg, pts), data) in enumerate(zip(clouds_with_angles, fid_data)):
        if i == ref_idx:
            aligned.append((0.0, pts))
            meta_list.append({
                "enabled": True, "applied": False, "role": "reference",
                "angle_deg": float(angle_deg),
                "n_valid_marks": int(np.count_nonzero(ref_valid)),
            })
            n_applied += 1
            continue
        if data is None or len(pts) == 0:
            aligned.append((angle_deg, pts))
            meta_list.append({
                "enabled": True, "applied": False,
                "reason": "fiducials_not_reconstructed",
                "angle_deg": float(angle_deg),
            })
            continue

        src_positions, src_valid = data
        src_pts = src_positions[src_valid]
        n_src = len(src_pts)
        n_ref = len(ref_pts)

        if n_src < 3 or n_ref < 3:
            aligned.append((angle_deg, pts))
            meta_list.append({
                "enabled": True, "applied": False,
                "reason": f"too_few_marks_src={n_src}_ref={n_ref}",
                "angle_deg": float(angle_deg),
            })
            continue

        R, t, matches, residual = _best_fiducial_transform(src_pts, ref_pts)

        transformed = ((R @ pts.astype(np.float64).T).T + t).astype(np.float32)
        src_match = matches[:, 0]
        ref_match = matches[:, 1]
        aligned_fid = (R @ src_pts[src_match].T).T + t
        per_mark_err = np.linalg.norm(aligned_fid - ref_pts[ref_match], axis=1)
        aligned.append((0.0, transformed))
        meta_list.append({
            "enabled": True, "applied": True,
            "angle_deg": float(angle_deg),
            "n_valid_marks_src": n_src,
            "n_valid_marks_ref": n_ref,
            "n_marks_used": int(len(matches)),
            "best_matches_src_to_ref": matches.tolist(),
            "fiducial_search_residual_mean_m": float(residual),
            "fiducial_residual_mean_m": float(np.mean(per_mark_err)),
            "fiducial_residual_max_m": float(np.max(per_mark_err)),
            "per_mark_error_m": per_mark_err.tolist(),
        })
        n_applied += 1

    return aligned, meta_list, n_applied == len(clouds_with_angles)


# ── axis estimation ───────────────────────────────────────────────────────────

def estimate_axis(clouds: list[np.ndarray]) -> np.ndarray:
    """
    Estimate the turntable rotation axis from multiple point cloud centroids.

    This is only a rough fallback.  For an asymmetric or partially visible
    object, a configured/calibrated axis centre is more reliable.

    Returns a (3,) array [x, y, z].
    """
    centroids = []
    for pts in clouds:
        finite = pts[np.all(np.isfinite(pts), axis=1)]
        if len(finite) > 0:
            centroids.append(np.mean(finite, axis=0))
    if not centroids:
        return np.zeros(3, dtype=np.float64)
    c = np.array(centroids)
    return np.array([np.mean(c[:, 0]), np.mean(c[:, 1]), np.mean(c[:, 2])], dtype=np.float64)


# ── optional registration refinement ─────────────────────────────────────────

def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points.astype(np.float64, copy=False)
    rng = random.Random(seed)
    idx = rng.sample(range(len(points)), max_points)
    return points[np.asarray(idx, dtype=np.int64)].astype(np.float64, copy=False)


def _rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_c = np.mean(source, axis=0)
    tgt_c = np.mean(target, axis=0)
    src0 = source - src_c
    tgt0 = target - tgt_c
    H = src0.T @ tgt0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = tgt_c - R @ src_c
    return R, t


def _rotation_angle_deg(R: np.ndarray) -> float:
    trace = float(np.trace(R))
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def refine_cloud_icp(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_points: int = 12000,
    max_iterations: int = 20,
    max_correspondence_m: float = 0.012,
    max_rotation_deg: float = 8.0,
    max_translation_m: float = 0.025,
    min_pairs: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Refine a coarsely aligned cloud against the accumulated target using rigid ICP.

    The turntable angle remains the coarse transform.  ICP is deliberately
    bounded so a bad/partial view cannot rotate the scan into a different pose.
    If scipy is unavailable or the fit is weak, the source is returned unchanged.
    """
    meta: dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "",
        "iterations": 0,
        "pairs": 0,
        "median_error_m": None,
        "rotation_deg": 0.0,
        "translation_m": [0.0, 0.0, 0.0],
    }
    if len(source) < min_pairs or len(target) < min_pairs:
        meta["reason"] = "too_few_points"
        return source, meta

    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        meta["reason"] = f"scipy_unavailable: {exc}"
        return source, meta

    src_work = _sample_points(source, max_points, seed)
    tgt_work = _sample_points(target, max_points, seed + 1009)
    tree = cKDTree(tgt_work)

    R_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros(3, dtype=np.float64)
    current = src_work.copy()
    last_median: float | None = None

    for iteration in range(max_iterations):
        distances, indices = tree.query(current, k=1, workers=-1)
        keep = np.isfinite(distances) & (distances <= max_correspondence_m)
        if int(np.count_nonzero(keep)) < min_pairs:
            meta["reason"] = "too_few_correspondences"
            break

        src_corr = current[keep]
        tgt_corr = tgt_work[indices[keep]]
        R_step, t_step = _rigid_transform(src_corr, tgt_corr)
        current = (R_step @ current.T).T + t_step
        R_total = R_step @ R_total
        t_total = R_step @ t_total + t_step

        median_error = float(np.median(distances[keep]))
        meta["iterations"] = iteration + 1
        meta["pairs"] = int(np.count_nonzero(keep))
        meta["median_error_m"] = median_error
        if last_median is not None and abs(last_median - median_error) < 1e-5:
            break
        last_median = median_error

    rotation_deg = _rotation_angle_deg(R_total)
    translation_norm = float(np.linalg.norm(t_total))
    meta["rotation_deg"] = rotation_deg
    meta["translation_m"] = [float(v) for v in t_total]

    if meta["pairs"] < min_pairs:
        return source, meta
    if rotation_deg > max_rotation_deg:
        meta["reason"] = "rotation_limit"
        return source, meta
    if translation_norm > max_translation_m:
        meta["reason"] = "translation_limit"
        return source, meta

    refined = ((R_total @ source.astype(np.float64).T).T + t_total).astype(np.float32)
    meta["applied"] = True
    meta["reason"] = "ok"
    return refined, meta


# ── merge entry point ─────────────────────────────────────────────────────────

def merge_scans(
    clouds_with_angles: list[tuple[float, np.ndarray]],
    axis: np.ndarray | None = None,
    rotation_sign: float = 1.0,
    rotation_axis: str | np.ndarray = "z",
    refine_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[np.ndarray], np.ndarray]:
    """
    Merge multiple Nx3 point clouds (each at a different turntable angle).

    Args:
        clouds_with_angles: List of (angle_deg, Nx3_array) pairs.
        axis:               Rotation axis centre; estimated if None.
        rotation_sign:      +1 = CW positive, -1 = CCW positive.
        rotation_axis:      "z" for top-down camera, "y" for side camera.

    Returns:
        (merged_points, axis, per_scan_meta, per_scan_parts, merged_colors).
    """
    valid = [(a, p) for a, p in clouds_with_angles if len(p) > 0]
    if not valid:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(3, dtype=np.float64), [], [], np.zeros((0, 3), dtype=np.uint8)

    if axis is None:
        axis = estimate_axis([p for _, p in valid])

    parts: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    per_scan_meta: list[dict[str, Any]] = []
    palette = np.array(
        [
            [230, 230, 230],
            [230, 75, 60],
            [70, 145, 240],
            [65, 180, 95],
            [245, 180, 55],
            [160, 95, 210],
        ],
        dtype=np.uint8,
    )
    refine_config = refine_config or {}
    refine_enabled = bool(refine_config.get("enabled", False))
    top_anchor_config = dict(refine_config.get("top_anchor") or {})
    top_anchor_enabled = bool(top_anchor_config.get("enabled", False))
    rotation_axis_dir = _axis_vector(rotation_axis)
    reference_top_anchor: np.ndarray | None = None
    reference_top_meta: dict[str, Any] | None = None

    for angle_deg, pts in valid:
        if angle_deg == 0.0:
            transformed = pts.astype(np.float32)
        else:
            transformed = transform_cloud(pts, angle_deg, axis, rotation_sign, rotation_axis)

        scan_meta: dict[str, Any] = {
            "angle_deg": float(angle_deg),
            "input_points": int(len(pts)),
            "output_points": int(len(transformed)),
            "centroid_before_m": np.mean(pts, axis=0).astype(float).tolist(),
            "centroid_after_m": np.mean(transformed, axis=0).astype(float).tolist(),
            "icp": {"enabled": False, "applied": False},
            "top_anchor": {"enabled": top_anchor_enabled, "applied": False},
        }

        if top_anchor_enabled:
            if reference_top_anchor is None:
                reference_top_anchor, reference_top_meta = top_anchor(
                    transformed,
                    rotation_axis_dir,
                    percentile=float(top_anchor_config.get("percentile", 90.0)),
                    side=str(top_anchor_config.get("side", "max")),
                    min_points=int(top_anchor_config.get("min_points", 200)),
                )
                scan_meta["top_anchor"] = {
                    "enabled": True,
                    "role": "reference",
                    "applied": False,
                    "reference_anchor": reference_top_meta,
                }
            else:
                transformed, top_meta = apply_top_anchor_refinement(
                    transformed,
                    reference_top_anchor,
                    rotation_axis_dir,
                    top_anchor_config,
                )
                scan_meta["top_anchor"] = top_meta
                scan_meta["centroid_after_m"] = np.mean(transformed, axis=0).astype(float).tolist()

        if refine_enabled and parts:
            target = np.concatenate(parts, axis=0)
            refined, icp_meta = refine_cloud_icp(
                transformed,
                target,
                max_points=int(refine_config.get("max_points", 12000)),
                max_iterations=int(refine_config.get("max_iterations", 20)),
                max_correspondence_m=float(refine_config.get("max_correspondence_m", 0.012)),
                max_rotation_deg=float(refine_config.get("max_rotation_deg", 8.0)),
                max_translation_m=float(refine_config.get("max_translation_m", 0.025)),
                min_pairs=int(refine_config.get("min_pairs", 200)),
                seed=int(refine_config.get("seed", 0)) + len(parts),
            )
            transformed = refined
            scan_meta["output_points"] = int(len(transformed))
            scan_meta["centroid_after_m"] = np.mean(transformed, axis=0).astype(float).tolist()
            scan_meta["icp"] = icp_meta

        color = palette[len(parts) % len(palette)]
        colors.append(np.tile(color, (len(transformed), 1)))
        parts.append(transformed.astype(np.float32))
        per_scan_meta.append(scan_meta)

    merged = np.concatenate(parts, axis=0)
    merged_colors = np.concatenate(colors, axis=0) if colors else np.zeros((0, 3), dtype=np.uint8)
    return merged, axis, per_scan_meta, parts, merged_colors


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_merged_ply(points: np.ndarray, path: Path) -> None:
    """Write Nx3 XYZ point cloud as an ASCII PLY file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)
        for x, y, z in points:
            fh.write(f"{float(x):.6f} {float(y):.6f} {float(z):.6f}\n")


def save_colored_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    """Write Nx3 XYZ plus uint8 RGB point cloud as an ASCII PLY file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)
        for (x, y, z), (r, g, b) in zip(points, colors):
            fh.write(f"{float(x):.6f} {float(y):.6f} {float(z):.6f} {int(r)} {int(g)} {int(b)}\n")


def load_scan_cloud(run_dir: Path) -> np.ndarray:
    """
    Load the reconstructed point cloud from a single-angle scan run directory.
    Returns an Nx3 float32 array (may be empty).
    """
    run_dir = Path(run_dir)
    xyz_path = run_dir / "reconstruct" / "xyz.npy"
    mask_path = run_dir / "reconstruct" / "masks" / "mask_reconstruct.npy"
    if not xyz_path.exists() or not mask_path.exists():
        return np.zeros((0, 3), dtype=np.float32)
    xyz = np.load(xyz_path)
    mask = np.load(mask_path).astype(bool)
    finite = np.all(np.isfinite(xyz), axis=-1)
    pts = xyz[mask & finite]
    return pts.astype(np.float32)


def merge_run_dirs(
    run_dirs_with_angles: list[tuple[float, Path]],
    config: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """
    Load clouds from scan run directories, merge them, and save the result.

    Args:
        run_dirs_with_angles: List of (angle_deg, run_dir) pairs.
        config:               Full application config dict.
        out_dir:              Directory to save merged.ply and merge_meta.json.

    Returns:
        Summary dict with totals and paths.
    """
    ms_cfg = config.get("multi_scan") or {}
    rotation_sign = float(ms_cfg.get("rotation_sign", 1.0))
    axis_vector = ms_cfg.get("rotation_axis_vector")
    rotation_axis: str | np.ndarray
    if axis_vector is not None:
        rotation_axis = np.asarray(axis_vector, dtype=np.float64)
    else:
        rotation_axis = str(ms_cfg.get("rotation_axis", "z"))
    refine_config = dict(ms_cfg.get("refine_icp") or {})

    axis: np.ndarray | None = None
    ax_x = ms_cfg.get("axis_x_m")
    ax_y = ms_cfg.get("axis_y_m")
    ax_z = ms_cfg.get("axis_z_m")

    clouds_with_angles: list[tuple[float, np.ndarray]] = []
    for angle_deg, run_dir in run_dirs_with_angles:
        pts = load_scan_cloud(run_dir)
        clouds_with_angles.append((angle_deg, pts))
        print(f"[merge] angle={angle_deg:.1f}° → {len(pts)} points from {run_dir.name}")

    fid_cfg = dict(ms_cfg.get("fiducials") or {})
    fid_meta: list[dict[str, Any]] = []
    fid_all_applied = False
    if bool(fid_cfg.get("enabled", False)):
        raw_coords = fid_cfg.get("pixel_coords", [])
        pixel_coords_list = [(int(c[0]), int(c[1])) for c in raw_coords if len(c) >= 2]
        if len(pixel_coords_list) >= 3:
            run_dirs_list = [rd for _, rd in run_dirs_with_angles]
            clouds_with_angles, fid_meta, fid_all_applied = align_clouds_from_fiducials(
                clouds_with_angles, run_dirs_list, pixel_coords_list
            )
            n_ok = sum(1 for m in fid_meta if m.get("applied") or m.get("role") == "reference")
            print(f"[merge] fiducials: {n_ok}/{len(fid_meta)} scans aligned via 4-point Procrustes")
        else:
            print(f"[merge] fiducials: need ≥3 pixel_coords, got {len(pixel_coords_list)} — skipping")

    if ax_x is not None and ax_y is not None:
        valid_clouds = [p for _, p in clouds_with_angles if len(p) > 0]
        mean_z = float(np.mean([np.mean(p[:, 2]) for p in valid_clouds])) if valid_clouds else 0.0
        axis = np.array([float(ax_x), float(ax_y), float(ax_z) if ax_z is not None else mean_z], dtype=np.float64)
        print(f"[merge] Using configured axis: x={axis[0]:.4f}m, y={axis[1]:.4f}m, z={axis[2]:.4f}m")

    aft_cfg = dict(ms_cfg.get("axis_from_top") or {})
    aft_meta: dict[str, Any] = {"enabled": bool(aft_cfg.get("enabled", False)), "applied": False}
    if fid_all_applied:
        aft_meta = {"enabled": False, "applied": False, "reason": "skipped_fiducials_active"}
    if aft_meta["enabled"] and axis is not None:
        axis, aft_meta = optimize_axis_from_top(
            clouds_with_angles,
            axis,
            rotation_sign,
            rotation_axis,
            percentile=float(aft_cfg.get("percentile", 95.0)),
            side=str(aft_cfg.get("side", "max")),
            min_points=int(aft_cfg.get("min_points", 500)),
        )
        if aft_meta.get("ok"):
            print(
                f"[merge] axis_from_top: P_opt={[f'{v:.5f}' for v in axis.tolist()]} "
                f"rms={aft_meta.get('anchor_residual_rms_m', float('nan')):.4f}m "
                f"max={aft_meta.get('anchor_residual_max_m', float('nan')):.4f}m"
            )
        else:
            print(f"[merge] axis_from_top failed: {aft_meta.get('reason', '?')} — using initial axis")

    afe_cfg = dict(ms_cfg.get("axis_from_edges") or {})
    afe_meta: dict[str, Any] = {"enabled": bool(afe_cfg.get("enabled", False)), "applied": False}
    if fid_all_applied:
        afe_meta = {"enabled": False, "applied": False, "reason": "skipped_fiducials_active"}
    if afe_meta["enabled"] and axis is not None:
        axis, afe_meta = optimize_axis_from_edges(
            clouds_with_angles,
            axis,
            rotation_sign,
            rotation_axis,
            max_shift_m=float(afe_cfg.get("max_shift_m", 0.015)),
            coarse_steps=int(afe_cfg.get("coarse_steps", 7)),
            sample_points=int(afe_cfg.get("sample_points", 3000)),
            seed=int(afe_cfg.get("seed", 1)),
        )
        if afe_meta.get("applied"):
            print(
                f"[merge] axis_from_edges: P_opt={[f'{v:.5f}' for v in axis.tolist()]} "
                f"score={afe_meta.get('score_m', float('nan')):.4f}m"
            )
        else:
            print(f"[merge] axis_from_edges failed: {afe_meta.get('reason', '?')} — using previous axis")

    merged, axis_used, per_scan_meta, _parts, colors = merge_scans(
        clouds_with_angles,
        axis=axis,
        rotation_sign=rotation_sign,
        rotation_axis=rotation_axis,
        refine_config=refine_config,
    )
    print(f"[merge] Merged {len(merged)} total points from {len(clouds_with_angles)} scans")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / "merged.ply"
    colored_ply_path = out_dir / "merged_by_angle.ply"
    save_merged_ply(merged, ply_path)
    if len(colors) == len(merged):
        save_colored_ply(merged, colors, colored_ply_path)

    meta = {
        "n_scans": len(run_dirs_with_angles),
        "angles_deg": [a for a, _ in run_dirs_with_angles],
        "points_per_scan": [len(p) for _, p in clouds_with_angles],
        "total_points": int(len(merged)),
        "axis_m": axis_used.tolist(),
        "rotation_sign": rotation_sign,
        "rotation_axis": rotation_axis.tolist() if isinstance(rotation_axis, np.ndarray) else rotation_axis,
        "fiducials": fid_meta,
        "axis_from_top": aft_meta,
        "axis_from_edges": afe_meta,
        "refine_icp": refine_config,
        "per_scan": per_scan_meta,
        "ply_path": str(ply_path),
        "colored_ply_path": str(colored_ply_path),
    }
    (out_dir / "merge_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
