"""
3D Reconstruction stage: Unwrapped phase → 3D point cloud via triangulation.

PIPELINE:
  1. Load unwrapped phase maps (horizontal & vertical)
  2. Convert phase → projector pixel coordinates (u, v)
  3. Build camera and projector ray models
  4. Triangulate ray pairs → 3D points
  5. Filter by reprojection error and depth range
  6. Save point cloud and diagnostic outputs
"""

from __future__ import annotations

from typing import Any
from pathlib import Path
import json

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy import ndimage

from fringe_app_v2.core.calibration import CalibrationService
from fringe_app_v2.utils.io import RunPaths, freq_tag, save_mask_png, write_json
from fringe_app_v2.pipeline.phase_to_projector import (
    phase_to_projector_coords,
    validate_uv_map,
)
from fringe_app_v2.pipeline.triangulation import (
    CameraModel,
    ProjectorModel,
    triangulate_from_uv,
)


def run_reconstruct_stage(
    run: RunPaths,
    calibration: CalibrationService,
    config: dict[str, Any],
    roi_mask: np.ndarray | None,
) -> dict[str, Any]:
    """
    Execute full 3D reconstruction pipeline.

    Args:
        run: RunPaths with input/output directories
        calibration: CalibrationService with camera/projector models
        config: Configuration dict with:
            - scan.frequencies: List of modulation frequencies
            - scan.frequency_semantics: "cycles_across_dimension" or "pixels_per_period"
            - scan.phase_origin_rad: Phase offset (default 0)
            - reconstruction: Triangulation config (z_min_m, z_max_m, max_reproj_err_px)
            - uv_gate: Validation config
        roi_mask: Optional HxW mask for region-of-interest

    Returns:
        Metadata dict with reconstruction statistics
    """
    # Load unwrapped phase
    freqs = [float(v) for v in (config.get("scan", {}) or {}).get("frequencies", [])]
    if not freqs:
        raise ValueError("scan.frequencies is required")
    high_freq = float(max(freqs))

    vertical_dir = run.unwrap / "vertical"
    horizontal_dir = run.unwrap / "horizontal"
    if not vertical_dir.exists() or not horizontal_dir.exists():
        raise FileNotFoundError(
            "Both vertical and horizontal unwrap outputs are required for 3D reconstruction"
        )

    print("[reconstruct] Loading unwrapped phase maps...")
    phi_vertical = np.load(vertical_dir / "phi_abs.npy").astype(np.float32)
    phi_horizontal = np.load(horizontal_dir / "phi_abs.npy").astype(np.float32)
    mask_vertical = np.load(vertical_dir / "mask_unwrap.npy").astype(bool)
    mask_horizontal = np.load(horizontal_dir / "mask_unwrap.npy").astype(bool)

    # Phase → Projector coordinates
    print("[reconstruct] Converting unwrapped phase to projector coordinates...")
    scan_cfg = config.get("scan", {}) or {}
    uv_map = phase_to_projector_coords(
        phi_horizontal=phi_horizontal,
        phi_vertical=phi_vertical,
        mask_horizontal=mask_horizontal,
        mask_vertical=mask_vertical,
        projector_width=calibration.projector.projector_size[0],
        projector_height=calibration.projector.projector_size[1],
        frequency_u=high_freq,
        frequency_v=high_freq,
        frequency_semantics=str(scan_cfg.get("frequency_semantics", "cycles_across_dimension")),
        phase_origin_u_rad=float(scan_cfg.get("phase_origin_u_rad", scan_cfg.get("phase_origin_rad", 0.0))),
        phase_origin_v_rad=float(scan_cfg.get("phase_origin_v_rad", scan_cfg.get("phase_origin_rad", 0.0))),
        projector_u_offset_px=float(scan_cfg.get("projector_u_offset_px", 0.0)),
        projector_v_offset_px=float(scan_cfg.get("projector_v_offset_px", 0.0)),
        roi_mask=roi_mask,
    )

    # Validate UV map
    is_valid, errors = validate_uv_map(
        uv_map,
        min_valid_ratio=float((config.get("uv_gate", {}) or {}).get("min_valid_ratio", 0.03)),
        min_u_range=float((config.get("uv_gate", {}) or {}).get("min_u_range_px", 40.0)),
        min_v_range=float((config.get("uv_gate", {}) or {}).get("min_v_range_px", 40.0)),
        max_edge_pct=float((config.get("uv_gate", {}) or {}).get("max_edge_pct", 0.10)),
        max_zero_pct=float((config.get("uv_gate", {}) or {}).get("max_zero_pct", 0.01)),
    )
    if not is_valid:
        print("[reconstruct] ⚠️  UV map validation warnings:")
        for error in errors:
            print(f"  - {error}")

    # Build camera and projector models
    cam_model = CameraModel(
        matrix=calibration.camera.matrix,
        dist_coeffs=calibration.camera.dist_coeffs,
        image_size=calibration.camera.image_size,
    )
    proj_model = ProjectorModel(
        matrix=calibration.projector.matrix,
        dist_coeffs=calibration.projector.dist_coeffs,
        rotation=calibration.projector.rotation,
        translation=calibration.projector.translation,
        size=calibration.projector.projector_size,
    )

    # Triangulate
    print("[reconstruct] Performing ray-ray triangulation...")
    recon_cfg = config.get("reconstruction", {}) or {}
    recon_cfg["debug"] = bool(config.get("debug_reconstruct", False))
    tri_result = triangulate_from_uv(
        u_map=uv_map.u,
        v_map=uv_map.v,
        mask_uv=uv_map.mask,
        camera_model=cam_model,
        projector_model=proj_model,
        config=recon_cfg,
    )

    # Stage 1: remove isolated 2D pixel islands (fast, image-space)
    n_comp_removed = 0
    min_comp_px = int(recon_cfg.get("min_component_px", 50))
    if min_comp_px > 0 and np.any(tri_result.mask):
        clean_mask, n_comp_removed = _remove_small_components(tri_result.mask, min_comp_px)
        tri_result.mask[:] = clean_mask
        tri_result.depth[~clean_mask] = np.nan
        tri_result.xyz[~clean_mask] = np.nan
        print(f"[reconstruct] Component filter removed {n_comp_removed} pixels from small islands")

    # Stage 2: statistical outlier removal (3D neighbor distance)
    n_sor_removed = 0
    if bool(recon_cfg.get("sor_enabled", True)) and np.any(tri_result.mask):
        sor_k = int(recon_cfg.get("sor_k", 20))
        sor_std = float(recon_cfg.get("sor_std_multiplier", 3.0))
        print(f"[reconstruct] Statistical outlier removal (k={sor_k}, std={sor_std})...")
        clean_mask, n_sor_removed = _statistical_outlier_removal(
            tri_result.xyz, tri_result.mask, k=sor_k, std_multiplier=sor_std
        )
        tri_result.mask[:] = clean_mask
        tri_result.depth[~clean_mask] = np.nan
        tri_result.xyz[~clean_mask] = np.nan
        print(f"[reconstruct] SOR removed {n_sor_removed} outlier points")

    # Save outputs
    print("[reconstruct] Saving reconstruction outputs...")
    _save_reconstruction(
        run.reconstruct,
        tri_result,
        uv_map,
        calibration,
        recon_cfg,
    )

    # Build metadata
    meta = _build_metadata(
        tri_result,
        uv_map,
        calibration,
        recon_cfg,
    )
    meta["component_filter_removed"] = n_comp_removed
    meta["sor_removed"] = n_sor_removed

    # Reconstruct fiducial mark pixels (bypasses ROI — marks on the turntable bed)
    ms_cfg = config.get("multi_scan") or {}
    fid_cfg = ms_cfg.get("fiducials") or {}
    if bool(fid_cfg.get("enabled", False)):
        pixel_coords_list = _detect_fiducial_pixels(run, fid_cfg)
        if pixel_coords_list:
            fid_results = _reconstruct_fiducial_pixels(run, config, calibration, pixel_coords_list)
            write_json(run.reconstruct / "fiducials.json", fid_results)
            n_ok = sum(1 for r in fid_results if r.get("ok"))
            print(f"[reconstruct] Fiducials: {n_ok}/{len(fid_results)} marks reconstructed OK")
            meta["fiducials"] = fid_results

    write_json(run.reconstruct / "reconstruction_meta.json", meta)

    # Export point cloud
    try:
        _export_pointcloud(run.reconstruct, tri_result.xyz, tri_result.mask)
    except Exception as e:
        print(f"[reconstruct] Warning: point cloud export failed: {e}")

    # Save debug images
    if config.get("debug_reconstruct", False):
        _save_debug_images(run.reconstruct, tri_result, uv_map)

    return meta


def _remove_small_components(
    mask: np.ndarray,
    min_px: int,
) -> tuple[np.ndarray, int]:
    """
    Remove connected pixel groups smaller than min_px from the 2D reconstruction mask.

    Scattered phase-noise pixels form tiny isolated islands in the camera image.
    They triangulate to valid-looking but wrong 3D positions. Removing them in
    image space is cheaper and more precise than 3D outlier methods.

    Returns updated mask and number of pixels removed.
    """
    labeled, n_comp = ndimage.label(mask)
    if n_comp == 0:
        return mask, 0

    new_mask = mask.copy()
    removed = 0
    for label in range(1, n_comp + 1):
        comp = labeled == label
        size = int(comp.sum())
        if size < min_px:
            new_mask[comp] = False
            removed += size

    return new_mask, removed


def _statistical_outlier_removal(
    xyz: np.ndarray,
    mask: np.ndarray,
    k: int = 20,
    std_multiplier: float = 2.0,
) -> tuple[np.ndarray, int]:
    """
    Remove isolated points whose mean distance to k nearest neighbors exceeds
    mean + std_multiplier * std across all points.

    Returns updated mask and number of points removed.
    """
    ys, xs = np.where(mask)
    pts = xyz[ys, xs, :]

    if len(pts) < k + 1:
        return mask, 0

    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k + 1)
    mean_dist = dists[:, 1:].mean(axis=1)  # exclude self (index 0)

    threshold = mean_dist.mean() + std_multiplier * mean_dist.std()
    inlier = mean_dist <= threshold

    new_mask = np.zeros_like(mask)
    new_mask[ys[inlier], xs[inlier]] = True
    return new_mask, int((~inlier).sum())


def _save_reconstruction(
    out_dir: Path,
    tri_result,  # TriangulationResult
    uv_map,  # UVMap
    calibration: CalibrationService,
    config: dict[str, Any],
) -> None:
    """Save 3D reconstruction data and diagnostic outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Core 3D geometry
    np.save(out_dir / "xyz.npy", tri_result.xyz.astype(np.float32))
    np.save(out_dir / "depth.npy", tri_result.depth.astype(np.float32))
    np.save(out_dir / "height.npy", tri_result.depth.astype(np.float32))  # Alias for depth
    np.save(out_dir / "height_map.npy", tri_result.depth.astype(np.float32))  # Compatibility

    # UV coordinates
    uv_stack = np.stack([uv_map.u.astype(np.float32), uv_map.v.astype(np.float32)], axis=-1)
    uv_stack[~uv_map.mask.astype(bool), :] = np.nan
    np.save(out_dir / "uv_map.npy", uv_stack.astype(np.float32))

    # Masks
    np.save(masks_dir / "mask_uv.npy", uv_map.mask.astype(bool))
    np.save(masks_dir / "mask_reconstruct.npy", tri_result.mask.astype(bool))
    np.save(masks_dir / "mask_recon.npy", tri_result.mask.astype(bool))  # Alias

    # Error maps
    np.save(out_dir / "reproj_err_cam.npy", tri_result.reproj_err_cam.astype(np.float32))
    np.save(out_dir / "reproj_err_proj.npy", tri_result.reproj_err_proj.astype(np.float32))

    # PNG visualizations
    save_mask_png(masks_dir / "mask_uv.png", uv_map.mask)
    save_mask_png(masks_dir / "mask_reconstruct.png", tri_result.mask)
    save_mask_png(masks_dir / "mask_recon.png", tri_result.mask)


def _build_metadata(
    tri_result,  # TriangulationResult
    uv_map,  # UVMap
    calibration: CalibrationService,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build comprehensive reconstruction metadata."""
    depth_values = tri_result.depth[tri_result.mask]
    reproj_cam_valid = tri_result.reproj_err_cam[tri_result.mask]
    reproj_proj_valid = tri_result.reproj_err_proj[tri_result.mask]

    meta = {
        "pipeline_version": "2.0",
        "algorithm": "ray-ray triangulation",
        "camera_intrinsics": str(calibration.camera.path),
        "projector_stereo": str(calibration.projector.path),
        "camera_size": list(calibration.camera.image_size),
        "projector_size": list(calibration.projector.projector_size),
        # Valid UV correspondences
        "valid_uv_points": int(np.count_nonzero(uv_map.mask)),
        # Valid 3D points after filtering
        "valid_reconstruct_points": int(np.count_nonzero(tri_result.mask)),
        # Filter thresholds
        "z_min_m": float(config.get("z_min_m", 0.05)),
        "z_max_m": float(config.get("z_max_m", 5.0)),
        "max_reproj_err_px": float(config.get("max_reproj_err_px", 2.0)),
        # Rejection statistics
        "rejected_by_depth_px": int(tri_result.debug.get("n_rejected_by_depth", 0)),
        "rejected_by_camera_reproj_px": int(tri_result.debug.get("n_rejected_by_cam_reproj", 0)),
        "rejected_by_projector_reproj_px": int(tri_result.debug.get("n_rejected_by_proj_reproj", 0)),
        # Depth statistics
        "height_min_m": float(np.nanmin(depth_values)) if depth_values.size > 0 else None,
        "height_max_m": float(np.nanmax(depth_values)) if depth_values.size > 0 else None,
        "height_mean_m": float(np.nanmean(depth_values)) if depth_values.size > 0 else None,
        "height_median_m": float(np.nanmedian(depth_values)) if depth_values.size > 0 else None,
        "height_std_m": float(np.nanstd(depth_values)) if depth_values.size > 0 else None,
        # Reprojection error statistics
        "reproj_cam_mean_px": float(np.nanmean(reproj_cam_valid)) if reproj_cam_valid.size > 0 else None,
        "reproj_cam_max_px": float(np.nanmax(reproj_cam_valid)) if reproj_cam_valid.size > 0 else None,
        "reproj_proj_mean_px": float(np.nanmean(reproj_proj_valid)) if reproj_proj_valid.size > 0 else None,
        "reproj_proj_max_px": float(np.nanmax(reproj_proj_valid)) if reproj_proj_valid.size > 0 else None,
        # UV map info
        "uv_meta": uv_map.meta,
        # Triangulation debug info
        "triangulation_debug": tri_result.debug,
    }

    return meta


def _save_debug_images(
    out_dir: Path,
    tri_result,  # TriangulationResult
    uv_map,  # UVMap
) -> None:
    """Save diagnostic images for visualization and debugging."""
    out_dir = Path(out_dir)
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    h, w = tri_result.mask.shape

    # Depth map
    depth_viz = np.full((h, w, 3), 0, dtype=np.uint8)
    if np.any(tri_result.mask):
        depth_valid = tri_result.depth[tri_result.mask]
        d_min, d_max = np.nanmin(depth_valid), np.nanmax(depth_valid)
        if d_max > d_min:
            depth_norm = ((tri_result.depth - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
            depth_viz[:, :, 0] = depth_norm
            depth_viz[:, :, 1] = depth_norm
            depth_viz[:, :, 2] = depth_norm
            depth_viz[~tri_result.mask] = [32, 32, 32]  # Dark for invalid
    Image.fromarray(depth_viz).save(debug_dir / "depth_map.png")

    # Reprojection error (camera)
    err_cam_viz = np.zeros((h, w), dtype=np.uint8)
    if np.any(np.isfinite(tri_result.reproj_err_cam)):
        err_valid = tri_result.reproj_err_cam[np.isfinite(tri_result.reproj_err_cam)]
        err_max = np.percentile(err_valid, 99)  # Use 99th percentile to avoid outliers
        mask_valid = np.isfinite(tri_result.reproj_err_cam) & (tri_result.reproj_err_cam <= err_max)
        err_cam_viz[mask_valid] = np.clip(
            (tri_result.reproj_err_cam[mask_valid] / err_max * 255.0), 0, 255
        ).astype(np.uint8)
    Image.fromarray(err_cam_viz).save(debug_dir / "reproj_err_camera.png")

    # Reprojection error (projector)
    err_proj_viz = np.zeros((h, w), dtype=np.uint8)
    if np.any(np.isfinite(tri_result.reproj_err_proj)):
        err_valid = tri_result.reproj_err_proj[np.isfinite(tri_result.reproj_err_proj)]
        err_max = np.percentile(err_valid, 99)
        mask_valid = np.isfinite(tri_result.reproj_err_proj) & (tri_result.reproj_err_proj <= err_max)
        err_proj_viz[mask_valid] = np.clip(
            (tri_result.reproj_err_proj[mask_valid] / err_max * 255.0), 0, 255
        ).astype(np.uint8)
    Image.fromarray(err_proj_viz).save(debug_dir / "reproj_err_projector.png")

    # UV coordinates visualization
    u_viz = np.zeros((h, w), dtype=np.uint8)
    v_viz = np.zeros((h, w), dtype=np.uint8)
    if np.any(uv_map.mask):
        u_valid = uv_map.u[uv_map.mask]
        v_valid = uv_map.v[uv_map.mask]
        u_norm = ((u_valid - np.min(u_valid)) / (np.max(u_valid) - np.min(u_valid)) * 255.0).astype(np.uint8)
        v_norm = ((v_valid - np.min(v_valid)) / (np.max(v_valid) - np.min(v_valid)) * 255.0).astype(np.uint8)
        u_viz[uv_map.mask] = u_norm
        v_viz[uv_map.mask] = v_norm
    Image.fromarray(u_viz).save(debug_dir / "projector_u.png")
    Image.fromarray(v_viz).save(debug_dir / "projector_v.png")

    print(f"[reconstruct] Debug images saved to {debug_dir}")


def _export_pointcloud(out_dir: Path, xyz: np.ndarray, mask: np.ndarray) -> None:
    """
    Export 3D point cloud to PLY format.

    Args:
        out_dir: Output directory
        xyz: HxWx3 array of 3D points
        mask: HxW boolean mask (True = valid)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError("xyz must be HxWx3 array")
    if mask.shape != xyz.shape[:2]:
        raise ValueError("mask shape must match xyz HxW")

    # Extract valid points
    flat_pts = xyz.reshape(-1, 3)
    flat_mask = mask.ravel()
    valid_pts = flat_pts[flat_mask]

    if valid_pts.size == 0:
        print("[reconstruct] Warning: No valid points to export")
        return

    # Write PLY header
    header = [
        "ply\n",
        "format ascii 1.0\n",
        f"element vertex {len(valid_pts)}\n",
        "property float x\n",
        "property float y\n",
        "property float z\n",
        "end_header\n",
    ]

    # Write to two filenames for compatibility
    for cloud_path in [out_dir / "cloud.ply", out_dir / "pointcloud.ply"]:
        try:
            with open(cloud_path, "w", encoding="utf8") as fh:
                fh.writelines(header)
                for x, y, z in valid_pts:
                    fh.write(f"{float(x):.6f} {float(y):.6f} {float(z):.6f}\n")
        except Exception as e:
            print(f"[reconstruct] Warning: Failed to write {cloud_path}: {e}")


def _detect_fiducial_pixels(
    run: RunPaths,
    fid_cfg: dict[str, Any],
) -> list[tuple[int, int]]:
    """
    Auto-detect fiducial mark pixel positions from the ROI capture image.

    Finds bright colored marks using HSV thresholding, clusters them, and
    returns the N largest plausible cluster centroids sorted top-to-bottom.
    Falls back to fixed pixel_coords from config if detection fails.
    """
    roi_capture = run.raw / "roi_capture.png"
    if not roi_capture.exists():
        return [(int(c[0]), int(c[1])) for c in fid_cfg.get("pixel_coords", [])]

    try:
        from PIL import Image as _PIL
        from scipy import ndimage as _ndi

        arr = np.array(_PIL.open(roi_capture)).astype(np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        r, g, b = arr[:, :, 0] / 255.0, arr[:, :, 1] / 255.0, arr[:, :, 2] / 255.0

        raw_hue_ranges = fid_cfg.get("detect_hue_ranges")
        if raw_hue_ranges:
            hue_ranges = [
                (float(pair[0]) % 360.0, float(pair[1]) % 360.0)
                for pair in raw_hue_ranges
                if isinstance(pair, (list, tuple)) and len(pair) >= 2
            ]
        else:
            hue_ranges = [
                (
                    float(fid_cfg.get("detect_hue_lo", 0.0)) % 360.0,
                    float(fid_cfg.get("detect_hue_hi", 30.0)) % 360.0,
                )
            ]
        sat_min = float(fid_cfg.get("detect_sat_min", 0.55))
        val_min = float(fid_cfg.get("detect_val_min", 0.35))
        min_px = int(fid_cfg.get("detect_min_px", 50))
        max_px = int(fid_cfg.get("detect_max_px", 6000))
        n_marks = int(fid_cfg.get("n_marks", 4))

        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        delta = maxc - minc
        sat = np.zeros_like(maxc)
        np.divide(delta, maxc, out=sat, where=maxc > 1e-6)
        val = maxc
        hue = np.zeros_like(r)
        m = delta > 1e-6
        mr = m & (maxc == r)
        mg = m & (maxc == g)
        mb = m & (maxc == b)
        hue[mr] = (60.0 * ((g[mr] - b[mr]) / delta[mr])) % 360.0
        hue[mg] = 60.0 * ((b[mg] - r[mg]) / delta[mg]) + 120.0
        hue[mb] = 60.0 * ((r[mb] - g[mb]) / delta[mb]) + 240.0

        in_hue = np.zeros_like(hue, dtype=bool)
        for hue_lo, hue_hi in hue_ranges:
            if hue_lo <= hue_hi:
                in_hue |= (hue >= hue_lo) & (hue <= hue_hi)
            else:
                in_hue |= (hue >= hue_lo) | (hue <= hue_hi)

        mask = in_hue & (sat >= sat_min) & (val >= val_min)
        labeled, n = _ndi.label(mask)
        if n == 0:
            raise ValueError("no colored marks detected")

        clusters = []
        for i in range(1, n + 1):
            comp = labeled == i
            sz = int(comp.sum())
            if sz < min_px or (max_px > 0 and sz > max_px):
                continue
            ys, xs = np.where(comp)
            clusters.append((sz, int(xs.mean()), int(ys.mean())))

        clusters.sort(key=lambda c: -c[0])
        clusters = clusters[:n_marks]
        if len(clusters) < 3:
            raise ValueError(f"only {len(clusters)} marks detected, need ≥3")

        # Sort top-to-bottom for stable ordering
        clusters.sort(key=lambda c: c[2])
        coords = [(c[1], c[2]) for c in clusters]
        print(f"[reconstruct] Fiducials auto-detected: {coords}")
        return coords

    except Exception as exc:
        print(f"[reconstruct] Fiducial auto-detect failed ({exc}), using fixed pixel_coords")
        return [(int(c[0]), int(c[1])) for c in fid_cfg.get("pixel_coords", [])]


def _reconstruct_fiducial_pixels(
    run: RunPaths,
    config: dict[str, Any],
    calibration: CalibrationService,
    pixel_coords: list[tuple[int, int]],
) -> list[dict]:
    """
    Reconstruct 3D positions of fiducial mark pixels, bypassing the ROI mask.

    Runs a mini unwrap→UV→triangulate pipeline for each specified pixel using
    only the raw per-frequency phase masks (not the ROI-filtered unwrap outputs).
    Saves results to reconstruct/fiducials.json so merge can load them.
    """
    scan_cfg = config.get("scan", {}) or {}
    freqs = sorted([float(v) for v in scan_cfg.get("frequencies", [])])
    if len(freqs) < 2:
        return [{"u": u, "v": v, "ok": False, "reason": "too_few_frequencies"} for u, v in pixel_coords]

    high_freq = max(freqs)
    two_pi = 2.0 * np.pi
    proj_w, proj_h = calibration.projector.projector_size
    phase_origin_u = float(scan_cfg.get("phase_origin_u_rad", scan_cfg.get("phase_origin_rad", 0.0)))
    phase_origin_v = float(scan_cfg.get("phase_origin_v_rad", scan_cfg.get("phase_origin_rad", 0.0)))
    u_offset = float(scan_cfg.get("projector_u_offset_px", 0.0))
    v_offset = float(scan_cfg.get("projector_v_offset_px", 0.0))
    freq_semantics = str(scan_cfg.get("frequency_semantics", "cycles_across_dimension"))

    cam_model = CameraModel(
        matrix=calibration.camera.matrix,
        dist_coeffs=calibration.camera.dist_coeffs,
        image_size=calibration.camera.image_size,
    )
    proj_model = ProjectorModel(
        matrix=calibration.projector.matrix,
        dist_coeffs=calibration.projector.dist_coeffs,
        rotation=calibration.projector.rotation,
        translation=calibration.projector.translation,
        size=calibration.projector.projector_size,
    )
    recon_cfg = dict(config.get("reconstruction", {}) or {})
    recon_cfg["sor_enabled"] = False
    # Marks are on the flat turntable bed where calibration accuracy may be lower
    # than for the object. The reproj filter is relaxed so marks are accepted;
    # their errors are stored in fiducials.json for inspection.
    recon_cfg["max_reproj_err_px"] = float(
        (config.get("multi_scan") or {}).get("fiducials", {}).get("max_reproj_err_px", 200.0)
    )

    results = []
    for px_u, px_v in pixel_coords:
        mark: dict = {"u": int(px_u), "v": int(px_v), "ok": False}
        try:
            phi_abs_per_orient: dict[str, float] = {}
            failed = False
            for orientation in ("vertical", "horizontal"):
                phi_low: float | None = None
                prev_freq: float | None = None
                for freq in freqs:
                    phase_path = run.phase / orientation / freq_tag(freq)
                    phi_w = float(np.load(phase_path / "phi_wrapped.npy")[px_v, px_u])
                    raw_mask = bool(np.load(phase_path / "mask.npy")[px_v, px_u])
                    if not raw_mask or not np.isfinite(phi_w):
                        mark["reason"] = f"phase_invalid_{orientation}_freq{freq}"
                        failed = True
                        break
                    if phi_low is None:
                        phi_low = phi_w % two_pi
                    else:
                        r = freq / prev_freq  # type: ignore[operator]
                        k = round((r * phi_low - phi_w) / two_pi)
                        phi_low = phi_w + two_pi * k
                    prev_freq = freq
                if failed:
                    break
                phi_abs_per_orient[orientation] = phi_low  # type: ignore[assignment]

            if failed:
                results.append(mark)
                continue

            phi_v = phi_abs_per_orient["vertical"]
            phi_h = phi_abs_per_orient["horizontal"]

            if freq_semantics == "cycles_across_dimension":
                u_proj = (phi_v - phase_origin_u) / (two_pi * high_freq) * proj_w + u_offset
                v_proj = (phi_h - phase_origin_v) / (two_pi * high_freq) * proj_h + v_offset
            else:
                u_proj = phi_v / two_pi * high_freq + u_offset
                v_proj = phi_h / two_pi * high_freq + v_offset

            mark["u_proj"] = float(u_proj)
            mark["v_proj"] = float(v_proj)

            if not (0 <= u_proj < proj_w and 0 <= v_proj < proj_h):
                mark["reason"] = "uv_out_of_projector_bounds"
                results.append(mark)
                continue

            # Build full-size UV arrays (triangulate_from_uv validates image size)
            cam_h, cam_w = calibration.camera.image_size[1], calibration.camera.image_size[0]
            u_full = np.full((cam_h, cam_w), np.nan, dtype=np.float32)
            v_full = np.full((cam_h, cam_w), np.nan, dtype=np.float32)
            mask_full = np.zeros((cam_h, cam_w), dtype=bool)
            u_full[px_v, px_u] = float(u_proj)
            v_full[px_v, px_u] = float(v_proj)
            mask_full[px_v, px_u] = True

            tri = triangulate_from_uv(
                u_map=u_full,
                v_map=v_full,
                mask_uv=mask_full,
                camera_model=cam_model,
                projector_model=proj_model,
                config=recon_cfg,
            )

            if tri.mask[px_v, px_u]:
                mark["ok"] = True
                mark["xyz_m"] = [float(c) for c in tri.xyz[px_v, px_u]]
                mark["reproj_cam_px"] = float(tri.reproj_err_cam[px_v, px_u])
                mark["reproj_proj_px"] = float(tri.reproj_err_proj[px_v, px_u])
            else:
                mark["reason"] = "triangulation_rejected"

        except Exception as exc:
            mark["reason"] = f"error: {exc}"

        results.append(mark)

    return results
