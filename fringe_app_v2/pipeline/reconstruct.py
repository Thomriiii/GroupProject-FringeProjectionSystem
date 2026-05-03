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
from fringe_app_v2.utils.io import RunPaths, save_mask_png, write_json
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
    quality = _reconstruction_quality(meta, recon_cfg)
    meta["quality_gate"] = quality

    write_json(run.reconstruct / "reconstruction_meta.json", meta)
    if bool(recon_cfg.get("fail_on_quality_error", True)) and not bool(quality.get("ok", True)):
        raise RuntimeError("Reconstruction quality gate failed: " + "; ".join(quality.get("reasons", [])))

    # Export point cloud
    try:
        _export_pointcloud(run.reconstruct, tri_result.xyz, tri_result.mask)
    except Exception as e:
        print(f"[reconstruct] Warning: point cloud export failed: {e}")

    # Save debug images
    if config.get("debug_reconstruct", False):
        _save_debug_images(run.reconstruct, tri_result, uv_map)

    return meta


def _reconstruction_quality(meta: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("quality_gate", {}) or {}
    if not bool(cfg.get("enabled", True)):
        return {"enabled": False, "ok": True, "reasons": []}

    valid_uv = int(meta.get("valid_uv_points") or 0)
    valid_reconstruct = int(meta.get("valid_reconstruct_points") or 0)
    rejected_proj = int(meta.get("rejected_by_projector_reproj_px") or 0)
    reconstruct_ratio = valid_reconstruct / float(valid_uv) if valid_uv > 0 else 0.0
    projector_reject_ratio = rejected_proj / float(valid_uv) if valid_uv > 0 else 1.0
    reasons: list[str] = []

    min_uv = int(cfg.get("min_valid_uv_points", 1000))
    if valid_uv < min_uv:
        reasons.append(f"valid_uv_points<{min_uv}")
    min_reconstruct = int(cfg.get("min_valid_reconstruct_points", 1000))
    if valid_reconstruct < min_reconstruct:
        reasons.append(f"valid_reconstruct_points<{min_reconstruct}")
    min_ratio = float(cfg.get("min_reconstruct_to_uv_ratio", 0.10))
    if valid_uv > 0 and reconstruct_ratio < min_ratio:
        reasons.append(f"reconstruct_to_uv_ratio<{min_ratio:.3f}")
    max_reject = float(cfg.get("max_projector_reject_ratio", 0.50))
    if projector_reject_ratio > max_reject:
        reasons.append(f"projector_reject_ratio>{max_reject:.3f}")
    max_reproj_mean = cfg.get("max_reproj_proj_mean_px")
    reproj_mean = meta.get("reproj_proj_mean_px")
    if max_reproj_mean is not None and reproj_mean is not None and float(reproj_mean) > float(max_reproj_mean):
        reasons.append(f"reproj_proj_mean_px>{float(max_reproj_mean):.2f}")

    return {
        "enabled": True,
        "ok": not reasons,
        "reasons": reasons,
        "valid_uv_points": valid_uv,
        "valid_reconstruct_points": valid_reconstruct,
        "reconstruct_to_uv_ratio": reconstruct_ratio,
        "projector_reject_ratio": projector_reject_ratio,
    }


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

