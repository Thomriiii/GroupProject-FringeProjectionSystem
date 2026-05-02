"""
Core 3D triangulation from stereo image pairs using camera and projector intrinsics.

This module implements the fundamental ray-ray intersection algorithm:
  - Camera rays: emanate from camera origin through camera pixels
  - Projector rays: emanate from projector origin through projector pixels
  - Intersection: Finds the midpoint of the closest point between two skew lines

MATHEMATICAL BASIS:
  Each pixel defines a ray in 3D space:
    ray = origin + t * direction    (t ∈ [0, ∞))

  For a stereo pair (camera pixel, projector pixel):
    P_cam = C_origin + λ * ray_cam
    P_proj = P_origin + μ * ray_proj

  We seek λ, μ such that |P_cam - P_proj| is minimized.
  This is a least-squares problem (SVD-based in DLT).

WHY BOTH PHASE DIRECTIONS ARE REQUIRED:
  - Horizontal phase patterns → projector Y coordinates (v)
  - Vertical phase patterns → projector X coordinates (u)
  - Without both u and v, we only have 1D correspondence → underdetermined
  - With both u and v, we have full 2D projector pixel correspondence → unique 3D point
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import numpy as np
import cv2


@dataclass(slots=True)
class CameraModel:
    """Camera intrinsic and extrinsic parameters."""
    matrix: np.ndarray          # 3x3 camera matrix K
    dist_coeffs: np.ndarray     # Distortion coefficients
    image_size: tuple[int, int] # (width, height) in pixels


@dataclass(slots=True)
class ProjectorModel:
    """Projector intrinsic and extrinsic parameters (relative to camera)."""
    matrix: np.ndarray      # 3x3 projector matrix K
    dist_coeffs: np.ndarray # Distortion coefficients
    rotation: np.ndarray    # 3x3 rotation matrix R
    translation: np.ndarray # 3x1 translation vector t
    size: tuple[int, int]   # (width, height) in pixels


@dataclass(slots=True)
class TriangulationResult:
    """Output of triangulation operation."""
    xyz: np.ndarray                 # HxWx3 point cloud
    depth: np.ndarray               # HxW depth (Z) values
    mask: np.ndarray                # HxW boolean mask (True = valid)
    reproj_err_cam: np.ndarray      # HxW reprojection error in camera
    reproj_err_proj: np.ndarray     # HxW reprojection error in projector
    debug: dict[str, Any]           # Debug info: ray directions, intersections, etc.


def undistort_points(
    points_2d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """
    Convert distorted image coordinates to normalized undistorted coordinates.

    Args:
        points_2d: Nx1x2 array of (x, y) pixel coordinates
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficient vector

    Returns:
        Nx2 array of normalized coordinates in camera frame
    """
    points_norm = cv2.undistortPoints(points_2d, camera_matrix, dist_coeffs)
    return points_norm.reshape(-1, 2)


def build_camera_rays(
    points_norm: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct ray directions from normalized coordinates.

    For each normalized point (x_n, y_n), the ray direction is:
        direction = inv(K) @ [x_n, y_n, 1] = [x_n, y_n, 1] (after K is identity in normalized space)

    Actually: For normalized coordinates, the ray direction is simply (x_n, y_n, 1) normalized.

    Args:
        points_norm: Nx2 array of normalized coordinates
        camera_matrix: 3x3 camera matrix (used for consistency, though normalized coords already apply K^-1)

    Returns:
        Tuple of:
          - ray_origins: Nx3 (all zeros = camera center)
          - ray_directions: Nx3 normalized direction vectors
    """
    n_points = points_norm.shape[0]

    # Camera is at origin
    ray_origins = np.zeros((n_points, 3), dtype=np.float64)

    # Ray directions from normalized coordinates
    # For a point (x_n, y_n) in normalized space, the ray direction is (x_n, y_n, 1)
    ones = np.ones((n_points, 1), dtype=np.float64)
    ray_dirs_unnormalized = np.hstack([points_norm, ones])

    # Normalize to unit vectors
    ray_directions = ray_dirs_unnormalized / np.linalg.norm(ray_dirs_unnormalized, axis=1, keepdims=True)

    return ray_origins, ray_directions


def build_projector_rays(
    points_norm: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct projector ray origins and directions.

    Projector rays originate at the projector center P and pass through pixels in projector space.
    In world coordinates:
        ray_origin = P_center = -R.T @ t
        ray_direction (unnormalized) = R.T @ [x_n, y_n, 1]

    Args:
        points_norm: Nx2 array of normalized coordinates in projector space
        rotation: 3x3 rotation matrix (world → projector)
        translation: 3x1 translation vector (world → projector)

    Returns:
        Tuple of:
          - ray_origins: Nx3 (projector center in world coords)
          - ray_directions: Nx3 normalized direction vectors in world coords
    """
    n_points = points_norm.shape[0]

    # Projector center in world coordinates
    # P_world = -R.T @ t
    projector_center = -rotation.T @ translation.reshape(3)
    ray_origins = np.tile(projector_center, (n_points, 1))

    # Ray directions in projector space: (x_n, y_n, 1)
    ones = np.ones((n_points, 1), dtype=np.float64)
    rays_proj_frame = np.hstack([points_norm, ones])

    # Transform to world frame: R.T @ ray_dir
    ray_dirs_world = (rotation.T @ rays_proj_frame.T).T

    # Normalize to unit vectors
    ray_directions = ray_dirs_world / np.linalg.norm(ray_dirs_world, axis=1, keepdims=True)

    return ray_origins, ray_directions


def triangulate_rays(
    ray_origins_cam: np.ndarray,
    ray_dirs_cam: np.ndarray,
    ray_origins_proj: np.ndarray,
    ray_dirs_proj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Triangulate 3D points from pairs of rays using linear triangulation.

    For each ray pair, we solve:
        cam_origin + λ * cam_dir ≈ proj_origin + μ * proj_dir

    This is solved via DLT (Direct Linear Triangulation) using SVD.

    Args:
        ray_origins_cam: Nx3 camera ray origins (all zeros)
        ray_dirs_cam: Nx3 normalized camera ray directions
        ray_origins_proj: Nx3 projector ray origins
        ray_dirs_proj: Nx3 normalized projector ray directions

    Returns:
        Tuple of:
          - xyz_points: Nx3 array of triangulated 3D points
          - triangulation_error: Nx1 array of residuals
    """
    n_points = ray_dirs_cam.shape[0]
    xyz_points = np.zeros((n_points, 3), dtype=np.float64)
    tri_errors = np.zeros(n_points, dtype=np.float64)

    for i in range(n_points):
        # Build the system matrix for this ray pair
        # [ ray_cam_dir | -ray_proj_dir ] @ [λ; μ] = proj_origin - cam_origin
        A = np.column_stack([ray_dirs_cam[i], -ray_dirs_proj[i]])
        b = ray_origins_proj[i] - ray_origins_cam[i]

        # Solve via least squares (SVD)
        try:
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            lambda_val = x[0]
            
            # Compute the point
            point = ray_origins_cam[i] + lambda_val * ray_dirs_cam[i]
            xyz_points[i] = point
            
            # Triangulation error
            if residuals.size > 0:
                tri_errors[i] = np.sqrt(residuals[0])
            else:
                # Perfect solution or under-determined system
                tri_errors[i] = 0.0
        except np.linalg.LinAlgError:
            xyz_points[i] = np.nan
            tri_errors[i] = np.inf

    return xyz_points, tri_errors


def compute_reprojection_errors(
    xyz_points: np.ndarray,
    camera_matrix: np.ndarray,
    camera_dist_coeffs: np.ndarray,
    projector_matrix: np.ndarray,
    projector_dist_coeffs: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_points_obs: np.ndarray,
    projector_points_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute reprojection errors: how well the triangulated points project back to observations.

    Args:
        xyz_points: Nx3 world coordinates
        camera_matrix: 3x3 camera intrinsic
        camera_dist_coeffs: Camera distortion coefficients
        projector_matrix: 3x3 projector intrinsic
        projector_dist_coeffs: Projector distortion coefficients
        rotation: 3x3 rotation (world → projector)
        translation: 3x1 translation (world → projector)
        camera_points_obs: Nx2 observed camera pixels
        projector_points_obs: Nx2 observed projector pixels

    Returns:
        Tuple of:
          - reproj_err_cam: Nx1 pixel errors in camera image
          - reproj_err_proj: Nx1 pixel errors in projector image
    """
    n_points = xyz_points.shape[0]
    reproj_err_cam = np.full(n_points, np.inf, dtype=np.float64)
    reproj_err_proj = np.full(n_points, np.inf, dtype=np.float64)

    # Filter finite points
    finite_mask = np.isfinite(xyz_points).all(axis=1)
    if not np.any(finite_mask):
        return reproj_err_cam, reproj_err_proj

    xyz_valid = xyz_points[finite_mask].reshape(-1, 1, 3).astype(np.float32)

    # Camera reprojection
    try:
        cam_proj, _ = cv2.projectPoints(
            xyz_valid,
            np.zeros((3, 1), dtype=np.float32),
            np.zeros((3, 1), dtype=np.float32),
            camera_matrix.astype(np.float32),
            camera_dist_coeffs.astype(np.float32),
        )
        cam_proj = cam_proj.reshape(-1, 2)
        cam_obs_valid = camera_points_obs[finite_mask]
        reproj_err_cam[finite_mask] = np.linalg.norm(cam_proj - cam_obs_valid, axis=1)
    except Exception:
        pass

    # Projector reprojection
    try:
        rvec, _ = cv2.Rodrigues(rotation.astype(np.float32))
        proj_proj, _ = cv2.projectPoints(
            xyz_valid,
            rvec,
            translation.reshape(3, 1).astype(np.float32),
            projector_matrix.astype(np.float32),
            projector_dist_coeffs.astype(np.float32),
        )
        proj_proj = proj_proj.reshape(-1, 2)
        proj_obs_valid = projector_points_obs[finite_mask]
        reproj_err_proj[finite_mask] = np.linalg.norm(proj_proj - proj_obs_valid, axis=1)
    except Exception:
        pass

    return reproj_err_cam, reproj_err_proj


def triangulate_from_uv(
    u_map: np.ndarray,
    v_map: np.ndarray,
    mask_uv: np.ndarray,
    camera_model: CameraModel,
    projector_model: ProjectorModel,
    config: dict[str, Any] | None = None,
) -> TriangulationResult:
    """
    Perform full 3D reconstruction via triangulation from UV projector coordinate maps.

    ALGORITHM:
    1. Load u_map (projector X) and v_map (projector Y) from phase extraction
    2. For each valid pixel (y, x):
       a. Camera pixel → camera ray
       b. Projector pixel (u[y,x], v[y,x]) → projector ray
       c. Triangulate ray pair → 3D point
       d. Compute reprojection errors
       e. Apply outlier filters (depth range, reprojection error)
    3. Assemble point cloud with masks

    Args:
        u_map: HxW projector X coordinates (or NaNs)
        v_map: HxW projector Y coordinates (or NaNs)
        mask_uv: HxW boolean valid/invalid mask
        camera_model: Camera intrinsics
        projector_model: Projector intrinsics + extrinsics
        config: Optional configuration dict with keys:
            - z_min_m, z_max_m: depth range (default 0.05, 5.0)
            - max_reproj_err_px: outlier threshold (default 2.0)
            - debug: boolean to enable debug outputs

    Returns:
        TriangulationResult with 3D points, depth, masks, and errors
    """
    config = config or {}
    h, w = u_map.shape
    debug_enabled = bool(config.get("debug", False))

    # Validate input sizes
    if (w, h) != camera_model.image_size:
        raise ValueError(
            f"UV map size {(w, h)} does not match camera {camera_model.image_size}"
        )

    # Find valid pixels
    valid = mask_uv.astype(bool) & np.isfinite(u_map) & np.isfinite(v_map)
    ys, xs = np.where(valid)
    n_valid = len(ys)

    if n_valid == 0:
        raise ValueError("No valid UV correspondences to triangulate")

    # Prepare pixel coordinates
    cam_pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1).reshape(-1, 1, 2)
    proj_pts = np.stack([u_map[ys, xs].astype(np.float64), v_map[ys, xs].astype(np.float64)], axis=1).reshape(-1, 1, 2)

    # Undistort points
    cam_norm = undistort_points(cam_pts, camera_model.matrix, camera_model.dist_coeffs)
    proj_norm = undistort_points(proj_pts, projector_model.matrix, projector_model.dist_coeffs)

    # Build rays
    cam_origins, cam_dirs = build_camera_rays(cam_norm, camera_model.matrix)
    proj_origins, proj_dirs = build_projector_rays(
        proj_norm,
        projector_model.rotation,
        projector_model.translation,
    )

    # Triangulate
    xyz_points, tri_errors = triangulate_rays(cam_origins, cam_dirs, proj_origins, proj_dirs)

    # Reprojection errors
    reproj_cam, reproj_proj = compute_reprojection_errors(
        xyz_points,
        camera_model.matrix,
        camera_model.dist_coeffs,
        projector_model.matrix,
        projector_model.dist_coeffs,
        projector_model.rotation,
        projector_model.translation,
        cam_pts.reshape(-1, 2),
        proj_pts.reshape(-1, 2),
    )

    # Apply filters
    z_min = float(config.get("z_min_m", 0.05))
    z_max = float(config.get("z_max_m", 5.0))
    max_reproj = float(config.get("max_reproj_err_px", 2.0))

    z = xyz_points[:, 2]
    depth_ok = np.isfinite(z) & (z > z_min) & (z < z_max)
    cam_ok = np.isfinite(reproj_cam) & (reproj_cam <= max_reproj)
    proj_ok = np.isfinite(reproj_proj) & (reproj_proj <= max_reproj)
    accepted = depth_ok & cam_ok & proj_ok

    # Assemble output
    xyz = np.full((h, w, 3), np.nan, dtype=np.float32)
    depth = np.full((h, w), np.nan, dtype=np.float32)
    mask = np.zeros((h, w), dtype=bool)
    reproj_cam_map = np.full((h, w), np.nan, dtype=np.float32)
    reproj_proj_map = np.full((h, w), np.nan, dtype=np.float32)

    ys_ok = ys[accepted]
    xs_ok = xs[accepted]
    xyz_ok = xyz_points[accepted].astype(np.float32)
    xyz[ys_ok, xs_ok, :] = xyz_ok
    depth[ys_ok, xs_ok] = xyz_ok[:, 2]
    mask[ys_ok, xs_ok] = True

    reproj_cam_map[ys, xs] = reproj_cam.astype(np.float32)
    reproj_proj_map[ys, xs] = reproj_proj.astype(np.float32)

    # Debug info
    debug_info = {
        "n_valid_pixels": int(n_valid),
        "n_accepted": int(np.count_nonzero(accepted)),
        "n_rejected_by_depth": int(np.count_nonzero(~depth_ok)),
        "n_rejected_by_cam_reproj": int(np.count_nonzero(depth_ok & ~cam_ok)),
        "n_rejected_by_proj_reproj": int(np.count_nonzero(depth_ok & cam_ok & ~proj_ok)),
    }

    if debug_enabled:
        debug_info.update({
            "triangulation_errors_mean": float(np.nanmean(tri_errors[accepted])) if np.any(accepted) else None,
            "triangulation_errors_max": float(np.nanmax(tri_errors[accepted])) if np.any(accepted) else None,
            "reproj_cam_mean": float(np.nanmean(reproj_cam[accepted])) if np.any(accepted) else None,
            "reproj_proj_mean": float(np.nanmean(reproj_proj[accepted])) if np.any(accepted) else None,
            "depth_mean": float(np.nanmean(depth[mask])) if np.any(mask) else None,
            "depth_std": float(np.nanstd(depth[mask])) if np.any(mask) else None,
        })

    return TriangulationResult(
        xyz=xyz,
        depth=depth,
        mask=mask,
        reproj_err_cam=reproj_cam_map,
        reproj_err_proj=reproj_proj_map,
        debug=debug_info,
    )
