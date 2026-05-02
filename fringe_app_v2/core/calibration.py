"""Compatibility layer for existing camera/projector calibration artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import fringe_app_v2
from fringe_app_v2.pipeline.phase_to_projector import phase_to_projector_coords
from fringe_app_v2.utils.io import load_yaml


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    path: Path
    matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProjectorModel:
    path: Path
    matrix: np.ndarray
    dist_coeffs: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    projector_size: tuple[int, int]
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class WorldMap:
    xyz: np.ndarray
    height: np.ndarray
    mask: np.ndarray
    reproj_err_cam: np.ndarray
    reproj_err_proj: np.ndarray
    meta: dict[str, Any]


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else fringe_app_v2.REPO_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    import json

    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _latest_projector_v2_export(root: Path) -> Path | None:
    v2_root = root / "projector_v2"
    if not v2_root.exists():
        return None
    candidates = sorted(v2_root.glob("*/export/stereo.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def find_camera_intrinsics(config: dict[str, Any]) -> Path:
    calib = config.get("calibration", {}) or {}
    root = _repo_path(calib.get("root", "data/calibration"))
    candidates = [
        calib.get("camera_intrinsics_path"),
        root / "camera" / "intrinsics_latest.json",
        root / "camera_intrinsics" / "intrinsics_latest.json",
        root / "intrinsics_latest.json",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        path = _repo_path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError("No compatible camera intrinsics file found")


def find_projector_stereo(config: dict[str, Any]) -> Path:
    calib = config.get("calibration", {}) or {}
    root = _repo_path(calib.get("root", "data/calibration"))
    v2_latest = _latest_projector_v2_export(root)
    candidates = [
        calib.get("projector_stereo_path"),
        root / "projector" / "stereo_latest.json",
        root / "projector" / "results" / "stereo_latest.json",
        v2_latest,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        path = _repo_path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError("No compatible projector stereo calibration file found")


def load_camera_intrinsics(path: Path) -> CameraIntrinsics:
    payload = _load_json(path)
    matrix = np.asarray(payload.get("camera_matrix"), dtype=np.float64)
    dist = np.asarray(payload.get("dist_coeffs"), dtype=np.float64).reshape(-1, 1)
    image_size = payload.get("image_size")
    if matrix.shape != (3, 3):
        raise ValueError(f"Invalid camera_matrix shape in {path}: {matrix.shape}")
    if dist.size == 0:
        raise ValueError(f"Missing dist_coeffs in {path}")
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError(f"Missing image_size [width, height] in {path}")
    return CameraIntrinsics(
        path=path,
        matrix=matrix,
        dist_coeffs=dist,
        image_size=(int(image_size[0]), int(image_size[1])),
        payload=payload,
    )


def load_projector_model(path: Path) -> ProjectorModel:
    payload = _load_json(path)
    matrix = np.asarray(payload.get("projector_matrix"), dtype=np.float64)
    dist = np.asarray(payload.get("projector_dist_coeffs"), dtype=np.float64).reshape(-1, 1)
    rotation = np.asarray(payload.get("R"), dtype=np.float64)
    translation = np.asarray(payload.get("T"), dtype=np.float64).reshape(3)
    projector = payload.get("projector", {}) or {}
    if matrix.shape != (3, 3):
        raise ValueError(f"Invalid projector_matrix shape in {path}: {matrix.shape}")
    if dist.size == 0:
        raise ValueError(f"Missing projector_dist_coeffs in {path}")
    if rotation.shape != (3, 3):
        raise ValueError(f"Invalid R shape in {path}: {rotation.shape}")
    if translation.shape != (3,):
        raise ValueError(f"Invalid T shape in {path}: {translation.shape}")
    if "width" in projector and "height" in projector:
        projector_size = (int(projector["width"]), int(projector["height"]))
    elif isinstance(payload.get("projector_size"), (list, tuple)) and len(payload["projector_size"]) == 2:
        projector_size = (int(payload["projector_size"][0]), int(payload["projector_size"][1]))
    else:
        raise ValueError(f"Missing projector size in {path}")
    return ProjectorModel(
        path=path,
        matrix=matrix,
        dist_coeffs=dist,
        rotation=rotation,
        translation=translation,
        projector_size=projector_size,
        payload=payload,
    )


class CalibrationService:
    """Loads and exposes existing calibration files without changing their math."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.camera = load_camera_intrinsics(find_camera_intrinsics(config))
        self.projector = load_projector_model(find_projector_stereo(config))

    def get_camera_matrix(self) -> np.ndarray:
        return self.camera.matrix.copy()

    def get_projector_model(self) -> ProjectorModel:
        return self.projector

    def status(self) -> dict[str, Any]:
        return {
            "camera_intrinsics": str(self.camera.path),
            "projector_stereo": str(self.projector.path),
            "camera_image_size": list(self.camera.image_size),
            "projector_size": list(self.projector.projector_size),
            "camera_matrix_shape": list(self.camera.matrix.shape),
            "projector_matrix_shape": list(self.projector.matrix.shape),
        }

    def phase_to_world(
        self,
        phase_vertical: np.ndarray,
        phase_horizontal: np.ndarray,
        mask_vertical: np.ndarray,
        mask_horizontal: np.ndarray,
        roi_mask: np.ndarray | None,
        freq_u: float,
        freq_v: float,
        frequency_semantics: str,
        phase_origin_u_rad: float,
        phase_origin_v_rad: float,
        recon_cfg: dict[str, Any] | None = None,
        uv_gate_cfg: dict[str, Any] | None = None,
    ) -> tuple[Any, WorldMap]:
        uv = phase_to_projector_coords(
            phi_horizontal=phase_horizontal,
            phi_vertical=phase_vertical,
            mask_horizontal=mask_horizontal,
            mask_vertical=mask_vertical,
            projector_width=self.projector.projector_size[0],
            projector_height=self.projector.projector_size[1],
            frequency_u=freq_u,
            frequency_v=freq_v,
            frequency_semantics=frequency_semantics,
            phase_origin_u_rad=phase_origin_u_rad,
            phase_origin_v_rad=phase_origin_v_rad,
            roi_mask=roi_mask,
        )
        world = triangulate_uv_maps(self.camera, self.projector, uv.u, uv.v, uv.mask, recon_cfg or {})
        world.meta["uv_meta"] = uv.meta
        return uv, world


def triangulate_uv_maps(
    camera: CameraIntrinsics,
    projector: ProjectorModel,
    u: np.ndarray,
    v: np.ndarray,
    mask_uv: np.ndarray,
    recon_cfg: dict[str, Any],
) -> WorldMap:
    if u.shape != v.shape or u.shape != mask_uv.shape:
        raise ValueError("u, v, and mask_uv must have matching shapes")
    h, w = u.shape
    if (w, h) != camera.image_size:
        raise ValueError(f"UV map size {(w, h)} does not match camera calibration {camera.image_size}")

    valid = mask_uv.astype(bool) & np.isfinite(u) & np.isfinite(v)
    ys, xs = np.where(valid)
    if ys.size == 0:
        raise ValueError("No valid UV correspondences")

    cam_pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1).reshape(-1, 1, 2)
    proj_pts = np.stack([u[ys, xs].astype(np.float64), v[ys, xs].astype(np.float64)], axis=1).reshape(-1, 1, 2)

    cam_norm = cv2.undistortPoints(cam_pts, camera.matrix, camera.dist_coeffs).reshape(-1, 2)
    proj_norm = cv2.undistortPoints(proj_pts, projector.matrix, projector.dist_coeffs).reshape(-1, 2)
    p_camera = np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)])
    p_projector = np.hstack([projector.rotation, projector.translation.reshape(3, 1)])
    x_h = cv2.triangulatePoints(p_camera, p_projector, cam_norm.T, proj_norm.T)
    xyz_points = (x_h[:3] / x_h[3]).T.astype(np.float64)

    reproj_cam = np.full((ys.size,), np.inf, dtype=np.float64)
    reproj_proj = np.full((ys.size,), np.inf, dtype=np.float64)
    finite_xyz = np.isfinite(xyz_points).all(axis=1)
    if np.any(finite_xyz):
        xyz_valid = xyz_points[finite_xyz].reshape(-1, 1, 3)
        cam_obs = cam_pts.reshape(-1, 2)[finite_xyz]
        proj_obs = proj_pts.reshape(-1, 2)[finite_xyz]
        cam_rep, _ = cv2.projectPoints(
            xyz_valid,
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            camera.matrix,
            camera.dist_coeffs,
        )
        rvec_proj, _ = cv2.Rodrigues(projector.rotation)
        proj_rep, _ = cv2.projectPoints(
            xyz_valid,
            rvec_proj,
            projector.translation.reshape(3, 1),
            projector.matrix,
            projector.dist_coeffs,
        )
        reproj_cam[finite_xyz] = np.linalg.norm(cam_rep.reshape(-1, 2) - cam_obs, axis=1)
        reproj_proj[finite_xyz] = np.linalg.norm(proj_rep.reshape(-1, 2) - proj_obs, axis=1)

    z = xyz_points[:, 2]
    z_min = float(recon_cfg.get("z_min_m", 0.05))
    z_max = float(recon_cfg.get("z_max_m", 5.0))
    max_reproj = float(recon_cfg.get("max_reproj_err_px", 3.0))
    depth_ok = finite_xyz & np.isfinite(z) & (z > z_min) & (z < z_max)
    cam_ok = np.isfinite(reproj_cam) & (reproj_cam <= max_reproj)
    proj_ok = np.isfinite(reproj_proj) & (reproj_proj <= max_reproj)
    accepted = depth_ok & cam_ok & proj_ok

    xyz = np.full((h, w, 3), np.nan, dtype=np.float32)
    height = np.full((h, w), np.nan, dtype=np.float32)
    mask = np.zeros((h, w), dtype=bool)
    reproj_cam_map = np.full((h, w), np.nan, dtype=np.float32)
    reproj_proj_map = np.full((h, w), np.nan, dtype=np.float32)

    ys_ok = ys[accepted]
    xs_ok = xs[accepted]
    xyz_ok = xyz_points[accepted].astype(np.float32)
    xyz[ys_ok, xs_ok, :] = xyz_ok
    height[ys_ok, xs_ok] = xyz_ok[:, 2]
    mask[ys_ok, xs_ok] = True
    reproj_cam_map[ys, xs] = reproj_cam.astype(np.float32)
    reproj_proj_map[ys, xs] = reproj_proj.astype(np.float32)

    values = height[mask]
    meta = {
        "camera_intrinsics": str(camera.path),
        "projector_stereo": str(projector.path),
        "camera_size": list(camera.image_size),
        "projector_size": list(projector.projector_size),
        "valid_uv_points": int(np.count_nonzero(valid)),
        "valid_reconstruct_points": int(np.count_nonzero(mask)),
        "z_min_m": z_min,
        "z_max_m": z_max,
        "max_reproj_err_px": max_reproj,
        "height_min_m": float(np.nanmin(values)) if values.size else None,
        "height_max_m": float(np.nanmax(values)) if values.size else None,
        "height_median_m": float(np.nanmedian(values)) if values.size else None,
        "rejected_by_depth_px": int(np.count_nonzero(~depth_ok)),
        "rejected_by_camera_reproj_px": int(np.count_nonzero(depth_ok & ~cam_ok)),
        "rejected_by_projector_reproj_px": int(np.count_nonzero(depth_ok & cam_ok & ~proj_ok)),
    }
    return WorldMap(
        xyz=xyz,
        height=height,
        mask=mask,
        reproj_err_cam=reproj_cam_map,
        reproj_err_proj=reproj_proj_map,
        meta=meta,
    )


def _default_config() -> dict[str, Any]:
    return load_yaml(fringe_app_v2.REPO_ROOT / "fringe_app_v2" / "config" / "default.yaml")


def get_camera_matrix(config: dict[str, Any] | None = None) -> np.ndarray:
    return CalibrationService(config or _default_config()).get_camera_matrix()


def get_projector_model(config: dict[str, Any] | None = None) -> ProjectorModel:
    return CalibrationService(config or _default_config()).get_projector_model()


def phase_to_world(
    phase_vertical: np.ndarray,
    phase_horizontal: np.ndarray,
    mask_vertical: np.ndarray,
    mask_horizontal: np.ndarray,
    roi_mask: np.ndarray | None,
    freq_u: float,
    freq_v: float,
    frequency_semantics: str = "cycles_across_dimension",
    phase_origin_u_rad: float = 0.0,
    phase_origin_v_rad: float = 0.0,
    config: dict[str, Any] | None = None,
) -> tuple[Any, WorldMap]:
    cfg = config or _default_config()
    return CalibrationService(cfg).phase_to_world(
        phase_vertical=phase_vertical,
        phase_horizontal=phase_horizontal,
        mask_vertical=mask_vertical,
        mask_horizontal=mask_horizontal,
        roi_mask=roi_mask,
        freq_u=freq_u,
        freq_v=freq_v,
        frequency_semantics=frequency_semantics,
        phase_origin_u_rad=phase_origin_u_rad,
        phase_origin_v_rad=phase_origin_v_rad,
        recon_cfg=cfg.get("reconstruction", {}) or {},
        uv_gate_cfg=cfg.get("uv_gate", {}) or {},
    )
