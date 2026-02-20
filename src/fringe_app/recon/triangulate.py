"""Stereo triangulation from UV correspondences."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    _cv2_import_error = exc
else:
    _cv2_import_error = None


def _require_cv2():
    if cv2 is None:
        raise RuntimeError(f"OpenCV is required for reconstruction: {_cv2_import_error}")
    return cv2


@dataclass(slots=True)
class StereoModel:
    Kc: np.ndarray
    distc: np.ndarray
    Kp: np.ndarray
    distp: np.ndarray
    R: np.ndarray
    T: np.ndarray
    proj_size: tuple[int, int]
    cam_size: tuple[int, int]


@dataclass(slots=True)
class ReconstructionResult:
    xyz: np.ndarray
    depth: np.ndarray
    mask_uv: np.ndarray
    mask_recon: np.ndarray
    reproj_err_cam: np.ndarray
    reproj_err_proj: np.ndarray
    rgb: np.ndarray | None
    meta: dict[str, Any]


def load_stereo_model(camera_intrinsics_path: Path, stereo_path: Path) -> StereoModel:
    if not camera_intrinsics_path.exists():
        raise FileNotFoundError(f"Camera intrinsics not found: {camera_intrinsics_path}")
    if not stereo_path.exists():
        raise FileNotFoundError(f"Projector stereo calibration not found: {stereo_path}")

    cam = json.loads(camera_intrinsics_path.read_text())
    st = json.loads(stereo_path.read_text())

    Kc = np.asarray(cam.get("camera_matrix"), dtype=np.float64)
    distc = np.asarray(cam.get("dist_coeffs"), dtype=np.float64).reshape(-1, 1)
    Kp = np.asarray(st.get("projector_matrix"), dtype=np.float64)
    distp = np.asarray(st.get("projector_dist_coeffs"), dtype=np.float64).reshape(-1, 1)
    R = np.asarray(st.get("R"), dtype=np.float64)
    T = np.asarray(st.get("T"), dtype=np.float64).reshape(3)

    if Kc.shape != (3, 3):
        raise ValueError("Invalid camera matrix shape in intrinsics file")
    if Kp.shape != (3, 3):
        raise ValueError("Invalid projector matrix shape in stereo file")
    if R.shape != (3, 3):
        raise ValueError("Invalid stereo rotation matrix shape")
    if T.shape != (3,):
        raise ValueError("Invalid stereo translation vector shape")

    img_size = cam.get("image_size")
    if not isinstance(img_size, (list, tuple)) or len(img_size) != 2:
        raise ValueError("Camera intrinsics missing image_size [width, height]")
    cam_size = (int(img_size[0]), int(img_size[1]))

    proj = st.get("projector", {}) or {}
    if "width" in proj and "height" in proj:
        proj_size = (int(proj["width"]), int(proj["height"]))
    elif "projector_size" in st and isinstance(st["projector_size"], (list, tuple)) and len(st["projector_size"]) == 2:
        proj_size = (int(st["projector_size"][0]), int(st["projector_size"][1]))
    else:
        raise ValueError("Stereo file missing projector size")

    return StereoModel(
        Kc=Kc,
        distc=distc,
        Kp=Kp,
        distp=distp,
        R=R,
        T=T,
        proj_size=proj_size,
        cam_size=cam_size,
    )


def projection_matrices(model: StereoModel) -> tuple[np.ndarray, np.ndarray]:
    P1 = model.Kc @ np.hstack([np.eye(3), np.zeros((3, 1), dtype=np.float64)])
    P2 = model.Kp @ np.hstack([model.R, model.T.reshape(3, 1)])
    return P1, P2


def _load_uv_inputs(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    uv_dir = run_dir / "projector_uv"
    if not uv_dir.exists():
        raise FileNotFoundError(f"UV folder not found: {uv_dir}")
    u_path = uv_dir / "u.npy"
    v_path = uv_dir / "v.npy"
    m_path = uv_dir / "mask_uv.npy"
    if not (u_path.exists() and v_path.exists() and m_path.exists()):
        raise FileNotFoundError(f"Missing UV files under {uv_dir}")
    u = np.load(u_path).astype(np.float32)
    v = np.load(v_path).astype(np.float32)
    mask_uv = np.load(m_path).astype(bool)
    uv_meta_path = uv_dir / "uv_meta.json"
    uv_meta = json.loads(uv_meta_path.read_text()) if uv_meta_path.exists() else {}
    return u, v, mask_uv, uv_meta


def _load_rgb_reference(run_dir: Path, shape_hw: tuple[int, int]) -> np.ndarray | None:
    h, w = shape_hw
    candidates: list[Path] = []
    # Combined UV run structure.
    v_caps = run_dir / "vertical" / "captures"
    if v_caps.exists():
        for freq_dir in sorted(v_caps.glob("f_*")):
            candidates.extend(sorted(freq_dir.glob("step_*.png")))
    # Single-run structure fallback.
    caps = run_dir / "captures"
    if caps.exists():
        for freq_dir in sorted(caps.glob("f_*")):
            candidates.extend(sorted(freq_dir.glob("step_*.png")))
        candidates.extend(sorted(caps.glob("frame_*.png")))

    for p in candidates:
        try:
            arr = np.array(Image.open(p))
        except Exception:
            continue
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=2)
        if arr.ndim != 3 or arr.shape[2] < 3:
            continue
        if arr.shape[0] != h or arr.shape[1] != w:
            continue
        return arr[:, :, :3].astype(np.uint8)
    return None


def _validate_model_and_uv(
    model: StereoModel,
    u: np.ndarray,
    v: np.ndarray,
    mask_uv: np.ndarray,
    uv_meta: dict[str, Any],
) -> None:
    if u.shape != v.shape or u.shape != mask_uv.shape:
        raise ValueError("u/v/mask_uv shapes do not match")
    h, w = u.shape
    cam_w, cam_h = model.cam_size
    if (w, h) != (cam_w, cam_h):
        raise ValueError(
            f"Camera size mismatch: UV map is {(w, h)} but camera calibration expects {(cam_w, cam_h)}"
        )
    uv_proj_w = uv_meta.get("projector_width")
    uv_proj_h = uv_meta.get("projector_height")
    if uv_proj_w is not None and uv_proj_h is not None:
        if (int(uv_proj_w), int(uv_proj_h)) != model.proj_size:
            raise ValueError(
                "Projector size mismatch: "
                f"UV map uses {(int(uv_proj_w), int(uv_proj_h))} but stereo model expects {model.proj_size}"
            )


def reconstruct_uv_run(
    run_dir: Path,
    model: StereoModel,
    recon_cfg: dict[str, Any] | None = None,
) -> ReconstructionResult:
    cv = _require_cv2()
    recon_cfg = recon_cfg or {}

    u, v, mask_uv, uv_meta = _load_uv_inputs(run_dir)
    _validate_model_and_uv(model, u, v, mask_uv, uv_meta)

    valid = mask_uv & np.isfinite(u) & np.isfinite(v)
    ys, xs = np.where(valid)
    if ys.size == 0:
        raise ValueError("No valid UV correspondences available for triangulation")

    cam_pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1).reshape(-1, 1, 2)
    proj_pts = np.stack([u[ys, xs].astype(np.float64), v[ys, xs].astype(np.float64)], axis=1).reshape(-1, 1, 2)

    cam_norm = cv.undistortPoints(cam_pts, model.Kc, model.distc).reshape(-1, 2)
    proj_norm = cv.undistortPoints(proj_pts, model.Kp, model.distp).reshape(-1, 2)

    P1n = np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)])
    P2n = np.hstack([model.R.astype(np.float64), model.T.astype(np.float64).reshape(3, 1)])
    X_h = cv.triangulatePoints(P1n, P2n, cam_norm.T, proj_norm.T)
    X = (X_h[:3] / X_h[3]).T.astype(np.float64)

    reproj_cam = np.full((ys.size,), np.inf, dtype=np.float64)
    reproj_proj = np.full((ys.size,), np.inf, dtype=np.float64)
    finite_xyz = np.isfinite(X).all(axis=1)
    if np.any(finite_xyz):
        X_f = X[finite_xyz].reshape(-1, 1, 3)
        cam_obs = cam_pts.reshape(-1, 2)[finite_xyz]
        proj_obs = proj_pts.reshape(-1, 2)[finite_xyz]

        cam_rep, _ = cv.projectPoints(
            X_f,
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            model.Kc,
            model.distc,
        )
        rvec_p, _ = cv.Rodrigues(model.R.astype(np.float64))
        proj_rep, _ = cv.projectPoints(
            X_f,
            rvec_p,
            model.T.astype(np.float64).reshape(3, 1),
            model.Kp,
            model.distp,
        )
        reproj_cam[finite_xyz] = np.linalg.norm(cam_rep.reshape(-1, 2) - cam_obs, axis=1)
        reproj_proj[finite_xyz] = np.linalg.norm(proj_rep.reshape(-1, 2) - proj_obs, axis=1)

    z = X[:, 2]
    z_min = float(recon_cfg.get("z_min_m", 0.05))
    z_max = float(recon_cfg.get("z_max_m", 5.0))
    max_reproj = float(recon_cfg.get("max_reproj_err_px", 2.0))

    ok = finite_xyz & np.isfinite(z) & (z > z_min) & (z < z_max)
    if max_reproj > 0:
        ok &= (reproj_cam <= max_reproj) & (reproj_proj <= max_reproj)

    h, w = u.shape
    xyz = np.full((h, w, 3), np.nan, dtype=np.float32)
    depth = np.full((h, w), np.nan, dtype=np.float32)
    mask_recon = np.zeros((h, w), dtype=bool)
    reproj_err_cam_map = np.full((h, w), np.nan, dtype=np.float32)
    reproj_err_proj_map = np.full((h, w), np.nan, dtype=np.float32)

    xs_ok = xs[ok]
    ys_ok = ys[ok]
    xyz_ok = X[ok].astype(np.float32)
    xyz[ys_ok, xs_ok, :] = xyz_ok
    depth[ys_ok, xs_ok] = xyz_ok[:, 2]
    mask_recon[ys_ok, xs_ok] = True
    reproj_err_cam_map[ys, xs] = reproj_cam.astype(np.float32)
    reproj_err_proj_map[ys, xs] = reproj_proj.astype(np.float32)

    rgb = _load_rgb_reference(run_dir, (h, w))
    valid_count = int(np.count_nonzero(mask_recon))
    uv_count = int(np.count_nonzero(valid))
    reject_reproj = int(np.count_nonzero(finite_xyz & (~ok)))
    dvals = depth[mask_recon]

    meta: dict[str, Any] = {
        "run_id": run_dir.name,
        "camera_size": [int(w), int(h)],
        "projector_size": [int(model.proj_size[0]), int(model.proj_size[1])],
        "valid_uv_points": uv_count,
        "valid_recon_points": valid_count,
        "rejected_reprojection_or_depth": reject_reproj,
        "valid_ratio_uv": float(uv_count / float(h * w)),
        "valid_ratio_recon": float(valid_count / float(h * w)),
        "max_reproj_err_px": max_reproj,
        "z_min_m": z_min,
        "z_max_m": z_max,
        "depth_min_m": float(np.nanmin(dvals)) if dvals.size else None,
        "depth_max_m": float(np.nanmax(dvals)) if dvals.size else None,
        "depth_median_m": float(np.nanmedian(dvals)) if dvals.size else None,
        "reproj_err_cam_median": float(np.nanmedian(reproj_err_cam_map[mask_recon])) if valid_count else None,
        "reproj_err_proj_median": float(np.nanmedian(reproj_err_proj_map[mask_recon])) if valid_count else None,
        "uv_meta": uv_meta,
    }

    return ReconstructionResult(
        xyz=xyz,
        depth=depth,
        mask_uv=mask_uv.astype(bool),
        mask_recon=mask_recon,
        reproj_err_cam=reproj_err_cam_map,
        reproj_err_proj=reproj_err_proj_map,
        rgb=rgb,
        meta=meta,
    )
