"""3D reconstruction stage."""

from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
from fringe_app.uv import save_uv_outputs

from fringe_app_v2.core.calibration import CalibrationService, WorldMap
from fringe_app_v2.utils.io import RunPaths, save_mask_png, write_json


def run_reconstruct_stage(
    run: RunPaths,
    calibration: CalibrationService,
    config: dict[str, Any],
    roi_mask: np.ndarray | None,
) -> dict[str, Any]:
    freqs = [float(v) for v in (config.get("scan", {}) or {}).get("frequencies", [])]
    if not freqs:
        raise ValueError("scan.frequencies is required")
    high_freq = float(max(freqs))
    vertical = run.unwrap / "vertical"
    horizontal = run.unwrap / "horizontal"
    if not vertical.exists() or not horizontal.exists():
        raise FileNotFoundError("Both vertical and horizontal unwrap outputs are required for 3D reconstruction")

    phi_v = np.load(vertical / "phi_abs.npy").astype(np.float32)
    phi_h = np.load(horizontal / "phi_abs.npy").astype(np.float32)
    mask_v = np.load(vertical / "mask_unwrap.npy").astype(bool)
    mask_h = np.load(horizontal / "mask_unwrap.npy").astype(bool)
    scan = config.get("scan", {}) or {}
    uv, world = calibration.phase_to_world(
        phase_vertical=phi_v,
        phase_horizontal=phi_h,
        mask_vertical=mask_v,
        mask_horizontal=mask_h,
        roi_mask=roi_mask,
        freq_u=high_freq,
        freq_v=high_freq,
        frequency_semantics=str(scan.get("frequency_semantics", "cycles_across_dimension")),
        phase_origin_u_rad=float(scan.get("phase_origin_rad", 0.0)),
        phase_origin_v_rad=float(scan.get("phase_origin_rad", 0.0)),
        recon_cfg=config.get("reconstruction", {}) or {},
        uv_gate_cfg=config.get("uv_gate", {}) or {},
    )

    uv_dir = run.reconstruct / "uv"
    save_uv_outputs(uv_dir, uv)
    _save_world(run.reconstruct, world, uv.u, uv.v, uv.mask_uv)
    write_json(run.reconstruct / "reconstruction_meta.json", world.meta)
    # Export a simple ASCII point cloud (uncolored) for convenience.
    try:
        _export_pointcloud(run.reconstruct, world.xyz, world.mask)
    except Exception:
        # Non-fatal: keep reconstruction success even if export fails.
        pass
    return world.meta


def _save_world(out_dir, world: WorldMap, u: np.ndarray, v: np.ndarray, mask_uv: np.ndarray) -> None:
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "xyz.npy", world.xyz.astype(np.float32))
    np.save(out_dir / "height.npy", world.height.astype(np.float32))
    np.save(out_dir / "depth.npy", world.height.astype(np.float32))
    uv_map = np.stack([u.astype(np.float32), v.astype(np.float32)], axis=-1)
    uv_map[~mask_uv.astype(bool), :] = np.nan
    np.save(out_dir / "uv_map.npy", uv_map.astype(np.float32))
    np.save(masks_dir / "mask_uv.npy", mask_uv.astype(bool))
    np.save(masks_dir / "mask_reconstruct.npy", world.mask.astype(bool))
    np.save(masks_dir / "mask_recon.npy", world.mask.astype(bool))
    np.save(out_dir / "reproj_err_cam.npy", world.reproj_err_cam.astype(np.float32))
    np.save(out_dir / "reproj_err_proj.npy", world.reproj_err_proj.astype(np.float32))
    save_mask_png(masks_dir / "mask_uv.png", mask_uv)
    save_mask_png(masks_dir / "mask_reconstruct.png", world.mask)


def _export_pointcloud(out_dir: Path, xyz: np.ndarray, mask: np.ndarray) -> None:
    """Write an ASCII PLY point cloud containing valid points from `xyz`.

    - `xyz` is expected shape HxWx3
    - `mask` is expected shape HxW boolean where True indicates valid points
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(xyz)
    if pts.ndim != 3 or pts.shape[2] != 3:
        raise ValueError("xyz must be HxWx3 array")
    m = np.asarray(mask).astype(bool)
    if m.shape != pts.shape[:2]:
        raise ValueError("mask shape must match xyz HxW")
    flat_pts = pts.reshape(-1, 3)
    flat_mask = m.ravel()
    valid_pts = flat_pts[flat_mask]
    if valid_pts.size == 0:
        return
    cloud_path = out_dir / "cloud.ply"
    pointcloud_path = out_dir / "pointcloud.ply"
    header = [
        "ply\n",
        "format ascii 1.0\n",
        f"element vertex {len(valid_pts)}\n",
        "property float x\n",
        "property float y\n",
        "property float z\n",
        "end_header\n",
    ]
    with open(cloud_path, "w", encoding="utf8") as fh:
        fh.writelines(header)
        for x, y, z in valid_pts:
            fh.write(f"{float(x):.6f} {float(y):.6f} {float(z):.6f}\n")
    # Create a second filename for compatibility with older UI
    try:
        out_path = pointcloud_path
        with open(out_path, "w", encoding="utf8") as fh:
            fh.writelines(header)
            for x, y, z in valid_pts:
                fh.write(f"{float(x):.6f} {float(y):.6f} {float(z):.6f}\n")
    except Exception:
        pass
    save_mask_png(masks_dir / "mask_recon.png", world.mask)
