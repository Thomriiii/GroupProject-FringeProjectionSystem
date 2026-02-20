"""I/O helpers for reconstruction outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .triangulate import ReconstructionResult


def _save_mask_png(mask: np.ndarray, path: Path) -> None:
    out = np.where(mask.astype(bool), 255, 0).astype(np.uint8)
    Image.fromarray(out).save(path)


def _save_depth_debug_png(
    depth: np.ndarray,
    mask: np.ndarray,
    fixed_path: Path,
    auto_path: Path,
    z_min: float,
    z_max: float,
) -> None:
    valid = mask.astype(bool) & np.isfinite(depth)
    fixed = np.zeros(depth.shape, dtype=np.uint8)
    auto = np.zeros(depth.shape, dtype=np.uint8)

    if np.any(valid):
        span = max(float(z_max - z_min), 1e-6)
        fixed[valid] = np.clip(((depth[valid] - z_min) / span) * 255.0, 0, 255).astype(np.uint8)

        vals = depth[valid]
        lo, hi = np.percentile(vals, [1, 99])
        if hi <= lo:
            hi = lo + 1e-6
        auto[valid] = np.clip(((depth[valid] - lo) / (hi - lo)) * 255.0, 0, 255).astype(np.uint8)

    Image.fromarray(fixed).save(fixed_path)
    Image.fromarray(auto).save(auto_path)


def save_ply(points_xyz: np.ndarray, colors_rgb: np.ndarray | None, path: Path) -> None:
    n = int(points_xyz.shape[0])
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors_rgb is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        if colors_rgb is None:
            for p in points_xyz:
                f.write(f"{float(p[0])} {float(p[1])} {float(p[2])}\n")
        else:
            c = np.clip(np.rint(colors_rgb), 0, 255).astype(np.uint8)
            for p, col in zip(points_xyz, c, strict=False):
                f.write(
                    f"{float(p[0])} {float(p[1])} {float(p[2])} "
                    f"{int(col[0])} {int(col[1])} {int(col[2])}\n"
                )


def save_reconstruction_outputs(
    out_dir: Path,
    result: ReconstructionResult,
    recon_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recon_cfg = recon_cfg or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "xyz.npy", result.xyz.astype(np.float32))
    np.save(out_dir / "depth.npy", result.depth.astype(np.float32))
    np.save(masks_dir / "mask_uv.npy", result.mask_uv.astype(bool))
    np.save(masks_dir / "mask_recon.npy", result.mask_recon.astype(bool))
    _save_mask_png(result.mask_uv, masks_dir / "mask_uv.png")
    _save_mask_png(result.mask_recon, masks_dir / "mask_recon.png")
    np.save(out_dir / "reproj_err_cam.npy", result.reproj_err_cam.astype(np.float32))
    np.save(out_dir / "reproj_err_proj.npy", result.reproj_err_proj.astype(np.float32))

    z_min = float(recon_cfg.get("z_min_m", 0.05))
    z_max = float(recon_cfg.get("z_max_m", 5.0))
    _save_depth_debug_png(
        result.depth,
        result.mask_recon,
        out_dir / "depth_debug_fixed.png",
        out_dir / "depth_debug_autoscale.png",
        z_min=z_min,
        z_max=z_max,
    )

    ys, xs = np.where(result.mask_recon)
    if ys.size == 0:
        raise ValueError("No valid 3D points after reconstruction filtering")

    export_mask = result.mask_recon.copy()
    ds_cfg = (recon_cfg.get("downsample", {}) or {})
    if bool(ds_cfg.get("enabled", True)):
        stride = max(1, int(ds_cfg.get("stride", 2)))
        if stride > 1:
            yy, xx = np.indices(export_mask.shape)
            export_mask &= ((yy % stride) == 0) & ((xx % stride) == 0)

    ys_e, xs_e = np.where(export_mask)
    max_points = int(recon_cfg.get("max_points", 800000))
    if ys_e.size > max_points > 0:
        step = int(math.ceil(ys_e.size / float(max_points)))
        ys_e = ys_e[::step]
        xs_e = xs_e[::step]

    pts = result.xyz[ys_e, xs_e, :].astype(np.float32)
    finite = np.isfinite(pts).all(axis=1)
    ys_e = ys_e[finite]
    xs_e = xs_e[finite]
    pts = pts[finite]

    colors = None
    xyzrgb = None
    if result.rgb is not None and bool(recon_cfg.get("ply", {}).get("write_rgb", True)):
        colors = result.rgb[ys_e, xs_e, :3].astype(np.float32)
        xyzrgb = np.concatenate([pts, colors], axis=1)
        np.save(out_dir / "xyzrgb.npy", xyzrgb.astype(np.float32))

    save_ply(pts, colors, out_dir / "cloud.ply")

    meta = dict(result.meta)
    meta.update(
        {
            "files": {
                "xyz": "xyz.npy",
                "xyzrgb": "xyzrgb.npy" if xyzrgb is not None else None,
                "depth": "depth.npy",
                "depth_debug_fixed": "depth_debug_fixed.png",
                "depth_debug_autoscale": "depth_debug_autoscale.png",
                "cloud": "cloud.ply",
                "masks": {
                    "mask_uv": "masks/mask_uv.npy",
                    "mask_recon": "masks/mask_recon.npy",
                },
                "reproj_err_cam": "reproj_err_cam.npy",
                "reproj_err_proj": "reproj_err_proj.npy",
            },
            "exported_points": int(pts.shape[0]),
            "downsample": ds_cfg,
        }
    )
    (out_dir / "reconstruction_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
