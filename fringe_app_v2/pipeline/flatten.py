"""Surface flattening post-process for reconstructed height maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fringe_app_v2.defect.utils import apply_mask_nan, gaussian_smooth_nan, normalise_to_u8
from fringe_app_v2.utils.io import RunPaths, save_image, write_json


@dataclass(frozen=True, slots=True)
class FlattenConfig:
    enabled: bool = True
    sigma: float = 8.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FlattenConfig":
        nested = config.get("flatten", {}) or {}
        enabled = config.get("enable_flatten", nested.get("enabled", cls.enabled))
        sigma = config.get("flatten_sigma", nested.get("sigma", cls.sigma))
        return cls(enabled=bool(enabled), sigma=float(sigma))


@dataclass(frozen=True, slots=True)
class FlattenResult:
    original_m: np.ndarray
    plane_m: np.ndarray
    plane_removed_m: np.ndarray
    low_frequency_m: np.ndarray
    flat_m: np.ndarray
    mask: np.ndarray
    plane_coefficients: tuple[float, float, float]
    config: FlattenConfig


def fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float, float]:
    design = np.c_[x.astype(np.float64), y.astype(np.float64), np.ones_like(x, dtype=np.float64)]
    coeffs, _, _, _ = np.linalg.lstsq(design, z.astype(np.float64), rcond=None)
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])


def remove_plane(height_map: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    h = np.asarray(height_map, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(h)
    if int(np.count_nonzero(valid)) < 3:
        raise ValueError("At least three valid height samples are required to fit a plane")

    rows, cols = np.mgrid[0 : h.shape[0], 0 : h.shape[1]]
    a, b, c = fit_plane(cols[valid], rows[valid], h[valid])
    plane = (a * cols + b * rows + c).astype(np.float32)
    plane_removed = (h - plane).astype(np.float32)
    plane_removed[~valid] = np.nan
    return plane_removed, plane, (a, b, c)


def remove_low_frequency(height_map: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(height_map, dtype=np.float32)
    smooth = gaussian_smooth_nan(h, sigma=sigma)
    flat = (h - smooth).astype(np.float32)
    flat[~np.isfinite(h)] = np.nan
    return flat, smooth.astype(np.float32)


def flatten_surface(height_map: np.ndarray, mask: np.ndarray, config: dict[str, Any]) -> FlattenResult:
    cfg = FlattenConfig.from_config(config)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(height_map)
    original = apply_mask_nan(height_map, valid)
    plane_removed, plane, coefficients = remove_plane(original, valid)
    if cfg.sigma > 0:
        flat, low_frequency = remove_low_frequency(plane_removed, sigma=cfg.sigma)
    else:
        low_frequency = np.zeros_like(plane_removed, dtype=np.float32)
        flat = plane_removed.copy()
    flat[~valid] = np.nan
    return FlattenResult(
        original_m=original,
        plane_m=plane,
        plane_removed_m=plane_removed,
        low_frequency_m=low_frequency,
        flat_m=flat,
        mask=valid,
        plane_coefficients=coefficients,
        config=cfg,
    )


def run_flatten_stage(run: RunPaths, config: dict[str, Any]) -> dict[str, Any]:
    return run_flatten_for_run(run.root, config)


def run_flatten_for_run(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(run_dir)
    cfg = FlattenConfig.from_config(config)
    if not cfg.enabled:
        summary = {"mode": "flatten", "enabled": False}
        write_json(run_dir / "flatten" / "flatten_meta.json", summary)
        return summary
    height_map, mask = load_flatten_inputs(run_dir)
    result = flatten_surface(height_map, mask, config)
    summary = save_flatten_outputs(run_dir, result)
    return summary


def load_flatten_inputs(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    recon = Path(run_dir) / "reconstruct"
    height_path = _first_existing([recon / "height_map.npy", recon / "height.npy", recon / "depth.npy"])
    mask_path = _first_existing(
        [
            recon / "masks" / "mask_reconstruct.npy",
            recon / "masks" / "mask_recon.npy",
            recon / "masks" / "mask_uv.npy",
        ]
    )
    return np.load(height_path).astype(np.float32), np.load(mask_path).astype(bool)


def save_flatten_outputs(run_dir: Path, result: FlattenResult) -> dict[str, Any]:
    out_dir = Path(run_dir) / "flatten"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "height_original.npy", result.original_m.astype(np.float32))
    np.save(out_dir / "height_plane.npy", result.plane_m.astype(np.float32))
    np.save(out_dir / "height_plane_removed.npy", result.plane_removed_m.astype(np.float32))
    np.save(out_dir / "height_low_frequency.npy", result.low_frequency_m.astype(np.float32))
    np.save(out_dir / "height_flat.npy", result.flat_m.astype(np.float32))
    np.save(out_dir / "mask_flat.npy", result.mask.astype(bool))

    save_image(out_dir / "original_height.png", normalise_to_u8(result.original_m, result.mask))
    save_image(out_dir / "plane_removed.png", normalise_to_u8(result.plane_removed_m, result.mask))
    save_image(out_dir / "low_frequency.png", normalise_to_u8(result.low_frequency_m, result.mask))
    save_image(out_dir / "final_flat.png", normalise_to_u8(result.flat_m, result.mask))

    values = result.flat_m[np.isfinite(result.flat_m) & result.mask]
    summary = {
        "mode": "flatten",
        "enabled": True,
        "height_units": "meters",
        "valid_pixels": int(np.count_nonzero(result.mask)),
        "plane_coefficients": {
            "a_dz_dx": result.plane_coefficients[0],
            "b_dz_dy": result.plane_coefficients[1],
            "c_z0": result.plane_coefficients[2],
        },
        "flatten_sigma": result.config.sigma,
        "flat_stats_m": {
            "min": float(np.min(values)) if values.size else None,
            "max": float(np.max(values)) if values.size else None,
            "mean": float(np.mean(values)) if values.size else None,
            "std": float(np.std(values)) if values.size else None,
        },
    }
    write_json(out_dir / "flatten_meta.json", summary)
    return summary


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these files exists: " + ", ".join(str(path) for path in paths))
