"""Phase quality validation from captured phase-step images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from fringe_app_v2.defect.utils import normalise_to_u8
from fringe_app_v2.utils.io import save_image, save_mask_png, write_json
from fringe_app_v2.utils.math_utils import to_gray_float


@dataclass(frozen=True, slots=True)
class PhaseQualityConfig:
    min_modulation: float = 10.0
    min_intensity: float = 5.0
    max_intensity: float = 250.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PhaseQualityConfig":
        cfg = config.get("phase_quality", {}) or {}
        defaults = cls()
        return cls(
            min_modulation=float(cfg.get("min_modulation", defaults.min_modulation)),
            min_intensity=float(cfg.get("min_intensity", defaults.min_intensity)),
            max_intensity=float(cfg.get("max_intensity", defaults.max_intensity)),
        )


def compute_intensity(images: Iterable[np.ndarray]) -> np.ndarray:
    stack = _gray_stack(images)
    return np.mean(stack, axis=0, dtype=np.float32).astype(np.float32)


def compute_modulation(images: Iterable[np.ndarray]) -> np.ndarray:
    stack = _gray_stack(images)
    count = int(stack.shape[0])
    angles = 2.0 * np.pi * np.arange(count, dtype=np.float32) / float(count)
    cos_terms = np.cos(angles).astype(np.float32)
    sin_terms = np.sin(angles).astype(np.float32)
    c_term = np.tensordot(cos_terms, stack, axes=(0, 0)).astype(np.float32)
    s_term = np.tensordot(sin_terms, stack, axes=(0, 0)).astype(np.float32)
    return ((2.0 / float(count)) * np.sqrt(c_term * c_term + s_term * s_term)).astype(np.float32)


def compute_valid_mask(modulation: np.ndarray, intensity: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    cfg = PhaseQualityConfig.from_config(config)
    return (
        (np.asarray(modulation) > cfg.min_modulation)
        & (np.asarray(intensity) > cfg.min_intensity)
        & (np.asarray(intensity) < cfg.max_intensity)
        & np.isfinite(modulation)
        & np.isfinite(intensity)
    )


def save_phase_quality_validation(
    out_dir: Path,
    images: Iterable[np.ndarray],
    config: dict[str, Any],
    roi_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    frames = list(images)
    intensity = compute_intensity(frames)
    modulation = compute_modulation(frames)
    valid_mask = compute_valid_mask(modulation, intensity, config)
    if roi_mask is not None:
        valid_mask &= np.asarray(roi_mask, dtype=bool)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "intensity.npy", intensity.astype(np.float32))
    np.save(out_dir / "modulation.npy", modulation.astype(np.float32))
    np.save(out_dir / "valid_mask.npy", valid_mask.astype(bool))
    save_image(out_dir / "intensity.png", normalise_to_u8(intensity, np.isfinite(intensity)))
    save_image(out_dir / "modulation.png", normalise_to_u8(modulation, np.isfinite(modulation)))
    save_mask_png(out_dir / "valid_mask.png", valid_mask)

    cfg = PhaseQualityConfig.from_config(config)
    domain = np.asarray(roi_mask, dtype=bool) if roi_mask is not None else np.ones_like(valid_mask, dtype=bool)
    values = modulation[domain & np.isfinite(modulation)]
    summary = {
        "min_modulation": cfg.min_modulation,
        "min_intensity": cfg.min_intensity,
        "max_intensity": cfg.max_intensity,
        "valid_pixels": int(np.count_nonzero(valid_mask)),
        "valid_fraction": float(np.count_nonzero(valid_mask) / max(np.count_nonzero(domain), 1)),
        "modulation_median": float(np.median(values)) if values.size else 0.0,
        "modulation_p10": float(np.percentile(values, 10)) if values.size else 0.0,
        "modulation_p90": float(np.percentile(values, 90)) if values.size else 0.0,
    }
    write_json(out_dir / "validation.json", summary)
    return summary


def _gray_stack(images: Iterable[np.ndarray]) -> np.ndarray:
    frames = [to_gray_float(image).astype(np.float32) for image in images]
    if not frames:
        raise ValueError("At least one image is required")
    return np.stack(frames, axis=0).astype(np.float32, copy=False)
