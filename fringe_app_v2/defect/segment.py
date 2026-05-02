"""Height-residual defect segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fringe_app_v2.defect.utils import (
    apply_mask_nan,
    binary_dilate,
    binary_erode,
    cleanup_mask,
    connected_components,
    gaussian_smooth_nan,
)


@dataclass(frozen=True, slots=True)
class DefectSegmentationConfig:
    smoothing: float = 12.0
    depth_threshold_mm: float = 0.25
    min_area_px: int = 80
    open_radius_px: int = 1
    close_radius_px: int = 2
    boundary_exclusion_px: int = 0
    max_area_px: int = 0
    max_area_fraction: float = 0.0
    max_area_floor_px: int = 0
    edge_suppression_enabled: bool = True
    edge_percentile: float = 92.0
    edge_exclusion_radius_px: int = 6
    edge_overlap_reject: float = 0.35

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DefectSegmentationConfig":
        cfg = config.get("defect", {}) or {}
        defaults = cls()
        return cls(
            smoothing=float(cfg.get("smoothing", defaults.smoothing)),
            depth_threshold_mm=float(cfg.get("depth_threshold_mm", defaults.depth_threshold_mm)),
            min_area_px=int(cfg.get("min_area_px", defaults.min_area_px)),
            open_radius_px=int(cfg.get("open_radius_px", defaults.open_radius_px)),
            close_radius_px=int(cfg.get("close_radius_px", defaults.close_radius_px)),
            boundary_exclusion_px=int(cfg.get("boundary_exclusion_px", defaults.boundary_exclusion_px)),
            max_area_px=int(cfg.get("max_area_px", defaults.max_area_px)),
            max_area_fraction=float(cfg.get("max_area_fraction", defaults.max_area_fraction)),
            max_area_floor_px=int(cfg.get("max_area_floor_px", defaults.max_area_floor_px)),
            edge_suppression_enabled=bool(
                cfg.get("edge_suppression_enabled", defaults.edge_suppression_enabled)
            ),
            edge_percentile=float(cfg.get("edge_percentile", defaults.edge_percentile)),
            edge_exclusion_radius_px=int(cfg.get("edge_exclusion_radius_px", defaults.edge_exclusion_radius_px)),
            edge_overlap_reject=float(cfg.get("edge_overlap_reject", defaults.edge_overlap_reject)),
        )


@dataclass(frozen=True, slots=True)
class DefectSegmentationResult:
    height_masked_m: np.ndarray
    reference_m: np.ndarray
    residual_m: np.ndarray
    raw_mask: np.ndarray
    defect_mask: np.ndarray
    valid_mask: np.ndarray
    config: DefectSegmentationConfig


def segment_defects(height_map: np.ndarray, mask: np.ndarray, config: dict[str, Any]) -> DefectSegmentationResult:
    cfg = DefectSegmentationConfig.from_config(config)
    valid_mask = np.asarray(mask, dtype=bool) & np.isfinite(height_map)
    if cfg.boundary_exclusion_px > 0:
        valid_mask &= binary_erode(valid_mask, cfg.boundary_exclusion_px)
    height_masked = apply_mask_nan(height_map, valid_mask)
    reference = gaussian_smooth_nan(height_masked, sigma=cfg.smoothing)
    residual = (height_masked - reference).astype(np.float32)
    residual[~valid_mask] = np.nan

    threshold_m = cfg.depth_threshold_mm / 1000.0
    raw_mask = (np.abs(residual) > threshold_m) & valid_mask & np.isfinite(residual)
    defect_mask = cleanup_mask(
        raw_mask,
        valid_mask=valid_mask,
        min_area=cfg.min_area_px,
        open_radius=cfg.open_radius_px,
        close_radius=cfg.close_radius_px,
    )
    defect_mask = _filter_component_area(defect_mask, valid_mask, cfg)
    defect_mask = _filter_geometry_edges(defect_mask, reference, valid_mask, cfg)
    return DefectSegmentationResult(
        height_masked_m=height_masked,
        reference_m=reference.astype(np.float32),
        residual_m=residual,
        raw_mask=raw_mask.astype(bool),
        defect_mask=defect_mask.astype(bool),
        valid_mask=valid_mask,
        config=cfg,
    )


def _filter_component_area(
    defect_mask: np.ndarray,
    valid_mask: np.ndarray,
    cfg: DefectSegmentationConfig,
) -> np.ndarray:
    max_area = cfg.max_area_px if cfg.max_area_px > 0 else None
    valid_count = int(np.count_nonzero(valid_mask))
    if cfg.max_area_fraction > 0 and valid_count > 0:
        by_fraction = max(cfg.max_area_floor_px, int(round(valid_count * cfg.max_area_fraction)))
        max_area = by_fraction if max_area is None else min(max_area, by_fraction)
    if max_area is None:
        return defect_mask.astype(bool)

    out = np.zeros_like(defect_mask, dtype=bool)
    for region in connected_components(defect_mask & valid_mask):
        if region.pixel_count <= max_area:
            out[region.rows, region.cols] = True
    return out


def _filter_geometry_edges(
    defect_mask: np.ndarray,
    reference_m: np.ndarray,
    valid_mask: np.ndarray,
    cfg: DefectSegmentationConfig,
) -> np.ndarray:
    if not cfg.edge_suppression_enabled:
        return defect_mask.astype(bool)
    edge_mask = _geometry_edge_mask(reference_m, valid_mask, cfg)
    out = np.zeros_like(defect_mask, dtype=bool)
    for region in connected_components(defect_mask & valid_mask):
        edge_pixels = int(np.count_nonzero(edge_mask[region.rows, region.cols]))
        overlap = edge_pixels / max(region.pixel_count, 1)
        if overlap < cfg.edge_overlap_reject:
            out[region.rows, region.cols] = True
    return out


def _geometry_edge_mask(reference_m: np.ndarray, valid_mask: np.ndarray, cfg: DefectSegmentationConfig) -> np.ndarray:
    finite = np.isfinite(reference_m) & valid_mask
    if not np.any(finite):
        return np.zeros_like(valid_mask, dtype=bool)
    filled = np.where(finite, reference_m, np.nanmedian(reference_m[finite])).astype(np.float32)
    grad_y, grad_x = np.gradient(filled)
    gradient = np.hypot(grad_x, grad_y)
    values = gradient[finite & np.isfinite(gradient)]
    if values.size == 0:
        return np.zeros_like(valid_mask, dtype=bool)
    threshold = float(np.percentile(values, cfg.edge_percentile))
    edge_mask = (gradient >= threshold) & finite
    if cfg.edge_exclusion_radius_px > 0:
        edge_mask = binary_dilate(edge_mask, cfg.edge_exclusion_radius_px)
    return edge_mask & valid_mask
