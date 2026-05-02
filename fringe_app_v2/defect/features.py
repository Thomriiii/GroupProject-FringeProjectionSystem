"""Geometric feature extraction for segmented defects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fringe_app_v2.defect.utils import connected_components


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    mm_per_pixel: float = 1.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FeatureConfig":
        cfg = config.get("defect", {}) or {}
        defaults = cls()
        return cls(mm_per_pixel=float(cfg.get("mm_per_pixel", defaults.mm_per_pixel)))


def extract_features(defect_mask: np.ndarray, h_res: np.ndarray, config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = FeatureConfig.from_config(config)
    residual_mm = np.asarray(h_res, dtype=np.float32) * 1000.0
    pixel_area_mm2 = cfg.mm_per_pixel * cfg.mm_per_pixel
    features: list[dict[str, Any]] = []
    for region in connected_components(defect_mask):
        values = residual_mm[region.rows, region.cols]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        features.append(_region_features(len(features) + 1, region, values, cfg.mm_per_pixel, pixel_area_mm2))
    return features


def _region_features(
    feature_id: int,
    region: Any,
    residual_mm: np.ndarray,
    mm_per_pixel: float,
    pixel_area_mm2: float,
) -> dict[str, Any]:
    x, y, width_px, height_px = region.bbox_xywh
    x_min = float(x) * mm_per_pixel
    x_max = float(x + width_px) * mm_per_pixel
    y_min = float(y) * mm_per_pixel
    y_max = float(y + height_px) * mm_per_pixel
    z_min = float(np.min(residual_mm))
    z_max = float(np.max(residual_mm))
    width_mm = x_max - x_min
    height_mm = y_max - y_min
    depth_mm = z_max - z_min
    bbox_volume_mm3 = width_mm * height_mm * max(depth_mm, 0.0)
    signed_volume_mm3 = float(np.sum(residual_mm) * pixel_area_mm2)
    absolute_volume_mm3 = float(np.sum(np.abs(residual_mm)) * pixel_area_mm2)
    positive_volume_mm3 = float(np.sum(np.clip(residual_mm, 0.0, None)) * pixel_area_mm2)
    negative_volume_mm3 = float(np.sum(np.clip(-residual_mm, 0.0, None)) * pixel_area_mm2)
    return {
        "id": int(feature_id),
        "type": "unknown",
        "pixel_count": int(region.pixel_count),
        "centroid": {
            "x_px": float(np.mean(region.cols)),
            "y_px": float(np.mean(region.rows)),
            "x_mm": float(np.mean(region.cols)) * mm_per_pixel,
            "y_mm": float(np.mean(region.rows)) * mm_per_pixel,
        },
        "bbox": {
            "xmin": x_min,
            "xmax": x_max,
            "ymin": y_min,
            "ymax": y_max,
            "zmin": z_min,
            "zmax": z_max,
            "units": "mm",
        },
        "bbox_px": {
            "xmin": int(x),
            "xmax": int(x + width_px - 1),
            "ymin": int(y),
            "ymax": int(y + height_px - 1),
            "width": int(width_px),
            "height": int(height_px),
        },
        "dimensions": {
            "width": width_mm,
            "height": height_mm,
            "depth": depth_mm,
            "major_axis": max(width_mm, height_mm),
            "minor_axis": min(width_mm, height_mm),
            "aspect_ratio": max(width_mm, height_mm) / max(min(width_mm, height_mm), 1e-9),
            "units": "mm",
        },
        "area": float(region.pixel_count) * pixel_area_mm2,
        "volume": signed_volume_mm3,
        "volume_abs": absolute_volume_mm3,
        "volume_positive": positive_volume_mm3,
        "volume_negative": negative_volume_mm3,
        "bbox_volume": bbox_volume_mm3,
        "volume_to_bbox_volume": absolute_volume_mm3 / bbox_volume_mm3 if bbox_volume_mm3 > 0 else 0.0,
        "surface": {
            "mean_depth": float(np.mean(residual_mm)),
            "std_depth": float(np.std(residual_mm)),
            "min_depth": z_min,
            "max_depth": z_max,
            "peak_abs_depth": float(np.max(np.abs(residual_mm))),
            "units": "mm",
        },
        "voxel_ready": {
            "num_points": int(region.pixel_count),
            "pixel_area_mm2": pixel_area_mm2,
            "height_samples_unit": "mm",
        },
    }
