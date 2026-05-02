"""Defect segmentation, geometry feature extraction, and labeling stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fringe_app_v2.defect.features import extract_features
from fringe_app_v2.defect.models import classify_defect
from fringe_app_v2.defect.segment import DefectSegmentationResult, segment_defects
from fringe_app_v2.defect.utils import normalise_to_u8, save_overlay
from fringe_app_v2.utils.io import RunPaths, load_image, save_image, save_mask_png, write_json
from fringe_app_v2.utils.math_utils import to_gray_float


def run_defect_stage(run: RunPaths, config: dict[str, Any]) -> dict[str, Any]:
    return run_defect_for_run(run.root, config)


def run_defect_for_run(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(run_dir)
    height_map, mask, source = load_height_inputs(run_dir)
    segmentation = segment_defects(height_map, mask, config)
    features = extract_features(segmentation.defect_mask, segmentation.residual_m, config)
    classified = [classify_defect(item, config) for item in features]
    summary = _summary(segmentation, classified, source)
    save_defect_outputs(run_dir, segmentation, classified, summary)
    return summary


def load_height_inputs(run_dir: Path) -> tuple[np.ndarray, np.ndarray, str]:
    flatten = Path(run_dir) / "flatten"
    recon = Path(run_dir) / "reconstruct"
    height_path = _first_existing(
        [flatten / "height_flat.npy", recon / "height_map.npy", recon / "height.npy", recon / "depth.npy"]
    )
    mask_path = _first_existing(
        [
            flatten / "mask_flat.npy",
            recon / "masks" / "mask_reconstruct.npy",
            recon / "masks" / "mask_recon.npy",
            recon / "masks" / "mask_uv.npy",
        ]
    )
    return np.load(height_path).astype(np.float32), np.load(mask_path).astype(bool), str(height_path.relative_to(run_dir))


def save_defect_outputs(
    run_dir: Path,
    segmentation: DefectSegmentationResult,
    features: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    out_dir = Path(run_dir) / "defect"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "defect_mask.npy", segmentation.defect_mask.astype(bool))
    np.save(out_dir / "defect_raw_mask.npy", segmentation.raw_mask.astype(bool))
    np.save(out_dir / "height_reference.npy", segmentation.reference_m.astype(np.float32))
    np.save(out_dir / "height_residual.npy", segmentation.residual_m.astype(np.float32))
    save_mask_png(out_dir / "defect_mask.png", segmentation.defect_mask)
    save_mask_png(out_dir / "defect_raw_mask.png", segmentation.raw_mask)
    save_image(out_dir / "residual.png", normalise_to_u8(segmentation.residual_m, segmentation.valid_mask))
    save_overlay(out_dir / "height_overlay.png", segmentation.residual_m, segmentation.defect_mask, segmentation.valid_mask)
    base = _load_gray_overlay_base(run_dir, segmentation.residual_m.shape)
    save_overlay(out_dir / "overlay.png", base, segmentation.defect_mask, segmentation.valid_mask)
    write_json(out_dir / "features.json", {**summary, "features": features})


def _summary(segmentation: DefectSegmentationResult, features: list[dict[str, Any]], source: str) -> dict[str, Any]:
    valid_px = int(np.count_nonzero(segmentation.valid_mask))
    defect_px = int(np.count_nonzero(segmentation.defect_mask))
    type_counts: dict[str, int] = {}
    for item in features:
        label = str(item.get("type", "unknown"))
        type_counts[label] = type_counts.get(label, 0) + 1
    cfg = segmentation.config
    return {
        "mode": "geometry_features",
        "source": source,
        "height_units": "meters",
        "feature_units": "millimeters",
        "valid_pixels": valid_px,
        "defect_pixels": defect_px,
        "defect_fraction": float(defect_px / valid_px) if valid_px else 0.0,
        "component_count": len(features),
        "type_counts": type_counts,
        "segmentation": {
            "smoothing": cfg.smoothing,
            "depth_threshold_mm": cfg.depth_threshold_mm,
            "min_area_px": cfg.min_area_px,
            "open_radius_px": cfg.open_radius_px,
            "close_radius_px": cfg.close_radius_px,
            "boundary_exclusion_px": cfg.boundary_exclusion_px,
            "max_area_px": cfg.max_area_px,
            "max_area_fraction": cfg.max_area_fraction,
            "max_area_floor_px": cfg.max_area_floor_px,
            "edge_suppression_enabled": cfg.edge_suppression_enabled,
            "edge_percentile": cfg.edge_percentile,
            "edge_exclusion_radius_px": cfg.edge_exclusion_radius_px,
            "edge_overlap_reject": cfg.edge_overlap_reject,
        },
    }


def _load_gray_overlay_base(run_dir: Path, shape: tuple[int, int]) -> np.ndarray:
    path = Path(run_dir) / "raw" / "roi_capture.png"
    if path.exists():
        gray = to_gray_float(load_image(path)).astype(np.float32)
        if gray.shape == shape:
            return gray
    return np.zeros(shape, dtype=np.float32)


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these files exists: " + ", ".join(str(path) for path in paths))
