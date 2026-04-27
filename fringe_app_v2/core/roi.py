"""ROI detection and persistence."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from fringe_app.vision.object_roi import ObjectRoiConfig, detect_object_roi

from fringe_app_v2.utils.io import RunPaths, save_image, save_mask_png, write_json


def roi_config_from_dict(config: dict[str, Any]) -> ObjectRoiConfig:
    roi = config.get("roi", {}) or {}
    post = roi.get("post", {}) or {}
    return ObjectRoiConfig(
        downscale_max_w=int(roi.get("downscale_max_w", 640)),
        blur_ksize=int(roi.get("blur_ksize", 7)),
        black_bg_percentile=float(roi.get("black_bg_percentile", 82)),
        threshold_offset=float(roi.get("threshold_offset", 22)),
        min_area_ratio=float(roi.get("min_area_ratio", 0.01)),
        max_area_ratio=float(roi.get("max_area_ratio", 0.50)),
        close_iters=int(roi.get("close_iters", 1)),
        open_iters=int(roi.get("open_iters", 2)),
        fill_holes=bool(roi.get("fill_holes", True)),
        ref_method=str(roi.get("ref_method", "median_over_frames")),  # type: ignore[arg-type]
        post_enabled=bool(post.get("enabled", True)),
        post_keep_largest_component=bool(post.get("keep_largest_component", True)),
        post_fill_small_holes=bool(post.get("fill_small_holes", True)),
        post_max_hole_area=int(post.get("max_hole_area", 2000)),
        post_dilate_radius_px=int(post.get("dilate_radius_px", 10)),
    )


def detect_and_save_roi(run: RunPaths, image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    cfg = roi_config_from_dict(config)
    result = detect_object_roi(image, cfg)
    save_image(run.roi / "roi_source.png", image)
    np.save(run.roi / "roi_mask.npy", result.roi_mask.astype(bool))
    np.save(run.roi / "roi_mask_core.npy", result.roi_core_mask.astype(bool))
    np.save(run.roi / "roi_mask_dilated.npy", result.roi_dilated_mask.astype(bool))
    save_mask_png(run.roi / "roi_mask.png", result.roi_mask)
    save_mask_png(run.roi / "roi_mask_core.png", result.roi_core_mask)
    save_mask_png(run.roi / "roi_mask_dilated.png", result.roi_dilated_mask)
    save_mask_png(run.roi / "roi_raw.png", result.raw_mask)
    save_mask_png(run.roi / "roi_post.png", result.post_mask)
    write_json(
        run.roi / "roi_meta.json",
        {
            "cfg": asdict(cfg),
            "bbox": result.bbox,
            "bbox_core": result.bbox_core,
            "bbox_dilated": result.bbox_dilated,
            "area_ratio": float(np.count_nonzero(result.roi_mask) / result.roi_mask.size),
            "debug": result.debug,
        },
    )
    return result.roi_mask.astype(bool)
