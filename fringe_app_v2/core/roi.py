"""ROI detection and persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from fringe_app_v2.core.object_roi import ObjectRoiConfig, detect_object_roi

from fringe_app_v2.utils.io import RunPaths, save_image, save_mask_png, write_json
from fringe_app_v2.utils.math_utils import to_gray_float


@dataclass(frozen=True, slots=True)
class RoiResult:
    roi_mask: np.ndarray
    bbox: tuple[int, int, int, int] | None
    roi_core_mask: np.ndarray
    roi_dilated_mask: np.ndarray
    bbox_core: tuple[int, int, int, int] | None
    bbox_dilated: tuple[int, int, int, int] | None
    raw_mask: np.ndarray
    post_mask: np.ndarray
    debug: dict[str, Any]


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
        fallback_mode=str(roi.get("fallback_mode", "empty")),  # type: ignore[arg-type]
    )


def detect_and_save_roi(run: RunPaths, image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    roi = config.get("roi", {}) or {}
    if str(roi.get("method", "not_dark")) == "legacy":
        cfg = roi_config_from_dict(config)
        result = detect_object_roi(image, cfg)
        cfg_payload = asdict(cfg)
    else:
        result = detect_not_dark_roi(image, config)
        cfg_payload = result.debug.get("cfg", {})
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
            "cfg": cfg_payload,
            "bbox": result.bbox,
            "bbox_core": result.bbox_core,
            "bbox_dilated": result.bbox_dilated,
            "area_ratio": float(np.count_nonzero(result.roi_mask) / result.roi_mask.size),
            "debug": result.debug,
        },
    )
    return result.roi_mask.astype(bool)


def detect_not_dark_roi(image: np.ndarray, config: dict[str, Any]) -> RoiResult:
    roi = config.get("roi", {}) or {}
    post = roi.get("post", {}) or {}
    gray = to_gray_float(image).astype(np.float32)
    h, w = gray.shape
    max_channel = _max_channel(image)
    border_width = int(roi.get("border_width_px", 24))
    border = _border_mask(gray.shape, border_width)

    gray_bg = float(np.percentile(gray[border], float(roi.get("background_percentile", 95.0))))
    max_bg = float(np.percentile(max_channel[border], float(roi.get("background_percentile", 95.0))))
    gray_threshold = max(float(roi.get("min_foreground_gray", 12.0)), gray_bg + float(roi.get("gray_offset", 6.0)))
    max_threshold = max(float(roi.get("min_foreground_max_channel", 16.0)), max_bg + float(roi.get("max_channel_offset", 8.0)))

    blurred_gray = _gaussian_blur(gray, float(roi.get("sigma_px", 1.5)))
    blurred_max = _gaussian_blur(max_channel, float(roi.get("sigma_px", 1.5)))
    raw_mask = (blurred_gray > gray_threshold) | (blurred_max > max_threshold)
    post_mask = _morphology(raw_mask, close_radius=int(roi.get("close_radius_px", 3)), open_radius=int(roi.get("open_radius_px", 1)))
    core_mask = _largest_component_mask(post_mask)
    fallback = core_mask is None
    fallback_reason = "no_component" if fallback else None
    min_area_ratio = float(roi.get("min_area_ratio", 0.0))
    max_area_ratio = float(roi.get("max_area_ratio", 1.0))
    if core_mask is None:
        core_mask = _roi_fallback_mask(gray.shape, str(roi.get("fallback_mode", "empty")), roi)
    else:
        core_mask = _fill_holes(core_mask)
        core_ratio = float(np.count_nonzero(core_mask) / core_mask.size)
        if core_ratio < min_area_ratio or core_ratio > max_area_ratio:
            fallback = True
            fallback_reason = "area_ratio_out_of_range"
            core_mask = _roi_fallback_mask(gray.shape, str(roi.get("fallback_mode", "empty")), roi)

    dilated = core_mask.copy()
    if bool(post.get("enabled", True)) and np.any(dilated):
        if bool(post.get("fill_small_holes", True)):
            dilated = _fill_holes(dilated)
        radius = int(post.get("dilate_radius_px", 12))
        if radius > 0:
            dilated = _binary_dilate(dilated, radius)
    final_ratio = float(np.count_nonzero(dilated) / float(h * w))
    if not fallback and (final_ratio < min_area_ratio or final_ratio > max_area_ratio):
        fallback = True
        fallback_reason = "post_area_ratio_out_of_range"
        core_mask = _roi_fallback_mask(gray.shape, str(roi.get("fallback_mode", "empty")), roi)
        dilated = core_mask.copy()
    bbox_core = _bbox_from_mask(core_mask)
    bbox_dilated = _bbox_from_mask(dilated)
    debug_cfg = {
        "method": "not_dark",
        "border_width_px": border_width,
        "background_percentile": float(roi.get("background_percentile", 95.0)),
        "gray_offset": float(roi.get("gray_offset", 6.0)),
        "max_channel_offset": float(roi.get("max_channel_offset", 8.0)),
        "min_foreground_gray": float(roi.get("min_foreground_gray", 12.0)),
        "min_foreground_max_channel": float(roi.get("min_foreground_max_channel", 16.0)),
        "sigma_px": float(roi.get("sigma_px", 1.5)),
        "close_radius_px": int(roi.get("close_radius_px", 3)),
        "open_radius_px": int(roi.get("open_radius_px", 1)),
        "post": post,
    }
    return RoiResult(
        roi_mask=dilated.astype(bool),
        bbox=bbox_dilated,
        roi_core_mask=core_mask.astype(bool),
        roi_dilated_mask=dilated.astype(bool),
        bbox_core=bbox_core,
        bbox_dilated=bbox_dilated,
        raw_mask=raw_mask.astype(bool),
        post_mask=dilated.astype(bool),
        debug={
            "cfg": debug_cfg,
            "gray_bg": gray_bg,
            "max_channel_bg": max_bg,
            "gray_threshold": gray_threshold,
            "max_channel_threshold": max_threshold,
            "roi_fallback": fallback,
            "fallback_reason": fallback_reason,
            "min_area_ratio": min_area_ratio,
            "max_area_ratio": max_area_ratio,
        },
    )


def _roi_fallback_mask(
    shape: tuple[int, int],
    mode: str,
    roi: dict[str, Any],
) -> np.ndarray:
    h, w = shape
    normalized = mode.strip().lower()
    if normalized in {"full", "full_frame", "all"}:
        return np.ones(shape, dtype=bool)
    if normalized in {"center", "center_crop"}:
        frac = float(roi.get("fallback_center_fraction", 0.5))
        frac = min(1.0, max(0.05, frac))
        crop_w = max(1, int(round(w * frac)))
        crop_h = max(1, int(round(h * frac)))
        x0 = max(0, (w - crop_w) // 2)
        y0 = max(0, (h - crop_h) // 2)
        mask = np.zeros(shape, dtype=bool)
        mask[y0:y0 + crop_h, x0:x0 + crop_w] = True
        return mask
    return np.zeros(shape, dtype=bool)


def _max_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim == 3:
        return np.max(image[:, :, :3].astype(np.float32), axis=2)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _border_mask(shape: tuple[int, int], width: int) -> np.ndarray:
    h, w = shape
    bw = max(1, min(int(width), h // 3, w // 3))
    mask = np.zeros((h, w), dtype=bool)
    mask[:bw, :] = True
    mask[-bw:, :] = True
    mask[:, :bw] = True
    mask[:, -bw:] = True
    return mask


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image.astype(np.float32)
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(image.astype(np.float32), sigma=sigma, mode="nearest")
    except Exception:
        return image.astype(np.float32)


def _morphology(mask: np.ndarray, close_radius: int, open_radius: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    if close_radius > 0:
        out = _binary_erode(_binary_dilate(out, close_radius), close_radius)
    if open_radius > 0:
        out = _binary_dilate(_binary_erode(out, open_radius), open_radius)
    return out


def _largest_component_mask(mask: np.ndarray) -> np.ndarray | None:
    try:
        from scipy import ndimage

        labels, count = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
        if count == 0:
            return None
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        return labels == int(np.argmax(sizes))
    except Exception:
        return mask.astype(bool) if np.any(mask) else None


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import binary_fill_holes

        return binary_fill_holes(mask).astype(bool)
    except Exception:
        return mask.astype(bool)


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(radius))):
        out = _shift_or(out)
    return out


def _binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(radius))):
        out = _shift_and(out)
    return out


def _shift_or(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    return np.logical_or.reduce(
        [
            p[0:h, 0:w],
            p[0:h, 1 : w + 1],
            p[0:h, 2 : w + 2],
            p[1 : h + 1, 0:w],
            p[1 : h + 1, 1 : w + 1],
            p[1 : h + 1, 2 : w + 2],
            p[2 : h + 2, 0:w],
            p[2 : h + 2, 1 : w + 1],
            p[2 : h + 2, 2 : w + 2],
        ]
    )


def _shift_and(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    return np.logical_and.reduce(
        [
            p[0:h, 0:w],
            p[0:h, 1 : w + 1],
            p[0:h, 2 : w + 2],
            p[1 : h + 1, 0:w],
            p[1 : h + 1, 1 : w + 1],
            p[1 : h + 1, 2 : w + 2],
            p[2 : h + 2, 0:w],
            p[2 : h + 2, 1 : w + 1],
            p[2 : h + 2, 2 : w + 2],
        ]
    )


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask.astype(bool))
    if ys.size == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
