"""Single-view 2D defect detection for controlled-light inspection."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from fringe_app.core.auto_normalise import NormaliseConfig, auto_normalise_capture
from fringe_app.core.models import ScanParams
from fringe_app.io.run_store import RunStore
from fringe_app.vision.object_roi import ObjectRoiConfig, detect_object_roi


DEFECT_2D_MODE = "2D_DEFECT_MODE"


@dataclass(slots=True)
class Defect2DConfig:
    """Image-processing parameters for direct 2D defect detection."""

    normalise_low_percentile: float = 1.0
    normalise_high_percentile: float = 99.0
    gaussian_blur: bool = True
    blur_sigma: float = 0.8
    histogram_equalisation: bool = False
    gradient_kernel: Literal["sobel", "scharr"] = "scharr"
    highpass_sigma: float = 7.0
    local_std_window: int = 15
    roi_inner_erode_px: int = 3
    threshold_auto: bool = True
    gradient_threshold: float = 0.22
    highpass_threshold: float = 0.08
    local_std_threshold: float = 0.055
    reference_diff_threshold: float = 0.10
    auto_mad_k: float = 6.0
    auto_percentile: float = 99.2
    morphology_open_iters: int = 1
    morphology_close_iters: int = 1
    min_area_px: int = 18
    max_area_px: int = 0
    save_heatmap: bool = True


@dataclass(slots=True)
class Defect2DCaptureConfig:
    """Acquisition parameters for one 2D defect run."""

    stack_size: int = 5
    use_averaging: bool = False
    inspection_dn: int = 180
    settle_ms: int = 180
    flush_frames: int = 2
    auto_normalise: bool = True
    save_debug: bool = False
    fullscreen: bool = True
    projector_screen_index: int | None = None


@dataclass(slots=True)
class Defect2DAnalysis:
    raw_rgb: np.ndarray
    gray_u8: np.ndarray
    preprocessed_u8: np.ndarray
    roi_mask: np.ndarray
    roi_core_mask: np.ndarray
    analysis_mask: np.ndarray
    bbox: tuple[int, int, int, int]
    defect_mask: np.ndarray
    overlay_rgb: np.ndarray
    heatmap_rgb: np.ndarray
    features: dict[str, np.ndarray]
    thresholds: dict[str, float | None]
    metrics: dict[str, Any]


def config_from_dict(data: dict[str, Any] | None) -> Defect2DConfig:
    cfg = Defect2DConfig()
    for key, value in (data or {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.gradient_kernel = "scharr" if str(cfg.gradient_kernel).lower() == "scharr" else "sobel"
    cfg.local_std_window = max(3, int(cfg.local_std_window) | 1)
    cfg.min_area_px = max(1, int(cfg.min_area_px))
    cfg.max_area_px = max(0, int(cfg.max_area_px))
    cfg.roi_inner_erode_px = max(0, int(cfg.roi_inner_erode_px))
    return cfg


def capture_config_from_dict(data: dict[str, Any] | None) -> Defect2DCaptureConfig:
    cfg = Defect2DCaptureConfig()
    for key, value in (data or {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.stack_size = max(1, int(cfg.stack_size))
    cfg.inspection_dn = int(np.clip(int(cfg.inspection_dn), 0, 255))
    cfg.settle_ms = max(0, int(cfg.settle_ms))
    cfg.flush_frames = max(0, int(cfg.flush_frames))
    return cfg


def analyse_image(
    image: np.ndarray,
    cfg: Defect2DConfig,
    roi_cfg: ObjectRoiConfig,
    reference_image: np.ndarray | None = None,
) -> Defect2DAnalysis:
    """Run Capture -> Preprocess -> ROI -> Features -> Detection -> Visualisation."""
    raw_rgb = _as_rgb_u8(image)
    gray_u8 = _to_gray_u8(raw_rgb)

    roi_res = detect_object_roi(gray_u8, roi_cfg)
    roi_core = roi_res.roi_core_mask.astype(bool)
    roi_mask = roi_res.roi_dilated_mask.astype(bool)
    if bool(roi_res.debug.get("roi_fallback", False)):
        roi_core = roi_mask.astype(bool)

    analysis_mask = roi_core.copy()
    for _ in range(int(cfg.roi_inner_erode_px)):
        analysis_mask = _erode(analysis_mask)
    if int(np.count_nonzero(analysis_mask)) < max(64, int(0.001 * analysis_mask.size)):
        analysis_mask = roi_core.copy()

    gray01 = gray_u8.astype(np.float32) / 255.0
    proc = _robust_normalise(
        gray01,
        analysis_mask,
        cfg.normalise_low_percentile,
        cfg.normalise_high_percentile,
    )
    if cfg.histogram_equalisation:
        proc = _equalise_histogram(proc, analysis_mask)
    if cfg.gaussian_blur and cfg.blur_sigma > 0:
        proc = _gaussian_blur(proc, float(cfg.blur_sigma))
    proc = np.clip(proc, 0.0, 1.0).astype(np.float32)

    gradient = _gradient_magnitude(proc, cfg.gradient_kernel)
    low = _gaussian_blur(proc, float(cfg.highpass_sigma)) if cfg.highpass_sigma > 0 else proc
    highpass = np.abs(proc - low).astype(np.float32)
    local_std = _local_std(proc, int(cfg.local_std_window))

    ref_diff = None
    if reference_image is not None:
        ref_gray = _to_gray_u8(_as_rgb_u8(reference_image))
        if ref_gray.shape != gray_u8.shape:
            raise ValueError(
                f"Reference image shape {ref_gray.shape} does not match capture shape {gray_u8.shape}"
            )
        ref01 = _robust_normalise(
            ref_gray.astype(np.float32) / 255.0,
            analysis_mask,
            cfg.normalise_low_percentile,
            cfg.normalise_high_percentile,
        )
        ref_diff = np.abs(proc - ref01).astype(np.float32)

    thresholds = _feature_thresholds(
        cfg,
        analysis_mask,
        gradient=gradient,
        highpass=highpass,
        local_std=local_std,
        reference_diff=ref_diff,
    )

    candidate = (
        (gradient > float(thresholds["gradient"]))
        | (highpass > float(thresholds["highpass"]))
        | (local_std > float(thresholds["local_std"]))
    )
    if ref_diff is not None and thresholds.get("reference_diff") is not None:
        candidate |= ref_diff > float(thresholds["reference_diff"])
    candidate &= analysis_mask

    cleaned = candidate.copy()
    for _ in range(int(cfg.morphology_open_iters)):
        cleaned = _dilate(_erode(cleaned))
    for _ in range(int(cfg.morphology_close_iters)):
        cleaned = _erode(_dilate(cleaned))
    cleaned &= analysis_mask

    defect_mask, regions = _filter_components(cleaned, int(cfg.min_area_px), int(cfg.max_area_px))
    score_map = _combined_score(
        gradient,
        highpass,
        local_std,
        ref_diff,
        thresholds,
        analysis_mask,
    )

    bbox = roi_res.bbox_dilated or _full_bbox(gray_u8)
    overlay = _overlay_defects(raw_rgb, defect_mask, roi_mask)
    heatmap = _heatmap(score_map, analysis_mask)

    roi_area_px = int(np.count_nonzero(analysis_mask))
    defect_area_px = int(np.count_nonzero(defect_mask))
    region_areas = [int(r["area_px"]) for r in regions]
    metrics: dict[str, Any] = {
        "mode": DEFECT_2D_MODE,
        "pipeline": [
            "capture",
            "preprocess",
            "roi",
            "feature_extraction",
            "defect_detection",
            "visualisation",
            "save",
        ],
        "roi_fallback": bool(roi_res.debug.get("roi_fallback", False)),
        "bbox": [int(v) for v in bbox],
        "roi_area_px": roi_area_px,
        "defect_area_px": defect_area_px,
        "defect_area_percent": float(100.0 * defect_area_px / max(1, roi_area_px)),
        "defect_region_count": int(len(regions)),
        "average_defect_size_px": float(np.mean(region_areas)) if region_areas else 0.0,
        "max_defect_size_px": int(max(region_areas)) if region_areas else 0,
        "regions": regions[:200],
        "thresholds": thresholds,
        "roi_debug": roi_res.debug,
        "feature_stats": {
            "gradient": _stats(gradient, analysis_mask),
            "highpass": _stats(highpass, analysis_mask),
            "local_std": _stats(local_std, analysis_mask),
            "reference_diff": _stats(ref_diff, analysis_mask) if ref_diff is not None else None,
        },
    }
    features = {
        "gradient": gradient,
        "highpass": highpass,
        "local_std": local_std,
        "score": score_map,
    }
    if ref_diff is not None:
        features["reference_diff"] = ref_diff

    return Defect2DAnalysis(
        raw_rgb=raw_rgb,
        gray_u8=gray_u8,
        preprocessed_u8=_float01_to_u8(proc),
        roi_mask=roi_mask,
        roi_core_mask=roi_core,
        analysis_mask=analysis_mask,
        bbox=bbox,
        defect_mask=defect_mask,
        overlay_rgb=overlay,
        heatmap_rgb=heatmap,
        features=features,
        thresholds=thresholds,
        metrics=metrics,
    )


def save_analysis_outputs(
    out_dir: Path,
    analysis: Defect2DAnalysis,
    cfg: Defect2DConfig,
    *,
    save_debug: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, w, h = analysis.bbox

    outputs = {
        "raw": "raw.png",
        "roi": "roi.png",
        "defect_mask": "defect_mask.png",
        "overlay": "overlay.png",
        "metrics": "metrics.json",
    }
    Image.fromarray(analysis.raw_rgb).save(out_dir / outputs["raw"])
    Image.fromarray(analysis.raw_rgb[y : y + h, x : x + w]).save(out_dir / outputs["roi"])
    Image.fromarray(np.where(analysis.defect_mask, 255, 0).astype(np.uint8)).save(
        out_dir / outputs["defect_mask"]
    )
    Image.fromarray(analysis.overlay_rgb).save(out_dir / outputs["overlay"])

    if cfg.save_heatmap:
        outputs["heatmap"] = "heatmap.png"
        Image.fromarray(analysis.heatmap_rgb).save(out_dir / outputs["heatmap"])

    if save_debug:
        debug_dir = out_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        Image.fromarray(analysis.gray_u8).save(debug_dir / "gray.png")
        Image.fromarray(analysis.preprocessed_u8).save(debug_dir / "preprocessed.png")
        Image.fromarray(np.where(analysis.roi_mask, 255, 0).astype(np.uint8)).save(debug_dir / "roi_mask.png")
        Image.fromarray(np.where(analysis.analysis_mask, 255, 0).astype(np.uint8)).save(
            debug_dir / "analysis_mask.png"
        )
        for name, fmap in analysis.features.items():
            Image.fromarray(_autoscale_u8(fmap, analysis.analysis_mask)).save(debug_dir / f"{name}.png")
        outputs["debug"] = "debug/"

    metrics = {
        **analysis.metrics,
        "config": asdict(cfg),
        "outputs": outputs,
    }
    (out_dir / outputs["metrics"]).write_text(json.dumps(_json_safe(metrics), indent=2))
    return metrics


def average_stack(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("Cannot average an empty frame stack")
    stack = np.stack([_as_rgb_u8(f).astype(np.float32) for f in frames], axis=0)
    return np.clip(np.rint(np.mean(stack, axis=0)), 0, 255).astype(np.uint8)


def run_existing_image(
    image_path: Path,
    store: RunStore,
    params: ScanParams,
    roi_cfg: ObjectRoiConfig,
    defect_cfg: Defect2DConfig,
    *,
    reference_path: Path | None = None,
    save_debug: bool = False,
) -> dict[str, Any]:
    """Run the 2D detector on an existing image while preserving run layout."""
    params.scan_mode = DEFECT_2D_MODE
    run_id, run_dir, _ = store.create_run(
        params,
        {"source": "existing_image", "image_path": str(image_path), "mode": DEFECT_2D_MODE},
        preview_enabled=False,
    )
    raw = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    store.save_capture(run_dir, 0, raw)
    reference = np.array(Image.open(reference_path).convert("RGB"), dtype=np.uint8) if reference_path else None
    try:
        analysis = analyse_image(raw, defect_cfg, roi_cfg, reference_image=reference)
        _save_roi_artifacts(store, run_id, analysis, roi_cfg)
        metrics = save_analysis_outputs(run_dir / "2d_defect", analysis, defect_cfg, save_debug=save_debug)
        metrics["run_id"] = run_id
        metrics["run_dir"] = str(run_dir)
        store.update_meta(
            run_dir,
            status="completed",
            error=None,
            finished_at=datetime.now().isoformat(),
            total_frames=1,
            saved_frames=1,
        )
        return metrics
    except Exception as exc:
        store.update_meta(
            run_dir,
            status="error",
            error=str(exc),
            finished_at=datetime.now().isoformat(),
            total_frames=1,
            saved_frames=1,
        )
        raise


def capture_and_run(
    *,
    display: Any,
    camera: Any,
    store: RunStore,
    params: ScanParams,
    roi_cfg: ObjectRoiConfig,
    defect_cfg: Defect2DConfig,
    capture_cfg: Defect2DCaptureConfig,
    normalise_cfg: NormaliseConfig | None = None,
    reference_path: Path | None = None,
) -> dict[str, Any]:
    """Capture a uniform-light image/stack and run direct 2D inspection."""
    params.scan_mode = DEFECT_2D_MODE
    n_frames = capture_cfg.stack_size if capture_cfg.use_averaging else 1
    params.n_steps = n_frames
    params.frequencies = [0.0]
    params.frequency = 0.0
    params.orientation = "vertical"

    device_info: dict[str, Any] = {
        "camera": type(camera).__name__,
        "mode": DEFECT_2D_MODE,
        "capture": asdict(capture_cfg),
    }
    run_id, run_dir, _ = store.create_run(params, device_info, preview_enabled=False)
    captured = 0
    status = "error"
    error_msg: str | None = None
    started = datetime.now().isoformat()

    try:
        display.open(
            fullscreen=bool(capture_cfg.fullscreen),
            screen_index=capture_cfg.projector_screen_index,
        )
        surface_size = getattr(display, "get_projector_surface_size", lambda: None)()
        if surface_size is not None:
            device_info["projector_surface_size"] = [int(surface_size[0]), int(surface_size[1])]

        camera.start(params)
        if capture_cfg.auto_normalise:
            norm = normalise_cfg or NormaliseConfig()
            normalise_res = auto_normalise_capture(
                display,
                camera,
                norm,
                params,
                roi_cfg=roi_cfg,
                debug_dir=run_dir / "normalise",
            )
            params.exposure_us = normalise_res.exposure_us
            params.analogue_gain = normalise_res.analogue_gain
            params.awb_enable = False
            params.ae_enable = False
            store.save_normalise(run_dir, normalise_res.to_dict())
            device_info["normalise"] = normalise_res.to_dict()

        if params.exposure_us is not None and params.analogue_gain is not None:
            try:
                camera.set_manual_controls(
                    int(params.exposure_us),
                    float(params.analogue_gain),
                    awb_enable=False if params.awb_enable is None else bool(params.awb_enable),
                )
            except Exception:
                pass

        width, height = params.resolution
        illum = np.full((int(height), int(width)), int(capture_cfg.inspection_dn), dtype=np.uint8)
        display.show_gray(illum)
        display.pump()
        time.sleep(float(capture_cfg.settle_ms) / 1000.0)

        for _ in range(max(0, int(capture_cfg.flush_frames))):
            try:
                camera.capture_pair()
            except Exception:
                break

        frames: list[np.ndarray] = []
        for idx in range(n_frames):
            main, _preview = camera.capture_pair()
            main_u8 = _as_rgb_u8(main)
            frames.append(main_u8)
            store.save_capture(run_dir, idx, main_u8)
            captured += 1

        raw = average_stack(frames) if capture_cfg.use_averaging else frames[0]
        reference = np.array(Image.open(reference_path).convert("RGB"), dtype=np.uint8) if reference_path else None
        analysis = analyse_image(raw, defect_cfg, roi_cfg, reference_image=reference)
        _save_roi_artifacts(store, run_id, analysis, roi_cfg)
        metrics = save_analysis_outputs(
            run_dir / "2d_defect",
            analysis,
            defect_cfg,
            save_debug=bool(capture_cfg.save_debug),
        )
        status = "completed"
        metrics["run_id"] = run_id
        metrics["run_dir"] = str(run_dir)
        metrics["capture"] = {
            "frames_captured": int(captured),
            "use_averaging": bool(capture_cfg.use_averaging),
            "inspection_dn": int(capture_cfg.inspection_dn),
        }
        return metrics
    except Exception as exc:
        error_msg = str(exc)
        raise
    finally:
        try:
            device_info["applied_controls"] = getattr(camera, "get_applied_controls", lambda: {})()
        except Exception:
            pass
        try:
            camera.stop()
        except Exception:
            pass
        try:
            display.close()
        except Exception:
            pass
        try:
            meta = store.load_meta(run_dir)
            meta.started_at = started
            meta.finished_at = datetime.now().isoformat()
            meta.status = status
            meta.error = error_msg
            meta.device_info = device_info
            meta.total_frames = int(n_frames)
            meta.saved_frames = int(captured)
            store.save_meta(run_dir, meta)
        except Exception:
            pass


def _as_rgb_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.repeat(np.clip(arr, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return np.clip(arr[:, :, :3], 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.clip(image, 0, 255).astype(np.uint8)
    rgb = _as_rgb_u8(image).astype(np.float32)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def _float01_to_u8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(image, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


def _robust_normalise(
    gray01: np.ndarray,
    mask: np.ndarray | None,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    vals = gray01[mask.astype(bool)] if mask is not None and np.any(mask) else gray01.reshape(-1)
    lo = float(np.percentile(vals, float(low_percentile)))
    hi = float(np.percentile(vals, float(high_percentile)))
    if hi <= lo + 1e-6:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if hi <= lo + 1e-6:
        return np.zeros_like(gray01, dtype=np.float32)
    return np.clip((gray01.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _equalise_histogram(gray01: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    u8 = _float01_to_u8(gray01)
    samples = u8[mask.astype(bool)] if mask is not None and np.any(mask) else u8.reshape(-1)
    hist = np.bincount(samples.reshape(-1), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist)
    nonzero = np.flatnonzero(cdf > 0)
    if nonzero.size == 0:
        return gray01.astype(np.float32)
    cdf_min = cdf[int(nonzero[0])]
    denom = max(1.0, float(cdf[-1] - cdf_min))
    lut = np.clip(np.rint((cdf - cdf_min) * 255.0 / denom), 0, 255).astype(np.uint8)
    return (lut[u8].astype(np.float32) / 255.0).astype(np.float32)


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image.astype(np.float32)
    radius = max(1, int(round(3.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    kernel /= float(np.sum(kernel))
    return _separable_filter(image.astype(np.float32), kernel)


def _separable_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    r = int(len(kernel) // 2)
    padded_x = np.pad(image, ((0, 0), (r, r)), mode="edge")
    tmp = np.zeros_like(image, dtype=np.float32)
    for i, weight in enumerate(kernel):
        tmp += float(weight) * padded_x[:, i : i + image.shape[1]]
    padded_y = np.pad(tmp, ((r, r), (0, 0)), mode="edge")
    out = np.zeros_like(image, dtype=np.float32)
    for i, weight in enumerate(kernel):
        out += float(weight) * padded_y[i : i + image.shape[0], :]
    return out


def _gradient_magnitude(image: np.ndarray, kernel: str) -> np.ndarray:
    p = np.pad(image.astype(np.float32), 1, mode="edge")
    a = p[:-2, :-2]
    b = p[:-2, 1:-1]
    c = p[:-2, 2:]
    d = p[1:-1, :-2]
    f = p[1:-1, 2:]
    g = p[2:, :-2]
    h = p[2:, 1:-1]
    i = p[2:, 2:]
    if kernel == "scharr":
        gx = (3.0 * c + 10.0 * f + 3.0 * i) - (3.0 * a + 10.0 * d + 3.0 * g)
        gy = (3.0 * g + 10.0 * h + 3.0 * i) - (3.0 * a + 10.0 * b + 3.0 * c)
        scale = 32.0
    else:
        gx = (c + 2.0 * f + i) - (a + 2.0 * d + g)
        gy = (g + 2.0 * h + i) - (a + 2.0 * b + c)
        scale = 8.0
    return np.clip(np.sqrt(gx * gx + gy * gy) / scale, 0.0, 1.0).astype(np.float32)


def _box_mean(image: np.ndarray, k: int) -> np.ndarray:
    k = max(3, int(k) | 1)
    pad = k // 2
    padded = np.pad(image.astype(np.float64), pad, mode="edge")
    integ = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    h, w = image.shape
    y0 = np.arange(h)
    x0 = np.arange(w)
    y1 = y0 + k
    x1 = x0 + k
    out = (
        integ[y1[:, None], x1]
        - integ[y0[:, None], x1]
        - integ[y1[:, None], x0]
        + integ[y0[:, None], x0]
    ) / float(k * k)
    return out.astype(np.float32)


def _local_std(image: np.ndarray, k: int) -> np.ndarray:
    mean = _box_mean(image, k)
    mean_sq = _box_mean(image * image, k)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var).astype(np.float32)


def _feature_thresholds(
    cfg: Defect2DConfig,
    mask: np.ndarray,
    *,
    gradient: np.ndarray,
    highpass: np.ndarray,
    local_std: np.ndarray,
    reference_diff: np.ndarray | None,
) -> dict[str, float | None]:
    if not cfg.threshold_auto:
        return {
            "gradient": float(cfg.gradient_threshold),
            "highpass": float(cfg.highpass_threshold),
            "local_std": float(cfg.local_std_threshold),
            "reference_diff": float(cfg.reference_diff_threshold) if reference_diff is not None else None,
        }
    return {
        "gradient": _auto_threshold(gradient, mask, cfg, float(cfg.gradient_threshold)),
        "highpass": _auto_threshold(highpass, mask, cfg, float(cfg.highpass_threshold)),
        "local_std": _auto_threshold(local_std, mask, cfg, float(cfg.local_std_threshold)),
        "reference_diff": (
            _auto_threshold(reference_diff, mask, cfg, float(cfg.reference_diff_threshold))
            if reference_diff is not None
            else None
        ),
    }


def _auto_threshold(feature: np.ndarray, mask: np.ndarray, cfg: Defect2DConfig, floor: float) -> float:
    vals = feature[mask.astype(bool)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(floor)
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    sigma = 1.4826 * mad
    robust = median + float(cfg.auto_mad_k) * sigma
    pct = float(np.percentile(vals, float(cfg.auto_percentile)))
    if sigma <= 1e-8:
        threshold = pct
    else:
        threshold = min(robust, pct)
    return float(max(float(floor), threshold))


def _combined_score(
    gradient: np.ndarray,
    highpass: np.ndarray,
    local_std: np.ndarray,
    reference_diff: np.ndarray | None,
    thresholds: dict[str, float | None],
    mask: np.ndarray,
) -> np.ndarray:
    score = np.maximum.reduce(
        [
            gradient / max(float(thresholds["gradient"] or 1e-6), 1e-6),
            highpass / max(float(thresholds["highpass"] or 1e-6), 1e-6),
            local_std / max(float(thresholds["local_std"] or 1e-6), 1e-6),
        ]
    )
    if reference_diff is not None and thresholds.get("reference_diff") is not None:
        score = np.maximum(score, reference_diff / max(float(thresholds["reference_diff"] or 1e-6), 1e-6))
    score = np.where(mask.astype(bool), score, 0.0)
    return np.clip(score.astype(np.float32), 0.0, 4.0)


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    p = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    n = [
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
    return np.logical_or.reduce(n)


def _erode(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    p = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    n = [
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
    return np.logical_and.reduce(n)


def _filter_components(
    mask: np.ndarray,
    min_area_px: int,
    max_area_px: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    m = mask.astype(bool)
    h, w = m.shape
    visited = np.zeros_like(m, dtype=bool)
    out = np.zeros_like(m, dtype=bool)
    regions: list[dict[str, Any]] = []
    coords = np.argwhere(m)
    for y0, x0 in coords:
        y = int(y0)
        x = int(x0)
        if visited[y, x] or not m[y, x]:
            continue
        stack = [(y, x)]
        visited[y, x] = True
        comp_y: list[int] = []
        comp_x: list[int] = []
        while stack:
            cy, cx = stack.pop()
            comp_y.append(cy)
            comp_x.append(cx)
            for ny in (cy - 1, cy, cy + 1):
                if ny < 0 or ny >= h:
                    continue
                for nx in (cx - 1, cx, cx + 1):
                    if nx < 0 or nx >= w or (ny == cy and nx == cx):
                        continue
                    if m[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        area = len(comp_y)
        if area < min_area_px:
            continue
        if max_area_px > 0 and area > max_area_px:
            continue
        ys = np.asarray(comp_y, dtype=np.int32)
        xs = np.asarray(comp_x, dtype=np.int32)
        out[ys, xs] = True
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())
        regions.append(
            {
                "area_px": int(area),
                "bbox": [x_min, y_min, int(x_max - x_min + 1), int(y_max - y_min + 1)],
                "centroid_px": [float(np.mean(xs)), float(np.mean(ys))],
            }
        )
    regions.sort(key=lambda item: int(item["area_px"]), reverse=True)
    return out, regions


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    return m & (~_erode(m))


def _overlay_defects(raw_rgb: np.ndarray, defect_mask: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    out = raw_rgb.copy().astype(np.uint8)
    defect = defect_mask.astype(bool)
    if np.any(defect):
        mixed = out[defect].astype(np.float32)
        mixed[:, 0] = 0.25 * mixed[:, 0] + 0.75 * 255.0
        mixed[:, 1] = 0.25 * mixed[:, 1]
        mixed[:, 2] = 0.25 * mixed[:, 2]
        out[defect] = np.clip(mixed, 0, 255).astype(np.uint8)
    edge = _boundary(roi_mask)
    out[edge] = np.array([0, 255, 0], dtype=np.uint8)
    return out


def _heatmap(score: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = np.clip(score / 3.0, 0.0, 1.0)
    r = np.clip(2.0 * v - 0.2, 0.0, 1.0)
    g = np.clip(2.0 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.2 - 2.0 * v, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=2)
    rgb[~mask.astype(bool)] = 0.0
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _autoscale_u8(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    vals = data[mask.astype(bool)] if mask is not None and np.any(mask) else data.reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    lo = float(np.percentile(vals, 1.0))
    hi = float(np.percentile(vals, 99.0))
    if hi <= lo + 1e-8:
        hi = float(np.max(vals))
        lo = float(np.min(vals))
    if hi <= lo + 1e-8:
        return np.zeros(data.shape, dtype=np.uint8)
    return np.clip(np.rint((data - lo) * 255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def _stats(arr: np.ndarray | None, mask: np.ndarray) -> dict[str, float] | None:
    if arr is None:
        return None
    vals = arr[mask.astype(bool)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "min": float(np.min(vals)),
        "p50": float(np.percentile(vals, 50)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "max": float(np.max(vals)),
    }


def _full_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    return (0, 0, int(image.shape[1]), int(image.shape[0]))


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask.astype(bool))
    if ys.size == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _save_roi_artifacts(
    store: RunStore,
    run_id: str,
    analysis: Defect2DAnalysis,
    roi_cfg: ObjectRoiConfig,
) -> None:
    bbox_core = _bbox_from_mask(analysis.roi_core_mask)
    store.save_roi(
        run_id,
        analysis.roi_mask,
        analysis.bbox,
        {
            "cfg": asdict(roi_cfg),
            "mode": DEFECT_2D_MODE,
            "bbox_core": bbox_core,
            "bbox_dilated": analysis.bbox,
            "area_ratio_core": float(np.count_nonzero(analysis.roi_core_mask) / analysis.roi_core_mask.size),
            "area_ratio_dilated": float(np.count_nonzero(analysis.roi_mask) / analysis.roi_mask.size),
            "roi_gate_mask": "core_eroded",
            "debug": analysis.metrics.get("roi_debug", {}),
        },
        roi_core=analysis.roi_core_mask,
        roi_dilated=analysis.roi_mask,
        bbox_core=bbox_core,
        bbox_dilated=analysis.bbox,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value
