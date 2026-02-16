"""Auto-normalisation of camera/projector operating point before scanning."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import time

import numpy as np
from PIL import Image

from fringe_app.core.models import ScanParams
from fringe_app.vision.object_roi import ObjectRoiConfig, detect_object_roi


@dataclass(slots=True)
class NormaliseConfig:
    enabled: bool = True
    target_A_mean: float = 120.0
    target_A_tolerance: float = 10.0
    sat_high: int = 250
    max_clip_roi: float = 0.01
    exposure_min_us: int = 500
    gain_min: float = 1.0
    exposure_max_safe_us: int = 4000
    gain_max_safe: float = 2.0
    allow_extend: bool = True
    exposure_max_extended_us: int = 12000
    gain_max_extended: float = 4.0
    extend_trigger_roi_mean_dn: float = 20.0
    extend_requires_clip_below: float = 0.0
    max_iters: int = 10
    allow_pattern_adjust: bool = True
    contrast_min: float = 0.4
    contrast_max: float = 0.8
    brightness_offset_min: float = 0.40
    brightness_offset_max: float = 0.55
    min_intensity: float = 0.10
    warmup_ms: int = 250
    settle_ms: int = 200
    flush_frames: int = 2
    calib_white_dn: int = 230


@dataclass(slots=True)
class NormaliseResult:
    exposure_us: int
    analogue_gain: float
    contrast: float
    brightness_offset: float
    measured_roi_mean: float
    measured_clip_roi: float
    ok: bool
    iters: int
    notes: list[str]
    roi_fallback: bool
    used_extended_envelope: bool
    stage_a_trace: list[dict[str, Any]]
    stage_b_trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 3:
        f = img.astype(np.float32)
        g = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
        return np.clip(np.rint(g), 0, 255).astype(np.uint8)
    raise ValueError("Unsupported image format")


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    h, w = m.shape
    p = np.pad(m, 1, mode="constant", constant_values=False)
    n = [
        p[0:h, 0:w], p[0:h, 1:w + 1], p[0:h, 2:w + 2],
        p[1:h + 1, 0:w], p[1:h + 1, 1:w + 1], p[1:h + 1, 2:w + 2],
        p[2:h + 2, 0:w], p[2:h + 2, 1:w + 1], p[2:h + 2, 2:w + 2],
    ]
    eroded = np.logical_and.reduce(n)
    return m & (~eroded)


def _save_overlay(base_gray: np.ndarray, roi_mask: np.ndarray, out_path: Path) -> None:
    base = np.repeat(base_gray[:, :, None], 3, axis=2).astype(np.uint8)
    edge = _boundary(roi_mask)
    base[edge] = np.array([0, 255, 0], dtype=np.uint8)
    Image.fromarray(base).save(out_path)


def _capture_gray(camera, flush_frames: int) -> np.ndarray:
    for _ in range(max(0, flush_frames)):
        try:
            camera.capture_pair()
        except Exception:
            break
    main, _ = camera.capture_pair()
    return _to_gray_u8(main)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = np.zeros_like(mask, dtype=bool)
    best_n = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            q = [(y, x)]
            seen[y, x] = True
            coords: list[tuple[int, int]] = []
            while q:
                cy, cx = q.pop()
                coords.append((cy, cx))
                if cy > 0 and mask[cy - 1, cx] and not seen[cy - 1, cx]:
                    seen[cy - 1, cx] = True
                    q.append((cy - 1, cx))
                if cy + 1 < h and mask[cy + 1, cx] and not seen[cy + 1, cx]:
                    seen[cy + 1, cx] = True
                    q.append((cy + 1, cx))
                if cx > 0 and mask[cy, cx - 1] and not seen[cy, cx - 1]:
                    seen[cy, cx - 1] = True
                    q.append((cy, cx - 1))
                if cx + 1 < w and mask[cy, cx + 1] and not seen[cy, cx + 1]:
                    seen[cy, cx + 1] = True
                    q.append((cy, cx + 1))
            if len(coords) > best_n:
                best_n = len(coords)
                best[:] = False
                ys = [c[0] for c in coords]
                xs = [c[1] for c in coords]
                best[ys, xs] = True
    return best


def _bright_roi_fallback(gray: np.ndarray) -> np.ndarray:
    p = float(np.percentile(gray, 97.5))
    t = max(5.0, p)
    m = gray >= t
    if np.count_nonzero(m) < max(256, int(gray.size * 0.002)):
        m = gray >= max(3.0, float(np.percentile(gray, 95.0)))
    m = _largest_component(m)
    if np.count_nonzero(m) < max(256, int(gray.size * 0.002)):
        return np.ones_like(gray, dtype=bool)
    return m


def _apply_controls(camera, exposure: int, gain: float) -> None:
    try:
        camera.set_manual_controls(exposure, gain, awb_enable=False)
    except Exception:
        pass


def _should_extend(mean_dn: float, clip: float, cfg: NormaliseConfig) -> bool:
    return (
        cfg.allow_extend
        and mean_dn < cfg.extend_trigger_roi_mean_dn
        and clip <= cfg.extend_requires_clip_below
    )


def _project_gray(display, base_params: ScanParams, dn: int, settle_ms: int, warmup: bool = False) -> None:
    frame = np.full((base_params.resolution[1], base_params.resolution[0]), int(np.clip(dn, 0, 255)), dtype=np.uint8)
    display.show_gray(frame)
    display.pump()
    if warmup:
        time.sleep(settle_ms / 1000.0)
    else:
        time.sleep(settle_ms / 1000.0)


def auto_normalise_capture(
    display,
    camera,
    cfg: NormaliseConfig,
    base_params: ScanParams,
    roi_cfg: ObjectRoiConfig,
    debug_dir: Path | None = None,
) -> NormaliseResult:
    exposure = int(base_params.exposure_us if base_params.exposure_us is not None else cfg.exposure_min_us)
    gain = float(base_params.analogue_gain if base_params.analogue_gain is not None else cfg.gain_min)
    contrast = float(base_params.contrast)
    brightness_offset = float(base_params.brightness_offset)
    exposure = int(_clamp(exposure, cfg.exposure_min_us, cfg.exposure_max_safe_us))
    gain = float(_clamp(gain, cfg.gain_min, cfg.gain_max_safe))
    contrast = float(_clamp(contrast, cfg.contrast_min, cfg.contrast_max))
    brightness_offset = float(_clamp(brightness_offset, cfg.brightness_offset_min, cfg.brightness_offset_max))

    notes: list[str] = []
    stage_a_trace: list[dict[str, Any]] = []
    stage_b_trace: list[dict[str, Any]] = []
    used_extended = False

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Stage A: bright frame for robust ROI acquisition + coarse exposure guard.
    white_dn = int(np.clip(cfg.calib_white_dn, 1, 254))
    _project_gray(display, base_params, white_dn, cfg.warmup_ms, warmup=True)
    _apply_controls(camera, exposure, gain)
    gray_white = _capture_gray(camera, cfg.flush_frames)
    roi_res = detect_object_roi(gray_white, roi_cfg)
    roi_mask = roi_res.roi_mask
    roi_fallback = bool(roi_res.debug.get("roi_fallback", False))
    if roi_fallback:
        roi_mask = _bright_roi_fallback(gray_white)
        if np.all(roi_mask):
            notes.append("ROI fallback during Stage A; using full frame ROI.")
        else:
            notes.append("ROI fallback during Stage A; using bright-component fallback ROI.")
    if debug_dir is not None:
        Image.fromarray(gray_white).save(debug_dir / "calib_white.png")
        Image.fromarray((roi_mask.astype(np.uint8) * 255)).save(debug_dir / "roi_mask.png")
        _save_overlay(gray_white, roi_mask, debug_dir / "roi_overlay_white.png")

    stage_a_iters = max(3, cfg.max_iters // 2)
    for i in range(stage_a_iters):
        _apply_controls(camera, exposure, gain)
        _project_gray(display, base_params, white_dn, cfg.settle_ms)
        gray = _capture_gray(camera, cfg.flush_frames)
        vals = gray[roi_mask]
        mean_dn = float(np.mean(vals)) if vals.size else 0.0
        clip = float(np.mean(vals >= cfg.sat_high)) if vals.size else 1.0
        stage_a_trace.append(
            {
                "iter": i + 1,
                "exposure_us": exposure,
                "analogue_gain": gain,
                "roi_mean_white": mean_dn,
                "roi_clip_white": clip,
            }
        )
        if clip > cfg.max_clip_roi:
            notes.append(f"stageA iter{i+1}: clipping {clip:.4f}, reducing exposure/gain")
            if exposure > cfg.exposure_min_us:
                exposure = int(max(cfg.exposure_min_us, int(exposure * 0.8)))
            elif gain > cfg.gain_min:
                gain = float(max(cfg.gain_min, gain * 0.85))
            continue
        if mean_dn < 30.0:
            safe_ceiling = exposure >= cfg.exposure_max_safe_us and gain >= cfg.gain_max_safe
            if safe_ceiling and _should_extend(mean_dn, clip, cfg):
                used_extended = True
            exp_max = cfg.exposure_max_extended_us if used_extended else cfg.exposure_max_safe_us
            gain_max = cfg.gain_max_extended if used_extended else cfg.gain_max_safe
            if exposure < exp_max:
                exposure = int(min(exp_max, int(exposure * 1.25)))
            elif gain < gain_max:
                gain = float(min(gain_max, gain * 1.2))
            notes.append(f"stageA iter{i+1}: dark {mean_dn:.2f}, increasing controls")
        else:
            break

    # Stage B: fringe-like operating point, target mean and anti-clipping.
    peak = _clamp(brightness_offset + 0.5 * contrast, cfg.min_intensity, 1.0)
    peak_dn = int(round(255.0 * peak))
    measured_mean = 0.0
    measured_clip = 1.0
    ok = False
    for i in range(cfg.max_iters):
        _apply_controls(camera, exposure, gain)
        _project_gray(display, base_params, peak_dn, cfg.settle_ms)
        gray = _capture_gray(camera, cfg.flush_frames)
        vals = gray[roi_mask]
        measured_mean = float(np.mean(vals)) if vals.size else 0.0
        measured_clip = float(np.mean(vals >= cfg.sat_high)) if vals.size else 1.0
        stage_b_trace.append(
            {
                "iter": i + 1,
                "exposure_us": exposure,
                "analogue_gain": gain,
                "contrast": contrast,
                "brightness_offset": brightness_offset,
                "peak_dn": peak_dn,
                "roi_mean_fringe": measured_mean,
                "roi_clip_fringe": measured_clip,
            }
        )

        if debug_dir is not None and i == cfg.max_iters - 1:
            Image.fromarray(gray).save(debug_dir / "calib_fringe.png")
            _save_overlay(gray, roi_mask, debug_dir / "roi_overlay_fringe.png")

        within_target = abs(measured_mean - cfg.target_A_mean) <= cfg.target_A_tolerance
        if measured_clip <= cfg.max_clip_roi and within_target:
            ok = True
            break
        # Extend envelope early in dim scenes when clipping is zero.
        if _should_extend(measured_mean, measured_clip, cfg):
            used_extended = True
        if measured_clip > cfg.max_clip_roi:
            notes.append(f"stageB iter{i+1}: clipping {measured_clip:.4f}, reducing controls")
            if exposure > cfg.exposure_min_us:
                exposure = int(max(cfg.exposure_min_us, int(exposure * 0.85)))
            elif gain > cfg.gain_min:
                gain = float(max(cfg.gain_min, gain * 0.9))
            elif cfg.allow_pattern_adjust:
                contrast = float(max(cfg.contrast_min, contrast - 0.05))
                brightness_offset = float(max(cfg.brightness_offset_min, brightness_offset - 0.02))
            continue
        if measured_mean < (cfg.target_A_mean - cfg.target_A_tolerance):
            exp_max = cfg.exposure_max_extended_us if used_extended else cfg.exposure_max_safe_us
            gain_max = cfg.gain_max_extended if used_extended else cfg.gain_max_safe
            # Multiplicative controller on exposure first.
            ratio = float(np.clip(cfg.target_A_mean / max(measured_mean, 1e-3), 0.7, 1.5))
            proposed = int(max(cfg.exposure_min_us, round(exposure * ratio)))
            if proposed > exposure and exposure < exp_max:
                exposure = int(min(exp_max, proposed))
            elif gain < gain_max:
                gain = float(min(gain_max, gain * 1.15))
            elif cfg.allow_pattern_adjust:
                contrast = float(min(cfg.contrast_max, contrast + 0.05))
                brightness_offset = float(min(cfg.brightness_offset_max, brightness_offset + 0.02))
            notes.append(f"stageB iter{i+1}: dark {measured_mean:.2f}, increasing controls")
        elif measured_mean > (cfg.target_A_mean + cfg.target_A_tolerance):
            notes.append(f"stageB iter{i+1}: bright {measured_mean:.2f}, reducing controls")
            if exposure > cfg.exposure_min_us:
                exposure = int(max(cfg.exposure_min_us, int(exposure * 0.9)))
            elif gain > cfg.gain_min:
                gain = float(max(cfg.gain_min, gain * 0.9))
            elif cfg.allow_pattern_adjust:
                contrast = float(max(cfg.contrast_min, contrast - 0.05))
                brightness_offset = float(max(cfg.brightness_offset_min, brightness_offset - 0.01))
        peak = _clamp(brightness_offset + 0.5 * contrast, cfg.min_intensity, 1.0)
        peak_dn = int(round(255.0 * peak))

    if debug_dir is not None and not (debug_dir / "calib_fringe.png").exists():
        _apply_controls(camera, exposure, gain)
        _project_gray(display, base_params, peak_dn, cfg.settle_ms)
        gray = _capture_gray(camera, cfg.flush_frames)
        Image.fromarray(gray).save(debug_dir / "calib_fringe.png")
        _save_overlay(gray, roi_mask, debug_dir / "roi_overlay_fringe.png")

    return NormaliseResult(
        exposure_us=int(exposure),
        analogue_gain=float(gain),
        contrast=float(contrast),
        brightness_offset=float(brightness_offset),
        measured_roi_mean=float(measured_mean),
        measured_clip_roi=float(measured_clip),
        ok=ok,
        iters=len(stage_a_trace) + len(stage_b_trace),
        notes=notes + ([] if ok else ["target not reached before max_iters"]),
        roi_fallback=roi_fallback,
        used_extended_envelope=used_extended,
        stage_a_trace=stage_a_trace,
        stage_b_trace=stage_b_trace,
    )
