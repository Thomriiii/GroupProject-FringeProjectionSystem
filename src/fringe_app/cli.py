"""CLI commands for scan/phase/score/autotune."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from fringe_app.core.models import ScanParams
from fringe_app.patterns.generator import FringePatternGenerator
from fringe_app.display.pygame_display import PygameProjectorDisplay
from fringe_app.camera.picamera2_impl import Picamera2Camera
from fringe_app.camera.mock import MockCamera
from fringe_app.io.run_store import RunStore
from fringe_app.web.preview import PreviewBroadcaster
from fringe_app.core.controller import ScanController
from fringe_app.core.quality_gate import (
    QualityThresholds,
    QualityReport,
    build_phase_quality_report,
    apply_unwrap_quality_report,
    save_quality_report,
    load_quality_state,
    reset_quality_state,
    update_quality_state,
)
from fringe_app.phase.psp import PhaseShiftProcessor, PhaseThresholds
from fringe_app.phase.masking_post import (
    cleanup_mask,
    build_unwrap_mask,
    build_mask_for_defects,
    build_mask_for_display,
)
from fringe_app.phase.metrics import score_mask
from fringe_app.vision.object_roi import detect_object_roi, ObjectRoiConfig
from fringe_app.overlays.generate import overlay_masks, overlay_clipping
from fringe_app.unwrap.temporal import unwrap_multi_frequency, save_unwrap_outputs
from fringe_app.uv import phase_to_uv, save_uv_outputs


@dataclass(slots=True)
class CaptureScore:
    score: float
    clipped_any_pct_roi: float
    clipped_step_max_roi: float
    roi_b_median: float
    roi_valid_ratio: float
    roi_edge_noise_ratio: float
    run_id: str
    high_freq: float
    thresholds_met: bool


def _load_config() -> dict:
    cfg_path = Path("config/default.yaml")
    if not cfg_path.exists():
        return {}
    import yaml
    return yaml.safe_load(cfg_path.read_text()) or {}


def _camera_from_cfg(cfg: dict):
    cam_cfg = cfg.get("camera", {})
    cam_type = cam_cfg.get("type", "picamera2")
    if cam_type == "mock":
        return MockCamera(data_dir=cam_cfg.get("mock_data", "mock_data"))
    return Picamera2Camera(
        lores_yuv_format=str(cam_cfg.get("lores_yuv_format", "nv12")),
        lores_uv_swap=bool(cam_cfg.get("lores_uv_swap", False)),
    )


def _build_params(args, cfg: dict) -> ScanParams:
    scan_cfg = cfg.get("scan", {})
    pat_cfg = cfg.get("patterns", {})
    width = getattr(args, "width", None)
    height = getattr(args, "height", None)
    width = int(scan_cfg.get("width", 1024) if width is None else width)
    height = int(scan_cfg.get("height", 768) if height is None else height)
    frequencies = getattr(args, "frequencies", None)
    freq_arg = getattr(args, "frequency", None)
    if frequencies:
        frequencies = [float(f) for f in frequencies]
    elif scan_cfg.get("frequencies") and freq_arg is None:
        frequencies = [float(f) for f in scan_cfg.get("frequencies", [])]
    else:
        frequencies = [float(scan_cfg.get("frequency", 8.0) if freq_arg is None else freq_arg)]
    arg_brightness_offset = getattr(args, "brightness_offset", None)
    arg_contrast = getattr(args, "contrast", None)
    arg_min_intensity = getattr(args, "min_intensity", None)
    arg_exposure = getattr(args, "exposure_us", None)
    arg_gain = getattr(args, "gain", None)
    arg_awb_enable = getattr(args, "awb_enable", None)
    arg_awb_mode = getattr(args, "awb_mode", None)
    arg_iso = getattr(args, "iso", None)
    brightness_offset = pat_cfg.get("brightness_offset", scan_cfg.get("brightness_offset", 0.45)) if arg_brightness_offset is None else arg_brightness_offset
    contrast = pat_cfg.get("contrast", scan_cfg.get("contrast", 0.6)) if arg_contrast is None else arg_contrast
    min_intensity = pat_cfg.get("min_intensity", scan_cfg.get("min_intensity", 0.10)) if arg_min_intensity is None else arg_min_intensity
    exposure_us = scan_cfg.get("exposure_us") if arg_exposure is None else arg_exposure
    gain = scan_cfg.get("analogue_gain", scan_cfg.get("gain")) if arg_gain is None else arg_gain
    awb_enable = scan_cfg.get("awb_enable") if arg_awb_enable is None else arg_awb_enable
    awb_mode = scan_cfg.get("awb_mode") if arg_awb_mode is None else arg_awb_mode
    iso = scan_cfg.get("iso") if arg_iso is None else arg_iso

    mode_arg = getattr(args, "mode", None)
    mode_cfg = str(scan_cfg.get("mode", "fast"))
    mode = str(mode_arg) if mode_arg is not None else mode_cfg
    n_arg = getattr(args, "n", None)
    if n_arg is None:
        if mode == "stable":
            n_steps = int(scan_cfg.get("n_steps_stable", scan_cfg.get("n_steps", 8)))
        else:
            n_steps = int(scan_cfg.get("n_steps_fast", scan_cfg.get("n_steps", 4)))
    else:
        n_steps = int(n_arg)

    return ScanParams(
        n_steps=n_steps,
        frequency=float(frequencies[0]),
        frequencies=frequencies,
        orientation=str(getattr(args, "orientation", scan_cfg.get("orientation", "vertical"))),
        brightness=float(getattr(args, "brightness", scan_cfg.get("brightness", 1.0))),
        resolution=(width, height),
        settle_ms=int(getattr(args, "settle_ms", scan_cfg.get("settle_ms", 50))),
        save_patterns=bool(getattr(args, "save_patterns", scan_cfg.get("save_patterns", False))),
        preview_fps=float(getattr(args, "preview_fps", scan_cfg.get("preview_fps", 10.0))),
        phase_convention=str(scan_cfg.get("phase_convention", "atan2(-S,C)")),
        exposure_us=None if exposure_us is None else int(exposure_us),
        analogue_gain=None if gain is None else float(gain),
        awb_mode=awb_mode,
        awb_enable=None if awb_enable is None else bool(awb_enable),
        ae_enable=getattr(args, "ae_enable", scan_cfg.get("ae_enable")),
        iso=None if iso is None else int(iso),
        brightness_offset=float(brightness_offset),
        contrast=float(contrast),
        min_intensity=float(min_intensity),
        frequency_semantics=str(scan_cfg.get("frequency_semantics", "cycles_across_dimension")),
        phase_origin_rad=float(scan_cfg.get("phase_origin_rad", 0.0)),
        auto_normalise=bool(getattr(args, "auto_normalise", scan_cfg.get("auto_normalise", True))),
        expert_mode=bool(getattr(args, "expert", False)),
    )


def _phase_thresholds(cfg: dict, overrides: Dict[str, Any] | None = None) -> PhaseThresholds:
    phase_cfg = cfg.get("phase", {})
    perc = phase_cfg.get("debug_percentiles", [1, 99])
    base = PhaseThresholds(
        sat_low=float(phase_cfg.get("sat_low", 0)),
        sat_high=float(phase_cfg.get("sat_high", 250)),
        B_thresh=float(phase_cfg.get("B_thresh", 7)),
        A_min=float(phase_cfg.get("A_min", 15)),
        debug_percentiles=(float(perc[0]), float(perc[1])),
    )
    if overrides:
        for k, v in overrides.items():
            setattr(base, k, v)
    return base


def _roi_config(cfg: dict) -> ObjectRoiConfig:
    roi_cfg = cfg.get("roi", {})
    roi_post = roi_cfg.get("post", {}) or {}
    return ObjectRoiConfig(
        downscale_max_w=int(roi_cfg.get("downscale_max_w", 640)),
        blur_ksize=int(roi_cfg.get("blur_ksize", 7)),
        black_bg_percentile=float(roi_cfg.get("black_bg_percentile", 70.0)),
        threshold_offset=float(roi_cfg.get("threshold_offset", 10.0)),
        min_area_ratio=float(roi_cfg.get("min_area_ratio", 0.01)),
        max_area_ratio=float(roi_cfg.get("max_area_ratio", 0.95)),
        close_iters=int(roi_cfg.get("close_iters", 2)),
        open_iters=int(roi_cfg.get("open_iters", 1)),
        fill_holes=bool(roi_cfg.get("fill_holes", True)),
        ref_method=str(roi_cfg.get("ref_method", "median_over_frames")),
        post_enabled=bool(roi_post.get("enabled", True)),
        post_keep_largest_component=bool(roi_post.get("keep_largest_component", True)),
        post_fill_small_holes=bool(roi_post.get("fill_small_holes", True)),
        post_max_hole_area=int(roi_post.get("max_hole_area", 2000)),
        post_dilate_radius_px=int(roi_post.get("dilate_radius_px", 10)),
    )


def _postmask_config(cfg: dict) -> dict:
    return cfg.get("phase", {}).get("postmask_cleanup", {}) or {}


def _unwrap_mask_post_config(cfg: dict) -> dict:
    return (cfg.get("unwrap", {}) or {}).get("mask_post", {}) or {}


def _quality_thresholds(cfg: dict) -> QualityThresholds:
    q = cfg.get("quality_gate", {})
    state = load_quality_state(cfg)
    return QualityThresholds(
        max_clipped_any_pct_roi=float(q.get("max_clipped_any_pct_roi", 0.01)),
        max_clipped_step_pct_roi=float(q.get("max_clipped_step_pct_roi", 0.01)),
        max_residual_p95=float(state.get("current_residual_threshold", q.get("max_residual_p95", 0.8))),
        min_roi_valid_ratio_high_freq=float(q.get("min_roi_valid_ratio_high_freq", 0.85)),
        min_unwrap_pixels=int(q.get("min_unwrap_pixels", 3000)),
    )


def cmd_scan(args) -> int:
    cfg = _load_config()
    params = _build_params(args, cfg)
    display = PygameProjectorDisplay()
    camera = _camera_from_cfg(cfg)
    generator = FringePatternGenerator()
    store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
    preview = PreviewBroadcaster()

    controller = ScanController(
        display=display,
        camera=camera,
        generator=generator,
        store=store,
        preview=preview,
        config=cfg,
    )

    run_id = controller.start_scan(params)
    print(f"run_id={run_id}")

    while True:
        status = controller.get_status()
        if status["state"] in ("IDLE", "ERROR"):
            print(json.dumps(status))
            break
        time.sleep(0.2)

    return 0


def cmd_phase(args) -> int:
    cfg = _load_config()
    store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
    params = _params_from_meta(args.run, store)
    thresholds = _phase_thresholds(cfg)
    roi_cfg = _roi_config(cfg)
    ref = store.load_reference_image(args.run, ref_method=roi_cfg.ref_method)
    roi_res = detect_object_roi(ref, roi_cfg)
    store.save_roi(
        args.run,
        roi_res.roi_mask,
        roi_res.bbox,
        {
            "cfg": asdict(roi_cfg),
            "debug": {
                **roi_res.debug,
                "ref_method": roi_cfg.ref_method,
                "ref_stats": {
                    "min": int(np.min(ref)),
                    "mean": float(np.mean(ref)),
                    "max": int(np.max(ref)),
                },
            },
        },
        roi_raw=roi_res.raw_mask,
        roi_post=roi_res.post_mask,
    )
    roi_mask_for_score = None if roi_res.debug.get("roi_fallback") else roi_res.roi_mask

    freqs = params.get_frequencies()
    processor = PhaseShiftProcessor()
    post_cfg = _postmask_config(cfg)
    mask_cfg = (cfg.get("phase", {}) or {}).get("masks", {}) or {}
    defects_cfg = (mask_cfg.get("defects", {}) or {})
    display_cfg = (mask_cfg.get("display", {}) or {})
    unwrap_post_cfg = _unwrap_mask_post_config(cfg)
    post_enabled = bool(post_cfg.get("enabled", True))
    min_component_area = int(post_cfg.get("min_component_area", 200))
    fill_small_holes = bool(post_cfg.get("fill_small_holes", True))
    max_hole_area = int(post_cfg.get("max_hole_area", 200))
    for freq in freqs:
        images = store.iter_captures(args.run, freq=freq) if len(freqs) > 1 else store.iter_captures(args.run)
        result = processor.compute_phase(images, params, thresholds, roi_mask=roi_mask_for_score)
        raw_mask = result.mask_raw.copy()
        if post_enabled:
            cleaned = cleanup_mask(
                raw_mask,
                roi_mask_for_score,
                min_component_area=min_component_area,
                fill_small_holes=fill_small_holes,
                max_hole_area=max_hole_area,
            )
            # Hard guard: clipped pixels remain invalid after cleanup.
            cleaned &= (~result.clipped_any_map)
            result.mask_clean = cleaned
            result.mask = cleaned
            result.debug["postmask_cleanup"] = {
                "enabled": True,
                "min_component_area": min_component_area,
                "fill_small_holes": fill_small_holes,
                "max_hole_area": max_hole_area,
                "raw_valid_ratio": float(np.count_nonzero(raw_mask) / raw_mask.size),
                "clean_valid_ratio": float(np.count_nonzero(cleaned) / cleaned.size),
            }
        else:
            result.mask_clean = raw_mask
            result.mask = raw_mask
            result.debug["postmask_cleanup"] = {"enabled": False}

        # Dedicated conservative mask for temporal unwrapping.
        if bool(unwrap_post_cfg.get("enabled", True)):
            unwrap_mask = build_unwrap_mask(
                result.mask_clean,
                roi_mask_for_score,
                result.clipped_any_map,
                unwrap_post_cfg,
            )
        else:
            unwrap_mask = result.mask_clean & (~result.clipped_any_map)
            if roi_mask_for_score is not None:
                unwrap_mask &= roi_mask_for_score
        result.mask_for_unwrap = unwrap_mask
        # Inclusive/stable mask for defect analysis and display mask for UI.
        defects_mask = build_mask_for_defects(
            raw_mask,
            roi_mask_for_score,
            result.clipped_any_map,
            defects_cfg,
        )
        display_mask = build_mask_for_display(
            result.B,
            raw_mask,
            roi_mask_for_score,
            result.clipped_any_map,
            display_cfg,
        )
        result.mask_for_defects = defects_mask
        result.mask_for_display = display_mask
        result.debug["mask_for_unwrap_policy"] = {
            "enabled": bool(unwrap_post_cfg.get("enabled", True)),
            "keep_largest_component": bool(unwrap_post_cfg.get("keep_largest_component", True)),
            "closing_radius_px": int(unwrap_post_cfg.get("closing_radius_px", 0)),
            "erosion_radius_px": int(unwrap_post_cfg.get("erosion_radius_px", 0)),
            "min_area_px": int(unwrap_post_cfg.get("min_area_px", 5000)),
            "unwrap_valid_ratio": float(np.count_nonzero(unwrap_mask) / unwrap_mask.size),
        }
        result.debug["mask_for_defects_policy"] = {
            "enabled": bool(defects_cfg.get("enabled", True)),
            "keep_largest_component": bool(defects_cfg.get("keep_largest_component", True)),
            "dilate_radius_px": int(defects_cfg.get("dilate_radius_px", 3)),
            "erosion_radius_px": int(defects_cfg.get("erosion_radius_px", 0)),
            "fill_small_holes": bool(defects_cfg.get("fill_small_holes", False)),
            "defects_valid_ratio": float(np.count_nonzero(defects_mask) / defects_mask.size),
        }
        result.debug["mask_for_display_policy"] = {
            "enabled": bool(display_cfg.get("enabled", True)),
            "use_b_smooth": bool(display_cfg.get("use_b_smooth", True)),
            "b_smooth_ksize": int(display_cfg.get("b_smooth_ksize", 5)),
            "b_thresh_display": float(display_cfg.get("b_thresh_display", 5)),
            "display_valid_ratio": float(np.count_nonzero(display_mask) / display_mask.size),
        }

        # Phase gate metrics should reflect stable/inclusive defect mask, not conservative unwrap mask.
        score = score_mask(result.mask_for_defects, result.B, roi_mask=roi_mask_for_score)
        result.debug.update({
            "roi_valid_ratio": score.roi_valid_ratio,
            "roi_largest_component_ratio": score.roi_largest_component_ratio,
            "roi_edge_noise_ratio": score.roi_edge_noise_ratio,
            "roi_b_median": score.roi_b_median,
            "roi_score": score.roi_score,
            "roi_fallback": roi_res.debug.get("roi_fallback", False),
        })
        store.save_phase_outputs(args.run, result, freq=freq if len(freqs) > 1 else None)
        phase_dir = Path(store.root) / args.run / "phase"
        if len(freqs) > 1:
            phase_dir = phase_dir / store._freq_tag(freq)
        try:
            from fringe_app.phase import visualize as _viz
            valid_in_roi = result.mask_for_defects & roi_res.roi_mask
            _viz.save_mask_png(valid_in_roi, str(phase_dir / "valid_in_roi.png"))
        except Exception:
            pass

    overlays_dir = Path(store.root) / args.run / "overlays"
    overlays_dir.mkdir(exist_ok=True)
    try:
        empty = np.zeros_like(roi_res.roi_mask, dtype=bool)
        overlay_masks(ref, roi_res.roi_mask, empty, overlays_dir / "roi_overlay.png")
        clip_dir = overlays_dir / "clipping"
        clip_dir.mkdir(exist_ok=True)
        for freq in freqs:
            phase_dir = Path(store.root) / args.run / "phase"
            if len(freqs) > 1:
                phase_dir = phase_dir / store._freq_tag(freq)
            clipped_npy = phase_dir / "clipped_any.npy"
            if clipped_npy.exists():
                suffix = "" if len(freqs) == 1 else f"_{store._freq_tag(freq)}"
                overlay_clipping(ref, roi_res.roi_mask, np.load(clipped_npy), clip_dir / f"clipping_overlay{suffix}.png")
    except Exception:
        pass
    # Save side-by-side mask comparison for highest frequency.
    try:
        high = max(freqs)
        phase_dir = Path(store.root) / args.run / "phase"
        if len(freqs) > 1:
            phase_dir = phase_dir / store._freq_tag(high)
        roi_mask = roi_res.roi_mask.astype(np.uint8) * 255
        raw = np.load(phase_dir / "mask_raw.npy").astype(np.uint8) * 255
        defects = np.load(phase_dir / "mask_for_defects.npy").astype(np.uint8) * 255
        unwrap = np.load(phase_dir / "mask_for_unwrap.npy").astype(np.uint8) * 255
        spacer = np.full((roi_mask.shape[0], 8), 20, dtype=np.uint8)
        montage = np.concatenate([roi_mask, spacer, raw, spacer, defects, spacer, unwrap], axis=1)
        from PIL import Image
        Image.fromarray(montage).save(overlays_dir / "mask_compare.png")
    except Exception:
        pass
    print(json.dumps(result.debug, indent=2))
    return 0


def cmd_score(args) -> int:
    cfg = _load_config()
    store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
    phase_dir = Path(store.root) / args.run / "phase"
    phase_dir = _select_phase_dir(phase_dir)
    if not phase_dir.exists():
        raise SystemExit("phase outputs not found; run phase first")
    mask_path = phase_dir / "mask_clean.npy"
    if not mask_path.exists():
        mask_path = phase_dir / "mask.npy"
    mask = np.load(mask_path) if mask_path.exists() else None
    if mask is None:
        mask_png = phase_dir / "mask.png"
        if not mask_png.exists():
            raise SystemExit("mask not found")
        from PIL import Image
        mask = np.array(Image.open(mask_png)) > 0
    run_dir = Path(store.root) / args.run
    roi_mask = None
    roi_path = run_dir / "roi" / "roi_mask.png"
    if roi_path.exists():
        from PIL import Image
        roi_mask = np.array(Image.open(roi_path)) > 0
        roi_meta_path = run_dir / "roi" / "roi_meta.json"
        if roi_meta_path.exists():
            roi_meta = json.loads(roi_meta_path.read_text())
            if roi_meta.get("debug", {}).get("roi_fallback"):
                roi_mask = None
    B = np.load(phase_dir / "B.npy")
    score = score_mask(mask, B, roi_mask=roi_mask)
    print(json.dumps(asdict(score), indent=2))
    _print_score(score)
    meta_path = phase_dir / "phase_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(
            "clipping: "
            f"global_any={float(meta.get('clipped_any_pct', 0.0)):.4f} "
            f"roi_any={float(meta.get('clipped_any_pct_roi', 0.0)):.4f} "
            f"roi_step_max={max(meta.get('clipped_per_step_pct_roi', [0.0])):.4f}"
        )
    # Unwrap coverage (if available)
    unwrap_dir = Path(store.root) / args.run / "unwrap"
    if (unwrap_dir / "mask_unwrap.npy").exists():
        unwrap_mask = np.load(unwrap_dir / "mask_unwrap.npy")
        if roi_mask is not None:
            roi_unwrap = np.count_nonzero(unwrap_mask & roi_mask) / max(int(roi_mask.sum()), 1)
        else:
            roi_unwrap = float(np.count_nonzero(unwrap_mask) / unwrap_mask.size)
        print(f"roi_unwrap_valid_ratio={roi_unwrap:.3f}")
    return 0


def cmd_overlay(args) -> int:
    cfg = _load_config()
    store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
    run_dir = Path(store.root) / args.run
    roi_cfg = _roi_config(cfg)
    ref = store.load_reference_image(args.run, ref_method=roi_cfg.ref_method)
    roi_path = run_dir / "roi" / "roi_mask.png"
    phase_root = run_dir / "phase"
    if not roi_path.exists() or not phase_root.exists():
        raise SystemExit("roi or phase mask missing; run phase first")
    from PIL import Image
    roi_mask = np.array(Image.open(roi_path)) > 0
    overlays_dir = run_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)
    empty = np.zeros_like(roi_mask, dtype=bool)
    overlay_masks(ref, roi_mask, empty, overlays_dir / "roi_overlay.png")
    phase_dirs = []
    if (phase_root / "mask.npy").exists():
        phase_dirs = [phase_root]
    else:
        phase_dirs = sorted([p for p in phase_root.iterdir() if p.is_dir() and p.name.startswith("f_")], key=lambda p: p.name)
    if not phase_dirs:
        raise SystemExit("phase masks not found")
    for phase_dir in phase_dirs:
        mask_path = phase_dir / "mask_for_defects.npy"
        if not mask_path.exists():
            mask_path = phase_dir / "mask.npy"
        mask = np.load(mask_path)
        valid_in_roi = mask & roi_mask
        suffix = "" if phase_dir == phase_root else f"_{phase_dir.name}"
        overlay_masks(ref, empty, mask, overlays_dir / f"mask_overlay{suffix}.png")
        overlay_masks(ref, roi_mask, valid_in_roi, overlays_dir / f"valid_in_roi_overlay{suffix}.png")
        clipped_map = phase_dir / "clipped_any.npy"
        if clipped_map.exists():
            clipping_dir = overlays_dir / "clipping"
            clipping_dir.mkdir(exist_ok=True)
            overlay_clipping(ref, roi_mask, np.load(clipped_map), clipping_dir / f"clipping_overlay{suffix}.png")
    print(f"overlays saved to {overlays_dir}")
    return 0


def cmd_unwrap(args) -> int:
    cfg = _load_config()
    store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
    params = _params_from_meta(args.run, store)
    freqs = params.get_frequencies()
    if len(freqs) < 2:
        raise SystemExit("Need at least two frequencies for unwrap")
    phases = []
    masks = []
    for freq in freqs:
        phase_dir = Path(store.root) / args.run / "phase" / store._freq_tag(freq)
        phases.append(np.load(phase_dir / "phi_wrapped.npy"))
        mask_path = phase_dir / "mask_for_unwrap.npy"
        if not mask_path.exists():
            mask_path = phase_dir / "mask_clean.npy"
        if not mask_path.exists():
            mask_path = phase_dir / "mask_raw.npy"
        if not mask_path.exists():
            mask_path = phase_dir / "mask.npy"
        masks.append(np.load(mask_path))
    roi_mask = None
    roi_path = Path(store.root) / args.run / "roi" / "roi_mask.png"
    if roi_path.exists():
        from PIL import Image
        roi_mask = np.array(Image.open(roi_path)) > 0
    use_roi = args.use_roi != "off"
    if args.use_roi == "auto" and roi_mask is None:
        use_roi = False
    phi_abs, mask_unwrap, meta, residual = unwrap_multi_frequency(phases, masks, freqs, roi_mask=roi_mask, use_roi=use_roi)
    save_unwrap_outputs(Path(store.root) / args.run, phi_abs, mask_unwrap, meta, f_max=max(freqs), residual=residual)
    print(json.dumps(meta, indent=2))
    return 0


def _load_phase_meta_high_freq(run_id: str, cfg: dict) -> tuple[dict, float]:
    store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
    params = _params_from_meta(run_id, store)
    freqs = params.get_frequencies()
    high = max(freqs)
    phase_dir = Path(store.root) / run_id / "phase"
    if len(freqs) > 1:
        phase_dir = phase_dir / store._freq_tag(high)
    meta_path = phase_dir / "phase_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing phase_meta.json at {meta_path}")
    return json.loads(meta_path.read_text()), high


def _load_unwrap_meta(run_id: str, cfg: dict) -> dict:
    store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
    meta_path = Path(store.root) / run_id / "unwrap" / "unwrap_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing unwrap_meta.json at {meta_path}")
    return json.loads(meta_path.read_text())


def _check_step_sanity_reports(run_id: str, cfg: dict) -> tuple[bool, list[str]]:
    store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
    params = _params_from_meta(run_id, store)
    reasons: list[str] = []
    for freq in params.get_frequencies():
        p = Path(store.root) / run_id / "captures" / store._freq_tag(freq) / "step_sanity.json"
        if not p.exists():
            reasons.append(f"missing step_sanity for f={freq}")
            continue
        d = json.loads(p.read_text())
        if not d.get("ok", False):
            first = (d.get("reasons") or ["unknown reason"])[0]
            reasons.append(f"f={freq} sanity failed: {first}")
    return (len(reasons) == 0), reasons


def _enforce_safe_envelope(params: ScanParams, force: bool, expert: bool) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    if expert:
        return True, warnings
    checks = [
        ("exposure_us", params.exposure_us, 500, 4000),
        ("analogue_gain", params.analogue_gain, 1.0, 2.0),
        ("contrast", params.contrast, 0.4, 0.8),
        ("brightness_offset", params.brightness_offset, 0.40, 0.55),
    ]
    ok = True
    for name, value, lo, hi in checks:
        if value is None:
            continue
        v = float(value)
        if v < lo or v > hi:
            warnings.append(f"{name}={v} out of safe envelope [{lo}, {hi}]")
            ok = False
    if ok:
        return True, warnings
    if not force:
        return False, warnings
    # --force in non-expert mode clamps unsafe values.
    if params.exposure_us is not None:
        params.exposure_us = int(max(500, min(4000, int(params.exposure_us))))
    if params.analogue_gain is not None:
        params.analogue_gain = float(max(1.0, min(2.0, float(params.analogue_gain))))
    params.contrast = float(max(0.4, min(0.8, float(params.contrast))))
    params.brightness_offset = float(max(0.40, min(0.55, float(params.brightness_offset))))
    return True, warnings


def cmd_pipeline_run_safe(args) -> int:
    cfg = _load_config()
    params = _build_params(args, cfg)
    params.auto_normalise = bool(getattr(args, "auto_normalise", True))
    params.expert_mode = bool(getattr(args, "expert", False))
    params.quality_retry_count = 0
    ok_env, env_warnings = _enforce_safe_envelope(params, force=bool(args.force), expert=params.expert_mode)
    if env_warnings:
        print("Warning: value out of safe envelope; use --expert to override.")
        for w in env_warnings:
            print(f"- {w}")
    if not ok_env:
        return 6

    display = PygameProjectorDisplay()
    camera = _camera_from_cfg(cfg)
    generator = FringePatternGenerator()
    store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
    preview = PreviewBroadcaster()
    controller = ScanController(display=display, camera=camera, generator=generator, store=store, preview=preview, config=cfg)

    def _run_scan_once(local_params: ScanParams) -> tuple[str, dict]:
        run_id_local = controller.start_scan(local_params)
        print(f"run_id={run_id_local}")
        while True:
            status_local = controller.get_status()
            if status_local["state"] in ("IDLE", "ERROR"):
                break
            time.sleep(0.2)
        return run_id_local, status_local

    def _write_retry_info(base_run_id: str, new_run_id: str, reason: str, adjustments: dict[str, float]) -> None:
        payload = {
            "base_run_id": base_run_id,
            "retry_run_id": new_run_id,
            "reason": reason,
            "adjustments": adjustments,
        }
        (Path(store.root) / base_run_id / "quality_retry.json").write_text(json.dumps(payload, indent=2))
        (Path(store.root) / new_run_id / "quality_retry.json").write_text(json.dumps(payload, indent=2))

    run_id, status = _run_scan_once(params)
    if status["state"] == "ERROR":
        print(json.dumps(status, indent=2))
        return 1

    sanity_ok, sanity_reasons = _check_step_sanity_reports(run_id, cfg)
    if not sanity_ok:
        print("CAPTURE INVALID: step stack sanity failed.")
        for r in sanity_reasons:
            print(f"- {r}")
        if not args.force:
            print("Rescan with increased settle/flush.")
            return 4

    phase_rc = cmd_phase(argparse.Namespace(run=run_id))
    if phase_rc != 0:
        return phase_rc

    state_before = load_quality_state(cfg)
    th = _quality_thresholds(cfg)
    print(f"residual_gate_threshold={th.max_residual_p95:.3f} (consecutive_passes={state_before.get('consecutive_passes', 0)})")
    phase_meta, high_freq = _load_phase_meta_high_freq(run_id, cfg)
    report = build_phase_quality_report(phase_meta, th)
    run_dir = Path(store.root) / run_id
    save_quality_report(run_dir, report)
    print(
        f"phase_gate high_freq={high_freq}: clip_any_roi={report.metrics.get('clipped_any_pct_roi', 0):.4f} "
        f"clip_step_max_roi={report.metrics.get('clipped_step_max_pct_roi', 0):.4f} "
        f"roi_valid={report.metrics.get('roi_valid_ratio_high_freq', 0):.3f} ok={report.ok}"
    )
    if not report.ok:
        retry_enabled = not bool(getattr(args, "no_retry", False))
        if retry_enabled and "CLIPPING" in report.reasons and params.quality_retry_count < 1:
            params.quality_retry_count += 1
            params.exposure_us = int(max(500, int((params.exposure_us or 2000) * 0.8)))
            params.contrast = float(max(0.4, params.contrast - 0.1))
            print(
                f"Retrying once due to clipping: exposure_us={params.exposure_us}, contrast={params.contrast:.2f}"
            )
            base_run = run_id
            run_id, status = _run_scan_once(params)
            _write_retry_info(
                base_run,
                run_id,
                "CLIPPING",
                {"exposure_us": float(params.exposure_us or 0), "contrast": float(params.contrast)},
            )
            if status["state"] == "ERROR":
                print(json.dumps(status, indent=2))
                return 1
            sanity_ok, sanity_reasons = _check_step_sanity_reports(run_id, cfg)
            if not sanity_ok:
                print("CAPTURE INVALID after retry.")
                for r in sanity_reasons:
                    print(f"- {r}")
                if not args.force:
                    return 4
            phase_rc = cmd_phase(argparse.Namespace(run=run_id))
            if phase_rc != 0:
                return phase_rc
            phase_meta, high_freq = _load_phase_meta_high_freq(run_id, cfg)
            report = build_phase_quality_report(phase_meta, th)
            run_dir = Path(store.root) / run_id
            save_quality_report(run_dir, report)
        elif retry_enabled and "LOW_ROI_VALID" in report.reasons and params.quality_retry_count < 1:
            clip_any = float(report.metrics.get("clipped_any_pct_roi", 1.0))
            if clip_any <= th.max_clipped_any_pct_roi:
                params.quality_retry_count += 1
                adjustments: dict[str, float] = {}
                exp_max = int(cfg.get("normalise", {}).get("exposure_max_extended_us", 12000))
                current_exp = int(params.exposure_us or 2000)
                bumped_exp = int(min(exp_max, max(current_exp + 1, round(current_exp * 1.25))))
                if bumped_exp > current_exp:
                    params.exposure_us = bumped_exp
                    adjustments["exposure_us"] = float(params.exposure_us)
                else:
                    # Exposure already at ceiling: keep conservative behavior and skip pattern changes.
                    adjustments["exposure_us"] = float(current_exp)
                print(f"Retrying once due to LOW_ROI_VALID with low clipping: {adjustments}")
                base_run = run_id
                run_id, status = _run_scan_once(params)
                _write_retry_info(base_run, run_id, "LOW_ROI_VALID", adjustments)
                if status["state"] == "ERROR":
                    print(json.dumps(status, indent=2))
                    return 1
                sanity_ok, sanity_reasons = _check_step_sanity_reports(run_id, cfg)
                if not sanity_ok:
                    print("CAPTURE INVALID after retry.")
                    for r in sanity_reasons:
                        print(f"- {r}")
                    if not args.force:
                        return 4
                phase_rc = cmd_phase(argparse.Namespace(run=run_id))
                if phase_rc != 0:
                    return phase_rc
                phase_meta, high_freq = _load_phase_meta_high_freq(run_id, cfg)
                report = build_phase_quality_report(phase_meta, th)
                run_dir = Path(store.root) / run_id
                save_quality_report(run_dir, report)
        if not report.ok:
            print(f"FAIL: PHASE {';'.join(report.reasons)}")
            normalise_path = run_dir / "normalise.json"
            if normalise_path.exists():
                nd = json.loads(normalise_path.read_text())
                print(
                    "normalise: "
                    f"exp={nd.get('exposure_us')} gain={nd.get('analogue_gain')} "
                    f"roi_mean={nd.get('measured_roi_mean')} clip_roi={nd.get('measured_clip_roi')}"
                )
            if args.print_hints:
                for h in report.hints:
                    print(f"hint: {h}")
            if not args.force:
                update_quality_state(cfg, passed=False)
                return 2

    unwrap_rc = cmd_unwrap(argparse.Namespace(run=run_id, use_roi="auto"))
    if unwrap_rc != 0:
        return unwrap_rc
    unwrap_meta = _load_unwrap_meta(run_id, cfg)
    report = apply_unwrap_quality_report(report, unwrap_meta, th)
    # Keep unwrap metadata in sync with the effective threshold used by the gate.
    unwrap_meta["effective_residual_p95_threshold"] = float(th.max_residual_p95)
    unwrap_meta_path = Path(store.root) / run_id / "unwrap" / "unwrap_meta.json"
    unwrap_meta_path.write_text(json.dumps(unwrap_meta, indent=2))
    save_quality_report(run_dir, report)
    print(
        f"unwrap_gate residual_p95={report.metrics.get('residual_p95', 0):.4f} ok={report.ok}"
    )
    if not report.ok and "RESIDUAL" in report.reasons:
        print("FAIL: RESIDUAL")
        if args.print_hints:
            for h in report.hints:
                print(f"hint: {h}")
        if not args.force:
            update_quality_state(cfg, passed=False)
            return 3

    score_rc = cmd_score(argparse.Namespace(run=run_id))
    print(f"quality_report={run_dir / 'quality_report.json'}")
    update_quality_state(cfg, passed=bool(report.ok))
    return score_rc


def _latest_run_after(root: Path, t0: float) -> str | None:
    candidates: list[tuple[float, str]] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        try:
            if p.stat().st_mtime >= t0 and (p / "meta.json").exists():
                candidates.append((p.stat().st_mtime, p.name))
        except FileNotFoundError:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _copy_orientation_outputs(src_run: Path, dst_orient: Path) -> None:
    dst_orient.mkdir(parents=True, exist_ok=True)
    for name in ("captures", "phase", "unwrap"):
        src = src_run / name
        if src.exists():
            shutil.copytree(src, dst_orient / name, dirs_exist_ok=True)


def _build_uv_from_orientation_runs(
    merged_dir: Path,
    vertical_run: Path,
    horizontal_run: Path,
    params: ScanParams,
    cfg: dict,
) -> dict[str, Any]:
    freqs = params.get_frequencies()
    f_high = float(max(freqs))
    tag = RunStore()._freq_tag(f_high)

    v_phase_dir = vertical_run / "phase" / tag
    h_phase_dir = horizontal_run / "phase" / tag
    v_unwrap_dir = vertical_run / "unwrap"
    h_unwrap_dir = horizontal_run / "unwrap"

    phi_v = np.load(v_unwrap_dir / "phi_abs.npy")
    phi_h = np.load(h_unwrap_dir / "phi_abs.npy")

    def _load_mask(phase_dir: Path) -> np.ndarray:
        for name in ("mask_for_unwrap.npy", "mask_clean.npy", "mask_raw.npy", "mask.npy"):
            p = phase_dir / name
            if p.exists():
                return np.load(p).astype(bool)
        raise FileNotFoundError(f"No mask found in {phase_dir}")

    mask_v = _load_mask(v_phase_dir)
    mask_h = _load_mask(h_phase_dir)

    roi_path = merged_dir / "roi" / "roi_mask.png"
    roi_mask = None
    if roi_path.exists():
        from PIL import Image
        roi_mask = (np.array(Image.open(roi_path)) > 0)

    proj_w, proj_h = params.resolution
    v_meta = json.loads((vertical_run / "meta.json").read_text())
    h_meta = json.loads((horizontal_run / "meta.json").read_text())
    v_params = v_meta.get("params", {})
    h_params = h_meta.get("params", {})
    freq_semantics = str(v_params.get("frequency_semantics", params.frequency_semantics))
    uv_res = phase_to_uv(
        phi_abs_vertical=phi_v,
        phi_abs_horizontal=phi_h,
        freq_u=f_high,
        freq_v=f_high,
        proj_width=int(proj_w),
        proj_height=int(proj_h),
        mask_u=mask_v,
        mask_v=mask_h,
        roi_mask=roi_mask,
        frequency_semantics=freq_semantics,
        phase_origin_u_rad=float(v_params.get("phase_origin_rad", 0.0)),
        phase_origin_v_rad=float(h_params.get("phase_origin_rad", 0.0)),
        gate_cfg=(cfg.get("uv_gate", {}) or {}),
    )

    # Use first vertical frame as overlay background.
    base_frame = None
    cap_dir = vertical_run / "captures" / tag
    frames = sorted(cap_dir.glob("step_*.png"))
    if frames:
        from PIL import Image
        base_frame = np.array(Image.open(frames[0]))

    uv_dir = merged_dir / "projector_uv"
    save_uv_outputs(uv_dir, uv_res, base_frame=base_frame)
    uv_meta_path = uv_dir / "uv_meta.json"
    uv_meta = json.loads(uv_meta_path.read_text())
    uv_meta["freq_used_high"] = f_high
    uv_meta["frequency_semantics"] = freq_semantics
    uv_meta["mask_policy"] = "mask_u_for_unwrap & mask_v_for_unwrap & roi"
    uv_meta_path.write_text(json.dumps(uv_meta, indent=2))
    return uv_meta


def cmd_pipeline_run_uv(args) -> int:
    cfg = _load_config()
    root = Path(cfg.get("storage", {}).get("run_root", "data/runs"))
    root.mkdir(parents=True, exist_ok=True)

    exp_max = int(cfg.get("normalise", {}).get("exposure_max_extended_us", 12000))

    def _run_orientation_with_low_roi_nudge(orientation: str, ns: argparse.Namespace) -> tuple[int, str | None]:
        attempt_args = argparse.Namespace(**vars(ns))
        attempt_args.orientation = orientation
        t_start = time.time()
        rc = cmd_pipeline_run_safe(attempt_args)
        run_id = _latest_run_after(root, t_start)
        if rc == 0:
            return rc, run_id
        if run_id is None:
            return rc, None
        q_path = root / run_id / "quality_report.json"
        if not q_path.exists():
            return rc, run_id
        q = json.loads(q_path.read_text())
        reasons = q.get("reasons", [])
        clip_any = float(q.get("metrics", {}).get("clipped_any_pct_roi", 1.0))
        if "LOW_ROI_VALID" in reasons and clip_any <= 0.0:
            bumped = argparse.Namespace(**vars(attempt_args))
            current_exp = int(getattr(bumped, "exposure_us", None) or 2000)
            bumped_exp = int(min(exp_max, max(current_exp + 1, round(current_exp * 1.25))))
            bumped.exposure_us = bumped_exp
            print(
                f"{orientation}: retrying once after LOW_ROI_VALID with exposure bump "
                f"{current_exp}->{bumped_exp}"
            )
            t_retry = time.time()
            rc_retry = cmd_pipeline_run_safe(bumped)
            run_id_retry = _latest_run_after(root, t_retry)
            if run_id_retry is not None:
                retry_payload = {
                    "base_run_id": run_id,
                    "retry_run_id": run_id_retry,
                    "reason": "LOW_ROI_VALID",
                    "adjustments": {"exposure_us": float(bumped_exp)},
                }
                (root / run_id / "quality_retry.json").write_text(json.dumps(retry_payload, indent=2))
                (root / run_id_retry / "quality_retry.json").write_text(json.dumps(retry_payload, indent=2))
            return rc_retry, run_id_retry or run_id
        return rc, run_id

    base = argparse.Namespace(**vars(args))
    rc_v, v_run_id = _run_orientation_with_low_roi_nudge("vertical", base)
    if rc_v != 0:
        return rc_v
    if not v_run_id:
        print("Failed to locate vertical run output")
        return 10

    v_run = root / v_run_id
    normalise_path = v_run / "normalise.json"
    normalise = json.loads(normalise_path.read_text()) if normalise_path.exists() else {}

    h_args = argparse.Namespace(**vars(args))
    h_args.orientation = "horizontal"
    h_args.auto_normalise = False
    # Reuse locked controls from vertical normalisation exactly as-is.
    h_args.expert = True
    h_args.exposure_us = int(normalise.get("exposure_us", getattr(args, "exposure_us", 2000) or 2000))
    h_args.gain = float(normalise.get("analogue_gain", getattr(args, "gain", 1.0) or 1.0))
    h_args.contrast = float(normalise.get("contrast", getattr(args, "contrast", 0.6) or 0.6))
    h_args.brightness_offset = float(normalise.get("brightness_offset", getattr(args, "brightness_offset", 0.45) or 0.45))
    h_args.min_intensity = float(normalise.get("min_intensity", getattr(args, "min_intensity", 0.10) or 0.10))

    rc_h, h_run_id = _run_orientation_with_low_roi_nudge("horizontal", h_args)
    if rc_h != 0:
        return rc_h
    if not h_run_id:
        print("Failed to locate horizontal run output")
        return 11

    h_run = root / h_run_id
    combined_run_id = datetime.now().strftime("%Y%m%d_%H%M%S_uv")
    merged_dir = root / combined_run_id
    merged_dir.mkdir(parents=True, exist_ok=True)

    _copy_orientation_outputs(v_run, merged_dir / "vertical")
    _copy_orientation_outputs(h_run, merged_dir / "horizontal")

    # Shared root artifacts from vertical run.
    for name in ("roi", "normalise", "normalise.json"):
        src = v_run / name
        if src.is_dir():
            shutil.copytree(src, merged_dir / name, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, merged_dir / name)

    params = _params_from_meta(v_run_id, RunStore(root=str(root)))
    uv_meta = _build_uv_from_orientation_runs(merged_dir, v_run, h_run, params, cfg)

    v_quality = json.loads((v_run / "quality_report.json").read_text()) if (v_run / "quality_report.json").exists() else {}
    h_quality = json.loads((h_run / "quality_report.json").read_text()) if (h_run / "quality_report.json").exists() else {}
    uv_ok = bool(uv_meta.get("uv_gate_ok", False))
    combined_ok = bool(v_quality.get("ok", False) and h_quality.get("ok", False) and uv_ok)
    report = {
        "ok": combined_ok,
        "vertical_run_id": v_run_id,
        "horizontal_run_id": h_run_id,
        "uv_gate_ok": uv_ok,
        "uv_gate_failed_checks": uv_meta.get("uv_gate_failed_checks", []),
        "uv_gate_hints": uv_meta.get("uv_gate_hints", []),
        "uv_gate_thresholds": uv_meta.get("uv_gate_thresholds", {}),
        "metrics": {
            "uv_valid_ratio": float(uv_meta.get("valid_ratio", 0.0)),
            "u_range": float(uv_meta.get("u_range", 0.0)),
            "v_range": float(uv_meta.get("v_range", 0.0)),
            "u_edge_pct": float(uv_meta.get("u_edge_pct", 1.0)),
            "v_edge_pct": float(uv_meta.get("v_edge_pct", 1.0)),
            "u_zero_pct": float(uv_meta.get("u_zero_pct", 1.0)),
            "v_zero_pct": float(uv_meta.get("v_zero_pct", 1.0)),
            "vertical_effective_residual_p95_threshold": float(v_quality.get("metrics", {}).get("effective_residual_p95_threshold", 0.0)),
            "horizontal_effective_residual_p95_threshold": float(h_quality.get("metrics", {}).get("effective_residual_p95_threshold", 0.0)),
        },
    }
    (merged_dir / "quality_report.json").write_text(json.dumps(report, indent=2))

    # Lightweight merged metadata for discovery.
    meta = {
        "run_id": combined_run_id,
        "status": "completed" if combined_ok else "failed",
        "params": {
            **params.to_dict(),
            "orientation": "both",
        },
        "started_at": datetime.now().isoformat(),
        "finished_at": datetime.now().isoformat(),
        "error": None if combined_ok else "UV gate failed or orientation gate failed",
        "device_info": {"composed_from": {"vertical": v_run_id, "horizontal": h_run_id}},
        "total_frames": 0,
        "saved_frames": 0,
        "preview_enabled": True,
    }
    (merged_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"combined_run_id={combined_run_id}")
    print(json.dumps(report, indent=2))
    if not uv_ok:
        failed = uv_meta.get("uv_gate_failed_checks", [])
        if failed:
            print(f"UV gate failed checks: {', '.join(failed)}")
        for hint in uv_meta.get("uv_gate_hints", []):
            print(f"uv_hint: {hint}")
    return 0 if combined_ok else 12


def cmd_experiment(args) -> int:
    cfg = _load_config()
    params = _build_params(args, cfg)
    run_id = _run_scan_for_autotune(params, cfg, Path("data/experiments"), 0)
    if run_id is None:
        raise SystemExit("scan failed")
    score, _ = _compute_score_for_run(run_id, cfg)
    print(f"run_id={run_id}")
    print(json.dumps(asdict(score), indent=2))
    _print_score(score)
    return 0


def cmd_quality_state(args) -> int:
    cfg = _load_config()
    if bool(getattr(args, "reset", False)):
        state = reset_quality_state(cfg)
        print(json.dumps({"status": "reset", **state}, indent=2))
        return 0
    state = load_quality_state(cfg)
    print(json.dumps(state, indent=2))
    return 0


def _print_score(score):
    print(
        f"valid_ratio={score.valid_ratio:.3f} | largest_component_ratio={score.largest_component_ratio:.3f} | "
        f"edge_noise_ratio={score.edge_noise_ratio:.3f} | b_median={score.b_median:.2f} | score={score.score:.2f}"
    )
    print(
        f"roi_valid_ratio={score.roi_valid_ratio:.3f} | roi_largest_component_ratio={score.roi_largest_component_ratio:.3f} | "
        f"roi_edge_noise_ratio={score.roi_edge_noise_ratio:.3f} | roi_b_median={score.roi_b_median:.2f} | roi_score={score.roi_score:.2f}"
    )


def _params_from_meta(run_id: str, store: RunStore) -> ScanParams:
    run_dir = Path(store.root) / run_id
    meta = json.loads((run_dir / "meta.json").read_text())
    p = meta.get("params", {})
    width, height = p.get("resolution", (1024, 768))
    frequencies = p.get("frequencies")
    if not frequencies:
        frequencies = [float(p.get("frequency", 8.0))]
    return ScanParams(
        n_steps=int(p.get("n_steps", p.get("N", 4))),
        frequency=float(frequencies[0]),
        frequencies=[float(f) for f in frequencies],
        orientation=str(p.get("orientation", "vertical")),
        brightness=float(p.get("brightness", 1.0)),
        resolution=(int(width), int(height)),
        settle_ms=int(p.get("settle_ms", 50)),
        save_patterns=bool(p.get("save_patterns", False)),
        preview_fps=float(p.get("preview_fps", 10.0)),
        phase_convention=str(p.get("phase_convention", "atan2(-S,C)")),
        exposure_us=p.get("exposure_us"),
        analogue_gain=p.get("analogue_gain"),
        awb_mode=p.get("awb_mode"),
        awb_enable=p.get("awb_enable"),
        ae_enable=p.get("ae_enable"),
        iso=p.get("iso"),
        brightness_offset=float(p.get("brightness_offset", 0.5)),
        contrast=float(p.get("contrast", 1.0)),
        min_intensity=float(p.get("min_intensity", 0.10)),
        frequency_semantics=str(p.get("frequency_semantics", "cycles_across_dimension")),
        phase_origin_rad=float(p.get("phase_origin_rad", 0.0)),
        auto_normalise=bool(p.get("auto_normalise", True)),
        expert_mode=bool(p.get("expert_mode", False)),
        quality_retry_count=int(p.get("quality_retry_count", 0)),
    )


def _select_phase_dir(phase_dir: Path) -> Path:
    if not phase_dir.exists():
        return phase_dir
    if (phase_dir / "phi_debug_fixed.png").exists() or (phase_dir / "phi_debug_autoscale.png").exists():
        return phase_dir
    subdirs = [p for p in phase_dir.iterdir() if p.is_dir() and p.name.startswith("f_")]
    if not subdirs:
        return phase_dir
    def _tag_to_freq(name: str) -> float:
        val = name.replace("f_", "").replace("p", ".")
        try:
            return float(val)
        except Exception:
            return 0.0
    subdirs.sort(key=lambda p: _tag_to_freq(p.name))
    return subdirs[-1]


def _score_thresholds_from_images(images: List[np.ndarray], params: ScanParams, thresholds: PhaseThresholds) -> tuple[float, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    proc = PhaseShiftProcessor()
    res = proc.compute_phase(images, params, thresholds)
    score = score_mask(res.mask, res.B)
    return score.score, score.__dict__, res.mask, res.A, res.B


def cmd_autotune(args) -> int:
    cfg = _load_config()
    exp_root = Path("data/experiments") / datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root.mkdir(parents=True, exist_ok=True)

    target_valid = float(args.target_valid)
    min_largest = float(args.min_largest_component)
    max_edge = float(args.max_edge_noise)

    use_roi = getattr(args, "use_roi", "auto")

    def passes(score, roi_fallback: bool):
        if use_roi != "off" and not roi_fallback:
            cond1 = score.roi_valid_ratio >= 0.60 and score.roi_largest_component_ratio >= 0.85 and score.roi_edge_noise_ratio <= 0.20 and score.roi_b_median >= 20
            cond2 = score.roi_largest_component_ratio >= 0.75 and score.roi_valid_ratio >= 0.15
            return cond1 or cond2
        cond1 = score.valid_ratio >= target_valid and score.largest_component_ratio >= min_largest and score.edge_noise_ratio <= max_edge
        cond2 = score.largest_component_ratio >= 0.75 and score.valid_ratio >= 0.15
        return cond1 or cond2

    params = _build_params(args, cfg)

    best = None
    best_info = None
    attempts = 0
    attempt_records = []

    # Baseline scan
    run_id = _run_scan_for_autotune(params, cfg, exp_root, 0)
    if run_id is None:
        print("Baseline scan failed; aborting autotune.")
        return 1

    score, roi_info = _compute_score_for_run(run_id, cfg)
    _record_attempt(exp_root, 0, params, score, run_id=run_id, roi_meta=roi_info.get("roi_meta"))
    attempt_records.append((0, run_id, score))
    best = score
    best_info = {"run_id": run_id, "params": params.to_dict(), "score": asdict(score)}
    attempts += 1

    if passes(score, roi_info.get("roi_fallback", False)):
        _write_best(exp_root, best_info)
        _print_summary(exp_root, best_info, attempt_records)
        print("PASS")
        return 0

    # Threshold sweep
    thresholds_list = []
    for b in [2, 3, 4, 5, 7, 10, 12, 15, 20]:
        for s_lo in [0, 2, 5]:
            for s_hi in [245, 250, 253, 255]:
                thresholds_list.append({"B_thresh": b, "sat_low": s_lo, "sat_high": s_hi})

    store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
    images = store.load_captures(run_id)
    params_from_meta = _params_from_meta(run_id, store)

    for i, th in enumerate(thresholds_list, start=1):
        if attempts >= args.max_iters:
            _write_best(exp_root, best_info)
            _print_summary(exp_root, best_info, attempt_records)
            print("MAX_ITERS")
            return 0
        thresholds = _phase_thresholds(cfg, overrides=th)
        proc = PhaseShiftProcessor()
        res = proc.compute_phase(images, params_from_meta, thresholds)
        score = score_mask(res.mask, res.B, roi_mask=roi_info.get("roi_mask"))
        _record_attempt(exp_root, i, params_from_meta, score, run_id=run_id, thresholds=th, roi_meta=roi_info.get("roi_meta"))
        attempt_records.append((i, run_id, score))
        if best is None or score.score > best.score:
            best = score
            best_info = {"run_id": run_id, "params": params_from_meta.to_dict(), "score": asdict(score), "thresholds": th}
        if passes(score, roi_info.get("roi_fallback", False)):
            _write_best(exp_root, best_info)
            _print_summary(exp_root, best_info, attempt_records)
            print("PASS")
            return 0
        attempts += 1

    # Exposure/gain ramp
    exposures = [4000, 6000, 8000, 12000, 16000, 24000]
    gains = [1.0, 1.5, 2.0, 3.0, 4.0]

    attempt = len(thresholds_list) + 1
    for exp in exposures:
        for gain in gains:
            if attempts >= args.max_iters:
                _write_best(exp_root, best_info)
                _print_summary(exp_root, best_info, attempt_records)
                print("MAX_ITERS")
                return 0
            params.exposure_us = exp
            params.analogue_gain = gain
            run_id = _run_scan_for_autotune(params, cfg, exp_root, attempt)
            if run_id is None:
                attempt += 1
                attempts += 1
                continue
            score, roi_info = _compute_score_for_run(run_id, cfg)
            _record_attempt(exp_root, attempt, params, score, run_id=run_id, roi_meta=roi_info.get("roi_meta"))
            attempt_records.append((attempt, run_id, score))
            if score.score > best.score:
                best = score
                best_info = {"run_id": run_id, "params": params.to_dict(), "score": asdict(score)}
            if passes(score, roi_info.get("roi_fallback", False)):
                _write_best(exp_root, best_info)
                _print_summary(exp_root, best_info, attempt_records)
                print("PASS")
                return 0
            attempt += 1
            attempts += 1

    # Frequency fallback
    base_freq = params.frequency
    for freq in [base_freq / 2.0, base_freq / 4.0]:
        if freq < 1.0 or attempts >= args.max_iters:
            break
        params.frequency = freq
        run_id = _run_scan_for_autotune(params, cfg, exp_root, attempt)
        if run_id is None:
            attempt += 1
            attempts += 1
            continue
        score, roi_info = _compute_score_for_run(run_id, cfg)
        _record_attempt(exp_root, attempt, params, score, run_id=run_id, roi_meta=roi_info.get("roi_meta"))
        attempt_records.append((attempt, run_id, score))
        if score.score > best.score:
            best = score
            best_info = {"run_id": run_id, "params": params.to_dict(), "score": asdict(score)}
        if passes(score, roi_info.get("roi_fallback", False)):
            _write_best(exp_root, best_info)
            _print_summary(exp_root, best_info, attempt_records)
            print("PASS")
            return 0
        attempt += 1
        attempts += 1

    # Projector brightness/contrast sweep
    for contrast in [0.6, 0.8, 1.0]:
        for brightness in [0.2, 0.35, 0.5]:
            if attempts >= args.max_iters:
                _write_best(exp_root, best_info)
                _print_summary(exp_root, best_info, attempt_records)
                print("MAX_ITERS")
                return 0
            params.contrast = contrast
            params.brightness_offset = brightness
            run_id = _run_scan_for_autotune(params, cfg, exp_root, attempt)
            if run_id is None:
                attempt += 1
                attempts += 1
                continue
            score, roi_info = _compute_score_for_run(run_id, cfg)
            _record_attempt(exp_root, attempt, params, score, run_id=run_id, roi_meta=roi_info.get("roi_meta"))
            attempt_records.append((attempt, run_id, score))
            if score.score > best.score:
                best = score
                best_info = {"run_id": run_id, "params": params.to_dict(), "score": asdict(score)}
            if passes(score, roi_info.get("roi_fallback", False)):
                _write_best(exp_root, best_info)
                _print_summary(exp_root, best_info, attempt_records)
                print("PASS")
                return 0
            attempt += 1
            attempts += 1

    _write_best(exp_root, best_info)
    _print_summary(exp_root, best_info, attempt_records)
    print("MAX_ITERS")
    return 0


def _capture_score_from_metrics(run_id: str, high_freq: float, score, per_freq: Dict[str, Any]) -> CaptureScore:
    high_key = str(high_freq)
    metrics = per_freq.get(high_key, {})
    clipped_any_pct_roi = float(metrics.get("clipped_any_pct_roi", 1.0))
    clipped_step_max_roi = float(metrics.get("clipped_step_max_roi", 1.0))
    roi_b_median = float(metrics.get("roi_b_median", score.roi_b_median))
    roi_valid_ratio = float(metrics.get("roi_valid_ratio", score.roi_valid_ratio))
    roi_edge_noise_ratio = float(metrics.get("roi_edge_noise_ratio", score.roi_edge_noise_ratio))

    def clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    capture_score = (
        3.0 * clamp(1.0 - clipped_any_pct_roi / 0.01, 0.0, 1.0) +
        2.0 * clamp(roi_b_median / 30.0, 0.0, 1.5) +
        2.0 * clamp(roi_valid_ratio / 0.7, 0.0, 1.5) -
        1.5 * roi_edge_noise_ratio
    )
    thresholds_met = (
        clipped_any_pct_roi <= 0.01 and
        clipped_step_max_roi <= 0.01 and
        roi_b_median >= 20.0 and
        roi_valid_ratio >= 0.60
    )
    return CaptureScore(
        score=float(capture_score),
        clipped_any_pct_roi=clipped_any_pct_roi,
        clipped_step_max_roi=clipped_step_max_roi,
        roi_b_median=roi_b_median,
        roi_valid_ratio=roi_valid_ratio,
        roi_edge_noise_ratio=roi_edge_noise_ratio,
        run_id=run_id,
        high_freq=float(high_freq),
        thresholds_met=thresholds_met,
    )


def cmd_autotune_capture(args) -> int:
    cfg = _load_config()
    exp_root = Path("data/experiments") / datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root.mkdir(parents=True, exist_ok=True)
    params = _build_params(args, cfg)
    if args.frequencies:
        params.frequencies = [float(f) for f in args.frequencies]
        params.frequency = float(params.frequencies[0])
    freqs = params.get_frequencies()
    high_freq = max(freqs)

    candidates = []
    for exposure in [2000, 3000, 4000, 6000, 8000, 12000]:
        for gain in [1.0, 1.5, 2.0, 3.0]:
            for contrast in [0.4, 0.6, 0.8, 1.0]:
                for brightness_offset in [0.45, 0.5, 0.55, 0.6]:
                    for min_intensity in [0.10, 0.15, 0.20]:
                        candidates.append((exposure, gain, contrast, brightness_offset, min_intensity))

    best: CaptureScore | None = None
    best_payload: Dict[str, Any] | None = None
    records: list[CaptureScore] = []

    max_iters = int(args.max_iters)
    for idx, (exposure, gain, contrast, brightness_offset, min_intensity) in enumerate(candidates[:max_iters]):
        params.exposure_us = exposure
        params.analogue_gain = gain
        params.contrast = contrast
        params.brightness_offset = brightness_offset
        params.min_intensity = min_intensity

        run_id = _run_scan_for_autotune(params, cfg, exp_root, idx)
        if run_id is None:
            continue
        score, roi_info = _compute_score_for_run(run_id, cfg)
        cscore = _capture_score_from_metrics(run_id, high_freq, score, roi_info.get("per_freq", {}))
        records.append(cscore)
        attempt_dir = exp_root / f"attempt_{idx:02d}"
        attempt_dir.mkdir(exist_ok=True)
        attempt_data = {
            "run_id": run_id,
            "params": params.to_dict(),
            "mask_score": asdict(score),
            "capture_score": asdict(cscore),
            "per_freq": roi_info.get("per_freq", {}),
            "roi_meta": roi_info.get("roi_meta", {}),
        }
        (attempt_dir / "attempt.json").write_text(json.dumps(attempt_data, indent=2))
        (attempt_dir / "run_id.txt").write_text(run_id)

        if best is None or cscore.score > best.score:
            best = cscore
            best_payload = attempt_data

        print(
            f"{idx:02d} run={run_id} exp={exposure} gain={gain:.1f} contrast={contrast:.2f} "
            f"bright={brightness_offset:.2f} minI={min_intensity:.2f} "
            f"clip_roi={cscore.clipped_any_pct_roi:.4f} clip_step_max_roi={cscore.clipped_step_max_roi:.4f} "
            f"roi_B={cscore.roi_b_median:.2f} roi_valid={cscore.roi_valid_ratio:.3f} score={cscore.score:.3f}"
        )
        if cscore.thresholds_met:
            print("PASS")
            break

    if best is None or best_payload is None:
        print("No valid attempts completed")
        return 1

    (exp_root / "best.json").write_text(json.dumps(best_payload, indent=2))
    print("attempt | run_id | clip_roi | clip_step_max_roi | roi_B | roi_valid | roi_edge | score | pass")
    for i, rec in enumerate(records):
        print(
            f"{i:>7} | {rec.run_id} | {rec.clipped_any_pct_roi:>8.4f} | {rec.clipped_step_max_roi:>16.4f} | "
            f"{rec.roi_b_median:>5.2f} | {rec.roi_valid_ratio:>8.3f} | {rec.roi_edge_noise_ratio:>8.3f} | "
            f"{rec.score:>5.3f} | {rec.thresholds_met}"
        )
    if not best.thresholds_met:
        print("BEST_ATTEMPT_FAILED_CONSTRAINTS")
    print(f"best.json: {exp_root / 'best.json'}")
    return 0


def _compute_unwrap_roi_valid_ratio(run_id: str, cfg: dict) -> tuple[float, dict]:
    store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
    params = _params_from_meta(run_id, store)
    freqs = params.get_frequencies()
    if len(freqs) < 2:
        return 0.0, {"warning": "Need at least two frequencies"}
    phases: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for freq in freqs:
        phase_dir = Path(store.root) / run_id / "phase" / store._freq_tag(freq)
        phases.append(np.load(phase_dir / "phi_wrapped.npy"))
        mask_path = phase_dir / "mask_for_unwrap.npy"
        if not mask_path.exists():
            mask_path = phase_dir / "mask_clean.npy"
        if not mask_path.exists():
            mask_path = phase_dir / "mask.npy"
        masks.append(np.load(mask_path))
    roi_mask = None
    roi_path = Path(store.root) / run_id / "roi" / "roi_mask.png"
    if roi_path.exists():
        from PIL import Image
        roi_mask = np.array(Image.open(roi_path)) > 0
    phi_abs, mask_unwrap, meta, residual = unwrap_multi_frequency(phases, masks, freqs, roi_mask=roi_mask, use_roi=True)
    save_unwrap_outputs(Path(store.root) / run_id, phi_abs, mask_unwrap, meta, f_max=max(freqs), residual=residual)
    if roi_mask is not None and np.any(roi_mask):
        roi_unwrap_valid_ratio = float(np.count_nonzero(mask_unwrap & roi_mask) / int(roi_mask.sum()))
    else:
        roi_unwrap_valid_ratio = float(np.count_nonzero(mask_unwrap) / mask_unwrap.size)
    outside_roi_true = 0
    if roi_mask is not None:
        outside_roi_true = int(np.count_nonzero(mask_unwrap & (~roi_mask)))
    meta["roi_unwrap_valid_ratio"] = roi_unwrap_valid_ratio
    meta["outside_roi_true_px"] = outside_roi_true
    (Path(store.root) / run_id / "unwrap" / "unwrap_meta.json").write_text(json.dumps(meta, indent=2))
    return roi_unwrap_valid_ratio, meta


def cmd_polish_run(args) -> int:
    cfg = _load_config()
    params = _build_params(args, cfg)
    if args.frequencies:
        params.frequencies = [float(f) for f in args.frequencies]
        params.frequency = float(params.frequencies[0])
    freqs = params.get_frequencies()
    high_freq = max(freqs)

    polish = cfg.get("polish", {})
    t_roi_valid = float(polish.get("target_roi_valid_ratio", 0.95))
    t_roi_largest = float(polish.get("target_roi_largest_component_ratio", 0.90))
    t_roi_edge = float(polish.get("target_roi_edge_noise_ratio", 0.20))
    t_clip = float(polish.get("target_clipped_any_pct_roi", 0.01))
    exp_candidates = [int(v) for v in polish.get("exposure_us_candidates", [2000, 2500, 3000, 3500])]
    gain_candidates = [float(v) for v in polish.get("gain_candidates", [1.0, 1.5])]
    contrast_candidates = [float(v) for v in polish.get("contrast_candidates", [0.6, 0.7, 0.8])]
    bright_candidates = [float(v) for v in polish.get("brightness_offset_candidates", [0.45, 0.50, 0.55])]
    min_int_candidates = [float(v) for v in polish.get("min_intensity_candidates", [0.10, 0.12, 0.15])]

    exp_root = Path("data/experiments") / datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root.mkdir(parents=True, exist_ok=True)
    max_tries = int(args.max_tries)

    print(
        "try | run_id | exp | gain | contrast | brightness | minI | "
        "f_high_roi_valid | f_high_roi_largest | f_high_roi_edge | f_high_clip_roi | unwrap_roi_valid | PASS"
    )

    best: dict | None = None
    tries = 0
    for exp in exp_candidates:
        for gain in gain_candidates:
            for contrast in contrast_candidates:
                for bright in bright_candidates:
                    for min_intensity in min_int_candidates:
                        if tries >= max_tries:
                            break
                        params.exposure_us = exp
                        params.analogue_gain = gain
                        params.contrast = contrast
                        params.brightness_offset = bright
                        params.min_intensity = min_intensity

                        run_id = _run_scan_for_autotune(params, cfg, exp_root, tries)
                        if run_id is None:
                            tries += 1
                            continue
                        score, info = _compute_score_for_run(run_id, cfg)
                        roi_unwrap_valid_ratio, unwrap_meta = _compute_unwrap_roi_valid_ratio(run_id, cfg)
                        per_freq = info.get("per_freq", {})
                        high = per_freq.get(str(high_freq), {})
                        roi_valid = float(high.get("roi_valid_ratio", score.roi_valid_ratio))
                        roi_largest = float(high.get("roi_largest_component_ratio", score.roi_largest_component_ratio))
                        roi_edge = float(high.get("roi_edge_noise_ratio", score.roi_edge_noise_ratio))
                        clip_roi = float(high.get("clipped_any_pct_roi", 1.0))
                        pass_flag = (
                            clip_roi <= t_clip and
                            roi_valid >= t_roi_valid and
                            roi_largest >= t_roi_largest and
                            roi_edge <= t_roi_edge
                        )
                        attempt = {
                            "run_id": run_id,
                            "params": params.to_dict(),
                            "high_freq": high_freq,
                            "high_freq_metrics": high,
                            "mask_score": asdict(score),
                            "unwrap_meta": unwrap_meta,
                            "roi_unwrap_valid_ratio": roi_unwrap_valid_ratio,
                            "pass": pass_flag,
                        }
                        attempt_dir = exp_root / f"attempt_{tries:02d}"
                        attempt_dir.mkdir(exist_ok=True)
                        (attempt_dir / "attempt.json").write_text(json.dumps(attempt, indent=2))
                        (attempt_dir / "run_id.txt").write_text(run_id)
                        print(
                            f"{tries:>3} | {run_id} | {exp:>4} | {gain:>4.1f} | {contrast:>7.2f} | {bright:>10.2f} | {min_intensity:>4.2f} | "
                            f"{roi_valid:>16.3f} | {roi_largest:>18.3f} | {roi_edge:>15.3f} | {clip_roi:>15.4f} | {roi_unwrap_valid_ratio:>16.3f} | {pass_flag}"
                        )
                        if best is None:
                            best = attempt
                        else:
                            cur = best.get("high_freq_metrics", {})
                            if clip_roi < float(cur.get("clipped_any_pct_roi", 1.0)) or (
                                clip_roi <= float(cur.get("clipped_any_pct_roi", 1.0)) and
                                roi_largest > float(best.get("mask_score", {}).get("roi_largest_component_ratio", 0.0))
                            ):
                                best = attempt
                        tries += 1
                        if pass_flag:
                            (exp_root / "best.json").write_text(json.dumps(attempt, indent=2))
                            print("PASS")
                            print(f"best.json: {exp_root / 'best.json'}")
                            return 0
                    if tries >= max_tries:
                        break
                if tries >= max_tries:
                    break
            if tries >= max_tries:
                break
        if tries >= max_tries:
            break

    if best is not None:
        (exp_root / "best.json").write_text(json.dumps(best, indent=2))
    print("MAX_TRIES")
    print(f"best.json: {exp_root / 'best.json'}")
    return 0


def _run_scan_for_autotune(params: ScanParams, cfg: dict, exp_root: Path, attempt: int) -> str | None:
    try:
        display = PygameProjectorDisplay()
        camera = _camera_from_cfg(cfg)
        generator = FringePatternGenerator()
        store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
        preview = PreviewBroadcaster()
        controller = ScanController(display=display, camera=camera, generator=generator, store=store, preview=preview, config=cfg)
        run_id = controller.start_scan(params)
        while True:
            status = controller.get_status()
            if status["state"] in ("IDLE", "ERROR"):
                break
            time.sleep(0.2)
        if status["state"] == "ERROR":
            print("scan error", status.get("last_error"))
            return None
        return run_id
    except Exception as exc:
        print(f"scan failed: {exc}")
        return None


def _compute_score_for_run(run_id: str, cfg: dict):
    store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
    params = _params_from_meta(run_id, store)
    thresholds = _phase_thresholds(cfg)
    proc = PhaseShiftProcessor()
    roi_cfg = _roi_config(cfg)
    ref = store.load_reference_image(run_id, ref_method=roi_cfg.ref_method)
    roi_res = detect_object_roi(ref, roi_cfg)
    roi_meta = {
        "cfg": asdict(roi_cfg),
        "debug": {
            **roi_res.debug,
            "ref_method": roi_cfg.ref_method,
            "ref_stats": {
                "min": int(np.min(ref)),
                "mean": float(np.mean(ref)),
                "max": int(np.max(ref)),
            },
        },
        "bbox": roi_res.bbox,
    }
    store.save_roi(run_id, roi_res.roi_mask, roi_res.bbox, {"cfg": asdict(roi_cfg), "debug": roi_meta["debug"]})
    roi_mask_for_score = None if roi_res.debug.get("roi_fallback") else roi_res.roi_mask
    freqs = params.get_frequencies()
    post_cfg = _postmask_config(cfg)
    post_enabled = bool(post_cfg.get("enabled", True))
    min_component_area = int(post_cfg.get("min_component_area", 200))
    fill_small_holes = bool(post_cfg.get("fill_small_holes", True))
    max_hole_area = int(post_cfg.get("max_hole_area", 200))
    score = None
    per_freq: Dict[str, Any] = {}
    for freq in freqs:
        images = store.load_captures(run_id, freq=freq) if len(freqs) > 1 else store.load_captures(run_id)
        res = proc.compute_phase(images, params, thresholds, roi_mask=roi_mask_for_score)
        raw_mask = res.mask_raw.copy()
        if post_enabled:
            cleaned = cleanup_mask(
                raw_mask,
                roi_mask_for_score,
                min_component_area=min_component_area,
                fill_small_holes=fill_small_holes,
                max_hole_area=max_hole_area,
            )
            cleaned &= (~res.clipped_any_map)
            res.mask_clean = cleaned
            res.mask = cleaned
            res.debug["postmask_cleanup"] = {
                "enabled": True,
                "min_component_area": min_component_area,
                "fill_small_holes": fill_small_holes,
                "max_hole_area": max_hole_area,
                "raw_valid_ratio": float(np.count_nonzero(raw_mask) / raw_mask.size),
                "clean_valid_ratio": float(np.count_nonzero(cleaned) / cleaned.size),
            }
        else:
            res.mask_clean = raw_mask
            res.mask = raw_mask
            res.debug["postmask_cleanup"] = {"enabled": False}
        s = score_mask(res.mask, res.B, roi_mask=roi_mask_for_score)
        res.debug.update({
            "roi_valid_ratio": s.roi_valid_ratio,
            "roi_largest_component_ratio": s.roi_largest_component_ratio,
            "roi_edge_noise_ratio": s.roi_edge_noise_ratio,
            "roi_b_median": s.roi_b_median,
            "roi_score": s.roi_score,
            "roi_fallback": roi_res.debug.get("roi_fallback", False),
        })
        store.save_phase_outputs(run_id, res, freq=freq if len(freqs) > 1 else None)
        phase_dir = Path(store.root) / run_id / "phase"
        if len(freqs) > 1:
            phase_dir = phase_dir / store._freq_tag(freq)
        try:
            from fringe_app.phase import visualize as _viz
            valid_in_roi = res.mask & roi_res.roi_mask
            _viz.save_mask_png(valid_in_roi, str(phase_dir / "valid_in_roi.png"))
        except Exception:
            pass
        per_freq[str(freq)] = {
            "clipped_any_pct": float(res.debug.get("clipped_any_pct", 0.0)),
            "clipped_any_pct_roi": float(res.debug.get("clipped_any_pct_roi", 0.0)),
            "clipped_step_max_roi": float(max(res.debug.get("clipped_per_step_pct_roi", [0.0]))),
            "roi_valid_ratio": float(s.roi_valid_ratio),
            "roi_largest_component_ratio": float(s.roi_largest_component_ratio),
            "roi_b_median": float(s.roi_b_median),
            "roi_edge_noise_ratio": float(s.roi_edge_noise_ratio),
            "score": float(s.score),
        }
        # Use highest frequency as primary quality signal.
        if score is None or float(freq) >= float(max(freqs)):
            score = s
    overlays_dir = Path(store.root) / run_id / "overlays"
    overlays_dir.mkdir(exist_ok=True)
    clip_dir = overlays_dir / "clipping"
    clip_dir.mkdir(exist_ok=True)
    try:
        empty = np.zeros_like(roi_res.roi_mask, dtype=bool)
        overlay_masks(ref, roi_res.roi_mask, empty, overlays_dir / "roi_overlay.png")
        for freq in freqs:
            phase_dir = Path(store.root) / run_id / "phase"
            if len(freqs) > 1:
                phase_dir = phase_dir / store._freq_tag(freq)
            clipped_npy = phase_dir / "clipped_any.npy"
            if clipped_npy.exists():
                suffix = "" if len(freqs) == 1 else f"_{store._freq_tag(freq)}"
                overlay_clipping(ref, roi_res.roi_mask, np.load(clipped_npy), clip_dir / f"clipping_overlay{suffix}.png")
    except Exception:
        pass
    if score is None:
        raise RuntimeError("No phase score computed")
    return score, {
        "roi_mask": roi_res.roi_mask,
        "roi_fallback": roi_res.debug.get("roi_fallback", False),
        "roi_meta": roi_meta,
        "per_freq": per_freq,
    }


def _record_attempt(exp_root: Path, attempt: int, params: ScanParams, score, run_id: str | None = None, thresholds: Dict[str, Any] | None = None, roi_meta: Dict[str, Any] | None = None) -> None:
    attempt_dir = exp_root / f"attempt_{attempt:02d}"
    attempt_dir.mkdir(exist_ok=True)
    data = {
        "params": params.to_dict(),
        "score": asdict(score),
        "thresholds": thresholds,
        "run_id": run_id,
    }
    (attempt_dir / "attempt.json").write_text(json.dumps(data, indent=2))
    if run_id:
        (attempt_dir / "run_id.txt").write_text(str(run_id))
    if roi_meta is not None:
        (attempt_dir / "roi_meta.json").write_text(json.dumps(roi_meta, indent=2))


def _write_best(exp_root: Path, best_info: Dict[str, Any] | None) -> None:
    if best_info is None:
        return
    (exp_root / "best.json").write_text(json.dumps(best_info, indent=2))


def _print_summary(exp_root: Path, best_info: Dict[str, Any] | None, records: List[tuple]) -> None:
    print("attempt | run_id | score | valid_ratio | largest_component_ratio | edge_noise_ratio | b_median | roi_valid | roi_largest | roi_edge | roi_b")
    for attempt, run_id, score in records:
        print(
            f"{attempt:>7} | {run_id} | {score.score:>5.2f} | {score.valid_ratio:>10.3f} | "
            f"{score.largest_component_ratio:>23.3f} | {score.edge_noise_ratio:>15.3f} | {score.b_median:>7.2f} | "
            f"{score.roi_valid_ratio:>9.3f} | {score.roi_largest_component_ratio:>10.3f} | "
            f"{score.roi_edge_noise_ratio:>7.3f} | {score.roi_b_median:>6.2f}"
        )
    if best_info is not None:
        print(f"best.json: {exp_root / 'best.json'}")
        print(json.dumps(best_info, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fringe_app")
    sub = p.add_subparsers(dest="cmd")

    scan = sub.add_parser("scan")
    scan.add_argument("--n", type=int, default=4)
    scan.add_argument("--frequency", type=float, default=None)
    scan.add_argument("--frequencies", type=float, nargs="+", default=None)
    scan.add_argument("--orientation", type=str, default="vertical")
    scan.add_argument("--settle-ms", type=int, default=150)
    scan.add_argument("--exposure-us", type=int, default=None)
    scan.add_argument("--gain", "--analogue-gain", dest="gain", type=float, default=None)
    scan.add_argument("--ae-enable", type=lambda v: str(v).lower() in {"1", "true", "yes", "on"}, default=None)
    scan.add_argument("--width", type=int, default=None)
    scan.add_argument("--height", type=int, default=None)
    scan.add_argument("--brightness", type=float, default=1.0)
    scan.add_argument("--brightness-offset", type=float, default=0.5)
    scan.add_argument("--contrast", type=float, default=1.0)
    scan.add_argument("--min-intensity", type=float, default=None)

    phase = sub.add_parser("phase")
    phase.add_argument("--run", required=True)

    score = sub.add_parser("score")
    score.add_argument("--run", required=True)

    overlay = sub.add_parser("overlay")
    overlay.add_argument("--run", required=True)

    unwrap = sub.add_parser("unwrap")
    unwrap.add_argument("--run", required=True)
    unwrap.add_argument("--use-roi", type=str, default="auto", choices=["auto", "on", "off"])

    autotune = sub.add_parser("autotune-mask")
    autotune.add_argument("--n", type=int, default=4)
    autotune.add_argument("--frequency", type=float, default=None)
    autotune.add_argument("--orientation", type=str, default="vertical")
    autotune.add_argument("--settle-ms", type=int, default=150)
    autotune.add_argument("--target-valid", type=float, default=0.30)
    autotune.add_argument("--min-largest-component", type=float, default=0.60)
    autotune.add_argument("--max-edge-noise", type=float, default=0.35)
    autotune.add_argument("--max-iters", type=int, default=25)
    autotune.add_argument("--use-roi", type=str, default="auto", choices=["auto", "on", "off"])

    autotune_capture = sub.add_parser("autotune-capture")
    autotune_capture.add_argument("--n", type=int, default=4)
    autotune_capture.add_argument("--frequency", type=float, default=None)
    autotune_capture.add_argument("--frequencies", type=float, nargs="+", default=None)
    autotune_capture.add_argument("--orientation", type=str, default="vertical")
    autotune_capture.add_argument("--settle-ms", type=int, default=150)
    autotune_capture.add_argument("--max-iters", type=int, default=25)
    autotune_capture.add_argument("--use-roi", type=str, default="auto", choices=["auto", "on", "off"])

    polish = sub.add_parser("polish-run")
    polish.add_argument("--n", type=int, default=4)
    polish.add_argument("--frequency", type=float, default=None)
    polish.add_argument("--frequencies", type=float, nargs="+", default=None)
    polish.add_argument("--orientation", type=str, default="vertical")
    polish.add_argument("--settle-ms", type=int, default=150)
    polish.add_argument("--max-tries", type=int, default=12)

    experiment = sub.add_parser("experiment-mask")
    experiment.add_argument("--n", type=int, default=None)
    experiment.add_argument("--frequency", type=float, default=None)
    experiment.add_argument("--orientation", type=str, default=None)
    experiment.add_argument("--settle-ms", type=int, default=None)
    experiment.add_argument("--exposure-us", type=int, default=None)
    experiment.add_argument("--gain", type=float, default=None)
    experiment.add_argument("--width", type=int, default=None)
    experiment.add_argument("--height", type=int, default=None)
    experiment.add_argument("--brightness", type=float, default=None)
    experiment.add_argument("--brightness-offset", type=float, default=None)
    experiment.add_argument("--contrast", type=float, default=None)
    experiment.add_argument("--min-intensity", type=float, default=None)

    pipeline = sub.add_parser("pipeline-run-safe")
    pipeline.add_argument("--n", type=int, default=None)
    pipeline.add_argument("--mode", type=str, choices=["stable", "fast"], default=None)
    pipeline.add_argument("--frequency", type=float, default=None)
    pipeline.add_argument("--frequencies", type=float, nargs="+", default=None)
    pipeline.add_argument("--orientation", type=str, default="vertical")
    pipeline.add_argument("--settle-ms", type=int, default=150)
    pipeline.add_argument("--exposure-us", type=int, default=None)
    pipeline.add_argument("--gain", "--analogue-gain", dest="gain", type=float, default=None)
    pipeline.add_argument("--ae-enable", type=lambda v: str(v).lower() in {"1", "true", "yes", "on"}, default=None)
    pipeline.add_argument("--brightness-offset", type=float, default=None)
    pipeline.add_argument("--contrast", type=float, default=None)
    pipeline.add_argument("--min-intensity", type=float, default=None)
    pipeline.add_argument("--auto-normalise", dest="auto_normalise", action="store_true", default=True)
    pipeline.add_argument("--no-auto-normalise", dest="auto_normalise", action="store_false")
    pipeline.add_argument("--expert", action="store_true")
    pipeline.add_argument("--force", action="store_true")
    pipeline.add_argument("--no-retry", action="store_true")
    pipeline.add_argument("--print-hints", action="store_true")

    pipeline_uv = sub.add_parser("pipeline-run-uv")
    pipeline_uv.add_argument("--n", type=int, default=None)
    pipeline_uv.add_argument("--mode", type=str, choices=["stable", "fast"], default=None)
    pipeline_uv.add_argument("--frequency", type=float, default=None)
    pipeline_uv.add_argument("--frequencies", type=float, nargs="+", default=None)
    pipeline_uv.add_argument("--settle-ms", type=int, default=150)
    pipeline_uv.add_argument("--exposure-us", type=int, default=None)
    pipeline_uv.add_argument("--gain", "--analogue-gain", dest="gain", type=float, default=None)
    pipeline_uv.add_argument("--ae-enable", type=lambda v: str(v).lower() in {"1", "true", "yes", "on"}, default=None)
    pipeline_uv.add_argument("--brightness-offset", type=float, default=None)
    pipeline_uv.add_argument("--contrast", type=float, default=None)
    pipeline_uv.add_argument("--min-intensity", type=float, default=None)
    pipeline_uv.add_argument("--auto-normalise", dest="auto_normalise", action="store_true", default=True)
    pipeline_uv.add_argument("--no-auto-normalise", dest="auto_normalise", action="store_false")
    pipeline_uv.add_argument("--expert", action="store_true")
    pipeline_uv.add_argument("--force", action="store_true")
    pipeline_uv.add_argument("--no-retry", action="store_true")
    pipeline_uv.add_argument("--print-hints", action="store_true")

    qstate = sub.add_parser("quality-state")
    qstate.add_argument("--reset", action="store_true")
    qstate.add_argument("--show", action="store_true")

    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "scan":
        return cmd_scan(args)
    if args.cmd == "phase":
        return cmd_phase(args)
    if args.cmd == "score":
        return cmd_score(args)
    if args.cmd == "overlay":
        return cmd_overlay(args)
    if args.cmd == "unwrap":
        return cmd_unwrap(args)
    if args.cmd == "autotune-mask":
        return cmd_autotune(args)
    if args.cmd == "autotune-capture":
        return cmd_autotune_capture(args)
    if args.cmd == "polish-run":
        return cmd_polish_run(args)
    if args.cmd == "experiment-mask":
        return cmd_experiment(args)
    if args.cmd == "pipeline-run-safe":
        return cmd_pipeline_run_safe(args)
    if args.cmd == "pipeline-run-uv":
        return cmd_pipeline_run_uv(args)
    if args.cmd == "quality-state":
        return cmd_quality_state(args)
    parser.print_help()
    return 0
