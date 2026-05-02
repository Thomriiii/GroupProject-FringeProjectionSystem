"""Phase extraction stage using the existing PSP implementation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from fringe_app_v2.core.models import ScanParams
from fringe_app_v2.core import phase_viz as visualize
from fringe_app_v2.core.masking import (
    build_mask_for_defects,
    build_mask_for_display,
    build_unwrap_mask,
    cleanup_mask,
)
from fringe_app_v2.core.psp import PhaseShiftProcessor, PhaseThresholds, PhaseResult

from fringe_app_v2.phase_quality.diagnostics import save_wrapped_phase_diagnostics
from fringe_app_v2.phase_quality.validation import save_phase_quality_validation
from fringe_app_v2.utils.io import RunPaths, freq_tag, load_image_stack, save_mask_png, sorted_step_images, write_json


def thresholds_from_config(config: dict[str, Any]) -> PhaseThresholds:
    phase = config.get("phase", {}) or {}
    percentiles = phase.get("debug_percentiles", [1.0, 99.0])
    return PhaseThresholds(
        sat_low=float(phase.get("sat_low", 0)),
        sat_high=float(phase.get("sat_high", 250)),
        B_thresh=float(phase.get("B_thresh", 7)),
        A_min=float(phase.get("A_min", 15)),
        debug_percentiles=(float(percentiles[0]), float(percentiles[1])),
    )


def run_phase_stage(
    run: RunPaths,
    params: ScanParams,
    config: dict[str, Any],
    roi_mask: np.ndarray | None,
) -> dict[str, Any]:
    processor = PhaseShiftProcessor()
    thresholds = thresholds_from_config(config)
    freqs = [float(v) for v in params.get_frequencies()]
    orientations = [p.name for p in sorted(run.structured.iterdir()) if p.is_dir() and p.name in {"vertical", "horizontal"}]
    if not orientations:
        orientations = [str(v) for v in (config.get("scan", {}) or {}).get("orientations", ["vertical", "horizontal"])]
    summaries: dict[str, Any] = {}

    for orientation in orientations:
        orient_params = replace(params, orientation=orientation)  # type: ignore[arg-type]
        summaries[orientation] = {}
        for freq in freqs:
            tag = freq_tag(freq)
            image_paths = sorted_step_images(run.structured / orientation / tag)
            if len(image_paths) != int(params.n_steps):
                raise FileNotFoundError(
                    f"Expected {params.n_steps} captures in {run.structured / orientation / tag}, found {len(image_paths)}"
                )
            images = load_image_stack(image_paths)
            result = processor.compute_phase(
                images,
                orient_params,
                thresholds,
                roi_mask=roi_mask,
            )
            _apply_mask_policies(result, thresholds, roi_mask, config)
            out_dir = run.phase / orientation / tag
            _save_phase_result(out_dir, result)
            quality_dir = run.phase_quality / "validation" / orientation / tag
            result.debug["phase_quality"] = save_phase_quality_validation(quality_dir, images, config, roi_mask=roi_mask)
            diag_dir = run.phase_quality / "diagnostics"
            result.debug["phase_diagnostics"] = save_wrapped_phase_diagnostics(
                diag_dir,
                result.phi_wrapped,
                result.mask_for_display,
                f"{orientation}_{tag}",
            )
            write_json(out_dir / "phase_meta.json", result.debug)
            summaries[orientation][tag] = result.debug

    write_json(run.phase / "phase_summary.json", summaries)
    return summaries


def _apply_mask_policies(
    result: PhaseResult,
    thresholds: PhaseThresholds,
    roi_mask: np.ndarray | None,
    config: dict[str, Any],
) -> None:
    phase = config.get("phase", {}) or {}
    post = phase.get("postmask_cleanup", {}) or {}
    unwrap_post = (config.get("unwrap", {}) or {}).get("mask_post", {}) or {}
    masks = phase.get("masks", {}) or {}
    defects_cfg = masks.get("defects", {}) or {}
    display_cfg = masks.get("display", {}) or {}
    raw_mask = result.mask_raw.copy()

    if bool(post.get("enabled", True)):
        cleaned = cleanup_mask(
            raw_mask,
            roi_mask,
            min_component_area=int(post.get("min_component_area", 200)),
            fill_small_holes=bool(post.get("fill_small_holes", True)),
            max_hole_area=int(post.get("max_hole_area", 200)),
        )
        cleaned &= ~result.clipped_any_map
        result.mask = cleaned
        result.mask_clean = cleaned
    else:
        result.mask = raw_mask
        result.mask_clean = raw_mask

    roi_domain = roi_mask if roi_mask is not None else np.ones_like(raw_mask, dtype=bool)
    roi_vals = result.B[raw_mask & roi_domain]
    roi_b_median = float(np.median(roi_vals)) if roi_vals.size else 0.0
    configured_b = float(unwrap_post.get("b_thresh_unwrap", 0.0))
    b_unwrap = configured_b if configured_b > 0 else (
        float(min(thresholds.B_thresh, 0.05 * roi_b_median)) if roi_b_median > 0 else float(thresholds.B_thresh)
    )
    unwrap_seed = (~result.clipped_any_map) & (result.A >= thresholds.A_min) & (result.B >= b_unwrap)
    if roi_mask is not None:
        unwrap_seed &= roi_mask
    if bool(unwrap_post.get("enabled", True)):
        result.mask_for_unwrap = build_unwrap_mask(unwrap_seed, roi_mask, result.clipped_any_map, unwrap_post)
    else:
        result.mask_for_unwrap = unwrap_seed
    result.mask_for_defects = build_mask_for_defects(raw_mask, roi_mask, result.clipped_any_map, defects_cfg)
    result.mask_for_display = build_mask_for_display(result.B, raw_mask, roi_mask, result.clipped_any_map, display_cfg)
    result.debug["v2_mask_policy"] = {
        "postmask_cleanup": post,
        "unwrap_mask_post": unwrap_post,
        "b_thresh_unwrap": b_unwrap,
        "roi_limited": roi_mask is not None,
    }


def _save_phase_result(out_dir, result: PhaseResult) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "phi_wrapped.npy", result.phi_wrapped.astype(np.float32))
    np.save(out_dir / "A.npy", result.A.astype(np.float32))
    np.save(out_dir / "B.npy", result.B.astype(np.float32))
    np.save(out_dir / "mask.npy", result.mask_raw.astype(bool))
    np.save(out_dir / "mask_raw.npy", result.mask_raw.astype(bool))
    np.save(out_dir / "mask_clean.npy", result.mask_clean.astype(bool))
    np.save(out_dir / "mask_for_unwrap.npy", result.mask_for_unwrap.astype(bool))
    np.save(out_dir / "mask_for_defects.npy", result.mask_for_defects.astype(bool))
    np.save(out_dir / "mask_for_display.npy", result.mask_for_display.astype(bool))
    np.save(out_dir / "clipped_any.npy", result.clipped_any_map.astype(bool))
    save_mask_png(out_dir / "mask.png", result.mask_raw)
    save_mask_png(out_dir / "mask_for_unwrap.png", result.mask_for_unwrap)
    save_mask_png(out_dir / "mask_for_defects.png", result.mask_for_defects)
    perc = result.debug.get("debug_percentiles", [1.0, 99.0])
    visualize.save_phase_png_autoscale(
        result.phi_wrapped,
        result.mask_for_display,
        (float(perc[0]), float(perc[1])),
        str(out_dir / "phi_debug_autoscale.png"),
    )
    visualize.save_phase_png_fixed(result.phi_wrapped, result.mask_for_display, str(out_dir / "phi_debug_fixed.png"))
    visualize.save_modulation_png(result.B, result.mask_for_display, str(out_dir / "B_debug.png"))
    write_json(out_dir / "phase_meta.json", result.debug)
