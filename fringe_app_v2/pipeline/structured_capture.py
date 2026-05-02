"""Structured-light capture stage."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np

from fringe_app_v2.core.models import ScanParams

from fringe_app_v2.core.camera import CameraService
from fringe_app_v2.core.patterns import PatternService
from fringe_app_v2.core.projector import ProjectorService
from fringe_app_v2.phase_quality.gamma import apply_gamma, gamma_from_config
from fringe_app_v2.utils.io import RunPaths, freq_tag, save_image, write_json


def run_structured_capture(
    run: RunPaths,
    camera: CameraService,
    projector: ProjectorService,
    patterns: PatternService,
    params: ScanParams,
    config: dict[str, Any],
) -> dict[str, Any]:
    scan = config.get("scan", {}) or {}
    orientations = [str(v) for v in scan.get("orientations", ["vertical", "horizontal"])]
    freqs = [float(v) for v in params.get_frequencies()]
    settle_ms = int(scan.get("settle_ms", params.settle_ms))
    first_ms = int(scan.get("settle_ms_first_step", 400))
    switch_ms = int(scan.get("settle_ms_freq_switch", 400))
    warmup_ms = int(scan.get("freq_switch_warmup_ms", 250))
    flush_step = int(scan.get("flush_frames_per_step", 1))
    flush_switch = int(scan.get("flush_frames_on_freq_switch", 2))
    save_patterns = bool(scan.get("save_patterns", params.save_patterns))
    gamma = gamma_from_config(config)
    captured: list[dict[str, Any]] = []

    for orientation in orientations:
        orient_params = replace(params, orientation=orientation)  # type: ignore[arg-type]
        for freq_index, freq in enumerate(freqs):
            tag = freq_tag(freq)
            capture_dir = run.structured / orientation / tag
            pattern_dir = run.structured / orientation / "patterns" / tag
            capture_dir.mkdir(parents=True, exist_ok=True)
            sequence = patterns.sequence(orient_params, frequency=freq)
            if freq_index > 0:
                projector.show_level(128)
                time.sleep(max(warmup_ms, switch_ms) / 1000.0)
                camera.flush(flush_switch)
            for step, pattern in enumerate(sequence):
                display_pattern = apply_gamma(pattern, gamma)
                projector.show_gray(display_pattern)
                time.sleep((first_ms if step == 0 else settle_ms) / 1000.0)
                frame = camera.capture(flush_frames=flush_step)
                save_image(capture_dir / f"step_{step:03d}.png", frame)
                if save_patterns:
                    save_image(pattern_dir / f"pattern_{step:03d}.png", display_pattern)
                captured.append(
                    {
                        "orientation": orientation,
                        "frequency": freq,
                        "step": step,
                        "capture": str((capture_dir / f"step_{step:03d}.png").relative_to(run.root)),
                    }
                )

    projector.show_level(0)
    meta = {
        "orientations": orientations,
        "frequencies": freqs,
        "n_steps": int(params.n_steps),
        "frequency_semantics": params.frequency_semantics,
        "phase_origin_rad": float(params.phase_origin_rad),
        "gamma": gamma,
        "captures": captured,
    }
    write_json(run.structured / "structured_meta.json", meta)
    return meta
