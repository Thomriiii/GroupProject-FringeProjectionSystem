"""Temporal unwrap stage using the existing multi-frequency implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

from fringe_app_v2.core.temporal_unwrap import unwrap_multi_frequency

from fringe_app_v2.phase_quality.diagnostics import save_unwrapped_phase_diagnostics
from fringe_app_v2.utils.io import RunPaths, freq_tag, save_mask_png, write_json


def run_unwrap_stage(
    run: RunPaths,
    config: dict[str, Any],
    roi_mask: np.ndarray | None,
) -> dict[str, Any]:
    freqs = [float(v) for v in (config.get("scan", {}) or {}).get("frequencies", [])]
    if len(freqs) < 2:
        raise ValueError("At least two frequencies are required for temporal unwrapping")
    orientations = [p.name for p in sorted(run.phase.iterdir()) if p.is_dir() and p.name in {"vertical", "horizontal"}]
    summary: dict[str, Any] = {}
    use_roi = bool((config.get("unwrap", {}) or {}).get("use_roi", True))

    for orientation in orientations:
        phases: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for freq in freqs:
            pdir = run.phase / orientation / freq_tag(freq)
            phases.append(np.load(pdir / "phi_wrapped.npy").astype(np.float32))
            masks.append(np.load(pdir / "mask_for_unwrap.npy").astype(bool))
        phi_abs, mask_unwrap, meta, residual = unwrap_multi_frequency(
            phases,
            masks,
            freqs,
            roi_mask=roi_mask,
            use_roi=use_roi,
        )
        out_dir = run.unwrap / orientation
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "phi_abs.npy", phi_abs.astype(np.float32))
        np.save(out_dir / "phase_final.npy", phi_abs.astype(np.float32))
        np.save(out_dir / "mask_unwrap.npy", mask_unwrap.astype(bool))
        np.save(out_dir / "mask_final.npy", mask_unwrap.astype(bool))
        np.save(out_dir / "residual.npy", residual.astype(np.float32))
        save_mask_png(out_dir / "mask_unwrap.png", mask_unwrap)
        save_mask_png(out_dir / "mask_final.png", mask_unwrap)
        meta["phase_diagnostics"] = save_unwrapped_phase_diagnostics(
            run.phase_quality / "diagnostics",
            phi_abs,
            mask_unwrap,
            orientation,
        )
        write_json(out_dir / "unwrap_meta.json", meta)
        summary[orientation] = meta

    write_json(run.unwrap / "unwrap_summary.json", summary)
    return summary
