"""Phase diagnostic visualizations and stripe artefact metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fringe_app_v2.defect.utils import normalise_to_u8
from fringe_app_v2.utils.io import load_image_stack, save_image, sorted_step_images, write_json


def phase_gradient(phi: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    arr = np.asarray(phi, dtype=np.float32)
    valid = np.isfinite(arr) if mask is None else np.asarray(mask, dtype=bool) & np.isfinite(arr)
    fill = float(np.nanmedian(arr[valid])) if np.any(valid) else 0.0
    filled = np.where(valid, arr, fill).astype(np.float32)
    grad_y, grad_x = np.gradient(filled)
    magnitude = np.hypot(grad_x, grad_y).astype(np.float32)
    magnitude[~valid] = np.nan
    stripe_score = float(np.nanstd(grad_y[valid]) + np.nanstd(grad_x[valid])) if np.any(valid) else 0.0
    return magnitude, stripe_score


def save_wrapped_phase_diagnostics(
    out_dir: Path,
    phi: np.ndarray,
    mask: np.ndarray,
    label: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(phi)
    gradient, stripe_score = phase_gradient(phi, valid)
    save_image(out_dir / f"phi_{label}.png", normalise_to_u8(phi, valid))
    save_image(out_dir / f"gradient_{label}.png", normalise_to_u8(gradient, np.isfinite(gradient)))
    return {
        "label": label,
        "kind": "wrapped",
        "valid_pixels": int(np.count_nonzero(valid)),
        "stripe_score": stripe_score,
    }


def save_unwrapped_phase_diagnostics(
    out_dir: Path,
    phi: np.ndarray,
    mask: np.ndarray,
    label: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(phi)
    gradient, stripe_score = phase_gradient(phi, valid)
    save_image(out_dir / f"unwrap_{label}.png", normalise_to_u8(phi, valid))
    save_image(out_dir / f"gradient_unwrap_{label}.png", normalise_to_u8(gradient, np.isfinite(gradient)))
    return {
        "label": label,
        "kind": "unwrapped",
        "valid_pixels": int(np.count_nonzero(valid)),
        "stripe_score": stripe_score,
    }


def run_phase_debug_for_run(
    run_dir: Path,
    config: dict[str, Any],
    calibration: Any | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    quality_dir = run_dir / "phase_quality"
    report: dict[str, Any] = {"wrapped": [], "unwrapped": [], "validation": [], "calibration": None}
    report["validation"] = _run_existing_validation(run_dir, config)
    report["wrapped"] = _run_existing_wrapped_diagnostics(run_dir)
    report["unwrapped"] = _run_existing_unwrap_diagnostics(run_dir)
    if calibration is not None:
        from fringe_app_v2.phase_quality.calibration_check import run_calibration_check

        report["calibration"] = run_calibration_check(run_dir, config, calibration, fail_on_error=False)
    write_json(quality_dir / "report.json", report)
    return report


def _run_existing_validation(run_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    from fringe_app_v2.phase_quality.validation import save_phase_quality_validation

    summaries: list[dict[str, Any]] = []
    structured_root = run_dir / "structured"
    if not structured_root.exists():
        return summaries
    roi_mask = _load_optional_mask(run_dir / "roi" / "roi_mask.npy")
    for orientation_dir in sorted(path for path in structured_root.iterdir() if path.is_dir()):
        if orientation_dir.name == "patterns":
            continue
        for freq_dir in sorted(path for path in orientation_dir.iterdir() if path.is_dir() and path.name.startswith("f_")):
            paths = sorted_step_images(freq_dir)
            if not paths:
                continue
            label = f"{orientation_dir.name}_{freq_dir.name}"
            out_dir = run_dir / "phase_quality" / "validation" / orientation_dir.name / freq_dir.name
            summary = save_phase_quality_validation(out_dir, load_image_stack(paths), config, roi_mask=roi_mask)
            summary["label"] = label
            summaries.append(summary)
    return summaries


def _run_existing_wrapped_diagnostics(run_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    phase_root = run_dir / "phase"
    if not phase_root.exists():
        return summaries
    out_dir = run_dir / "phase_quality" / "diagnostics"
    for orientation_dir in sorted(path for path in phase_root.iterdir() if path.is_dir()):
        for freq_dir in sorted(path for path in orientation_dir.iterdir() if path.is_dir() and path.name.startswith("f_")):
            phi_path = freq_dir / "phi_wrapped.npy"
            mask_path = freq_dir / "mask_for_display.npy"
            if not phi_path.exists() or not mask_path.exists():
                continue
            label = f"{orientation_dir.name}_{freq_dir.name}"
            summaries.append(
                save_wrapped_phase_diagnostics(
                    out_dir,
                    np.load(phi_path).astype(np.float32),
                    np.load(mask_path).astype(bool),
                    label,
                )
            )
    return summaries


def _run_existing_unwrap_diagnostics(run_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    unwrap_root = run_dir / "unwrap"
    if not unwrap_root.exists():
        return summaries
    out_dir = run_dir / "phase_quality" / "diagnostics"
    for orientation_dir in sorted(path for path in unwrap_root.iterdir() if path.is_dir()):
        phi_path = orientation_dir / "phase_final.npy"
        mask_path = orientation_dir / "mask_final.npy"
        if not phi_path.exists() or not mask_path.exists():
            continue
        summaries.append(
            save_unwrapped_phase_diagnostics(
                out_dir,
                np.load(phi_path).astype(np.float32),
                np.load(mask_path).astype(bool),
                orientation_dir.name,
            )
        )
    return summaries


def _load_optional_mask(path: Path) -> np.ndarray | None:
    return np.load(path).astype(bool) if path.exists() else None
