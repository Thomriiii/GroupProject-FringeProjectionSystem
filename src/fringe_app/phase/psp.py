"""Phase-shifting processing for N-step PSP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Iterable

import numpy as np

from fringe_app.core.models import ScanParams


@dataclass(slots=True)
class PhaseThresholds:
    sat_low: float = 0.0
    sat_high: float = 250.0
    B_thresh: float = 10.0
    A_min: float = 10.0
    debug_percentiles: tuple[float, float] = (1.0, 99.0)


@dataclass(slots=True)
class PhaseResult:
    phi_wrapped: np.ndarray
    A: np.ndarray
    B: np.ndarray
    mask: np.ndarray
    mask_raw: np.ndarray
    mask_clean: np.ndarray
    mask_for_unwrap: np.ndarray
    mask_for_defects: np.ndarray
    mask_for_display: np.ndarray
    clipped_any_map: np.ndarray
    debug: Dict[str, Any]


class PhaseShiftProcessor:
    """
    Compute wrapped phase for a single-frequency, single-orientation run.

    Sign convention: phi = atan2(-S, C)
    """

    def compute_phase(
        self,
        images: Iterable[np.ndarray],
        params: ScanParams,
        thresholds: PhaseThresholds,
        roi_mask: np.ndarray | None = None,
    ) -> PhaseResult:
        n_expected = int(params.n_steps)
        if n_expected <= 0:
            raise ValueError("n_steps must be positive")

        gray_frames: list[np.ndarray] = []
        stack_u8_frames: list[np.ndarray] = []
        for idx, img in enumerate(images):
            if idx >= n_expected:
                break
            gray = self._to_gray(img)
            gray_frames.append(gray)
            stack_u8_frames.append(np.clip(np.rint(gray), 0, 255).astype(np.uint8))

        count = len(gray_frames)
        if count == 0:
            raise ValueError("No images provided")
        if count != n_expected:
            raise ValueError(f"Expected {n_expected} frames, got {count}")

        stack = np.stack(gray_frames, axis=0).astype(np.float32, copy=False)
        stack_u8 = np.stack(stack_u8_frames, axis=0)

        angles = 2.0 * np.pi * np.arange(n_expected, dtype=np.float32) / float(n_expected)
        cos_terms = np.cos(angles).astype(np.float32)
        sin_terms = np.sin(angles).astype(np.float32)

        A = np.mean(stack, axis=0, dtype=np.float32).astype(np.float32)
        C = np.tensordot(cos_terms, stack, axes=(0, 0)).astype(np.float32)
        S = np.tensordot(sin_terms, stack, axes=(0, 0)).astype(np.float32)
        phi_wrapped = np.arctan2(-S, C).astype(np.float32)
        B = ((2.0 / float(count)) * np.sqrt(C * C + S * S)).astype(np.float32)

        # Low-side intensity is intentionally ignored for masking: dark troughs are valid.
        invalid_high = (stack_u8 >= int(thresholds.sat_high)).any(axis=0)
        mask = (~invalid_high) & (B >= thresholds.B_thresh)
        if thresholds.A_min > 0:
            mask &= (A >= thresholds.A_min)
        mask_raw = mask.copy()

        valid = mask & np.isfinite(phi_wrapped)
        phi_vals = phi_wrapped[valid]
        if phi_vals.size:
            p_low, p_high = np.percentile(phi_vals, thresholds.debug_percentiles)
            phi_min = float(phi_vals.min())
            phi_max = float(phi_vals.max())
        else:
            p_low, p_high = -np.pi, np.pi
            phi_min, phi_max = float(-np.pi), float(np.pi)

        roi_for_clip = roi_mask.astype(bool) if roi_mask is not None else np.ones_like(mask, dtype=bool)
        clipped_any_map = invalid_high.astype(bool)
        clipped_any_pct = float(clipped_any_map.mean())
        clipped_any_pct_roi = float(clipped_any_map[roi_for_clip].mean()) if np.any(roi_for_clip) else 0.0
        clipped_per_step_pct = [float((stack_u8[k] >= int(thresholds.sat_high)).mean()) for k in range(count)]
        clipped_per_step_pct_roi = [
            float((stack_u8[k] >= int(thresholds.sat_high))[roi_for_clip].mean()) if np.any(roi_for_clip) else 0.0
            for k in range(count)
        ]

        debug = {
            "n_steps": int(count),
            "phi_min": phi_min,
            "phi_max": phi_max,
            "phi_p_low": float(p_low),
            "phi_p_high": float(p_high),
            "debug_percentiles": list(thresholds.debug_percentiles),
            "valid_pct": float(100.0 * np.count_nonzero(mask) / mask.size),
            "B_median": float(np.nanmedian(B[mask])) if np.any(mask) else 0.0,
            "A_median": float(np.nanmedian(A[mask])) if np.any(mask) else 0.0,
            "sat_low": thresholds.sat_low,
            "sat_high": thresholds.sat_high,
            "B_thresh": thresholds.B_thresh,
            "A_min": thresholds.A_min,
            "clipped_any_pct": clipped_any_pct,
            "clipped_any_pct_roi": clipped_any_pct_roi,
            "clipped_per_step_pct": clipped_per_step_pct,
            "clipped_per_step_pct_roi": clipped_per_step_pct_roi,
            "phase_convention": params.phase_convention,
            "mask_rules": {
                "high_saturation_rejected": True,
                "low_saturation_rejected": False,
                "A_min": thresholds.A_min,
                "B_thresh": thresholds.B_thresh,
            },
        }

        return PhaseResult(
            phi_wrapped=phi_wrapped,
            A=A,
            B=B,
            mask=mask,
            mask_raw=mask_raw,
            mask_clean=mask.copy(),
            mask_for_unwrap=mask.copy(),
            mask_for_defects=mask.copy(),
            mask_for_display=mask.copy(),
            clipped_any_map=clipped_any_map,
            debug=debug,
        )

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img.astype(np.float32)
        if img.ndim == 3 and img.shape[2] == 3:
            img_f = img.astype(np.float32)
            return 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]
        raise ValueError("Unsupported image format")
