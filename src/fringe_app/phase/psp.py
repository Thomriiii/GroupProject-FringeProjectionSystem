"""Phase-shifting processing for N-step PSP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np

from fringe_app.core.models import ScanParams


@dataclass(slots=True)
class PhaseThresholds:
    sat_low: float = 5.0
    sat_high: float = 250.0
    B_thresh: float = 10.0
    A_min: float = 0.0
    debug_percentiles: tuple[float, float] = (1.0, 99.0)


@dataclass(slots=True)
class PhaseResult:
    phi_wrapped: np.ndarray
    A: np.ndarray
    B: np.ndarray
    mask: np.ndarray
    debug: Dict[str, Any]


class PhaseShiftProcessor:
    """
    Compute wrapped phase for a single-frequency, single-orientation run.

    Sign convention: phi = atan2(-S, C)
    """

    def compute_phase(
        self,
        images: List[np.ndarray],
        params: ScanParams,
        thresholds: PhaseThresholds,
    ) -> PhaseResult:
        if len(images) == 0:
            raise ValueError("No images provided")

        gray_stack = self._to_gray_stack(images)
        n = gray_stack.shape[0]
        if n != int(params.n_steps):
            raise ValueError(f"Expected {params.n_steps} frames, got {n}")

        k = np.arange(n, dtype=np.float32)
        angles = 2.0 * np.pi * k / float(n)
        cos_terms = np.cos(angles)[:, None, None]
        sin_terms = np.sin(angles)[:, None, None]

        A = np.mean(gray_stack, axis=0, dtype=np.float32)
        C = np.sum(gray_stack * cos_terms, axis=0, dtype=np.float32)
        S = np.sum(gray_stack * sin_terms, axis=0, dtype=np.float32)

        phi_wrapped = np.arctan2(-S, C).astype(np.float32)
        B = (2.0 / float(n)) * np.sqrt(C * C + S * S)
        B = B.astype(np.float32)

        sat_mask = (gray_stack < thresholds.sat_low) | (gray_stack > thresholds.sat_high)
        saturated = np.any(sat_mask, axis=0)
        mask = (~saturated) & (B >= thresholds.B_thresh)
        if thresholds.A_min > 0:
            mask &= (A >= thresholds.A_min)

        valid = mask & np.isfinite(phi_wrapped)
        phi_vals = phi_wrapped[valid]
        if phi_vals.size:
            p_low, p_high = np.percentile(phi_vals, thresholds.debug_percentiles)
            phi_min = float(phi_vals.min())
            phi_max = float(phi_vals.max())
        else:
            p_low, p_high = -np.pi, np.pi
            phi_min, phi_max = float(-np.pi), float(np.pi)

        debug = {
            "n_steps": int(n),
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
            "phase_convention": params.phase_convention,
        }

        return PhaseResult(phi_wrapped=phi_wrapped, A=A, B=B, mask=mask, debug=debug)

    @staticmethod
    def _to_gray_stack(images: List[np.ndarray]) -> np.ndarray:
        gray_list = []
        for img in images:
            if img.ndim == 2:
                gray = img.astype(np.float32)
            elif img.ndim == 3 and img.shape[2] == 3:
                img_f = img.astype(np.float32)
                gray = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]
            else:
                raise ValueError("Unsupported image format")
            gray_list.append(gray)
        return np.stack(gray_list, axis=0).astype(np.float32)
