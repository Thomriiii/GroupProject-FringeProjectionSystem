"""
psp.py

Phase-Shifting Profilometry (PSP) with robust masking.

This version is intended for **NORMAL SCANS** (not calibration):

  - Uses your original thresholds:
        TH1_GAMMA = 0.12
        TH2_MOD   = 10.0
  - Gaussian blur applied to modulation B before thresholding (B_eff).
  - Gamma computed from B_eff / A.
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy.ndimage import median_filter
from typing import Dict, Tuple, List


IntensityDict = Dict[Tuple[int, int], np.ndarray]
PhaseDict = Dict[int, np.ndarray]
FloatImage = np.ndarray


class PSPResult:
    def __init__(self):
        self.phi_wrapped: PhaseDict = {}
        self.A: Dict[int, FloatImage] = {}
        self.B: Dict[int, FloatImage] = {}      # raw modulation
        self.gamma: Dict[int, FloatImage] = {}  # from blurred B
        self.mask: Dict[int, np.ndarray] = {}   # raw per-frequency mask


# Global thresholds for normal scanning
TH1_GAMMA = 0.12    # relative modulation B/A
TH2_MOD   = 10.0    # absolute modulation (8-bit intensity units)


def run_psp_per_frequency(
    I_dict: IntensityDict,
    freqs: List[int],
    n_phase: int,
    apply_input_blur: bool = False,
    blur_kernel: Tuple[int, int] = (3, 3),
    median_phase_filter: bool = True,
) -> PSPResult:
    """
    Compute wrapped phase, modulation and masks for each frequency.

    Intended for normal scans (object reconstruction). For calibration,
    use psp_calib.run_psp_calibration instead.
    """
    result = PSPResult()
    freqs_sorted = sorted(freqs)

    deltas = 2.0 * np.pi * np.arange(n_phase) / n_phase
    sin_d = np.sin(deltas)[:, None, None]
    cos_d = np.cos(deltas)[:, None, None]

    for f in freqs_sorted:
        stack = np.stack([I_dict[(f, k)] for k in range(n_phase)], axis=0)

        if apply_input_blur:
            for k in range(n_phase):
                stack[k] = cv2.GaussianBlur(stack[k], blur_kernel, 0)

        # Mean intensity
        A_f = stack.mean(axis=0)

        # Phase sums
        S = np.sum(stack * sin_d, axis=0)
        C = np.sum(stack * cos_d, axis=0)

        # Raw modulation amplitude
        B_f = (2.0 / n_phase) * np.sqrt(S * S + C * C)

        # Optional median smoothing of wrapped phase
        phi = np.arctan2(S, C)
        if median_phase_filter:
            phi = median_filter(phi, size=3)

        # Blur B for stable masking (B_eff)
        B_eff = cv2.GaussianBlur(B_f.astype(np.float32), (7, 7), 0)

        # Gamma from blurred B
        gamma_f = np.zeros_like(A_f, dtype=np.float32)
        nz = A_f > 1e-6
        gamma_f[nz] = B_eff[nz] / A_f[nz]

        # Saturation mask
        I_min = stack.min(axis=0)
        I_max = stack.max(axis=0)
        saturated = (I_max > 250) | (I_min < 5)

        # Thresholds for normal scans
        mask_f = (gamma_f > TH1_GAMMA) & (B_eff > TH2_MOD) & (~saturated)

        # Store outputs
        result.phi_wrapped[f] = phi
        result.A[f] = A_f
        result.B[f] = B_f          # keep raw B for debugging
        result.gamma[f] = gamma_f  # blurred B-based gamma
        result.mask[f] = mask_f

    return result
