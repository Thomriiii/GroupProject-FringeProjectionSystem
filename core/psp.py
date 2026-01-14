"""
psp.py

Phase-Shifting Profilometry (PSP) for normal scan captures.

This module estimates wrapped phase and per-frequency quality masks using
the scan-time thresholds and a blurred modulation estimate to stabilize
gamma-based masking.
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
    """
    Container for per-frequency PSP outputs.

    Attributes
    ----------
    phi_wrapped : dict[int, ndarray]
        Wrapped phase for each frequency.
    A : dict[int, ndarray]
        Mean intensity per frequency.
    B : dict[int, ndarray]
        Raw modulation amplitude per frequency.
    gamma : dict[int, ndarray]
        Gamma estimate (blurred modulation divided by mean).
    mask : dict[int, ndarray]
        Per-frequency validity mask.
    saturated : dict[int, ndarray]
        Per-frequency saturation mask.
    """

    def __init__(self):
        self.phi_wrapped: PhaseDict = {}
        self.A: Dict[int, FloatImage] = {}
        self.B: Dict[int, FloatImage] = {}
        self.gamma: Dict[int, FloatImage] = {}
        self.mask: Dict[int, np.ndarray] = {}
        self.saturated: Dict[int, np.ndarray] = {}


# Thresholds tuned for normal scanning.
TH1_GAMMA = 0.12    # Relative modulation B/A.
TH2_MOD   = 10.0    # Absolute modulation in 8-bit intensity units.


def run_psp_per_frequency(
    I_dict: IntensityDict,
    freqs: List[int],
    n_phase: int,
    apply_input_blur: bool = False,
    blur_kernel: Tuple[int, int] = (3, 3),
    median_phase_filter: bool = True,
) -> PSPResult:
    """
    Compute wrapped phase, modulation, and masks for each frequency.

    Parameters
    ----------
    I_dict : dict[(int, int), ndarray]
        Captured intensity frames keyed by (frequency, phase index).
    freqs : list[int]
        Frequencies to process.
    n_phase : int
        Number of phase steps per frequency.
    apply_input_blur : bool
        If True, blur each input frame before phase estimation.
    blur_kernel : tuple[int, int]
        Kernel size for the optional input blur.
    median_phase_filter : bool
        If True, apply a small median filter to the wrapped phase.

    Returns
    -------
    PSPResult
        Per-frequency wrapped phase, modulation, and masks for normal scans.
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

        # Mean intensity across the phase steps.
        A_f = stack.mean(axis=0)

        # Sine/cosine projections for phase estimation.
        S = np.sum(stack * sin_d, axis=0)
        C = np.sum(stack * cos_d, axis=0)

        # Raw modulation amplitude.
        B_f = (2.0 / n_phase) * np.sqrt(S * S + C * C)

        # Optional median smoothing of wrapped phase.
        phi = np.arctan2(S, C)
        if median_phase_filter:
            phi = median_filter(phi, size=3)

        # Blur B for more stable masking (B_eff).
        B_eff = cv2.GaussianBlur(B_f.astype(np.float32), (7, 7), 0)

        # Gamma from blurred modulation.
        gamma_f = np.zeros_like(A_f, dtype=np.float32)
        nz = A_f > 1e-6
        gamma_f[nz] = B_eff[nz] / A_f[nz]

        # Saturation mask to reject clipped pixels.
        I_min = stack.min(axis=0)
        I_max = stack.max(axis=0)
        saturated = (I_max > 250) | (I_min < 5)

        # Thresholds for normal scans.
        mask_f = (gamma_f > TH1_GAMMA) & (B_eff > TH2_MOD) & (~saturated)

        # Store outputs by frequency.
        result.phi_wrapped[f] = phi
        result.A[f] = A_f
        result.B[f] = B_f
        result.gamma[f] = gamma_f
        result.mask[f] = mask_f
        result.saturated[f] = saturated

    return result
