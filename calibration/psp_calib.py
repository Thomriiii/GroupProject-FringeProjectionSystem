"""
psp_calib.py

PSP for camera/projector calibration only.

Compared to normal scans, this uses lower frequencies and looser thresholds
to retain more valid pixels on tilted calibration boards.
"""

from __future__ import annotations
import numpy as np
import cv2
from typing import Dict, Tuple, List

IntensityDict = Dict[Tuple[int, int], np.ndarray]
PhaseDict = Dict[int, np.ndarray]
FloatImage = np.ndarray

CALIB_TH1_GAMMA = 0.05   # Relaxed gamma threshold.
CALIB_TH2_MOD   = 4.0    # Relaxed modulation threshold.
CALIB_BLUR = True
CALIB_BLUR_KERNEL = (5, 5)


def run_psp_calibration(
    I_dict: IntensityDict,
    freqs: List[int],
    n_phase: int,
) -> tuple[PhaseDict, Dict[int, np.ndarray]]:
    """
    PSP for calibration.

    Parameters
    ----------
    I_dict : dict[(f, k) -> HxW float]
        Intensity images per frequency f and phase index k.
    freqs : list[int]
        Frequencies to use for calibration (e.g. [4, 8]).
        High frequencies (16, 32) should be excluded.
    n_phase : int
        Number of phase steps (e.g. 4).

    Returns
    -------
    phi_wrapped : dict[f] -> HxW float
        Wrapped phase per frequency.
    masks : dict[f] -> HxW bool
        Relaxed per-frequency masks.
    """
    freqs_sorted = sorted(freqs)
    deltas = 2.0 * np.pi * np.arange(n_phase) / n_phase
    sin_d = np.sin(deltas)[:, None, None]
    cos_d = np.cos(deltas)[:, None, None]

    phi_wrapped: PhaseDict = {}
    masks: Dict[int, np.ndarray] = {}

    for f in freqs_sorted:
        stack = np.stack([I_dict[(f, k)] for k in range(n_phase)], axis=0)

        # Optional blur to stabilize noisy or low-modulation areas.
        if CALIB_BLUR:
            for k in range(n_phase):
                stack[k] = cv2.GaussianBlur(stack[k], CALIB_BLUR_KERNEL, 0)

        # Mean intensity.
        A_f = stack.mean(axis=0)

        # Phase sums.
        S = np.sum(stack * sin_d, axis=0)
        C = np.sum(stack * cos_d, axis=0)

        # Modulation amplitude.
        B = (2.0 / n_phase) * np.sqrt(S * S + C * C)

        # Wrapped phase.
        phi = np.arctan2(S, C)
        phi_wrapped[f] = phi

        # Blurred modulation for a stable mask.
        B_eff = cv2.GaussianBlur(B.astype(np.float32), (7, 7), 0)

        # Gamma = modulation / average brightness.
        gamma = np.zeros_like(A_f, dtype=np.float32)
        nz = A_f > 1e-6
        gamma[nz] = B_eff[nz] / A_f[nz]

        # Very tolerant mask (no saturation check; add if needed).
        mask_f = (gamma > CALIB_TH1_GAMMA) & (B_eff > CALIB_TH2_MOD)

        masks[f] = mask_f

    return phi_wrapped, masks
