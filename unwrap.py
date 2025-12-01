"""
unwrap.py

Temporal multi-frequency phase unwrapping.

This module performs the classical temporal phase unwrapping used
in multi-frequency PSP scanners:

    Φ_low   -> unwrap of lowest frequency
    Φ_high = wrapped_high + 2π * round( r * Φ_low - wrapped_high ) / (2π)

where r = f_high / f_low.

The result is a fully unwrapped phase map at the highest frequency,
plus intermediate unwrapped maps if needed.

Input:
  - Frequency → wrapped phase map (from psp.PSPResult.phi_wrapped)
  - Frequency → cleaned mask (from masking.merge_frequency_masks)

Output:
  - final unwrapped phase (highest frequency)
  - optionally intermediate unwrapped maps

This module *does not perform any spatial unwrapping* — only temporal.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List


# Type aliases
PhaseDict = Dict[int, np.ndarray]
MaskDict = Dict[int, np.ndarray]


class UnwrapResult:
    """
    Container for final and intermediate unwrapped phases.

    Attributes
    ----------
    Phi : dict[int -> HxW float]
        Unwrapped phase per frequency.
    Phi_final : HxW float
        Final unwrapped phase (highest frequency).
    mask_final : HxW bool
        Validity mask for the final phase (same as mask of f_low..f_high merged).
    """
    def __init__(self):
        self.Phi: PhaseDict = {}
        self.Phi_final = None
        self.mask_final = None


# =====================================================================
# TEMPORAL UNWRAPPING CORE
# =====================================================================

def temporal_unwrap(
    phi_wrapped: PhaseDict,
    mask_merged: MaskDict,
    freqs: List[int],
) -> UnwrapResult:
    """
    Perform temporal unwrapping from low → high frequency.

    Parameters
    ----------
    phi_wrapped : dict
        phi_wrapped[f] is the wrapped phase for frequency f.
    mask_merged : dict
        mask_merged[f] is the cleaned mask for each frequency f.
        These masks are combined externally (majority vote, morphological cleanup)
        before being passed here.
    freqs : list of int
        Sorted frequency list, e.g. [4, 8, 16, 32].

    Returns
    -------
    UnwrapResult
        Contains unwrapped phases at each step and final unwrapped phase.
    """
    result = UnwrapResult()
    freqs_sorted = sorted(freqs)

    # Initial phase is just the wrapped phase at lowest frequency
    f0 = freqs_sorted[0]
    Phi_prev = phi_wrapped[f0]
    result.Phi[f0] = Phi_prev

    # Combined mask — start with the lowest frequency
    mask_final = mask_merged[f0].copy()

    for i in range(1, len(freqs_sorted)):
        f_low = freqs_sorted[i - 1]
        f_high = freqs_sorted[i]

        wrapped_high = phi_wrapped[f_high]
        Phi_low = result.Phi[f_low]

        # Ratio between frequencies
        r = f_high / f_low

        # Temporal correction term
        k = np.round((r * Phi_low - wrapped_high) / (2.0 * np.pi))

        Phi_high = wrapped_high + 2.0 * np.pi * k

        # Save
        result.Phi[f_high] = Phi_high

        # Merge masks (logical AND of cleaned per-frequency masks)
        mask_final &= mask_merged[f_high]

    # Output assignments
    result.Phi_final = result.Phi[freqs_sorted[-1]]
    result.mask_final = mask_final

    return result
