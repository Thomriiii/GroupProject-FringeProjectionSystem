"""
unwrap.py

Temporal multi-frequency phase unwrapping.

This module performs the standard temporal unwrapping used in multi-frequency
PSP scanners:

    Phi_low  = unwrap of lowest frequency
    Phi_high = wrapped_high + 2*pi * round((r * Phi_low - wrapped_high) / (2*pi))

where r = f_high / f_low.

Inputs are per-frequency wrapped phase maps and their masks. Outputs are
the final unwrapped phase at the highest frequency plus intermediates.
This module does not perform spatial unwrapping except for the optional
low-frequency line unwrap used to recover fringe order.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List

# Constant used when aligning spatially unwrapped lines.
TWO_PI = 2.0 * np.pi


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
        Validity mask for the final phase (merged across all frequencies).
    """

    def __init__(self):
        self.Phi: PhaseDict = {}
        self.Phi_final = None
        self.mask_final = None


def _unwrap_phase_spatial(phi_wrapped: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    """
    Unwrap the lowest-frequency phase spatially along one axis.

    This restores the coarse fringe order before temporal unwrapping, which
    improves absolute projector coordinates.
    """
    phi_wrapped = phi_wrapped.astype(np.float64)
    mask_bool = mask.astype(bool)
    out = np.full_like(phi_wrapped, np.nan, dtype=np.float64)

    H, W = phi_wrapped.shape

    if axis == 1:
        # Unwrap each row independently (vertical fringes vary along x).
        for y in range(H):
            valid = mask_bool[y]
            if valid.sum() < 2:
                continue
            unwrapped_line = np.unwrap(phi_wrapped[y, valid])
            out[y, valid] = unwrapped_line

        # Align row offsets to a common reference using overlap.
        ref_idx = next((i for i in range(H) if np.isfinite(out[i]).any()), None)
        if ref_idx is not None:
            ref_line = out[ref_idx]
            ref_valid = np.isfinite(ref_line) & mask_bool[ref_idx]
            for y in range(H):
                if y == ref_idx or not np.isfinite(out[y]).any():
                    continue
                cur_valid = np.isfinite(out[y]) & mask_bool[y] & ref_valid
                if cur_valid.sum() < 8:
                    continue
                diff = np.nanmedian(out[y, cur_valid] - ref_line[cur_valid])
                offset_cycles = np.round(diff / TWO_PI)
                out[y, mask_bool[y]] -= offset_cycles * TWO_PI

    elif axis == 0:
        # Unwrap each column independently (horizontal fringes vary along y).
        for x in range(W):
            valid = mask_bool[:, x]
            if valid.sum() < 2:
                continue
            unwrapped_col = np.unwrap(phi_wrapped[valid, x])
            out[valid, x] = unwrapped_col

        # Align column offsets.
        ref_idx = next((i for i in range(W) if np.isfinite(out[:, i]).any()), None)
        if ref_idx is not None:
            ref_col = out[:, ref_idx]
            ref_valid = np.isfinite(ref_col) & mask_bool[:, ref_idx]
            for x in range(W):
                if x == ref_idx or not np.isfinite(out[:, x]).any():
                    continue
                cur_valid = np.isfinite(out[:, x]) & mask_bool[:, x] & ref_valid
                if cur_valid.sum() < 8:
                    continue
                diff = np.nanmedian(out[cur_valid, x] - ref_col[cur_valid])
                offset_cycles = np.round(diff / TWO_PI)
                out[mask_bool[:, x], x] -= offset_cycles * TWO_PI

    else:
        raise ValueError("axis must be 0 (vertical unwrap) or 1 (horizontal unwrap)")

    # Fall back to wrapped values where unwrapping failed.
    out = np.where(np.isfinite(out), out, phi_wrapped)
    return out


def temporal_unwrap(
    phi_wrapped: PhaseDict,
    mask_merged: MaskDict,
    freqs: List[int],
    spatial_axis: int | None = None,
) -> UnwrapResult:
    """
    Perform temporal unwrapping from low to high frequency.

    Parameters
    ----------
    phi_wrapped : dict
        phi_wrapped[f] is the wrapped phase for frequency f.
    mask_merged : dict
        mask_merged[f] is the cleaned mask for each frequency f.
    freqs : list of int
        Sorted frequency list, for example [4, 8, 16, 32].
    spatial_axis : {0, 1} or None
        Optionally unwrap the lowest frequency spatially along the given axis
        before temporal unwrapping. This restores the coarse fringe order
        needed for absolute projector coordinates.

    Returns
    -------
    UnwrapResult
        Contains unwrapped phases at each step and final unwrapped phase.
    """
    result = UnwrapResult()
    freqs_sorted = sorted(freqs)

    # Start from the lowest frequency and optionally recover fringe order spatially.
    f0 = freqs_sorted[0]
    Phi_prev = phi_wrapped[f0]
    if spatial_axis is not None:
        Phi_prev = _unwrap_phase_spatial(Phi_prev, mask_merged[f0], axis=spatial_axis)
    result.Phi[f0] = Phi_prev

    # Combined mask starts at the lowest frequency.
    mask_final = mask_merged[f0].copy()

    for i in range(1, len(freqs_sorted)):
        f_low = freqs_sorted[i - 1]
        f_high = freqs_sorted[i]

        wrapped_high = phi_wrapped[f_high]
        Phi_low = result.Phi[f_low]

        # Ratio between frequencies.
        r = f_high / f_low

        # Temporal correction term.
        k = np.round((r * Phi_low - wrapped_high) / TWO_PI)
        Phi_high = wrapped_high + TWO_PI * k

        # Save.
        result.Phi[f_high] = Phi_high

        # Merge masks (logical AND of cleaned per-frequency masks).
        mask_final &= mask_merged[f_high]

    # Output assignments.
    result.Phi_final = result.Phi[freqs_sorted[-1]]
    result.mask_final = mask_final

    return result
