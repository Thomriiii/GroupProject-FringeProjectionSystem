"""
geometry.py

Utilities for mapping unwrapped PSP phase maps to projector pixel
coordinates (u, v). The mapping assumes:
  - Vertical pattern set varies along projector X (width) → u
  - Horizontal pattern set varies along projector Y (height) → v
The highest frequency determines scaling from phase to pixels.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple

TWO_PI = 2.0 * np.pi


def _best_cycle_offset(coord_map: np.ndarray, mask: np.ndarray, size: int, freq: float) -> float:
    """
    Find an integer-cycle offset that maximises how many samples fall inside
    the projector bounds. Shifting by one cycle corresponds to size/freq pixels.
    """
    mask_valid = mask & np.isfinite(coord_map)
    if not mask_valid.any():
        return 0.0

    step = size / float(freq)
    best_k = 0
    best_count = -1

    # Search a generous window of possible cycle offsets
    for k in range(-int(freq) * 2, int(freq) * 2 + 1):
        shifted = coord_map + k * step
        count = np.count_nonzero(mask_valid & (shifted >= -0.5 * step) & (shifted <= size + 0.5 * step))
        if count > best_count:
            best_count = count
            best_k = k

    return best_k * step


def _affine_normalise_minmax(coord_map: np.ndarray, mask: np.ndarray, size: int) -> tuple[float, float]:
    """
    Compute a scale/offset so the valid coord_map spans the projector pixel range
    using simple min/max → [0..size).
    Returns (scale, offset) such that coord' = coord * scale + offset.
    """
    valid = mask & np.isfinite(coord_map)
    if valid.sum() < 16:
        return 1.0, 0.0

    vals = coord_map[valid]
    lo = float(vals.min())
    hi = float(vals.max())
    if hi - lo < 1e-6:
        return 1.0, 0.0

    scale = (size - 1.0) / (hi - lo)
    offset = -lo * scale
    return float(scale), float(offset)


def compute_projector_uv_from_phase(
    phase_vert: np.ndarray,
    phase_horiz: np.ndarray,
    freqs: Iterable[int],
    proj_size: Tuple[int, int],
    mask_vert: np.ndarray | None = None,
    mask_horiz: np.ndarray | None = None,
    offset_u: float = 0.0,
    offset_v: float = 0.0,
    auto_cycle_alignment: bool = True,
    apply_affine_normalisation: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert unwrapped vertical/horizontal phase maps into projector pixel maps.

    Parameters
    ----------
    phase_vert : ndarray (H, W)
        Final unwrapped vertical phase (varies along projector X).
    phase_horiz : ndarray (H, W)
        Final unwrapped horizontal phase (varies along projector Y).
    freqs : iterable[int]
        Frequencies used in PSP; highest frequency defines scale.
    proj_size : (width, height)
        Projector resolution that patterns were generated for.
    mask_vert, mask_horiz : ndarray bool or None
        Validity masks for vertical/horizontal phases. If None, treated as all True.
    offset_u, offset_v : float
        Optional offsets in projector pixels (defaults zero).
    auto_cycle_alignment : bool
        If True, shifts the maps by integer fringe periods so the majority of
        samples fall inside the projector bounds. This anchors the absolute
        fringe order and prevents negative/overflowing coordinates.
    apply_affine_normalisation : bool
        If True, compute a per-axis affine (scale + offset) so the observed UV
        distribution maps min→0, max→proj_size.

    Returns
    -------
    u_map, v_map, mask_final : float32 arrays (H, W)
        Dense projector coordinates. Invalid pixels set to NaN.
    """
    freqs_sorted = sorted(freqs)
    f_high = float(freqs_sorted[-1])

    proj_w, proj_h = proj_size

    mask_v = np.ones_like(phase_vert, dtype=bool) if mask_vert is None else mask_vert
    mask_h = np.ones_like(phase_horiz, dtype=bool) if mask_horiz is None else mask_horiz
    mask_final = mask_v & mask_h

    denom_u = (TWO_PI * f_high)
    denom_v = (TWO_PI * f_high)

    u_map = (phase_vert / denom_u) * proj_w + offset_u
    v_map = (phase_horiz / denom_v) * proj_h + offset_v

    # Optional integer-cycle shift to keep coordinates within projector bounds
    if auto_cycle_alignment:
        u_shift = _best_cycle_offset(u_map, mask_final, proj_w, f_high)
        v_shift = _best_cycle_offset(v_map, mask_final, proj_h, f_high)
        if abs(u_shift) > 1e-6:
            u_map += u_shift
        if abs(v_shift) > 1e-6:
            v_map += v_shift

    if apply_affine_normalisation:
        raise RuntimeError(
            "Affine normalisation of projector UVs breaks calibrated geometry "
            "and must remain disabled."
        )

    u_map = u_map.astype(np.float32)
    v_map = v_map.astype(np.float32)

    # Mask out invalids with NaN for clarity downstream and enforce projector bounds
    in_bounds = (
        (u_map >= 0.0) & (u_map <= proj_w - 1) &
        (v_map >= 0.0) & (v_map <= proj_h - 1)
    )
    valid_mask = mask_final & in_bounds
    u_map[~valid_mask] = np.nan
    v_map[~valid_mask] = np.nan

    mask_final = valid_mask

    return u_map, v_map, mask_final
