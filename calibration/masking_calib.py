"""
masking_calib.py

Calibration-only mask merging.

Differences from normal masking:
  - Uses a **logical OR** across frequencies instead of majority/AND.
  - A pixel is valid if **ANY** low frequency mask is True.
  - Returns a dict[f] -> merged_mask so it can be passed to temporal
    unwrapping (unwrap.temporal_unwrap).
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List

from core.masking import clean_mask  # reuse your existing morphology


MaskDict = Dict[int, np.ndarray]


def merge_masks_calibration(
    masks_raw: MaskDict,
    freqs: List[int],
    kernel_size: int = 3,
) -> MaskDict:
    """
    Merge per-frequency masks for calibration.

    Parameters
    ----------
    masks_raw : dict[f] -> HxW bool
        Raw per-frequency masks from psp_calib.
    freqs : list[int]
        Frequencies used (e.g. [4, 8]).
    kernel_size : int
        Kernel for clean_mask (morphological open/close).

    Returns
    -------
    mask_merged : dict[f] -> HxW bool
        For each f, the SAME merged mask, which is the OR of all
        cleaned per-frequency masks. This is convenient for passing
        into unwrap.temporal_unwrap().
    """
    freqs_sorted = sorted(freqs)

    # Clean each mask individually
    cleaned: MaskDict = {}
    for f in freqs_sorted:
        cleaned[f] = clean_mask(masks_raw[f], kernel_size=kernel_size)

    # OR them all together
    merged = None
    for f in freqs_sorted:
        m = cleaned[f]
        if merged is None:
            merged = m.copy()
        else:
            merged |= m

    if merged is None:
        raise ValueError("merge_masks_calibration: no frequencies provided")

    # For convenience, return dict[f] -> same merged mask
    mask_merged: MaskDict = {f: merged.copy() for f in freqs_sorted}
    return mask_merged
