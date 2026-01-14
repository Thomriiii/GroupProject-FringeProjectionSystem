"""
masking.py

Mask cleanup and merging helpers for structured-light PSP.
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Dict, List

MaskDict = Dict[int, np.ndarray]


def clean_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply morphological open/close to remove speckles and fill small gaps.
    """
    k = np.ones((kernel_size, kernel_size), np.uint8)
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return (m > 0)


def merge_frequency_masks(
    masks_raw: MaskDict,
    freqs: List[int],
    kernel_size: int = 3,
    majority_threshold: int | None = None,
) -> MaskDict:
    """
    Clean per-frequency masks and merge them with a majority vote.

    Parameters
    ----------
    masks_raw : dict[int, ndarray]
        Raw boolean masks for each frequency.
    freqs : list[int]
        Frequencies in the mask set.
    kernel_size : int
        Kernel size for morphological cleaning.
    majority_threshold : int or None
        Minimum number of votes to keep a pixel. Defaults to simple majority.

    Returns
    -------
    dict[int, ndarray]
        Merged mask replicated for each frequency.
    """
    freqs_sorted = sorted(freqs)
    cleaned_masks: MaskDict = {}

    for f in freqs_sorted:
        cleaned_masks[f] = clean_mask(masks_raw[f], kernel_size)

    mask_stack = np.stack([cleaned_masks[f] for f in freqs_sorted], axis=0)

    if majority_threshold is None:
        majority_threshold = len(freqs_sorted) // 2 + 1

    votes = mask_stack.sum(axis=0)
    merged = votes >= majority_threshold

    final_masks = {f: merged.copy() for f in freqs_sorted}
    return final_masks
