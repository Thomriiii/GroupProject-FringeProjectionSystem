"""
masking.py

Mask cleanup and merging for structured-light PSP.
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Dict, List

MaskDict = Dict[int, np.ndarray]


def clean_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
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
