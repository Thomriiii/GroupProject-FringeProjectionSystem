#!/usr/bin/env python3
"""
patterns_graycode/graycode_decode.py

Decode captured Gray code patterns into projector coordinates.
"""

from __future__ import annotations

import numpy as np


def _decode_axis(frames: list[np.ndarray], bits: int, dim: int, thresh: float = 5.0):
    """
    Decode Gray code for one axis.
    frames: [bit0_on, bit0_off, bit1_on, bit1_off, ...]
    """
    if len(frames) != bits * 2:
        raise ValueError("Frame count mismatch.")

    H, W = frames[0].shape
    codes = np.zeros((H, W), dtype=np.int32)
    valid = np.ones((H, W), dtype=bool)

    for bit in range(bits):
        f_on = frames[2 * bit].astype(np.float32)
        f_off = frames[2 * bit + 1].astype(np.float32)
        bit_mask = f_on > f_off
        uncertain = np.abs(f_on - f_off) < thresh
        valid &= ~uncertain
        codes |= (bit_mask.astype(np.int32) << bit)

    # Gray -> binary (vectorized)
    binary = codes.copy()
    shift = 1
    while shift < bits:
        binary ^= (codes >> shift)
        shift += 1

    coords = binary.astype(np.float32)
    coords[~valid] = -1.0
    coords = np.clip(coords, -1.0, dim - 1)
    return coords, valid


def decode_graycode(images: dict[str, np.ndarray], width: int, height: int, bits_x: int, bits_y: int, thresh: float = 5.0):
    """
    Decode Gray code images into projector X/Y maps.
    images: dict[label -> img], labels like x_bit00_on/off, y_bit00_on/off
    """
    # Ensure we have required labels
    for bit in range(bits_x):
        if f"x_bit{bit:02d}_on" not in images or f"x_bit{bit:02d}_off" not in images:
            raise RuntimeError(f"Missing X bit {bit} images for Gray code decoding.")
    for bit in range(bits_y):
        if f"y_bit{bit:02d}_on" not in images or f"y_bit{bit:02d}_off" not in images:
            raise RuntimeError(f"Missing Y bit {bit} images for Gray code decoding.")
    frames_x = []
    frames_y = []
    for bit in range(bits_x):
        frames_x.append(images[f"x_bit{bit:02d}_on"])
        frames_x.append(images[f"x_bit{bit:02d}_off"])
    for bit in range(bits_y):
        frames_y.append(images[f"y_bit{bit:02d}_on"])
        frames_y.append(images[f"y_bit{bit:02d}_off"])

    proj_x, valid_x = _decode_axis(frames_x, bits_x, width, thresh)
    proj_y, valid_y = _decode_axis(frames_y, bits_y, height, thresh)
    mask = valid_x & valid_y & (proj_x >= 0) & (proj_y >= 0)
    return proj_x, proj_y, mask
