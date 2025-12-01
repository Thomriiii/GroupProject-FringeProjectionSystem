#!/usr/bin/env python3
"""
calibration/graycode.py

Generate and decode Gray code patterns for projector calibration.
"""

from __future__ import annotations

import math
import numpy as np
import pygame


def _gray_code(n: int) -> int:
    return n ^ (n >> 1)


def _surface_from_mask(mask: np.ndarray) -> pygame.Surface:
    """
    Create a pygame surface from an HxW mask. Pygame expects array shape (W, H, 3).
    """
    arr = np.stack([mask]*3, axis=2).swapaxes(0, 1)
    return pygame.surfarray.make_surface(arr)


def generate_graycode_patterns(width: int, height: int):
    """
    Generate Gray code patterns (normal + inverse) for X and Y axes.

    Returns
    -------
    patterns_x : list[pygame.Surface]
    patterns_y : list[pygame.Surface]
    bits_x : int
    bits_y : int
    """
    bits_x = math.ceil(math.log2(width))
    bits_y = math.ceil(math.log2(height))

    patterns_x = []
    for bit in range(bits_x):
        period = 2 ** (bit + 1)
        stripe = np.zeros((height, width), dtype=np.uint8)
        for x in range(width):
            gray = _gray_code(x)
            bitval = (gray >> bit) & 1
            stripe[:, x] = 255 if bitval else 0
        stripe_inv = 255 - stripe
        patterns_x.append(_surface_from_mask(stripe))
        patterns_x.append(_surface_from_mask(stripe_inv))

    patterns_y = []
    for bit in range(bits_y):
        stripe = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            gray = _gray_code(y)
            bitval = (gray >> bit) & 1
            stripe[y, :] = 255 if bitval else 0
        stripe_inv = 255 - stripe
        patterns_y.append(_surface_from_mask(stripe))
        patterns_y.append(_surface_from_mask(stripe_inv))

    return patterns_x, patterns_y, bits_x, bits_y


def _decode_axis(frames: list[np.ndarray], bits: int, dim: int) -> np.ndarray:
    """
    Decode Gray code frames for one axis. frames expected as [bit0, bit0_inv, bit1, bit1_inv, ...]
    Returns decoded coordinate per pixel (float32), invalid pixels as -1.
    """
    if len(frames) != bits * 2:
        raise ValueError("Frame count mismatch for Gray code decoding.")

    H, W = frames[0].shape
    codes = np.zeros((H, W), dtype=np.int32)
    valid = np.ones((H, W), dtype=bool)

    for bit in range(bits):
        f_on = frames[2 * bit]
        f_off = frames[2 * bit + 1]
        # Simple threshold: on > off
        bit_mask = f_on > f_off
        # If difference is tiny, mark invalid
        uncertain = np.abs(f_on - f_off) < 5
        valid &= ~uncertain
        codes |= (bit_mask.astype(np.int32) << bit)

    # Convert Gray to binary
    binary = np.zeros_like(codes)
    for bit in reversed(range(bits)):
        if bit == bits - 1:
            binary |= (codes >> bit) & 1
        else:
            binary |= ((codes >> bit) & 1) ^ (binary >> (bit + 1) << (bit + 1))

    coords = binary.astype(np.float32)
    coords[~valid] = -1.0
    # Clamp to dimension
    coords = np.clip(coords, -1.0, dim - 1)
    return coords


def decode_graycode(frames_x: list[np.ndarray], frames_y: list[np.ndarray], width: int, height: int, bits_x: int, bits_y: int):
    """
    Decode captured Gray code frames into projector coordinates.

    Returns
    -------
    proj_x : np.ndarray float32 (H x W)
    proj_y : np.ndarray float32 (H x W)
    mask   : np.ndarray bool (H x W)
    """
    proj_x = _decode_axis(frames_x, bits_x, width)
    proj_y = _decode_axis(frames_y, bits_y, height)
    mask = (proj_x >= 0) & (proj_y >= 0)
    return proj_x, proj_y, mask
