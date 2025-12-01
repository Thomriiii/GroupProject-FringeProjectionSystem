#!/usr/bin/env python3
"""
patterns_graycode/graycode_generator.py

Generate Gray code patterns (normal + inverse) for projector calibration.
Provides numpy masks and pygame.Surface objects.
"""

from __future__ import annotations

import math
import os
import numpy as np
import pygame


def _gray_code(n: int) -> int:
    return n ^ (n >> 1)


def _surface_from_mask(mask: np.ndarray) -> pygame.Surface:
    """
    Pygame expects array shape (W, H, 3); swap axes accordingly.
    """
    arr = np.stack([mask] * 3, axis=2).swapaxes(0, 1)
    return pygame.surfarray.make_surface(arr)


def generate_graycode_patterns(width: int, height: int, out_dir: str | None = None):
    """
    Generate Gray code patterns for X and Y axes (with inverse).

    Returns
    -------
    patterns : list[tuple[str, pygame.Surface]]
        List of (label, surface) in order.
    bits_x, bits_y : int
        Number of bits used for X/Y axes.
    """
    bits_x = math.ceil(math.log2(width))
    bits_y = math.ceil(math.log2(height))

    patterns: list[tuple[str, pygame.Surface]] = []

    # X axis
    for bit in range(bits_x):
        stripe = np.zeros((height, width), dtype=np.uint8)
        for x in range(width):
            gray = _gray_code(x)
            bitval = (gray >> bit) & 1
            stripe[:, x] = 255 if bitval else 0
        stripe_inv = 255 - stripe

        patterns.append((f"x_bit{bit:02d}_on", _surface_from_mask(stripe)))
        patterns.append((f"x_bit{bit:02d}_off", _surface_from_mask(stripe_inv)))

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, f"x_bit{bit:02d}_on.npy"), stripe)
            np.save(os.path.join(out_dir, f"x_bit{bit:02d}_off.npy"), stripe_inv)

    # Y axis
    for bit in range(bits_y):
        stripe = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            gray = _gray_code(y)
            bitval = (gray >> bit) & 1
            stripe[y, :] = 255 if bitval else 0
        stripe_inv = 255 - stripe

        patterns.append((f"y_bit{bit:02d}_on", _surface_from_mask(stripe)))
        patterns.append((f"y_bit{bit:02d}_off", _surface_from_mask(stripe_inv)))

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, f"y_bit{bit:02d}_on.npy"), stripe)
            np.save(os.path.join(out_dir, f"y_bit{bit:02d}_off.npy"), stripe_inv)

    return patterns, bits_x, bits_y
