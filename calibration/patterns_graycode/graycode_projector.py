#!/usr/bin/env python3
"""
patterns_graycode/graycode_projector.py

Helper to generate Gray code patterns as pygame surfaces for projection.
"""

from __future__ import annotations

import os
from calibration.patterns_graycode.graycode_generator import generate_graycode_patterns


def prepare_graycode_patterns(width: int, height: int, pattern_dir: str | None = None):
    """
    Generate Gray code patterns for the given projector size.

    Returns
    -------
    patterns : list[(str, surface)]
    bits_x, bits_y : int
    """
    if pattern_dir:
        os.makedirs(pattern_dir, exist_ok=True)
    patterns, bits_x, bits_y = generate_graycode_patterns(width, height, out_dir=pattern_dir)
    return patterns, bits_x, bits_y
