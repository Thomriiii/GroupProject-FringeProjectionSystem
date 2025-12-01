#!/usr/bin/env python3
"""
calibration/projector_checkerboard.py

Utilities for projecting a checkerboard from the projector and computing
projector-space corner coordinates for calibration.
"""

from __future__ import annotations

import numpy as np
import pygame


def make_checkerboard_surface(width: int, height: int, pattern_size=(8, 6), square_px: int = 80):
    """
    Create a pygame.Surface with a centered checkerboard.

    Parameters
    ----------
    width, height : int
        Projector resolution.
    pattern_size : tuple[int, int]
        Inner corners (cols, rows), e.g. (8, 6).
    square_px : int
        Size of one checker square in projector pixels.
    """
    cols, rows = pattern_size
    squares_x = cols + 1
    squares_y = rows + 1
    board_w = squares_x * square_px
    board_h = squares_y * square_px

    offset_x = int((width - board_w) // 2)
    offset_y = int((height - board_h) // 2)

    surface = pygame.Surface((width, height))
    surface.fill((128, 128, 128))

    for y in range(squares_y):
        for x in range(squares_x):
            color = 255 if (x + y) % 2 == 0 else 0
            rect = pygame.Rect(
                offset_x + x * square_px,
                offset_y + y * square_px,
                square_px,
                square_px,
            )
            pygame.draw.rect(surface, (color, color, color), rect)

    return surface


def projector_inner_corners(width: int, height: int, pattern_size=(8, 6), square_px: int = 80) -> np.ndarray:
    """
    Compute projector pixel coordinates of the inner checkerboard corners.

    Returns
    -------
    proj_pts : (N x 2) float32
        Projector pixel coordinates of inner corners, row-major.
    """
    cols, rows = pattern_size
    squares_x = cols + 1
    squares_y = rows + 1
    board_w = squares_x * square_px
    board_h = squares_y * square_px

    offset_x = (width - board_w) / 2.0
    offset_y = (height - board_h) / 2.0

    xs = offset_x + (np.arange(cols) + 1) * square_px
    ys = offset_y + (np.arange(rows) + 1) * square_px
    grid_x, grid_y = np.meshgrid(xs, ys)
    proj_pts = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1).astype(np.float32)
    return proj_pts
