"""
patterns.py

Pattern generation for structured-light fringe projection.

This version adds:
  - BRIGHTNESS_SCALE to dim the projected image (prevents saturation).
  - CONTRAST_SCALE to slightly reduce sinusoid contrast if needed.
  - Support for vertical and horizontal fringe orientations.
"""

from __future__ import annotations

import numpy as np
import pygame
from typing import Dict, List, Tuple


# Global controls for image intensity
BRIGHTNESS_SCALE = 0.38   # overall brightness scaler (0..1)
CONTRAST_SCALE   = 0.70   # sinusoid contrast scaler (0..1)


SurfaceDict = Dict[int, List[pygame.Surface]]
RawPatternDict = Dict[int, List[np.ndarray]]


def _apply_projector_gamma(img: np.ndarray, gamma: float | None) -> np.ndarray:
    """
    Optionally apply projector gamma compensation.

    img : float array in [0, 1]
    gamma : projector gamma (if known). We apply img ** (1/gamma).
    """
    if gamma is None:
        return img

    eps = 1e-6
    img = np.clip(img, eps, 1.0 - eps)
    return img ** (1.0 / gamma)


def generate_psp_patterns(
    width: int,
    height: int,
    freqs: List[int],
    n_phase: int,
    gamma_proj: float | None = None,
    orientation: str = "vertical",
) -> Tuple[SurfaceDict, RawPatternDict]:
    """
    Generate sinusoidal PSP patterns for all frequencies and phases.

    Parameters
    ----------
    width, height : int
        Projector resolution.
    freqs : list[int]
        Number of periods across the varying dimension.
    n_phase : int
        Number of phase steps (e.g. 4).
    gamma_proj : float or None
        Projector gamma for compensation (optional).
    orientation : {"vertical", "horizontal"}
        - "vertical"   : fringes vary along x (columns)
        - "horizontal" : fringes vary along y (rows)

    Returns
    -------
    patterns : dict[freq -> list[pygame.Surface]]
        Displayable pygame surfaces.
    raw_patterns : dict[freq -> list[np.ndarray]]
        Linear float32 patterns in [0..1] before brightness scaling & gamma.
    """
    orientation = orientation.lower()
    if orientation not in ("vertical", "horizontal"):
        raise ValueError(f"Invalid orientation '{orientation}', use 'vertical' or 'horizontal'.")

    patterns: SurfaceDict = {}
    raw_patterns: RawPatternDict = {}

    deltas = 2.0 * np.pi * np.arange(n_phase) / n_phase

    # Coordinate axis depending on orientation
    if orientation == "vertical":
        axis_len = width
    else:  # "horizontal"
        axis_len = height

    coords = np.linspace(0.0, 1.0, axis_len, endpoint=False)

    for f in freqs:
        freq_surfaces: List[pygame.Surface] = []
        freq_raw: List[np.ndarray] = []

        for delta in deltas:
            phase = 2.0 * np.pi * f * coords + delta
            stripe = 0.5 + 0.5 * CONTRAST_SCALE * np.sin(phase)  # 0..1

            if orientation == "vertical":
                # stripe varies along x, replicate along y
                img = np.tile(stripe, (height, 1)).astype(np.float32)
            else:
                # stripe varies along y, replicate along x
                img = np.tile(stripe[:, None], (1, width)).astype(np.float32)

            # Store linear pattern before brightness scaling
            freq_raw.append(img.copy())

            # Apply projector gamma compensation (optional)
            img = _apply_projector_gamma(img, gamma_proj)

            # Global brightness scaling
            img_scaled = img * BRIGHTNESS_SCALE

            # Convert to 8-bit RGB
            img_u8 = np.clip(img_scaled * 255.0, 0, 255).astype(np.uint8)
            rgb = np.dstack([img_u8] * 3)

            surf = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
            freq_surfaces.append(surf)

        patterns[f] = freq_surfaces
        raw_patterns[f] = freq_raw

    return patterns, raw_patterns


def generate_midgrey_surface(
    width: int,
    height: int,
    level: float = 0.5,
    gamma_proj: float | None = None,
) -> pygame.Surface:
    """
    Generate a uniform mid-grey surface for auto-exposure.

    'level' is the linear intensity before brightness scaling.
    """
    level = float(np.clip(level, 0.0, 1.0))
    img = np.full((height, width), level, dtype=np.float32)

    img = _apply_projector_gamma(img, gamma_proj)

    # Apply same brightness scaling as the fringes
    img *= BRIGHTNESS_SCALE

    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    rgb = np.dstack([img_u8] * 3)
    surf = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
    return surf
