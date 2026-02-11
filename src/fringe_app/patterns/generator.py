"""Fringe pattern generator."""

from __future__ import annotations

import numpy as np

from fringe_app.core.models import ScanParams


class FringePatternGenerator:
    """
    Generate N-step phase-shifted sinusoidal fringe patterns.
    """

    def generate_sequence(self, params: ScanParams) -> list[np.ndarray]:
        width, height = params.resolution
        n = max(1, int(params.n_steps))
        freq = float(params.frequency)
        contrast = float(np.clip(params.contrast, 0.0, 1.5))
        brightness_offset = float(np.clip(params.brightness_offset, 0.0, 1.0))

        if params.orientation == "vertical":
            axis = np.linspace(0.0, 1.0, width, endpoint=False)
            grid = np.tile(axis, (height, 1))
        else:
            axis = np.linspace(0.0, 1.0, height, endpoint=False)
            grid = np.tile(axis[:, None], (1, width))

        patterns: list[np.ndarray] = []
        for k in range(n):
            phase = 2.0 * np.pi * (k / n)
            signal = np.sin(2.0 * np.pi * freq * grid + phase)
            img = brightness_offset + 0.5 * contrast * signal
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            patterns.append(img_u8)

        return patterns
