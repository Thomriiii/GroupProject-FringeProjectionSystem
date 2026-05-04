"""Fringe pattern generator."""

from __future__ import annotations

import numpy as np

from fringe_app.core.models import ScanParams


class FringePatternGenerator:
    """
    Generate N-step phase-shifted sinusoidal fringe patterns.
    """

    def __init__(self) -> None:
        self._gamma_lut: np.ndarray | None = None
        self._gamma_enabled: bool = True

    def set_gamma_lut(self, lut: np.ndarray | None, enabled: bool = True) -> None:
        if lut is None:
            self._gamma_lut = None
            self._gamma_enabled = bool(enabled)
            return
        arr = np.asarray(lut, dtype=np.uint8).reshape(-1)
        if arr.shape[0] != 256:
            raise ValueError("Gamma LUT must contain 256 entries")
        self._gamma_lut = arr.copy()
        self._gamma_enabled = bool(enabled)

    def clear_gamma_lut(self) -> None:
        self._gamma_lut = None

    def generate_sequence(self, params: ScanParams, frequency: float | None = None) -> list[np.ndarray]:
        width, height = params.resolution
        n = max(1, int(params.n_steps))
        freq = float(params.frequency if frequency is None else frequency)
        contrast = float(np.clip(params.contrast, 0.0, 1.5))
        brightness_offset = float(np.clip(params.brightness_offset, 0.0, 1.0))
        min_intensity = float(np.clip(params.min_intensity, 0.0, 1.0))

        if params.orientation == "vertical":
            axis = np.linspace(0.0, 1.0, width, endpoint=False)
            grid = np.tile(axis, (height, 1))
        else:
            axis = np.linspace(0.0, 1.0, height, endpoint=False)
            grid = np.tile(axis[:, None], (1, width))

        patterns: list[np.ndarray] = []
        for k in range(n):
            phase = 2.0 * np.pi * (k / n)
            signal = np.cos(2.0 * np.pi * freq * grid + phase)
            # Base cosine in [0, 1].
            w = 0.5 + 0.5 * signal
            # Enforce minimum intensity by lifting the floor.
            img = min_intensity + (1.0 - min_intensity) * w
            # Apply contrast as a blend toward mid-gray (does not dip below min_intensity).
            img = (1.0 - contrast) * 0.5 + contrast * img
            # Apply brightness offset as a final add then clamp.
            img = np.clip(img + (brightness_offset - 0.5), min_intensity, 1.0)
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            if self._gamma_lut is not None and self._gamma_enabled:
                img_u8 = self._gamma_lut[img_u8]
            patterns.append(img_u8)

        return patterns

    def pattern_metadata(self, params: ScanParams, frequency: float | None = None) -> dict:
        """
        Metadata describing phase origin and frequency semantics for one sequence.
        """
        freq = float(params.frequency if frequency is None else frequency)
        width, height = params.resolution
        axis = "x" if params.orientation == "vertical" else "y"
        dim = width if axis == "x" else height
        semantics = str(params.frequency_semantics)
        if semantics == "pixels_per_period":
            period_px = max(freq, 1e-6)
            cycles = float(dim) / period_px
        else:
            cycles = freq
            period_px = float(dim) / max(cycles, 1e-6)
        return {
            "phase_origin_rad": float(params.phase_origin_rad),
            "axis": axis,
            "frequency_semantics": semantics,
            "cycles": float(cycles),
            "period_px": float(period_px),
        }
