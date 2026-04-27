"""Pattern generation wrappers for v2 pipeline stages."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from fringe_app.core.models import ScanParams
from fringe_app.patterns.generator import FringePatternGenerator


class PatternService:
    def __init__(self) -> None:
        self._generator = FringePatternGenerator()

    def for_orientation(self, params: ScanParams, orientation: str) -> ScanParams:
        return replace(params, orientation=orientation)  # type: ignore[arg-type]

    def sequence(self, params: ScanParams, frequency: float) -> list[np.ndarray]:
        return self._generator.generate_sequence(params, frequency=frequency)

    def metadata(self, params: ScanParams, frequency: float) -> dict:
        return self._generator.pattern_metadata(params, frequency=frequency)
