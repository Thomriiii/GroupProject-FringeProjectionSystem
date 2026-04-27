"""Stable projector generator wrapper."""

from __future__ import annotations

from fringe_app.core.models import ScanParams
from fringe_app.patterns.generator import FringePatternGenerator


def generate_sequence(params: ScanParams, frequency: float | None = None):
    """Generate patterns via the existing generator unchanged."""
    return FringePatternGenerator().generate_sequence(params, frequency=frequency)
