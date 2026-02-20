"""
Core data models for scans and run metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Tuple, Dict, Any, Optional


Orientation = Literal["vertical", "horizontal"]


@dataclass(slots=True)
class ScanParams:
    """
    Parameters for a single fringe scan.
    """
    n_steps: int
    frequency: float
    orientation: Orientation
    brightness: float
    resolution: Tuple[int, int]
    settle_ms: int
    save_patterns: bool
    preview_fps: float
    projector_screen_index: Optional[int] = None
    phase_convention: str = "atan2(-S,C)"
    exposure_us: Optional[int] = None
    analogue_gain: Optional[float] = None
    awb_mode: Optional[str] = None
    awb_enable: Optional[bool] = None
    ae_enable: Optional[bool] = None
    iso: Optional[int] = None
    brightness_offset: float = 0.5
    contrast: float = 1.0
    min_intensity: float = 0.10
    frequencies: Optional[list[float]] = None
    frequency_semantics: str = "cycles_across_dimension"
    phase_origin_rad: float = 0.0
    auto_normalise: bool = True
    expert_mode: bool = False
    quality_retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Persist a single canonical field for scan frequencies in new runs.
        data["frequencies"] = self.get_frequencies()
        data.pop("frequency", None)
        # Explicit metadata for phase-to-projector mapping.
        data["phase_axis"] = "x" if self.orientation == "vertical" else "y"
        data["phase_origin_rad"] = float(self.phase_origin_rad)
        if self.frequency_semantics == "cycles_across_dimension":
            data["cycles_by_frequency"] = [float(f) for f in self.get_frequencies()]
        elif self.frequency_semantics == "pixels_per_period":
            # In this mode ScanParams.frequency values encode period in pixels.
            data["period_px_by_frequency"] = [float(f) for f in self.get_frequencies()]
        return data

    @property
    def N(self) -> int:
        return int(self.n_steps)

    def get_frequencies(self) -> list[float]:
        if self.frequencies:
            return list(self.frequencies)
        return [float(self.frequency)]


@dataclass(slots=True)
class RunMeta:
    """
    Metadata persisted to meta.json for each run.
    """
    run_id: str
    params: Dict[str, Any]
    started_at: str
    finished_at: str | None
    status: str
    error: str | None
    device_info: Dict[str, Any]
    total_frames: int
    saved_frames: int
    preview_enabled: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
