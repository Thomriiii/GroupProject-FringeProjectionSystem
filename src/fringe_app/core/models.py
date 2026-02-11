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
    iso: Optional[int] = None
    brightness_offset: float = 0.5
    contrast: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def N(self) -> int:
        return int(self.n_steps)


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
