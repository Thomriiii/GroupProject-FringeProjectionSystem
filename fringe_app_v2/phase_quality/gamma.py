"""Projector gamma correction for generated fringe patterns."""

from __future__ import annotations

from typing import Any

import numpy as np


def gamma_from_config(config: dict[str, Any]) -> float:
    value = float(config.get("gamma", (config.get("phase_quality", {}) or {}).get("gamma", 2.2)))
    if not 1.8 <= value <= 2.4:
        raise ValueError(f"gamma must be between 1.8 and 2.4, got {value}")
    return value


def apply_gamma(pattern: np.ndarray, gamma: float) -> np.ndarray:
    arr = np.asarray(pattern)
    is_integer = np.issubdtype(arr.dtype, np.integer)
    if is_integer:
        normalised = arr.astype(np.float32) / 255.0
    else:
        normalised = arr.astype(np.float32)
    corrected = np.clip(normalised, 0.0, 1.0) ** (1.0 / float(gamma))
    if is_integer:
        return np.clip(np.rint(corrected * 255.0), 0, 255).astype(arr.dtype)
    return corrected.astype(arr.dtype, copy=False)
