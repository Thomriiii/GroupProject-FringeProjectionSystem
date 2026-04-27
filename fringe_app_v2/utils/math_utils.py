"""Small numerical helpers shared by pipeline stages."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim == 3 and image.shape[2] >= 3:
        rgb = image[:, :, :3].astype(np.float32)
        return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    raise ValueError(f"Unsupported image shape: {image.shape}")


def to_gray_u8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(to_gray_float(image)), 0, 255).astype(np.uint8)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value
