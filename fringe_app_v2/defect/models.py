"""Rule-based defect labels built from geometric features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DefectClassifierConfig:
    linear_aspect_ratio: float = 5.0
    linear_max_minor_mm: float = 0.0
    point_aspect_min: float = 0.8
    point_max_major_mm: float = 2.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DefectClassifierConfig":
        cfg = config.get("defect", {}) or {}
        defaults = cls()
        return cls(
            linear_aspect_ratio=float(cfg.get("linear_aspect_ratio", defaults.linear_aspect_ratio)),
            linear_max_minor_mm=float(cfg.get("linear_max_minor_mm", defaults.linear_max_minor_mm)),
            point_aspect_min=float(cfg.get("point_aspect_min", defaults.point_aspect_min)),
            point_max_major_mm=float(cfg.get("point_max_major_mm", defaults.point_max_major_mm)),
        )


def classify_defect(features: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    cfg = DefectClassifierConfig.from_config(config)
    dimensions = features.get("dimensions", {}) or {}
    width = float(dimensions.get("width", 0.0))
    height = float(dimensions.get("height", 0.0))
    major = float(dimensions.get("major_axis", max(width, height)))
    minor = float(dimensions.get("minor_axis", min(width, height)))
    aspect = major / max(minor, 1e-9)

    label = "surface"
    reason = "remaining geometry after linear and point checks"
    linear_size_ok = cfg.linear_max_minor_mm <= 0 or minor <= cfg.linear_max_minor_mm
    point_square_like = minor / max(major, 1e-9) >= cfg.point_aspect_min
    if aspect >= cfg.linear_aspect_ratio and linear_size_ok:
        label = "scratch"
        reason = "elongated geometry"
    elif point_square_like and major <= cfg.point_max_major_mm:
        label = "pit"
        reason = "small near-isotropic geometry"

    out = dict(features)
    out["type"] = label
    out["classification"] = {
        "method": "rule_based_geometry_v1",
        "label": label,
        "reason": reason,
        "rules": {
            "linear_aspect_ratio": cfg.linear_aspect_ratio,
            "linear_max_minor_mm": cfg.linear_max_minor_mm,
            "point_aspect_min": cfg.point_aspect_min,
            "point_max_major_mm": cfg.point_max_major_mm,
        },
    }
    return out
