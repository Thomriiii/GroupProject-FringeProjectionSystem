"""
Convert unwrapped phase maps to projector pixel coordinates.

PHYSICAL PRINCIPLE:
  Structured light encodes distance via phase. Each projector pixel (u, v) emits
  light at a specific phase Φ(u, v). When light reflects from a surface at distance d,
  the surface's phase-time delay is proportional to d.

  By capturing at multiple frequencies, we can unwrap phase to get continuous
  phase values (not wrapped to [0, 2π)).

  Phase → Projector Pixel Mapping:
    u = (phase_vertical / (2π × cycles)) × projector_width
    v = (phase_horizontal / (2π × cycles)) × projector_height

WHY TWO DIRECTIONS:
  - Vertical patterns: stripes are vertical lines, brightness varies left-right → encode projector X (u)
  - Horizontal patterns: stripes are horizontal lines, brightness varies top-bottom → encode projector Y (v)
  - Both are needed for 2D correspondence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class UVMap:
    """Projector coordinate maps from unwrapped phase."""
    u: np.ndarray              # HxW projector X coordinates
    v: np.ndarray              # HxW projector Y coordinates
    mask: np.ndarray           # HxW boolean mask (valid pixels)
    meta: dict[str, Any]       # Metadata (dimensions, ranges, etc.)


def phase_to_projector_coords(
    phi_horizontal: np.ndarray,
    phi_vertical: np.ndarray,
    mask_horizontal: np.ndarray,
    mask_vertical: np.ndarray,
    projector_width: int,
    projector_height: int,
    frequency_u: float,
    frequency_v: float,
    frequency_semantics: str = "cycles_across_dimension",
    phase_origin_u_rad: float = 0.0,
    phase_origin_v_rad: float = 0.0,
    roi_mask: np.ndarray | None = None,
) -> UVMap:
    """
    Convert unwrapped phase maps to projector pixel coordinates.

    INPUTS:
      phi_horizontal: Phase from horizontal patterns (stripes vary top-bottom) → projector Y (v)
      phi_vertical: Phase from vertical patterns (stripes vary left-right) → projector X (u)
      mask_horizontal: Boolean mask for horizontal phase (True = valid)
      mask_vertical: Boolean mask for vertical phase (True = valid)
      frequency_u/v: Modulation frequency (cycles across dimension or pixels/period)
      phase_origin_u_rad, phase_origin_v_rad: Phase offsets (typically 0)

    OUTPUT:
      UVMap with u[y,x] = projector X, v[y,x] = projector Y

    ALGORITHM:
      1. Compute number of cycles from frequency
      2. For each valid pixel, map phase to [0, 2π × cycles)
      3. Normalize: coord = (phase / (2π × cycles)) × dimension
      4. Apply ROI mask if provided
      5. Compute statistics and return
    """
    if phi_horizontal.shape != phi_vertical.shape:
        raise ValueError("Horizontal and vertical phase shapes must match")

    h, w = phi_horizontal.shape

    # Initialize maps with NaN
    u = np.full((h, w), np.nan, dtype=np.float32)
    v = np.full((h, w), np.nan, dtype=np.float32)

    # Compute number of cycles
    if frequency_semantics == "pixels_per_period":
        # frequency means pixels per period (wavelength)
        cycles_u = float(projector_width) / max(float(frequency_u), 1e-6)
        cycles_v = float(projector_height) / max(float(frequency_v), 1e-6)
    else:
        # frequency_semantics == "cycles_across_dimension"
        cycles_u = float(frequency_u)
        cycles_v = float(frequency_v)

    # Compute phase spans
    two_pi = 2.0 * np.pi
    span_u = two_pi * cycles_u
    span_v = two_pi * cycles_v

    # Vertical patterns vary left-right → projector X (u)
    valid_u = np.isfinite(phi_vertical) & mask_vertical.astype(bool)
    if np.any(valid_u):
        phi_u = phi_vertical[valid_u] - float(phase_origin_u_rad)
        u[valid_u] = (phi_u / span_u) * float(projector_width)

    # Horizontal patterns vary top-bottom → projector Y (v)
    valid_v = np.isfinite(phi_horizontal) & mask_horizontal.astype(bool)
    if np.any(valid_v):
        phi_v = phi_horizontal[valid_v] - float(phase_origin_v_rad)
        v[valid_v] = (phi_v / span_v) * float(projector_height)

    # Combine masks
    mask = np.isfinite(u) & np.isfinite(v)

    # Apply ROI mask if provided
    if roi_mask is not None:
        if roi_mask.shape != (h, w):
            raise ValueError(f"ROI mask shape {roi_mask.shape} does not match phase {(h, w)}")
        mask = mask & roi_mask.astype(bool)
        u[~mask] = np.nan
        v[~mask] = np.nan

    # Compute statistics for debugging
    valid_pixels = mask
    u_vals = u[valid_pixels]
    v_vals = v[valid_pixels]

    if u_vals.size > 0:
        u_min, u_max = float(np.min(u_vals)), float(np.max(u_vals))
        u_percentiles = np.percentile(u_vals, [1, 99])
        u_p01, u_p99 = float(u_percentiles[0]), float(u_percentiles[1])
        u_range = u_max - u_min
    else:
        u_min = u_max = u_p01 = u_p99 = u_range = np.nan

    if v_vals.size > 0:
        v_min, v_max = float(np.min(v_vals)), float(np.max(v_vals))
        v_percentiles = np.percentile(v_vals, [1, 99])
        v_p01, v_p99 = float(v_percentiles[0]), float(v_percentiles[1])
        v_range = v_max - v_min
    else:
        v_min = v_max = v_p01 = v_p99 = v_range = np.nan

    # Detect edge/boundary pixels
    u_edge_pct = float(np.mean((u_vals < 1.0) | (u_vals > float(projector_width - 2))) if u_vals.size else 1.0)
    v_edge_pct = float(np.mean((v_vals < 1.0) | (v_vals > float(projector_height - 2))) if v_vals.size else 1.0)
    u_zero_pct = float(np.mean(u_vals <= 1e-6) if u_vals.size else 1.0)
    v_zero_pct = float(np.mean(v_vals <= 1e-6) if v_vals.size else 1.0)

    meta = {
        "projector_width": int(projector_width),
        "projector_height": int(projector_height),
        "frequency_semantics": str(frequency_semantics),
        "freq_u": float(frequency_u),
        "freq_v": float(frequency_v),
        "cycles_u": float(cycles_u),
        "cycles_v": float(cycles_v),
        "phase_origin_u_rad": float(phase_origin_u_rad),
        "phase_origin_v_rad": float(phase_origin_v_rad),
        "valid_pixel_count": int(np.count_nonzero(valid_pixels)),
        "valid_ratio": float(np.count_nonzero(valid_pixels) / valid_pixels.size),
        "u_min": u_min,
        "u_max": u_max,
        "u_p01": u_p01,
        "u_p99": u_p99,
        "u_range": u_range,
        "v_min": v_min,
        "v_max": v_max,
        "v_p01": v_p01,
        "v_p99": v_p99,
        "v_range": v_range,
        "u_edge_pct": u_edge_pct,
        "v_edge_pct": v_edge_pct,
        "u_zero_pct": u_zero_pct,
        "v_zero_pct": v_zero_pct,
    }

    return UVMap(u=u, v=v, mask=mask, meta=meta)


def validate_uv_map(
    uv_map: UVMap,
    min_valid_ratio: float = 0.03,
    min_u_range: float = 40.0,
    min_v_range: float = 40.0,
    max_edge_pct: float = 0.10,
    max_zero_pct: float = 0.01,
) -> tuple[bool, list[str]]:
    """
    Validate UV map for quality issues.

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []
    meta = uv_map.meta

    # Check valid pixel ratio
    if meta["valid_ratio"] < min_valid_ratio:
        errors.append(
            f"Valid pixel ratio {meta['valid_ratio']:.2%} < {min_valid_ratio:.2%}. "
            f"Check ROI or thresholds."
        )

    # Check coordinate range
    if meta["u_range"] < min_u_range:
        errors.append(
            f"U range {meta['u_range']:.1f}px < {min_u_range:.1f}px. "
            f"Object too small or projector misconfigured."
        )
    if meta["v_range"] < min_v_range:
        errors.append(
            f"V range {meta['v_range']:.1f}px < {min_v_range:.1f}px. "
            f"Object too small or projector misconfigured."
        )

    # Check for edge pinning (saturated coordinates)
    if meta["u_edge_pct"] > max_edge_pct:
        errors.append(
            f"U edge pixels {meta['u_edge_pct']:.1%} > {max_edge_pct:.1%}. "
            f"Coordinates may be clipping. Check phase origin."
        )
    if meta["v_edge_pct"] > max_edge_pct:
        errors.append(
            f"V edge pixels {meta['v_edge_pct']:.1%} > {max_edge_pct:.1%}. "
            f"Coordinates may be clipping. Check phase origin."
        )

    # Check for zero-value masking
    if meta["u_zero_pct"] > max_zero_pct:
        errors.append(
            f"U zero pixels {meta['u_zero_pct']:.2%} > {max_zero_pct:.2%}. "
            f"Possibly unwrap or mask issues."
        )
    if meta["v_zero_pct"] > max_zero_pct:
        errors.append(
            f"V zero pixels {meta['v_zero_pct']:.2%} > {max_zero_pct:.2%}. "
            f"Possibly unwrap or mask issues."
        )

    is_valid = len(errors) == 0
    return is_valid, errors
