"""Projector UV map generation from unwrapped phase."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(slots=True)
class UvResult:
    u: np.ndarray
    v: np.ndarray
    mask_uv: np.ndarray
    meta: dict


def _save_mask_png(mask: np.ndarray, path: Path) -> None:
    img = np.where(mask.astype(bool), 255, 0).astype(np.uint8)
    Image.fromarray(img).save(path)


def _save_debug_fixed(values: np.ndarray, valid: np.ndarray, lo: float, hi: float, path: Path) -> None:
    out = np.zeros(values.shape, dtype=np.uint8)
    span = max(hi - lo, 1e-6)
    out[valid] = np.clip(((values[valid] - lo) / span) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(path)


def _save_debug_autoscale(values: np.ndarray, valid: np.ndarray, path: Path) -> None:
    out = np.zeros(values.shape, dtype=np.uint8)
    if np.any(valid):
        vals = values[valid]
        lo, hi = np.percentile(vals, [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        out[valid] = np.clip(((values[valid] - lo) / (hi - lo)) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(path)


def _save_uv_overlay(base_rgb: np.ndarray, mask_uv: np.ndarray, path: Path) -> None:
    if base_rgb.ndim == 2:
        rgb = np.stack([base_rgb, base_rgb, base_rgb], axis=2).astype(np.uint8)
    else:
        rgb = base_rgb[:, :, :3].astype(np.uint8).copy()
    # Tint valid UV pixels cyan for quick visual check.
    rgb[mask_uv, 1] = np.clip(rgb[mask_uv, 1].astype(np.int16) + 70, 0, 255).astype(np.uint8)
    rgb[mask_uv, 2] = np.clip(rgb[mask_uv, 2].astype(np.int16) + 70, 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def phase_to_uv(
    phi_abs_vertical: np.ndarray,
    phi_abs_horizontal: np.ndarray,
    freq_u: float,
    freq_v: float,
    proj_width: int,
    proj_height: int,
    mask_u: np.ndarray,
    mask_v: np.ndarray,
    roi_mask: np.ndarray | None,
    frequency_semantics: str = "cycles_across_dimension",
    phase_origin_u_rad: float = 0.0,
    phase_origin_v_rad: float = 0.0,
    gate_cfg: dict | None = None,
) -> UvResult:
    if phi_abs_vertical.shape != phi_abs_horizontal.shape:
        raise ValueError("vertical/horizontal phase shapes do not match")

    u = np.full(phi_abs_vertical.shape, np.nan, dtype=np.float32)
    v = np.full(phi_abs_horizontal.shape, np.nan, dtype=np.float32)

    valid_u = np.isfinite(phi_abs_vertical) & mask_u.astype(bool)
    valid_v = np.isfinite(phi_abs_horizontal) & mask_v.astype(bool)

    two_pi = 2.0 * np.pi
    if frequency_semantics == "pixels_per_period":
        # If frequency means period (px), cycles = dimension/period.
        cycles_u = float(proj_width) / max(float(freq_u), 1e-6)
        cycles_v = float(proj_height) / max(float(freq_v), 1e-6)
    else:
        cycles_u = float(freq_u)
        cycles_v = float(freq_v)

    # Use explicit phase-origin + modulo mapping to avoid negative clipping plateaus.
    span_u = two_pi * cycles_u
    span_v = two_pi * cycles_v
    phi_u = np.mod(phi_abs_vertical[valid_u] - float(phase_origin_u_rad), span_u)
    phi_v = np.mod(phi_abs_horizontal[valid_v] - float(phase_origin_v_rad), span_v)
    u[valid_u] = (phi_u / span_u) * float(proj_width)
    v[valid_v] = (phi_v / span_v) * float(proj_height)

    mask_uv = np.isfinite(u) & np.isfinite(v)
    if roi_mask is not None:
        mask_uv &= roi_mask.astype(bool)
        u[~mask_uv] = np.nan
        v[~mask_uv] = np.nan

    valid = mask_uv
    u_vals = u[valid]
    v_vals = v[valid]
    if u_vals.size:
        u_p01, u_p99 = np.percentile(u_vals, [1, 99])
        v_p01, v_p99 = np.percentile(v_vals, [1, 99])
    else:
        u_p01 = u_p99 = v_p01 = v_p99 = np.nan
    u_range = float(u_p99 - u_p01) if np.isfinite(u_p01) and np.isfinite(u_p99) else 0.0
    v_range = float(v_p99 - v_p01) if np.isfinite(v_p01) and np.isfinite(v_p99) else 0.0
    u_edge_pct = float(np.mean((u_vals < 1.0) | (u_vals > float(proj_width - 2))) if u_vals.size else 1.0)
    v_edge_pct = float(np.mean((v_vals < 1.0) | (v_vals > float(proj_height - 2))) if v_vals.size else 1.0)
    u_zero_pct = float(np.mean(u_vals <= 1e-6) if u_vals.size else 1.0)
    v_zero_pct = float(np.mean(v_vals <= 1e-6) if v_vals.size else 1.0)
    gate_cfg = gate_cfg or {}
    gate_enabled = bool(gate_cfg.get("enabled", True))
    min_u_range_px = float(gate_cfg.get("min_u_range_px", 40.0))
    min_v_range_px = float(gate_cfg.get("min_v_range_px", 40.0))
    max_edge_pct = float(gate_cfg.get("max_edge_pct", 0.10))
    max_zero_pct = float(gate_cfg.get("max_zero_pct", 0.01))
    min_valid_ratio = float(gate_cfg.get("min_valid_ratio", 0.03))

    checks = {
        "valid_ratio": float(np.count_nonzero(valid) / valid.size) >= min_valid_ratio,
        "u_range": u_range >= min_u_range_px,
        "v_range": v_range >= min_v_range_px,
        "u_edge_pct": u_edge_pct <= max_edge_pct,
        "v_edge_pct": v_edge_pct <= max_edge_pct,
        "u_zero_pct": u_zero_pct <= max_zero_pct,
        "v_zero_pct": v_zero_pct <= max_zero_pct,
    }
    failed_checks = [k for k, ok in checks.items() if not ok]
    hints: list[str] = []
    if "u_range" in failed_checks or "v_range" in failed_checks:
        hints.append(
            "Object occupies small projector area; move object closer / enlarge ROI / or accept small-object mode."
        )
    if "valid_ratio" in failed_checks:
        hints.append(
            "ROI/mask too strict; consider B_thresh adjustment or exposure increase."
        )
    if "u_edge_pct" in failed_checks or "v_edge_pct" in failed_checks or "u_zero_pct" in failed_checks or "v_zero_pct" in failed_checks:
        hints.append(
            "Mapping pinned to edges; check phase origin / modulo mapping / clipping."
        )
    uv_gate_ok = True if not gate_enabled else (len(failed_checks) == 0)

    uv_meta = {
        "projector_width": int(proj_width),
        "projector_height": int(proj_height),
        "frequency_semantics": frequency_semantics,
        "freq_u": float(freq_u),
        "freq_v": float(freq_v),
        "cycles_u": float(cycles_u),
        "cycles_v": float(cycles_v),
        "phase_origin_u_rad": float(phase_origin_u_rad),
        "phase_origin_v_rad": float(phase_origin_v_rad),
        "valid_ratio": float(np.count_nonzero(valid) / valid.size),
        "u_min": float(np.min(u[valid])) if np.any(valid) else None,
        "u_max": float(np.max(u[valid])) if np.any(valid) else None,
        "v_min": float(np.min(v[valid])) if np.any(valid) else None,
        "v_max": float(np.max(v[valid])) if np.any(valid) else None,
        "u_p01": float(u_p01) if np.isfinite(u_p01) else None,
        "u_p99": float(u_p99) if np.isfinite(u_p99) else None,
        "v_p01": float(v_p01) if np.isfinite(v_p01) else None,
        "v_p99": float(v_p99) if np.isfinite(v_p99) else None,
        "u_range": float(u_range),
        "v_range": float(v_range),
        "u_edge_pct": float(u_edge_pct),
        "v_edge_pct": float(v_edge_pct),
        "u_zero_pct": float(u_zero_pct),
        "v_zero_pct": float(v_zero_pct),
        "uv_gate_ok": uv_gate_ok,
        "uv_gate_thresholds": {
            "enabled": gate_enabled,
            "min_u_range_px": min_u_range_px,
            "min_v_range_px": min_v_range_px,
            "max_edge_pct": max_edge_pct,
            "max_zero_pct": max_zero_pct,
            "min_valid_ratio": min_valid_ratio,
        },
        "uv_gate_failed_checks": failed_checks,
        "uv_gate_hints": hints,
    }

    return UvResult(u=u, v=v, mask_uv=mask_uv.astype(bool), meta=uv_meta)


def save_uv_outputs(out_dir: Path, result: UvResult, base_frame: np.ndarray | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "u.npy", result.u.astype(np.float32))
    np.save(out_dir / "v.npy", result.v.astype(np.float32))
    np.save(out_dir / "mask_uv.npy", result.mask_uv.astype(np.bool_))
    _save_mask_png(result.mask_uv, out_dir / "mask_uv.png")

    valid = result.mask_uv.astype(bool)
    _save_debug_fixed(result.u, valid, 0.0, float(result.meta["projector_width"]), out_dir / "u_debug_fixed.png")
    _save_debug_fixed(result.v, valid, 0.0, float(result.meta["projector_height"]), out_dir / "v_debug_fixed.png")
    _save_debug_autoscale(result.u, valid, out_dir / "u_debug_autoscale.png")
    _save_debug_autoscale(result.v, valid, out_dir / "v_debug_autoscale.png")

    if base_frame is not None:
        _save_uv_overlay(base_frame, result.mask_uv, out_dir / "uv_overlay.png")

    (out_dir / "uv_meta.json").write_text(json.dumps(result.meta, indent=2))
