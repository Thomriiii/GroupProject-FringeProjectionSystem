"""Projector-camera gamma calibration utilities (calibration-only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
from PIL import Image, ImageDraw


@dataclass(slots=True)
class GammaConfig:
    enabled: bool = True
    samples: int = 33
    settle_ms: int = 120
    flush_frames: int = 1
    board_roi_only: bool = False

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "GammaConfig":
        c = dict(cfg or {})
        return cls(
            enabled=bool(c.get("enabled", True)),
            samples=max(8, int(c.get("samples", 33))),
            settle_ms=max(0, int(c.get("settle_ms", 120))),
            flush_frames=max(0, int(c.get("flush_frames", 1))),
            board_roi_only=bool(c.get("board_roi_only", False)),
        )


def _fit_gamma(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    Fit y = a * x^gamma + b in normalized [0,1] domain (coarse robust fit).
    """
    xx = np.clip(x.astype(np.float64), 0.0, 1.0)
    yy = np.clip(y.astype(np.float64), 0.0, 1.0)
    # Robust linearization around offset b from low-intensity percentile.
    b = float(np.percentile(yy, 2))
    z = np.clip(yy - b, 1e-6, 1.0)
    lx = np.log(np.clip(xx, 1e-6, 1.0))
    lz = np.log(z)
    A = np.column_stack([lx, np.ones_like(lx)])
    coeff, *_ = np.linalg.lstsq(A, lz, rcond=None)
    gamma = float(np.clip(coeff[0], 0.2, 5.0))
    loga = float(coeff[1])
    a = float(np.clip(np.exp(loga), 1e-6, 10.0))
    return a, gamma, b


def _invert_model_to_lut(a: float, gamma: float, b: float) -> np.ndarray:
    """
    Invert y=a*x^gamma+b to x=((y-b)/a)^(1/gamma) for y target in [0,1].
    Output LUT maps intended linear level->projector command [0..255].
    """
    target = np.linspace(0.0, 1.0, 256, endpoint=True)
    base = np.clip((target - b) / max(a, 1e-9), 0.0, 1.0)
    inv = np.power(base, 1.0 / max(gamma, 1e-9))
    lut = np.clip(np.rint(inv * 255.0), 0, 255).astype(np.uint8)
    return lut


def _save_gamma_plot(levels: np.ndarray, means: np.ndarray, fit_means: np.ndarray, out_path: Path) -> None:
    w, h = 960, 600
    m = 70
    img = Image.new("RGB", (w, h), (20, 20, 24))
    draw = ImageDraw.Draw(img)
    draw.text((20, 18), "Gamma fit", fill=(235, 235, 235))

    def _pt(x: float, y: float) -> tuple[int, int]:
        xx = int(m + x * (w - 2 * m))
        yy = int(h - m - y * (h - 2 * m))
        return xx, yy

    # Axes
    draw.line((_pt(0.0, 0.0), _pt(1.0, 0.0)), fill=(130, 130, 130), width=1)
    draw.line((_pt(0.0, 0.0), _pt(0.0, 1.0)), fill=(130, 130, 130), width=1)

    # Measured points
    for x, y in zip(levels, means):
        p = _pt(float(x), float(y))
        draw.ellipse((p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2), fill=(255, 180, 90))

    # Fitted curve
    last = None
    for x, y in zip(levels, fit_means):
        p = _pt(float(x), float(y))
        if last is not None:
            draw.line((last, p), fill=(90, 210, 120), width=2)
        last = p

    img.save(out_path)


def calibrate_gamma_lut(
    controller,
    session_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Project ramp levels, capture means, fit gamma, save LUT and diagnostics.
    """
    gcfg = GammaConfig.from_config((cfg.get("calibration", {}) or {}).get("gamma", {}))
    out_dir = session_dir / "results" / "gamma"
    out_dir.mkdir(parents=True, exist_ok=True)
    pcal = cfg.get("projector_calibration", {}) or {}
    cap_cfg = (pcal.get("capture", {}) or {})
    dn_levels = np.linspace(0, 255, gcfg.samples, endpoint=True).astype(np.uint8)
    measured: list[float] = []
    # Rebind projector display context in the current worker thread before
    # presenting gamma frames to avoid EGL thread-affinity issues.
    controller.set_projector_calibration_light(enabled=True, dn=0)
    try:
        for dn in dn_levels:
            controller.set_projector_calibration_light(enabled=True, dn=int(dn))
            if gcfg.settle_ms > 0:
                import time
                time.sleep(float(gcfg.settle_ms) / 1000.0)
            img = controller.capture_single_frame(
                flush_frames=gcfg.flush_frames,
                exposure_us=(pcal.get("camera", {}) or {}).get("exposure_us"),
                analogue_gain=(pcal.get("camera", {}) or {}).get("analogue_gain"),
                awb_enable=(pcal.get("camera", {}) or {}).get("awb_enable"),
                settle_ms=0,
            )
            if img.ndim == 3:
                gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            else:
                gray = img.astype(np.float32)
            measured.append(float(np.mean(gray) / 255.0))
    finally:
        # Restore white-board illumination for interactive calibration UI.
        try:
            controller.set_projector_calibration_light(
                enabled=True,
                dn=int(cap_cfg.get("checkerboard_white_dn", 230)),
            )
        except Exception:
            pass

    x = dn_levels.astype(np.float64) / 255.0
    y = np.asarray(measured, dtype=np.float64)
    a, gamma, b = _fit_gamma(x, y)
    fit = np.clip(a * np.power(np.clip(x, 0.0, 1.0), gamma) + b, 0.0, 1.0)
    lut = _invert_model_to_lut(a, gamma, b)
    np.save(out_dir / "gamma_lut.npy", lut.astype(np.uint8))
    _save_gamma_plot(x, y, fit, out_dir / "gamma_plot.png")
    payload = {
        "enabled": bool(gcfg.enabled),
        "calibration_ran": True,
        "samples": int(gcfg.samples),
        "a": float(a),
        "gamma": float(gamma),
        "b": float(b),
        "levels": [int(v) for v in dn_levels.tolist()],
        "measured_mean_norm": [float(v) for v in y.tolist()],
        "fit_mean_norm": [float(v) for v in fit.tolist()],
        "lut_path": str((out_dir / "gamma_lut.npy").relative_to(session_dir)),
        "plot_path": str((out_dir / "gamma_plot.png").relative_to(session_dir)),
    }
    (out_dir / "gamma_fit.json").write_text(json.dumps(payload, indent=2))
    return payload


def load_gamma_lut(session_dir: Path) -> np.ndarray | None:
    p = session_dir / "results" / "gamma" / "gamma_lut.npy"
    if not p.exists():
        return None
    try:
        lut = np.load(p)
    except Exception:
        return None
    lut = np.asarray(lut)
    if lut.shape != (256,):
        return None
    return np.clip(lut.astype(np.uint8), 0, 255)
