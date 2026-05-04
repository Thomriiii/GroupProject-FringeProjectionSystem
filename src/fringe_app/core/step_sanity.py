"""Per-frequency phase-step stack sanity checks."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(slots=True)
class StepSanityThresholds:
    min_step_mean_dn: float = 5.0
    min_step_mean_ratio: float = 0.15
    max_step_mean_ratio: float = 2.50
    low_light_dn: float = 20.0
    min_step_mean_dn_low_light: float = 1.0
    use_center_patch: bool = True
    center_patch_frac: float = 0.30


@dataclass(slots=True)
class StepSanityReport:
    ok: bool
    step_means: list[float]
    step_stds: list[float]
    reasons: list[str]
    roi_median_dn: float = 0.0
    mode: str = "normal"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_step_stack(stack_u8: np.ndarray, roi_mask: np.ndarray, thr: StepSanityThresholds) -> StepSanityReport:
    if stack_u8.ndim != 3:
        raise ValueError("stack_u8 must be (N,H,W)")
    if roi_mask.ndim != 2:
        raise ValueError("roi_mask must be (H,W)")
    if stack_u8.shape[1:] != roi_mask.shape:
        raise ValueError("stack_u8 and roi_mask shapes are incompatible")

    roi = roi_mask.astype(bool)
    if not np.any(roi):
        roi = np.ones_like(roi, dtype=bool)
    if thr.use_center_patch:
        h, w = roi.shape
        frac = float(np.clip(thr.center_patch_frac, 0.05, 1.0))
        ph = max(8, int(round(h * frac)))
        pw = max(8, int(round(w * frac)))
        y0 = (h - ph) // 2
        x0 = (w - pw) // 2
        patch = np.zeros_like(roi, dtype=bool)
        patch[y0:y0 + ph, x0:x0 + pw] = True
        roi = roi & patch
        if not np.any(roi):
            roi = roi_mask.astype(bool)

    means: list[float] = []
    stds: list[float] = []
    reasons: list[str] = []
    roi_vals = stack_u8[:, roi].astype(np.float32)
    roi_median = float(np.median(roi_vals)) if roi_vals.size else 0.0
    low_light = roi_median < thr.low_light_dn
    min_step_dn = thr.min_step_mean_dn_low_light if low_light else thr.min_step_mean_dn
    for i in range(stack_u8.shape[0]):
        vals = stack_u8[i][roi]
        m = float(np.mean(vals)) if vals.size else 0.0
        s = float(np.std(vals)) if vals.size else 0.0
        means.append(m)
        stds.append(s)

    med = float(np.median(means)) if means else 0.0
    ok = True
    for i, m in enumerate(means):
        if m < min_step_dn:
            ok = False
            reasons.append(f"step{i} mean too low: {m:.2f}")
        if med > 0:
            r = m / med
            if r < thr.min_step_mean_ratio:
                ok = False
                reasons.append(f"step{i} mean ratio too low: {r:.3f}")
            if r > thr.max_step_mean_ratio:
                ok = False
                reasons.append(f"step{i} mean ratio too high: {r:.3f}")

    return StepSanityReport(
        ok=ok,
        step_means=means,
        step_stds=stds,
        reasons=reasons,
        roi_median_dn=roi_median,
        mode="low_light" if low_light else "normal",
    )
