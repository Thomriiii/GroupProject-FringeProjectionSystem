"""Quality gate checks for safe phase/unwrap pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any


@dataclass(slots=True)
class QualityThresholds:
    max_clipped_any_pct_roi: float = 0.01
    max_clipped_step_pct_roi: float = 0.01
    max_residual_p95: float = 0.2
    min_roi_valid_ratio_high_freq: float = 0.85
    min_unwrap_pixels: int = 3000


@dataclass(slots=True)
class QualityReport:
    ok: bool
    reasons: list[str]
    hints: list[str]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


STATE_PATH = Path("data/system_state.json")


def _residual_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return (cfg.get("quality", {}) or {}).get("residual", {}) or {}


def load_quality_state(cfg: dict[str, Any]) -> dict[str, Any]:
    rcfg = _residual_cfg(cfg)
    default_thr = float(rcfg.get("p95_threshold_start", 0.8))
    state = {"consecutive_passes": 0, "current_residual_threshold": default_thr}
    if STATE_PATH.exists():
        try:
            state.update(json.loads(STATE_PATH.read_text()))
        except Exception:
            pass
    return state


def save_quality_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def reset_quality_state(cfg: dict[str, Any]) -> dict[str, Any]:
    state = load_quality_state(cfg)
    rcfg = _residual_cfg(cfg)
    state["consecutive_passes"] = 0
    state["current_residual_threshold"] = float(rcfg.get("p95_threshold_start", 0.8))
    save_quality_state(state)
    return state


def update_quality_state(cfg: dict[str, Any], passed: bool) -> dict[str, Any]:
    state = load_quality_state(cfg)
    rcfg = _residual_cfg(cfg)
    start = float(rcfg.get("p95_threshold_start", 0.8))
    tight = float(rcfg.get("p95_threshold_tight", 0.4))
    final = float(rcfg.get("p95_threshold_final", 0.2))
    tighten_after = int(rcfg.get("tighten_after_passes", 10))
    final_after = int(rcfg.get("final_after_passes", 30))
    if passed:
        state["consecutive_passes"] = int(state.get("consecutive_passes", 0)) + 1
    else:
        state["consecutive_passes"] = 0
    n = int(state["consecutive_passes"])
    if n >= final_after:
        state["current_residual_threshold"] = final
    elif n >= tighten_after:
        state["current_residual_threshold"] = tight
    else:
        state["current_residual_threshold"] = start
    save_quality_state(state)
    return state


def build_phase_quality_report(phase_meta: dict[str, Any], th: QualityThresholds) -> QualityReport:
    reasons: list[str] = []
    hints: list[str] = []
    clipped_any = float(phase_meta.get("clipped_any_pct_roi", 1.0))
    clipped_step_max = float(max(phase_meta.get("clipped_per_step_pct_roi", [1.0])))
    roi_valid = float(phase_meta.get("roi_valid_ratio", 0.0))
    metrics = {
        "clipped_any_pct_roi": clipped_any,
        "clipped_step_max_pct_roi": clipped_step_max,
        "roi_valid_ratio_high_freq": roi_valid,
    }
    if clipped_any > th.max_clipped_any_pct_roi or clipped_step_max > th.max_clipped_step_pct_roi:
        reasons.append("CLIPPING")
        hints.extend([
            "Lower exposure_us or analogue_gain.",
            "Lower contrast or brightness_offset.",
            "Keep sat_high at 250.",
        ])
    if roi_valid < th.min_roi_valid_ratio_high_freq:
        reasons.append("LOW_ROI_VALID")
    ok = not reasons
    return QualityReport(ok=ok, reasons=reasons, hints=sorted(set(hints)), metrics=metrics)


def apply_unwrap_quality_report(base: QualityReport, unwrap_meta: dict[str, Any], th: QualityThresholds) -> QualityReport:
    reasons = list(base.reasons)
    hints = list(base.hints)
    metrics = dict(base.metrics)
    residual_p95 = float(unwrap_meta.get("residual_p95", 999.0))
    unwrap_valid_px = int(unwrap_meta.get("unwrap_valid_px", 0))
    metrics["residual_p95"] = residual_p95
    metrics["unwrap_valid_px"] = float(unwrap_valid_px)
    metrics["effective_residual_p95_threshold"] = float(th.max_residual_p95)
    if unwrap_valid_px < int(th.min_unwrap_pixels):
        reasons.append("RESIDUAL_INSUFFICIENT_PIXELS")
        hints.append("Mask overlap too small; increase exposure/gain or reduce erosion radius.")
        return QualityReport(ok=False, reasons=sorted(set(reasons)), hints=sorted(set(hints)), metrics=metrics)
    if residual_p95 > th.max_residual_p95:
        reasons.append("RESIDUAL")
        hints.extend([
            "Check step ordering.",
            "Ensure AE is disabled.",
            "Increase settle_ms.",
            "Reduce frequency ratio (try 4->12 instead of 4->16).",
        ])
    return QualityReport(ok=(len(reasons) == 0), reasons=reasons, hints=sorted(set(hints)), metrics=metrics)


def save_quality_report(run_dir: Path, report: QualityReport) -> Path:
    out = run_dir / "quality_report.json"
    out.write_text(json.dumps(report.to_dict(), indent=2))
    return out
