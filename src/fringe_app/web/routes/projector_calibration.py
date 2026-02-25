"""Projector calibration route helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable


def _json_safe(value: Any) -> Any:
    """Convert NaN/Inf values recursively to JSON-safe None."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def build_projector_session_payload(
    calibration_root: Path,
    session_id: str,
    list_views_fn: Callable[[Path], list[dict[str, Any]]],
) -> dict[str, Any]:
    session_dir = calibration_root / "projector" / "sessions" / session_id
    if not session_dir.exists():
        raise FileNotFoundError("session not found")
    session_path = session_dir / "session.json"
    if not session_path.exists():
        raise FileNotFoundError("session.json missing")
    session = json.loads(session_path.read_text())
    coverage_path = session_dir / "session_coverage.json"
    coverage = None
    if coverage_path.exists():
        try:
            coverage = json.loads(coverage_path.read_text())
        except Exception:
            coverage = None
    if coverage is None:
        coverage = session.get("coverage_map")
    payload = {
        "session": session,
        "views": list_views_fn(session_dir),
        "coverage": coverage or {},
        "last_capture_result": session.get("last_capture_result"),
    }
    return _json_safe(payload)


def build_capture_response_payload(view_summary: dict[str, Any]) -> dict[str, Any]:
    accept = bool(view_summary.get("accept", False))
    reasons = list(view_summary.get("reject_reasons", []) or [])
    hints = list(view_summary.get("hints", []) or [])
    if accept:
        message = (
            f"View accepted. Valid corners: {100.0 * float(view_summary.get('valid_corner_ratio', 0.0)):.1f}%"
        )
    else:
        reason = reasons[0] if reasons else str(view_summary.get("reason", "view rejected"))
        hint = hints[0] if hints else ""
        message = f"View rejected: {reason}"
        if hint:
            message += f" | Hint: {hint}"
    payload = {
        "view": view_summary,
        "accept": accept,
        "message": message,
        "reject_reasons": reasons,
        "hints": hints,
    }
    return _json_safe(payload)
