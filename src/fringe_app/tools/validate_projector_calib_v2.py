"""Offline validator for projector calibration v2 sessions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_PLOT_FILES = [
    "reproj_cam.png",
    "reproj_proj.png",
    "coverage.png",
    "residual_hist.png",
]


def _err(errors: list[str], msg: str) -> None:
    errors.append(msg)


def _load_json(path: Path, errors: list[str], label: str) -> dict[str, Any] | None:
    if not path.exists():
        _err(errors, f"Missing {label}: {path}")
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        _err(errors, f"Invalid JSON for {label}: {path} ({exc})")
        return None
    if not isinstance(payload, dict):
        _err(errors, f"JSON root for {label} must be object: {path}")
        return None
    return payload


def _check_session_json(session_json: dict[str, Any], errors: list[str]) -> None:
    for key in ("schema_version", "session_id", "created_at", "config_snapshot", "state"):
        if key not in session_json:
            _err(errors, f"session.json missing key '{key}'")
    state = session_json.get("state", {})
    if not isinstance(state, dict):
        _err(errors, "session.json state must be object")
        return
    if not isinstance(state.get("views", []), list):
        _err(errors, "session.json state.views must be array")
    if not isinstance(state.get("coverage", {}), dict):
        _err(errors, "session.json state.coverage must be object")


def _validate_view(session_dir: Path, view_entry: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    view_id = str(view_entry.get("view_id", ""))
    view_dir = session_dir / "views" / view_id
    report: dict[str, Any] | None = None

    required = [
        view_dir / "camera.png",
        view_dir / "charuco.json",
        view_dir / "overlay.png",
        view_dir / "view_report.json",
    ]
    for path in required:
        if not path.exists():
            _err(errors, f"View {view_id}: missing required file {path}")

    charuco = _load_json(view_dir / "charuco.json", errors, f"view {view_id} charuco")
    report = _load_json(view_dir / "view_report.json", errors, f"view {view_id} report")

    if charuco is not None:
        ids = charuco.get("ids", [])
        corners = charuco.get("corners_px", [])
        obj = charuco.get("object_points_m", [])
        if not isinstance(ids, list):
            _err(errors, f"View {view_id}: charuco.ids must be array")
        if not isinstance(corners, list):
            _err(errors, f"View {view_id}: charuco.corners_px must be array")
        if not isinstance(obj, list):
            _err(errors, f"View {view_id}: charuco.object_points_m must be array")

    if report is not None:
        status = str(report.get("status", ""))
        uv = report.get("uv", {}) or {}
        if status not in {"accepted", "rejected"}:
            _err(errors, f"View {view_id}: report.status must be accepted|rejected")

        if status == "accepted":
            uv_dir = view_dir / "uv"
            for path in [uv_dir / "u.npy", uv_dir / "v.npy", uv_dir / "mask.npy", uv_dir / "uv_overlay.png"]:
                if not path.exists():
                    _err(errors, f"View {view_id}: missing accepted-view UV file {path}")

            if (uv_dir / "u.npy").exists() and (uv_dir / "v.npy").exists() and (uv_dir / "mask.npy").exists():
                try:
                    u = np.load(uv_dir / "u.npy")
                    v = np.load(uv_dir / "v.npy")
                    m = np.load(uv_dir / "mask.npy")
                    if u.shape != v.shape or u.shape != m.shape:
                        _err(errors, f"View {view_id}: uv/mask shape mismatch u={u.shape} v={v.shape} m={m.shape}")
                except Exception as exc:
                    _err(errors, f"View {view_id}: failed loading UV npy files ({exc})")

            valid_mask = uv.get("valid_mask", [])
            proj_corners = uv.get("projector_corners_px", [])
            if not isinstance(valid_mask, list) or not isinstance(proj_corners, list):
                _err(errors, f"View {view_id}: uv.valid_mask/projector_corners_px must be arrays")
            if isinstance(valid_mask, list) and isinstance(proj_corners, list):
                if len(valid_mask) != len(proj_corners):
                    _err(errors, f"View {view_id}: uv.valid_mask length does not match projector_corners_px")

    return {
        "view_id": view_id,
        "status": str(view_entry.get("status", "")),
    }


def _validate_solve_outputs(session_dir: Path, solved: bool, errors: list[str]) -> None:
    solve_dir = session_dir / "solve"
    plots_dir = solve_dir / "plots"
    export_dir = session_dir / "export"

    if solved:
        required_json = [
            solve_dir / "inputs_summary.json",
            solve_dir / "stereo_raw.json",
            solve_dir / "stereo_pruned.json",
            solve_dir / "solve_report.json",
            export_dir / "stereo.json",
        ]
        for path in required_json:
            if not path.exists():
                _err(errors, f"Missing solve artifact: {path}")

        for name in REQUIRED_PLOT_FILES:
            p = plots_dir / name
            if not p.exists():
                _err(errors, f"Missing solve plot: {p}")

        stereo = _load_json(export_dir / "stereo.json", errors, "export stereo")
        if stereo is not None:
            for key in ("projector_matrix", "projector_dist_coeffs", "R", "T", "projector"):
                if key not in stereo:
                    _err(errors, f"export/stereo.json missing key '{key}'")


def validate_session_folder(session_dir: Path) -> tuple[bool, dict[str, Any], list[str]]:
    errors: list[str] = []
    if not session_dir.exists() or not session_dir.is_dir():
        errors.append(f"Session folder not found: {session_dir}")
        return False, {}, errors

    session = _load_json(session_dir / "session.json", errors, "session")
    if session is None:
        return False, {}, errors
    _check_session_json(session, errors)

    state = (session.get("state", {}) or {})
    views = [v for v in (state.get("views", []) or []) if isinstance(v, dict)]

    validated_views: list[dict[str, Any]] = []
    for view in views:
        validated_views.append(_validate_view(session_dir, view, errors))

    solved = bool(state.get("solve")) or (session_dir / "export" / "stereo.json").exists()
    _validate_solve_outputs(session_dir, solved, errors)

    accepted = [v for v in validated_views if v.get("status") == "accepted"]
    rejected = [v for v in validated_views if v.get("status") == "rejected"]

    summary = {
        "session_id": session.get("session_id"),
        "session_dir": str(session_dir),
        "views_total": int(len(validated_views)),
        "views_accepted": int(len(accepted)),
        "views_rejected": int(len(rejected)),
        "coverage_indicator": (state.get("coverage", {}) or {}).get("indicator"),
        "solved": bool(solved),
        "errors": int(len(errors)),
    }
    return len(errors) == 0, summary, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate projector calibration v2 session folder")
    parser.add_argument("session_folder", help="Path to data/calibration/projector_v2/<session_id>")
    args = parser.parse_args(argv)

    session_dir = Path(args.session_folder)
    ok, summary, errors = validate_session_folder(session_dir)

    print(json.dumps(summary, indent=2))
    if not ok:
        print("\nValidation failed:", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
