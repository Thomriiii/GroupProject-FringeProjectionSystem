"""
Projector calibration session persistence.

Filesystem layout:
    data/calibration/projector_v2/
        <session_id>/
            session.json
            views/
                view_0001/
                    image.png           — white-light capture
                    overlay.png         — ChArUco overlay
                    charuco.json        — detected corners + object points
                    view_report.json    — gating + UV sampling report
                    uv/
                        u_map.npy       — H×W projector X coordinates
                        v_map.npy       — H×W projector Y coordinates
                        mask_uv.npy     — H×W boolean valid mask
            solve/
                stereo_raw.json
                stereo_pruned.json
            export/
                stereo.json             — final output (compatible with load_projector_model)
"""

from __future__ import annotations

import json
import math
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from fringe_app_v2.calibration.projector.coverage import init_coverage, recompute_coverage


SCHEMA_VERSION = 1

_DEFAULT_CFG: dict[str, Any] = {
    "charuco": {
        "dict": "DICT_4X4_50",
        "squares_x": 9,
        "squares_y": 7,
        "square_length_m": 0.010,
        "marker_length_m": 0.0075,
    },
    "gating": {
        "min_corners": 16,
        "min_markers": 8,
        "border_margin_px": 20,
        "min_area_ratio": 0.05,
        "max_area_ratio": 0.60,
    },
    "coverage": {
        "grid_x": 3,
        "grid_y": 3,
        "min_bins_filled": 5,
        "tilt_buckets_deg": [0, 10, 20, 35],
        "min_bucket_counts": [2, 2, 2],
    },
    "solve": {
        "min_views": 8,
        "min_improvement_px": 0.05,
        "max_prune_steps": 6,
    },
    "uv_refine": {
        "enabled": True,
        "patch_radius_px": 12,
        "plane_fit": True,
        "max_delta_px": 3.0,
    },
}


def _json_safe(v: Any) -> Any:
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {str(k): _json_safe(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(vv) for vv in v]
    return v


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], val)
        else:
            out[k] = deepcopy(val)
    return out


def get_v2_config(config: dict[str, Any]) -> dict[str, Any]:
    user_cfg = (config.get("calibration_v2") or {}) if isinstance(config, dict) else {}
    return _deep_merge(_DEFAULT_CFG, user_cfg)


def _proj_root(config: dict[str, Any]) -> Path:
    calib = config.get("calibration", {}) or {}
    root = Path(str(calib.get("root", "data/calibration")))
    return root / "projector_v2"


def _find_camera_intrinsics(config: dict[str, Any]) -> Path | None:
    calib = config.get("calibration", {}) or {}
    root = Path(str(calib.get("root", "data/calibration")))
    cam_root = Path(str(calib.get("camera_root", str(root / "camera"))))
    candidates = [
        cam_root / "intrinsics_latest.json",
        root / "camera" / "intrinsics_latest.json",
        root / "intrinsics_latest.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _ensure_layout(session_path: Path) -> None:
    (session_path / "views").mkdir(parents=True, exist_ok=True)
    (session_path / "solve" / "plots").mkdir(parents=True, exist_ok=True)
    (session_path / "export").mkdir(parents=True, exist_ok=True)


def _new_session_id(root: Path) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sid, i = stamp, 0
    while (root / sid).exists():
        i += 1
        sid = f"{stamp}_{i:02d}"
    return sid


def create_session(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    root = _proj_root(config)
    root.mkdir(parents=True, exist_ok=True)

    intr_path = _find_camera_intrinsics(config)
    if intr_path is None:
        raise FileNotFoundError(
            "Camera intrinsics not found. Run camera calibration first.\n"
            "Expected: data/calibration/camera/intrinsics_latest.json"
        )

    v2_cfg = get_v2_config(config)
    sid = _new_session_id(root)
    session_path = root / sid
    _ensure_layout(session_path)

    coverage = init_coverage(v2_cfg.get("coverage", {}) or {})
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "session_id": sid,
        "created_at": datetime.now().isoformat(),
        "config_snapshot": {
            "calibration_v2": v2_cfg,
            "camera_intrinsics_path": str(intr_path),
        },
        "state": {
            "capture_index": 0,
            "views": [],
            "accepted_view_ids": [],
            "rejected_view_ids": [],
            "coverage": coverage,
            "coverage_indicator": str(coverage.get("indicator", "Need more variety")),
            "coverage_sufficient": bool(coverage.get("sufficient", False)),
            "solve": None,
        },
    }
    _save_session(session_path, payload)
    return session_path, payload


def load_session(session_path: Path) -> dict[str, Any]:
    p = session_path / "session.json"
    if not p.exists():
        raise FileNotFoundError(f"Projector calibration session not found: {session_path}")
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid session file: {p}")
    return data


def _save_session(session_path: Path, payload: dict[str, Any]) -> None:
    _ensure_layout(session_path)
    (session_path / "session.json").write_text(json.dumps(_json_safe(payload), indent=2))


def get_session(config: dict[str, Any], session_id: str) -> tuple[Path, dict[str, Any]]:
    root = _proj_root(config)
    session_path = root / session_id
    if not session_path.exists():
        raise FileNotFoundError(f"Projector session not found: {session_id}")
    return session_path, load_session(session_path)


def list_sessions(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = _proj_root(config)
    if not root.exists():
        return []
    result: list[dict[str, Any]] = []
    for p in sorted(root.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        sf = p / "session.json"
        if not sf.exists():
            continue
        try:
            sess = json.loads(sf.read_text())
        except Exception:
            continue
        state = sess.get("state", {}) or {}
        views = [v for v in (state.get("views") or []) if isinstance(v, dict)]
        accepted = [v for v in views if str(v.get("status")) == "accepted"]
        result.append({
            "session_id": str(sess.get("session_id", p.name)),
            "created_at": sess.get("created_at"),
            "views_total": len(views),
            "views_accepted": len(accepted),
            "coverage_indicator": str(state.get("coverage_indicator", "Need more variety")),
            "solved": bool(state.get("solve")),
        })
    return result


def allocate_view_id(session_path: Path, session: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    state = session.get("state", {}) or {}
    idx = int(state.get("capture_index", 0)) + 1
    views_dir = session_path / "views"
    while (views_dir / f"view_{idx:04d}").exists():
        idx += 1
    state["capture_index"] = idx
    session["state"] = state
    _save_session(session_path, session)
    return f"view_{idx:04d}", session


def _refresh_derived(session: dict[str, Any]) -> dict[str, Any]:
    state = session.get("state", {}) or {}
    views = sorted(
        [v for v in (state.get("views") or []) if isinstance(v, dict)],
        key=lambda v: str(v.get("view_id", "")),
    )
    state["views"] = views
    state["accepted_view_ids"] = [str(v["view_id"]) for v in views if str(v.get("status")) == "accepted"]
    state["rejected_view_ids"] = [str(v["view_id"]) for v in views if str(v.get("status")) == "rejected"]

    v2_cfg = ((session.get("config_snapshot") or {}).get("calibration_v2") or {})
    accepted = [v for v in views if str(v.get("status")) == "accepted"]
    cov = recompute_coverage(v2_cfg.get("coverage", {}) or {}, accepted)
    state["coverage"] = cov
    state["coverage_indicator"] = str(cov.get("indicator", "Need more variety"))
    state["coverage_sufficient"] = bool(cov.get("sufficient", False))

    session["state"] = state
    return session


def register_view(session_path: Path, view_record: dict[str, Any]) -> dict[str, Any]:
    session = load_session(session_path)
    state = session.get("state", {}) or {}
    views = [v for v in (state.get("views") or []) if isinstance(v, dict)]
    view_id = str(view_record.get("view_id", "")).strip()
    if not view_id:
        raise ValueError("view_record.view_id is required")
    views = [v for v in views if str(v.get("view_id")) != view_id]
    views.append(view_record)
    state["views"] = views
    session["state"] = state
    session = _refresh_derived(session)
    _save_session(session_path, session)
    return session


def delete_view(session_path: Path, view_id: str) -> dict[str, Any]:
    session = load_session(session_path)
    state = session.get("state", {}) or {}
    views = [v for v in (state.get("views") or []) if isinstance(v, dict)]
    before = len(views)
    views = [v for v in views if str(v.get("view_id")) != view_id]
    if len(views) == before:
        raise FileNotFoundError(f"View not found: {view_id}")
    vdir = session_path / "views" / view_id
    if vdir.exists():
        shutil.rmtree(vdir)
    state["views"] = views
    session["state"] = state
    session = _refresh_derived(session)
    _save_session(session_path, session)
    return session


def build_session_payload(
    session_path: Path,
    api_prefix: str = "/api/calibration/projector",
) -> dict[str, Any]:
    session = load_session(session_path)
    sid = str(session.get("session_id", session_path.name))
    state = session.get("state", {}) or {}
    views = [v for v in (state.get("views") or []) if isinstance(v, dict)]

    items: list[dict[str, Any]] = []
    for view in sorted(views, key=lambda v: str(v.get("view_id", ""))):
        vid = str(view.get("view_id", ""))
        items.append({
            "view_id": vid,
            "status": str(view.get("status", "rejected")),
            "reason": view.get("reason"),
            "hints": [str(h) for h in (view.get("hints") or [])],
            "metrics": view.get("metrics", {}),
            "image_url": f"{api_prefix}/session/{sid}/view/{vid}/image",
            "overlay_url": f"{api_prefix}/session/{sid}/view/{vid}/overlay",
            "uv_overlay_url": f"{api_prefix}/session/{sid}/view/{vid}/uv_overlay",
        })

    return _json_safe({
        "session_id": sid,
        "session": session,
        "views": items,
        "accepted_views": len([v for v in items if v["status"] == "accepted"]),
        "rejected_views": len([v for v in items if v["status"] == "rejected"]),
        "coverage": state.get("coverage", {}),
        "coverage_indicator": str(state.get("coverage_indicator", "Need more variety")),
        "coverage_sufficient": bool(state.get("coverage_sufficient", False)),
        "solve": state.get("solve"),
    })
