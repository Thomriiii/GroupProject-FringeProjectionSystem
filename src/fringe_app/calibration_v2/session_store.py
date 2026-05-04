"""Session persistence and filesystem schema for projector calibration v2."""

from __future__ import annotations

import json
import math
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from .coverage import init_coverage, recompute_coverage


SCHEMA_VERSION = 1

DEFAULT_CONFIG: dict[str, Any] = {
    "charuco": {
        "dict": "DICT_4X4_50",
        "squares_x": 9,
        "squares_y": 7,
        "square_length_m": 0.015,
        "marker_length_m": 0.011,
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def get_v2_config(cfg: dict[str, Any]) -> dict[str, Any]:
    user_cfg = (cfg.get("calibration_v2", {}) or {}) if isinstance(cfg, dict) else {}
    merged = _deep_merge(DEFAULT_CONFIG, user_cfg)
    return merged


def calibration_root(cfg: dict[str, Any]) -> Path:
    calib = cfg.get("calibration", {}) or {}
    base = Path(str(calib.get("root", "data/calibration")))
    return base / "projector_v2"


def session_dir(cfg: dict[str, Any], session_id: str) -> Path:
    return calibration_root(cfg) / str(session_id)


def _session_json_path(session_path: Path) -> Path:
    return session_path / "session.json"


def find_camera_intrinsics_latest(cfg: dict[str, Any]) -> Path | None:
    calib = cfg.get("calibration", {}) or {}
    root = Path(str(calib.get("root", "data/calibration")))
    camera_root = Path(str(calib.get("camera_root", str(root / "camera"))))
    candidates = [
        camera_root / "intrinsics_latest.json",
        root / "camera" / "intrinsics_latest.json",
        root / "camera_intrinsics" / "intrinsics_latest.json",
        root / "intrinsics_latest.json",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _init_state(v2_cfg: dict[str, Any]) -> dict[str, Any]:
    coverage = init_coverage(v2_cfg.get("coverage", {}) or {})
    return {
        "capture_index": 0,
        "views": [],
        "accepted_view_ids": [],
        "rejected_view_ids": [],
        "coverage": coverage,
        "coverage_indicator": str(coverage.get("indicator", "Need more variety")),
        "coverage_sufficient": bool(coverage.get("sufficient", False)),
        "solve": None,
    }


def _ensure_layout(session_path: Path) -> None:
    (session_path / "views").mkdir(parents=True, exist_ok=True)
    (session_path / "solve" / "plots").mkdir(parents=True, exist_ok=True)
    (session_path / "export").mkdir(parents=True, exist_ok=True)


def _new_session_id(root: Path) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sid = stamp
    idx = 0
    while (root / sid).exists():
        idx += 1
        sid = f"{stamp}_{idx:02d}"
    return sid


def create_session(cfg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    root = calibration_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    v2_cfg = get_v2_config(cfg)
    intr_path = find_camera_intrinsics_latest(cfg)
    if intr_path is None:
        raise FileNotFoundError(
            "Camera intrinsics not found. Expected data/calibration/camera/intrinsics_latest.json"
        )

    sid = _new_session_id(root)
    session_path = root / sid
    _ensure_layout(session_path)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "session_id": sid,
        "created_at": datetime.now().isoformat(),
        "config_snapshot": {
            "calibration_v2": v2_cfg,
            "camera_intrinsics_path": str(intr_path),
        },
        "state": _init_state(v2_cfg),
    }
    save_session(session_path, payload)
    return session_path, payload


def load_session(session_path: Path) -> dict[str, Any]:
    path = _session_json_path(session_path)
    if not path.exists():
        raise FileNotFoundError(f"Projector calibration v2 session not found: {session_path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid session file: {path}")
    return data


def save_session(session_path: Path, payload: dict[str, Any]) -> None:
    _ensure_layout(session_path)
    _session_json_path(session_path).write_text(json.dumps(_json_safe(payload), indent=2))


def get_session(cfg: dict[str, Any], session_id: str) -> tuple[Path, dict[str, Any]]:
    sdir = session_dir(cfg, session_id)
    if not sdir.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    return sdir, load_session(sdir)


def allocate_view_id(session_path: Path, session: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    state = (session.get("state", {}) or {})
    idx = int(state.get("capture_index", 0)) + 1
    views_dir = session_path / "views"
    while (views_dir / f"view_{idx:04d}").exists():
        idx += 1
    state["capture_index"] = int(idx)
    session["state"] = state
    save_session(session_path, session)
    return f"view_{idx:04d}", session


def _refresh_state_derived(session: dict[str, Any]) -> dict[str, Any]:
    state = (session.get("state", {}) or {})
    views = [v for v in (state.get("views", []) or []) if isinstance(v, dict)]
    views_sorted = sorted(views, key=lambda v: str(v.get("view_id", "")))
    state["views"] = views_sorted

    accepted = [str(v.get("view_id")) for v in views_sorted if str(v.get("status", "")) == "accepted"]
    rejected = [str(v.get("view_id")) for v in views_sorted if str(v.get("status", "")) == "rejected"]
    state["accepted_view_ids"] = accepted
    state["rejected_view_ids"] = rejected

    v2_cfg = (((session.get("config_snapshot", {}) or {}).get("calibration_v2", {}) or {}))
    cov = recompute_coverage(v2_cfg.get("coverage", {}) or {}, [v for v in views_sorted if str(v.get("status", "")) == "accepted"])
    state["coverage"] = cov
    state["coverage_indicator"] = str(cov.get("indicator", "Need more variety"))
    state["coverage_sufficient"] = bool(cov.get("sufficient", False))

    session["state"] = state
    return session


def register_view(session_path: Path, view_record: dict[str, Any]) -> dict[str, Any]:
    session = load_session(session_path)
    state = (session.get("state", {}) or {})
    views = [v for v in (state.get("views", []) or []) if isinstance(v, dict)]
    view_id = str(view_record.get("view_id", "")).strip()
    if not view_id:
        raise ValueError("view_record.view_id is required")

    views = [v for v in views if str(v.get("view_id", "")) != view_id]
    views.append(view_record)
    state["views"] = views
    session["state"] = state

    session = _refresh_state_derived(session)
    save_session(session_path, session)
    return session


def delete_view(session_path: Path, view_id: str) -> dict[str, Any]:
    session = load_session(session_path)
    state = (session.get("state", {}) or {})
    views = [v for v in (state.get("views", []) or []) if isinstance(v, dict)]
    before = len(views)
    kept = [v for v in views if str(v.get("view_id", "")) != str(view_id)]
    if len(kept) == before:
        raise FileNotFoundError(f"View not found: {view_id}")

    vdir = session_path / "views" / str(view_id)
    if vdir.exists():
        shutil.rmtree(vdir)

    state["views"] = kept
    session["state"] = state
    session = _refresh_state_derived(session)
    save_session(session_path, session)
    return session


def reset_session(session_path: Path) -> dict[str, Any]:
    session = load_session(session_path)

    for rel in ("views", "solve", "export"):
        target = session_path / rel
        if target.exists():
            shutil.rmtree(target)
    _ensure_layout(session_path)

    v2_cfg = (((session.get("config_snapshot", {}) or {}).get("calibration_v2", {}) or {}))
    session["state"] = _init_state(v2_cfg)
    save_session(session_path, session)
    return session


def update_solve_summary(session_path: Path, solve_summary: dict[str, Any]) -> dict[str, Any]:
    session = load_session(session_path)
    state = (session.get("state", {}) or {})
    state["solve"] = solve_summary
    session["state"] = state
    save_session(session_path, session)
    return session


def build_session_payload(session_path: Path, api_prefix: str = "/api/calibration/projector_v2") -> dict[str, Any]:
    session = load_session(session_path)
    sid = str(session.get("session_id", session_path.name))
    state = (session.get("state", {}) or {})
    views = [v for v in (state.get("views", []) or []) if isinstance(v, dict)]

    items: list[dict[str, Any]] = []
    for view in sorted(views, key=lambda x: str(x.get("view_id", ""))):
        vid = str(view.get("view_id", ""))
        item = {
            "view_id": vid,
            "status": str(view.get("status", "rejected")),
            "reason": view.get("reason"),
            "hints": [str(h) for h in (view.get("hints", []) or [])],
            "metrics": view.get("metrics", {}),
            "files": view.get("files", {}),
            "image_url": f"{api_prefix}/session/{sid}/view/{vid}/image",
            "overlay_url": f"{api_prefix}/session/{sid}/view/{vid}/overlay",
            "uv_overlay_url": f"{api_prefix}/session/{sid}/view/{vid}/uv_overlay",
        }
        items.append(item)

    payload = {
        "session_id": sid,
        "session_dir": str(session_path),
        "session": session,
        "views": items,
        "accepted_views": int(len([v for v in items if v["status"] == "accepted"])),
        "rejected_views": int(len([v for v in items if v["status"] == "rejected"])),
        "coverage": state.get("coverage", {}),
        "coverage_indicator": str(state.get("coverage_indicator", "Need more variety")),
        "coverage_sufficient": bool(state.get("coverage_sufficient", False)),
        "solve": state.get("solve"),
        "download_url": f"{api_prefix}/session/{sid}/download_zip",
    }
    return _json_safe(payload)


def build_session_zip(session_path: Path) -> Path:
    zip_base = session_path.parent / session_path.name
    zip_path = Path(
        shutil.make_archive(
            base_name=str(zip_base),
            format="zip",
            root_dir=str(session_path.parent),
            base_dir=session_path.name,
        )
    )
    return zip_path


def list_sessions(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    root = calibration_root(cfg)
    if not root.exists():
        return []

    sessions: list[dict[str, Any]] = []
    for p in sorted(root.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        session_path = p / "session.json"
        if not session_path.exists():
            continue
        try:
            sess = json.loads(session_path.read_text())
        except Exception:
            continue
        if not isinstance(sess, dict):
            continue

        state = (sess.get("state", {}) or {})
        views = [v for v in (state.get("views", []) or []) if isinstance(v, dict)]
        accepted = int(len([v for v in views if str(v.get("status", "")) == "accepted"]))
        rejected = int(len([v for v in views if str(v.get("status", "")) == "rejected"]))
        sessions.append(
            {
                "session_id": str(sess.get("session_id", p.name)),
                "created_at": sess.get("created_at"),
                "views_total": int(len(views)),
                "views_accepted": accepted,
                "views_rejected": rejected,
                "coverage_indicator": str(state.get("coverage_indicator", "Need more variety")),
                "solved": bool(state.get("solve")),
            }
        )
    return sessions
