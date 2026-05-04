"""Projector calibration session helpers (accepted-view only + coverage map)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .conditioning import ConditioningAccumulator, ConditioningConfig
from .projector_view_gating import ProjectorViewDiagnostics


def session_json_path(session_dir: Path) -> Path:
    return session_dir / "session.json"


def session_coverage_path(session_dir: Path) -> Path:
    return session_dir / "session_coverage.json"


def load_session(session_dir: Path) -> dict[str, Any]:
    path = session_json_path(session_dir)
    if not path.exists():
        raise FileNotFoundError(f"Projector calibration session not found: {session_dir}")
    return json.loads(path.read_text())


def save_session(session_dir: Path, data: dict[str, Any]) -> None:
    session_json_path(session_dir).write_text(json.dumps(_json_safe(data), indent=2))


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def init_coverage_map(
    projector_size: tuple[int, int],
    conditioning_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = ConditioningConfig.from_config(conditioning_cfg)
    acc = ConditioningAccumulator(projector_size, cfg)
    payload = acc.to_dict()
    payload["conditioning_cfg"] = conditioning_cfg or {}
    return payload


def _conditioning_cfg(session: dict[str, Any]) -> dict[str, Any]:
    session_cfg = session.get("config", {}) if isinstance(session, dict) else {}
    cond = (session_cfg.get("conditioning", {}) or {}) if isinstance(session_cfg, dict) else {}
    return dict(cond)


def _ensure_session_state(
    session: dict[str, Any],
    projector_size: tuple[int, int],
) -> dict[str, Any]:
    if not isinstance(session.get("views"), list):
        session["views"] = []
    if not isinstance(session.get("accepted_pose_history"), list):
        session["accepted_pose_history"] = []
    if not isinstance(session.get("capture_attempts"), int):
        session["capture_attempts"] = 0
    cond_cfg = _conditioning_cfg(session)
    cov = session.get("coverage_map")
    if not isinstance(cov, dict):
        session["coverage_map"] = init_coverage_map(projector_size, conditioning_cfg=cond_cfg)
    else:
        cov["projector_size"] = [int(projector_size[0]), int(projector_size[1])]
        cov.setdefault("grid_size", [int(cond_cfg.get("grid_w", 64)), int(cond_cfg.get("grid_h", 36))])
        cov.setdefault("bins_x", int(cov.get("grid_size", [64, 36])[0]))
        cov.setdefault("bins_y", int(cov.get("grid_size", [64, 36])[1]))
        cov.setdefault("grid", [[0 for _ in range(int(cov["bins_x"]))] for _ in range(int(cov["bins_y"]))])
        cov.setdefault("covered_bins", 0)
        cov.setdefault("total_bins", int(cov["bins_x"] * cov["bins_y"]))
        cov.setdefault("coverage_ratio", 0.0)
        cov.setdefault("guidance", [])
        cov.setdefault("uniformity_metric", 0.0)
        cov.setdefault("edge_coverage_ratio", 0.0)
        cov.setdefault("sufficient", False)
        cov["conditioning_cfg"] = cond_cfg
        session["coverage_map"] = cov
    return session


def _coverage_size(coverage: dict[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(coverage, dict):
        return None
    proj = coverage.get("projector_size")
    if isinstance(proj, (list, tuple)) and len(proj) == 2:
        try:
            return (int(proj[0]), int(proj[1]))
        except Exception:
            return None
    return None


def _rebuild_coverage_from_session_views(
    session_dir: Path,
    session: dict[str, Any],
    projector_size: tuple[int, int],
    conditioning_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cov = init_coverage_map(projector_size, conditioning_cfg=conditioning_cfg)
    for view in (session.get("views", []) or []):
        if not isinstance(view, dict):
            continue
        view_id = str(view.get("view_id", "")).strip()
        if not view_id:
            continue
        corr_path = session_dir / "views" / view_id / "correspondences.json"
        if not corr_path.exists():
            continue
        try:
            corr = json.loads(corr_path.read_text())
        except Exception:
            continue
        pts = np.asarray(corr.get("projector_corners_px", []), dtype=np.float64).reshape(-1, 2)
        vm = np.asarray(corr.get("valid_mask", []), dtype=bool).reshape(-1)
        if pts.shape[0] != vm.shape[0] or pts.size == 0:
            continue
        valid_pts = pts[vm]
        if valid_pts.size == 0:
            continue
        cov = update_coverage_map(cov, valid_pts, conditioning_cfg=conditioning_cfg)
    return cov


def next_view_id(session_dir: Path, projector_size: tuple[int, int]) -> str:
    session = load_session(session_dir)
    session = _ensure_session_state(session, projector_size)
    idx = int(session.get("capture_attempts", 0)) + 1
    views_dir = session_dir / "views"
    while (views_dir / f"view_{idx:04d}").exists():
        idx += 1
    session["capture_attempts"] = int(idx)
    save_session(session_dir, session)
    return f"view_{idx:04d}"


def recent_accepted_poses(session: dict[str, Any], n: int = 3) -> list[dict[str, Any]]:
    poses = session.get("accepted_pose_history", []) or []
    if not isinstance(poses, list):
        return []
    return [p for p in poses if isinstance(p, dict)][-int(max(1, n)) :]


def update_pose_history(session: dict[str, Any], pose_entry: dict[str, Any] | None) -> None:
    if pose_entry is None:
        return
    poses = session.setdefault("accepted_pose_history", [])
    if not isinstance(poses, list):
        poses = []
        session["accepted_pose_history"] = poses
    poses.append(pose_entry)
    if len(poses) > 128:
        del poses[:-128]


def update_coverage_map(
    coverage: dict[str, Any],
    projector_points: np.ndarray,
    conditioning_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    proj_size = coverage.get("projector_size", [0, 0])
    pw = max(1, int(proj_size[0]))
    ph = max(1, int(proj_size[1]))
    cfg_dict = conditioning_cfg if conditioning_cfg is not None else (coverage.get("conditioning_cfg", {}) or {})
    cfg = ConditioningConfig.from_config(cfg_dict)
    grid = np.asarray(coverage.get("grid", []), dtype=np.int32)
    acc = ConditioningAccumulator((pw, ph), cfg, grid=grid)
    acc.update(np.asarray(projector_points, dtype=np.float64).reshape(-1, 2))
    payload = acc.to_dict()
    payload["conditioning_cfg"] = cfg_dict
    return payload


def add_view_if_valid(
    session_dir: Path,
    view_data: dict[str, Any],
    diagnostics: ProjectorViewDiagnostics,
    projector_points: np.ndarray | None,
    projector_size: tuple[int, int],
    pose_entry: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    session = load_session(session_dir)
    session = _ensure_session_state(session, projector_size)
    cond_cfg = _conditioning_cfg(session)
    cov_size = _coverage_size(session.get("coverage_map"))
    target_size = (int(projector_size[0]), int(projector_size[1]))
    if cov_size is None or cov_size != target_size:
        session["coverage_map"] = _rebuild_coverage_from_session_views(
            session_dir,
            session,
            projector_size=target_size,
            conditioning_cfg=cond_cfg,
        )

    capture_result = {
        "view_id": view_data.get("view_id"),
        "accept": bool(diagnostics.accept),
        "reject_reasons": list(diagnostics.reject_reasons),
        "hints": list(diagnostics.hints),
        "valid_corner_ratio": float(diagnostics.valid_corner_ratio),
        "coverage_sufficient": False,
    }
    session["last_capture_result"] = capture_result

    accepted = bool(diagnostics.accept)
    if accepted:
        views = session.setdefault("views", [])
        if not isinstance(views, list):
            views = []
            session["views"] = views
        views.append(view_data)
        update_pose_history(session, pose_entry)
        cov = session.get("coverage_map", init_coverage_map(projector_size))
        if projector_points is None:
            projector_points = np.empty((0, 2), dtype=np.float32)
        session["coverage_map"] = update_coverage_map(cov, projector_points, conditioning_cfg=cond_cfg)
        capture_result["coverage_sufficient"] = bool((session["coverage_map"] or {}).get("sufficient", False))
        if capture_result["coverage_sufficient"]:
            capture_result["hints"] = sorted(set(capture_result["hints"] + ["Coverage target reached; you can run Solve."]))
        session_coverage_path(session_dir).write_text(
            json.dumps(session["coverage_map"], indent=2)
        )

    save_session(session_dir, session)
    return accepted, session
