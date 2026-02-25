"""Projector calibration session helpers (accepted-view only + coverage map)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

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
    bins_x: int = 8,
    bins_y: int = 8,
) -> dict[str, Any]:
    w, h = int(projector_size[0]), int(projector_size[1])
    grid = [[0 for _ in range(int(bins_x))] for _ in range(int(bins_y))]
    return {
        "projector_size": [w, h],
        "bins_x": int(bins_x),
        "bins_y": int(bins_y),
        "grid": grid,
        "covered_bins": 0,
        "total_bins": int(bins_x * bins_y),
        "coverage_ratio": 0.0,
    }


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
    cov = session.get("coverage_map")
    if not isinstance(cov, dict):
        session["coverage_map"] = init_coverage_map(projector_size)
    else:
        cov.setdefault("projector_size", [int(projector_size[0]), int(projector_size[1])])
        cov.setdefault("bins_x", 8)
        cov.setdefault("bins_y", 8)
        cov.setdefault("grid", [[0 for _ in range(int(cov["bins_x"]))] for _ in range(int(cov["bins_y"]))])
        cov.setdefault("covered_bins", 0)
        cov.setdefault("total_bins", int(cov["bins_x"] * cov["bins_y"]))
        cov.setdefault("coverage_ratio", 0.0)
        session["coverage_map"] = cov
    return session


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
) -> dict[str, Any]:
    grid = np.asarray(coverage.get("grid", []), dtype=np.int32)
    if grid.ndim != 2 or grid.size == 0:
        bins_y = int(coverage.get("bins_y", 8))
        bins_x = int(coverage.get("bins_x", 8))
        grid = np.zeros((bins_y, bins_x), dtype=np.int32)
    else:
        bins_y, bins_x = int(grid.shape[0]), int(grid.shape[1])
    proj_size = coverage.get("projector_size", [0, 0])
    pw = max(1, int(proj_size[0]))
    ph = max(1, int(proj_size[1]))

    pts = np.asarray(projector_points, dtype=np.float64).reshape(-1, 2)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.size > 0:
        u = np.clip(pts[:, 0], 0.0, float(pw - 1))
        v = np.clip(pts[:, 1], 0.0, float(ph - 1))
        bx = np.clip((u / float(pw)) * bins_x, 0.0, bins_x - 1e-6).astype(np.int32)
        by = np.clip((v / float(ph)) * bins_y, 0.0, bins_y - 1e-6).astype(np.int32)
        grid[by, bx] = 1

    covered = int(np.count_nonzero(grid))
    total = int(grid.size)
    coverage["grid"] = grid.astype(int).tolist()
    coverage["covered_bins"] = covered
    coverage["total_bins"] = total
    coverage["coverage_ratio"] = float(covered / max(total, 1))
    return coverage


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

    capture_result = {
        "view_id": view_data.get("view_id"),
        "accept": bool(diagnostics.accept),
        "reject_reasons": list(diagnostics.reject_reasons),
        "hints": list(diagnostics.hints),
        "valid_corner_ratio": float(diagnostics.valid_corner_ratio),
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
        session["coverage_map"] = update_coverage_map(cov, projector_points)
        session_coverage_path(session_dir).write_text(
            json.dumps(session["coverage_map"], indent=2)
        )

    save_session(session_dir, session)
    return accepted, session
