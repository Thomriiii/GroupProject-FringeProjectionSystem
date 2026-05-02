"""
Projector stereo calibration solve with deterministic view pruning.

Algorithm:
  1. Load camera intrinsics (K_cam, D_cam) — held fixed throughout.
  2. For each accepted view, load:
       - ChArUco object points  (Nx3 metres, board frame)
       - Camera image points    (Nx2 pixels)
       - Projector image points (Nx2 pixels, from UV sampling)
  3. Solve projector intrinsics alone via cv2.calibrateCamera on projector points.
  4. Run cv2.stereoCalibrate with CALIB_FIX_INTRINSIC to solve R and T only.
  5. Compute per-view reprojection errors:
       - For each view: solvePnP(obj, cam_pts, K_cam, D_cam) → R_c, t_c
       - Then R_p = R_stereo @ R_c,  t_p = R_stereo @ t_c + T_stereo
       - Reproject obj through projector: compare to observed proj_pts
  6. Iterative pruning: remove the worst view if RMS improves by >= threshold.
  7. Choose raw or pruned model (whichever has lower stereo RMS).
  8. Export stereo.json in format compatible with load_projector_model().

Output JSON keys (see fringe_app_v2/core/calibration.py load_projector_model):
    projector_matrix      — 3x3 float list
    projector_dist_coeffs — flat float list
    R                     — 3x3 float list (camera → projector rotation)
    T                     — float list length 3 (camera → projector translation, metres)
    projector             — {width: int, height: int}
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    _cv2_err = exc
else:
    _cv2_err = None


def _require_cv2():
    if cv2 is None:
        raise RuntimeError(f"OpenCV required: {_cv2_err}")
    return cv2


def _json_safe(v: Any) -> Any:
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {str(k): _json_safe(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(vv) for vv in v]
    if isinstance(v, np.ndarray):
        return _json_safe(v.tolist())
    if isinstance(v, (np.floating,)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# ── Load helpers ──────────────────────────────────────────────────────────────

def _load_camera_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Camera intrinsics not found: {path}")
    payload = json.loads(path.read_text())
    K = np.asarray(payload["camera_matrix"], dtype=np.float64)
    D = np.asarray(payload["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    sz = payload["image_size"]
    if K.shape != (3, 3):
        raise ValueError(f"camera_matrix must be (3,3), got {K.shape}")
    if not isinstance(sz, (list, tuple)) or len(sz) != 2:
        raise ValueError("image_size must be [width, height]")
    return K, D, (int(sz[0]), int(sz[1]))


def _load_view_inputs(
    session_path: Path,
    session: dict[str, Any],
    min_uv_corners: int,
    projector_size_hint: tuple[int, int] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[int, int]]:
    """
    Load accepted views from disk.

    Returns:
        (inputs, summary, projector_size)

    Each item in inputs:
        {view_id, object_points (Nx3 float32), camera_points (Nx2 float32),
         projector_points (Nx2 float32)}
    """
    state = session.get("state", {}) or {}
    all_views = [v for v in (state.get("views") or []) if isinstance(v, dict)]
    accepted = [v for v in all_views if str(v.get("status")) == "accepted"]

    inputs: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    proj_size: tuple[int, int] | None = projector_size_hint

    for view in accepted:
        vid = str(view.get("view_id", ""))
        view_dir = session_path / "views" / vid

        charuco_path = view_dir / "charuco.json"
        report_path = view_dir / "view_report.json"

        if not charuco_path.exists() or not report_path.exists():
            dropped.append({"view_id": vid, "reason": "missing_files"})
            continue

        charuco = json.loads(charuco_path.read_text())
        report = json.loads(report_path.read_text())

        corners = np.asarray(charuco.get("corners_px", []), dtype=np.float32).reshape(-1, 2)
        obj = np.asarray(charuco.get("object_points_m", []), dtype=np.float32).reshape(-1, 3)

        uv_info = report.get("uv", {}) or {}
        proj = np.asarray(uv_info.get("projector_corners_px", []), dtype=np.float32).reshape(-1, 2)
        valid = np.asarray(uv_info.get("valid_mask", []), dtype=bool).reshape(-1)

        if corners.shape[0] == 0 or obj.shape[0] == 0:
            dropped.append({"view_id": vid, "reason": "no_charuco_points"})
            continue

        if corners.shape[0] != obj.shape[0] or proj.shape[0] != obj.shape[0] or valid.shape[0] != obj.shape[0]:
            dropped.append({"view_id": vid, "reason": "shape_mismatch"})
            continue

        # Use only corners that both ChArUco found and UV sampling validated
        finite_proj = np.isfinite(proj[:, 0]) & np.isfinite(proj[:, 1])
        use = valid & finite_proj
        n_use = int(np.count_nonzero(use))

        if n_use < min_uv_corners:
            dropped.append({"view_id": vid, "reason": f"valid_uv={n_use} < {min_uv_corners}"})
            continue

        # Get projector size from this view's report
        psize_raw = uv_info.get("projector_size")
        if isinstance(psize_raw, (list, tuple)) and len(psize_raw) == 2:
            psize: tuple[int, int] = (int(psize_raw[0]), int(psize_raw[1]))
            if proj_size is None:
                proj_size = psize
            elif proj_size != psize:
                dropped.append({"view_id": vid, "reason": f"projector_size_mismatch {psize}"})
                continue

        inputs.append({
            "view_id": vid,
            "object_points": obj[use].astype(np.float32),
            "camera_points": corners[use].astype(np.float32),
            "projector_points": proj[use].astype(np.float32),
        })

    if proj_size is None:
        proj_size = (1024, 768)

    summary = {
        "accepted_in_session": len(accepted),
        "usable": len(inputs),
        "dropped": dropped,
    }
    return inputs, summary, proj_size


# ── Per-view error ─────────────────────────────────────────────────────────────

def _per_view_errors(
    view_inputs: list[dict[str, Any]],
    Kc: np.ndarray,
    Dc: np.ndarray,
    Kp: np.ndarray,
    Dp: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Compute per-view reprojection errors.

    For each view:
      1. solvePnP(obj, cam_pts, Kc, Dc) → R_c, t_c  (camera pose)
      2. R_p = R @ R_c,  t_p = R @ t_c + T          (projector pose via stereo)
      3. Reproject obj through projector → compare to observed proj_pts
    """
    cv = _require_cv2()
    per: list[dict[str, Any]] = []
    all_cam: list[float] = []
    all_proj: list[float] = []

    R_st = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T_st = np.asarray(T, dtype=np.float64).reshape(3, 1)

    for item in view_inputs:
        vid = str(item["view_id"])
        obj = np.asarray(item["object_points"], dtype=np.float64).reshape(-1, 3)
        cam = np.asarray(item["camera_points"], dtype=np.float64).reshape(-1, 2)
        proj = np.asarray(item["projector_points"], dtype=np.float64).reshape(-1, 2)

        if obj.shape[0] < 4:
            per.append({
                "view_id": vid,
                "n_corners": int(obj.shape[0]),
                "camera_reproj_px": float("nan"),
                "projector_reproj_px": float("nan"),
            })
            continue

        ok, rvec_c, tvec_c = cv.solvePnP(
            obj.astype(np.float32),
            cam.reshape(-1, 1, 2).astype(np.float32),
            Kc, Dc,
        )
        if not ok:
            per.append({
                "view_id": vid,
                "n_corners": int(obj.shape[0]),
                "camera_reproj_px": float("nan"),
                "projector_reproj_px": float("nan"),
            })
            continue

        # Camera reprojection
        cam_rep, _ = cv.projectPoints(obj, rvec_c, tvec_c, Kc, Dc)
        cam_err = np.linalg.norm(cam_rep.reshape(-1, 2) - cam, axis=1)

        # Projector reprojection (apply stereo transform on top of camera pose)
        R_c, _ = cv.Rodrigues(rvec_c)
        R_p = R_st @ R_c
        t_p = R_st @ np.asarray(tvec_c, dtype=np.float64).reshape(3, 1) + T_st
        rvec_p, _ = cv.Rodrigues(R_p)
        proj_rep, _ = cv.projectPoints(obj, rvec_p, t_p, Kp, Dp)
        proj_err = np.linalg.norm(proj_rep.reshape(-1, 2) - proj, axis=1)

        all_cam.extend(cam_err.tolist())
        all_proj.extend(proj_err.tolist())

        cam_rms = float(np.sqrt(np.mean(np.square(cam_err)))) if cam_err.size else float("nan")
        proj_rms = float(np.sqrt(np.mean(np.square(proj_err)))) if proj_err.size else float("nan")
        per.append({
            "view_id": vid,
            "n_corners": int(obj.shape[0]),
            "camera_reproj_px": cam_rms,
            "projector_reproj_px": proj_rms,
        })

    return per, np.asarray(all_cam, dtype=np.float64), np.asarray(all_proj, dtype=np.float64)


# ── Single solve ──────────────────────────────────────────────────────────────

def _build_cv_lists(
    view_inputs: list[dict[str, Any]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
    obj_l: list[np.ndarray] = []
    cam_l: list[np.ndarray] = []
    proj_l: list[np.ndarray] = []
    ids: list[str] = []
    for item in view_inputs:
        obj_l.append(np.asarray(item["object_points"], dtype=np.float32).reshape(-1, 1, 3))
        cam_l.append(np.asarray(item["camera_points"], dtype=np.float32).reshape(-1, 1, 2))
        proj_l.append(np.asarray(item["projector_points"], dtype=np.float32).reshape(-1, 1, 2))
        ids.append(str(item["view_id"]))
    return obj_l, cam_l, proj_l, ids


def _solve_once(
    *,
    session_id: str,
    view_inputs: list[dict[str, Any]],
    Kc: np.ndarray,
    Dc: np.ndarray,
    cam_size: tuple[int, int],   # (width, height) for cv2 — same as image_size
    proj_size: tuple[int, int],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """
    Run one stereo calibration solve.

    Returns:
        (result_dict, all_cam_residuals, all_proj_residuals)
    """
    cv = _require_cv2()
    obj_l, cam_l, proj_l, view_ids = _build_cv_lists(view_inputs)

    if not obj_l:
        raise ValueError("No usable views for solve")

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-7)

    # Step 1: Calibrate projector intrinsics alone
    _, Kp, Dp, _, _ = cv.calibrateCamera(
        obj_l, proj_l, proj_size, None, None, criteria=criteria
    )

    # Step 2: Stereo calibrate (camera intrinsics fixed)
    rms_stereo, Kc_o, Dc_o, Kp_o, Dp_o, R, T, E, F = cv.stereoCalibrate(
        obj_l, cam_l, proj_l,
        Kc.copy(), Dc.copy(),
        Kp, Dp,
        cam_size,
        criteria=criteria,
        flags=cv.CALIB_FIX_INTRINSIC,
    )

    per_view, cam_resid, proj_resid = _per_view_errors(
        view_inputs, Kc_o, Dc_o, Kp_o, Dp_o, R, T
    )

    result: dict[str, Any] = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "views_used": len(view_ids),
        "view_ids": view_ids,
        "rms_stereo": float(rms_stereo),
        # Keys must match load_projector_model() in fringe_app_v2/core/calibration.py
        "camera_matrix": Kc_o.tolist(),
        "camera_dist_coeffs": Dc_o.reshape(-1).tolist(),
        "projector_matrix": Kp_o.tolist(),
        "projector_dist_coeffs": Dp_o.reshape(-1).tolist(),
        "R": R.tolist(),
        "T": T.reshape(-1).tolist(),
        "E": E.tolist(),
        "F": F.tolist(),
        "projector": {
            "width": int(proj_size[0]),
            "height": int(proj_size[1]),
        },
        "per_view_errors": per_view,
    }
    return result, cam_resid, proj_resid


# ── Main solve entry point ────────────────────────────────────────────────────

def solve_session(session_path: Path, projector_size: tuple[int, int] | None = None) -> dict[str, Any]:
    """
    Run the full projector stereo calibration solve for a session.

    Loads all accepted views, runs stereo calibration, optionally prunes the
    worst views and re-solves, then exports the best model.

    Args:
        session_path:    Directory containing session.json and views/
        projector_size:  Override projector (width, height); inferred from view
                         reports if None.

    Returns:
        Summary dict with paths to output files and final RMS.
    """
    session_json_path = session_path / "session.json"
    if not session_json_path.exists():
        raise FileNotFoundError(f"session.json not found: {session_path}")
    session = json.loads(session_json_path.read_text())

    v2_cfg = ((session.get("config_snapshot") or {}).get("calibration_v2") or {})
    solve_cfg = v2_cfg.get("solve", {}) or {}
    min_views = max(2, int(solve_cfg.get("min_views", 8)))
    min_improvement_px = float(solve_cfg.get("min_improvement_px", 0.05))
    max_prune_steps = max(0, int(solve_cfg.get("max_prune_steps", 6)))
    min_uv_corners = int((v2_cfg.get("gating", {}) or {}).get("min_corners", 16))

    intr_path_str = ((session.get("config_snapshot") or {}).get("camera_intrinsics_path") or "")
    intr_path = Path(intr_path_str)
    Kc, Dc, cam_size = _load_camera_intrinsics(intr_path)

    inputs, summary, proj_size = _load_view_inputs(
        session_path, session, min_uv_corners, projector_size
    )

    sid = str(session.get("session_id", session_path.name))

    if len(inputs) < min_views:
        raise ValueError(
            f"Need at least {min_views} accepted usable views; found {len(inputs)}. "
            "Capture more board positions."
        )

    # ── Raw solve ─────────────────────────────────────────────────────────────
    raw_result, raw_cam_resid, raw_proj_resid = _solve_once(
        session_id=sid,
        view_inputs=inputs,
        Kc=Kc,
        Dc=Dc,
        cam_size=cam_size,
        proj_size=proj_size,
    )

    # ── Iterative pruning ─────────────────────────────────────────────────────
    current_result = raw_result
    current_ids = list(raw_result["view_ids"])
    current_cam_resid = raw_cam_resid
    current_proj_resid = raw_proj_resid
    prune_log: list[dict[str, Any]] = []

    for step in range(max_prune_steps):
        if len(current_ids) <= min_views:
            break

        per_view = [e for e in (current_result.get("per_view_errors") or []) if isinstance(e, dict)]
        valid_scores = [e for e in per_view if np.isfinite(float(e.get("projector_reproj_px", float("nan"))))]
        if not valid_scores:
            break

        worst = max(valid_scores, key=lambda e: float(e["projector_reproj_px"]))
        worst_id = str(worst.get("view_id", ""))
        if not worst_id:
            break

        candidate_ids = [vid for vid in current_ids if vid != worst_id]
        if len(candidate_ids) < min_views:
            break

        candidate_inputs = [v for v in inputs if str(v["view_id"]) in set(candidate_ids)]
        candidate_result, cand_cam_resid, cand_proj_resid = _solve_once(
            session_id=sid,
            view_inputs=candidate_inputs,
            Kc=Kc, Dc=Dc,
            cam_size=cam_size,
            proj_size=proj_size,
        )

        rms_before = float(current_result["rms_stereo"])
        rms_after = float(candidate_result["rms_stereo"])
        improvement = rms_before - rms_after
        accepted = np.isfinite(improvement) and improvement >= min_improvement_px

        prune_log.append({
            "step": step + 1,
            "removed_view_id": worst_id,
            "rms_before": rms_before,
            "rms_after": rms_after,
            "improvement_px": float(improvement),
            "accepted": bool(accepted),
        })

        if not accepted:
            break

        current_result = candidate_result
        current_ids = candidate_ids
        current_cam_resid = cand_cam_resid
        current_proj_resid = cand_proj_resid

    pruned_result = current_result

    # ── Choose model ──────────────────────────────────────────────────────────
    raw_rms = float(raw_result["rms_stereo"])
    pruned_rms = float(pruned_result["rms_stereo"])
    selected_model = "pruned" if np.isfinite(pruned_rms) and pruned_rms < raw_rms else "raw"
    selected = pruned_result if selected_model == "pruned" else raw_result
    selected_cam_resid = current_cam_resid if selected_model == "pruned" else raw_cam_resid
    selected_proj_resid = current_proj_resid if selected_model == "pruned" else raw_proj_resid

    # ── Write outputs ─────────────────────────────────────────────────────────
    solve_dir = session_path / "solve"
    export_dir = session_path / "export"
    solve_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    (solve_dir / "stereo_raw.json").write_text(json.dumps(_json_safe(raw_result), indent=2))
    (solve_dir / "stereo_pruned.json").write_text(json.dumps(_json_safe(pruned_result), indent=2))
    (solve_dir / "prune_log.json").write_text(json.dumps(_json_safe(prune_log), indent=2))
    (solve_dir / "inputs_summary.json").write_text(json.dumps(_json_safe({
        "session_id": sid,
        "camera_intrinsics_path": str(intr_path),
        "camera_size": list(cam_size),
        "projector_size": list(proj_size),
        "selected_model": selected_model,
        **summary,
        "usable_view_ids": [str(v["view_id"]) for v in inputs],
    }), indent=2))

    np.save(solve_dir / "cam_residuals.npy", selected_cam_resid.astype(np.float32))
    np.save(solve_dir / "proj_residuals.npy", selected_proj_resid.astype(np.float32))

    export_path = export_dir / "stereo.json"
    export_path.write_text(json.dumps(_json_safe(selected), indent=2))

    # Update session state
    state = session.get("state", {}) or {}
    state["solve"] = {
        "solved_at": datetime.now().isoformat(),
        "raw_rms": raw_rms,
        "pruned_rms": pruned_rms,
        "selected_model": selected_model,
        "views_used": int(selected.get("views_used", 0)),
        "export_stereo_path": "export/stereo.json",
    }
    session["state"] = state
    session_json_path.write_text(json.dumps(_json_safe(session), indent=2))

    return _json_safe({
        "session_id": sid,
        "raw_rms": raw_rms,
        "pruned_rms": pruned_rms,
        "selected_model": selected_model,
        "views_used": int(selected.get("views_used", 0)),
        "export_stereo_path": str(export_path),
        "prune_steps": len(prune_log),
    })
