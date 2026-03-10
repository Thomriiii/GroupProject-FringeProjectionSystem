"""Stereo solver with deterministic pruning for projector calibration v2."""

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
    _cv2_import_error = exc
else:
    _cv2_import_error = None

from .reporting import (
    build_solve_report,
    save_coverage_plot,
    save_reprojection_plot,
    save_residual_histogram,
)


SCHEMA_VERSION = 1


def _require_cv2():
    if cv2 is None:
        raise RuntimeError(f"OpenCV is required for projector calibration solve: {_cv2_import_error}")
    return cv2


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _load_camera_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Camera intrinsics not found: {path}")
    payload = json.loads(path.read_text())
    K = np.asarray(payload.get("camera_matrix"), dtype=np.float64)
    D = np.asarray(payload.get("dist_coeffs"), dtype=np.float64).reshape(-1, 1)
    image_size = payload.get("image_size")
    if K.shape != (3, 3):
        raise ValueError("camera_matrix must have shape (3,3)")
    if D.size == 0:
        raise ValueError("dist_coeffs missing in camera intrinsics")
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError("image_size missing in camera intrinsics")
    cam_size = (int(image_size[0]), int(image_size[1]))
    return K, D, cam_size


def _load_view_data(view_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    charuco_path = view_dir / "charuco.json"
    report_path = view_dir / "view_report.json"
    if not charuco_path.exists():
        raise FileNotFoundError(f"Missing charuco.json: {charuco_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Missing view_report.json: {report_path}")
    return json.loads(charuco_path.read_text()), json.loads(report_path.read_text())


def _projector_size_from_report(report: dict[str, Any]) -> tuple[int, int] | None:
    uv = (report.get("uv", {}) or {}) if isinstance(report, dict) else {}
    size = uv.get("projector_size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        try:
            return int(size[0]), int(size[1])
        except Exception:
            return None
    return None


def _gather_inputs(
    session_dir: Path,
    session: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[int, int]]:
    state = (session.get("state", {}) or {})
    views = [v for v in (state.get("views", []) or []) if isinstance(v, dict)]
    accepted = [v for v in views if str(v.get("status", "")) == "accepted"]

    v2_cfg = (((session.get("config_snapshot", {}) or {}).get("calibration_v2", {}) or {}))
    min_corners = int(((v2_cfg.get("gating", {}) or {}).get("min_corners", 16)))

    inputs: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    proj_size: tuple[int, int] | None = None

    for view in accepted:
        view_id = str(view.get("view_id", ""))
        view_dir = session_dir / "views" / view_id
        try:
            charuco, report = _load_view_data(view_dir)
        except Exception as exc:
            dropped.append({"view_id": view_id, "reason": f"missing_files: {exc}"})
            continue

        corners = np.asarray(charuco.get("corners_px", []), dtype=np.float32).reshape(-1, 2)
        obj = np.asarray(charuco.get("object_points_m", []), dtype=np.float32).reshape(-1, 3)
        uv = (report.get("uv", {}) or {})
        proj = np.asarray(uv.get("projector_corners_px", []), dtype=np.float32).reshape(-1, 2)
        valid = np.asarray(uv.get("valid_mask", []), dtype=bool).reshape(-1)

        if corners.shape[0] == 0 or obj.shape[0] == 0:
            dropped.append({"view_id": view_id, "reason": "missing_charuco_points"})
            continue
        if corners.shape[0] != obj.shape[0] or proj.shape[0] != obj.shape[0] or valid.shape[0] != obj.shape[0]:
            dropped.append({"view_id": view_id, "reason": "shape_mismatch"})
            continue

        finite_proj = np.isfinite(proj[:, 0]) & np.isfinite(proj[:, 1])
        use = valid & finite_proj
        count = int(np.count_nonzero(use))
        if count < min_corners:
            dropped.append({"view_id": view_id, "reason": f"valid_uv_corners={count} < {min_corners}"})
            continue

        psize = _projector_size_from_report(report)
        if psize is None:
            dropped.append({"view_id": view_id, "reason": "missing_projector_size"})
            continue
        if proj_size is None:
            proj_size = psize
        elif proj_size != psize:
            dropped.append({"view_id": view_id, "reason": f"projector_size_mismatch={psize} expected={proj_size}"})
            continue

        inputs.append(
            {
                "view_id": view_id,
                "object_points": obj[use, :].astype(np.float32),
                "camera_points": corners[use, :].astype(np.float32),
                "projector_points": proj[use, :].astype(np.float32),
            }
        )

    if proj_size is None:
        proj_size = (
            int(((session.get("state", {}) or {}).get("projector_size", [1024, 768])[0])),
            int(((session.get("state", {}) or {}).get("projector_size", [1024, 768])[1])),
        )

    summary = {
        "accepted_views_in_session": int(len(accepted)),
        "usable_views": int(len(inputs)),
        "dropped_views": dropped,
    }
    return inputs, summary, proj_size


def _build_cv_lists(view_inputs: list[dict[str, Any]]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
    obj: list[np.ndarray] = []
    cam: list[np.ndarray] = []
    proj: list[np.ndarray] = []
    ids: list[str] = []
    for item in view_inputs:
        obj_pts = np.asarray(item["object_points"], dtype=np.float32).reshape(-1, 1, 3)
        cam_pts = np.asarray(item["camera_points"], dtype=np.float32).reshape(-1, 1, 2)
        proj_pts = np.asarray(item["projector_points"], dtype=np.float32).reshape(-1, 1, 2)
        obj.append(obj_pts)
        cam.append(cam_pts)
        proj.append(proj_pts)
        ids.append(str(item["view_id"]))
    return obj, cam, proj, ids


def _per_view_errors(
    view_inputs: list[dict[str, Any]],
    Kc: np.ndarray,
    Dc: np.ndarray,
    Kp: np.ndarray,
    Dp: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    cv = _require_cv2()
    per: list[dict[str, Any]] = []
    all_cam_resid: list[float] = []
    all_proj_resid: list[float] = []

    tvec_st = np.asarray(T, dtype=np.float64).reshape(3, 1)
    R_st = np.asarray(R, dtype=np.float64).reshape(3, 3)

    for item in view_inputs:
        view_id = str(item["view_id"])
        obj = np.asarray(item["object_points"], dtype=np.float64).reshape(-1, 3)
        cam = np.asarray(item["camera_points"], dtype=np.float64).reshape(-1, 2)
        proj = np.asarray(item["projector_points"], dtype=np.float64).reshape(-1, 2)

        if obj.shape[0] < 4:
            per.append(
                {
                    "view_id": view_id,
                    "corners_used": int(obj.shape[0]),
                    "camera_reproj_error_px": float("nan"),
                    "projector_reproj_error_px": float("nan"),
                    "combined_reproj_error_px": float("nan"),
                }
            )
            continue

        ok, rvec_c, tvec_c = cv.solvePnP(
            obj.astype(np.float32),
            cam.reshape(-1, 1, 2).astype(np.float32),
            Kc,
            Dc,
        )
        if not ok:
            per.append(
                {
                    "view_id": view_id,
                    "corners_used": int(obj.shape[0]),
                    "camera_reproj_error_px": float("nan"),
                    "projector_reproj_error_px": float("nan"),
                    "combined_reproj_error_px": float("nan"),
                }
            )
            continue

        cam_rep, _ = cv.projectPoints(obj, rvec_c, tvec_c, Kc, Dc)
        cam_rep_xy = cam_rep.reshape(-1, 2)
        cam_err = np.linalg.norm(cam_rep_xy - cam, axis=1)

        Rc, _ = cv.Rodrigues(rvec_c)
        Rp = R_st @ Rc
        tp = R_st @ np.asarray(tvec_c, dtype=np.float64).reshape(3, 1) + tvec_st
        rvec_p, _ = cv.Rodrigues(Rp)
        proj_rep, _ = cv.projectPoints(obj, rvec_p, tp, Kp, Dp)
        proj_rep_xy = proj_rep.reshape(-1, 2)
        proj_err = np.linalg.norm(proj_rep_xy - proj, axis=1)

        all_cam_resid.extend(cam_err.tolist())
        all_proj_resid.extend(proj_err.tolist())

        cam_rms = float(np.sqrt(np.mean(np.square(cam_err)))) if cam_err.size > 0 else float("nan")
        proj_rms = float(np.sqrt(np.mean(np.square(proj_err)))) if proj_err.size > 0 else float("nan")
        combined = np.concatenate([cam_err, proj_err], axis=0)
        comb_rms = float(np.sqrt(np.mean(np.square(combined)))) if combined.size > 0 else float("nan")

        per.append(
            {
                "view_id": view_id,
                "corners_used": int(obj.shape[0]),
                "camera_reproj_error_px": cam_rms,
                "projector_reproj_error_px": proj_rms,
                "combined_reproj_error_px": comb_rms,
            }
        )

    return per, np.asarray(all_cam_resid, dtype=np.float64), np.asarray(all_proj_resid, dtype=np.float64)


def _solve_once(
    *,
    session_id: str,
    view_inputs: list[dict[str, Any]],
    Kc: np.ndarray,
    Dc: np.ndarray,
    cam_size: tuple[int, int],
    proj_size: tuple[int, int],
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    cv = _require_cv2()
    obj, cam, proj, view_ids = _build_cv_lists(view_inputs)

    if len(obj) == 0:
        raise ValueError("No usable views for solve")

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-7)

    rms_proj, Kp, Dp, _, _ = cv.calibrateCamera(
        obj,
        proj,
        proj_size,
        None,
        None,
        criteria=criteria,
    )

    rms_stereo, Kc_o, Dc_o, Kp_o, Dp_o, R, T, E, F = cv.stereoCalibrate(
        obj,
        cam,
        proj,
        Kc.copy(),
        Dc.copy(),
        Kp,
        Dp,
        cam_size,
        criteria=criteria,
        flags=cv.CALIB_FIX_INTRINSIC,
    )

    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(
        Kc_o,
        Dc_o,
        Kp_o,
        Dp_o,
        cam_size,
        R,
        T,
    )

    per_view, cam_resid, proj_resid = _per_view_errors(
        view_inputs,
        Kc=Kc_o,
        Dc=Dc_o,
        Kp=Kp_o,
        Dp=Dp_o,
        R=R,
        T=T,
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "views_used": int(len(view_ids)),
        "view_ids": list(view_ids),
        "rms_projector_intrinsics": float(rms_proj),
        "rms_stereo": float(rms_stereo),
        "camera_matrix": Kc_o.astype(float).tolist(),
        "camera_dist_coeffs": Dc_o.reshape(-1).astype(float).tolist(),
        "projector_matrix": Kp_o.astype(float).tolist(),
        "projector_dist_coeffs": Dp_o.reshape(-1).astype(float).tolist(),
        "R": R.astype(float).tolist(),
        "T": T.reshape(-1).astype(float).tolist(),
        "E": E.astype(float).tolist(),
        "F": F.astype(float).tolist(),
        "rectification": {
            "R1": R1.astype(float).tolist(),
            "R2": R2.astype(float).tolist(),
            "P1": P1.astype(float).tolist(),
            "P2": P2.astype(float).tolist(),
            "Q": Q.astype(float).tolist(),
        },
        "per_view_errors": per_view,
        "projector": {
            "width": int(proj_size[0]),
            "height": int(proj_size[1]),
        },
    }

    mats = {
        "camera_matrix": Kc_o,
        "camera_dist": Dc_o,
        "projector_matrix": Kp_o,
        "projector_dist": Dp_o,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
    }
    return result, mats, cam_resid, proj_resid


def _subset_inputs(inputs: list[dict[str, Any]], view_ids: list[str]) -> list[dict[str, Any]]:
    keep = set(view_ids)
    return [v for v in inputs if str(v.get("view_id")) in keep]


def solve_session(session_dir: Path) -> dict[str, Any]:
    session_path = session_dir / "session.json"
    if not session_path.exists():
        raise FileNotFoundError(f"session.json missing: {session_path}")
    session = json.loads(session_path.read_text())

    v2_cfg = (((session.get("config_snapshot", {}) or {}).get("calibration_v2", {}) or {}))
    solve_cfg = (v2_cfg.get("solve", {}) or {})
    min_views = max(2, int(solve_cfg.get("min_views", 8)))
    min_improvement_px = float(solve_cfg.get("min_improvement_px", 0.05))
    max_prune_steps = max(0, int(solve_cfg.get("max_prune_steps", 6)))

    camera_intrinsics_path = Path(str((session.get("config_snapshot", {}) or {}).get("camera_intrinsics_path", "")))
    Kc, Dc, cam_size = _load_camera_intrinsics(camera_intrinsics_path)

    inputs, inputs_summary, proj_size = _gather_inputs(session_dir, session)
    if len(inputs) < min_views:
        raise ValueError(
            f"Need at least {min_views} accepted usable views for solve; found {len(inputs)}"
        )

    sid = str(session.get("session_id", session_dir.name))

    raw_result, raw_mats, raw_cam_resid, raw_proj_resid = _solve_once(
        session_id=sid,
        view_inputs=inputs,
        Kc=Kc,
        Dc=Dc,
        cam_size=cam_size,
        proj_size=proj_size,
    )

    current_ids = list(raw_result.get("view_ids", []))
    current_result = raw_result
    current_mats = raw_mats
    current_cam_resid = raw_cam_resid
    current_proj_resid = raw_proj_resid

    prune_steps: list[dict[str, Any]] = []

    for step_idx in range(max_prune_steps):
        if len(current_ids) <= min_views:
            break

        per_view = [v for v in (current_result.get("per_view_errors", []) or []) if isinstance(v, dict)]
        if len(per_view) == 0:
            break

        def _score(entry: dict[str, Any]) -> float:
            try:
                return float(entry.get("projector_reproj_error_px", float("nan")))
            except Exception:
                return float("nan")

        valid_scores = [e for e in per_view if np.isfinite(_score(e))]
        if len(valid_scores) == 0:
            break

        worst = max(valid_scores, key=_score)
        worst_id = str(worst.get("view_id", ""))
        if not worst_id:
            break

        candidate_ids = [vid for vid in current_ids if vid != worst_id]
        if len(candidate_ids) < min_views:
            break

        candidate_inputs = _subset_inputs(inputs, candidate_ids)
        candidate_result, candidate_mats, candidate_cam_resid, candidate_proj_resid = _solve_once(
            session_id=sid,
            view_inputs=candidate_inputs,
            Kc=Kc,
            Dc=Dc,
            cam_size=cam_size,
            proj_size=proj_size,
        )

        current_rms = float(current_result.get("rms_stereo", float("nan")))
        candidate_rms = float(candidate_result.get("rms_stereo", float("nan")))
        improvement = current_rms - candidate_rms

        step = {
            "step": int(step_idx + 1),
            "removed_view_id": worst_id,
            "rms_before": current_rms,
            "rms_after": candidate_rms,
            "improvement_px": float(improvement),
            "accepted": bool(np.isfinite(improvement) and improvement >= min_improvement_px),
        }
        prune_steps.append(step)

        if not step["accepted"]:
            break

        current_ids = candidate_ids
        current_result = candidate_result
        current_mats = candidate_mats
        current_cam_resid = candidate_cam_resid
        current_proj_resid = candidate_proj_resid

    pruned_result = current_result
    pruned_mats = current_mats
    pruned_cam_resid = current_cam_resid
    pruned_proj_resid = current_proj_resid

    raw_rms = float(raw_result.get("rms_stereo", float("nan")))
    pruned_rms = float(pruned_result.get("rms_stereo", float("nan")))
    selected_model = "pruned" if np.isfinite(pruned_rms) and np.isfinite(raw_rms) and pruned_rms < raw_rms else "raw"

    suggestions: list[str] = []
    coverage_sufficient = bool(((session.get("state", {}) or {}).get("coverage_sufficient", False)))
    if not coverage_sufficient:
        suggestions.append("Coverage is not yet sufficient. Capture more diverse board poses.")
    if len(prune_steps) == 0:
        suggestions.append("No pruning step improved RMS by threshold.")
    if len(inputs) == min_views:
        suggestions.append("Solve used the minimum number of views. Capture more views for robustness.")

    solve_dir = session_dir / "solve"
    plots_dir = solve_dir / "plots"
    export_dir = session_dir / "export"
    solve_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    inputs_summary_payload = {
        "session_id": sid,
        "camera_intrinsics_path": str(camera_intrinsics_path),
        "camera_size": [int(cam_size[0]), int(cam_size[1])],
        "projector_size": [int(proj_size[0]), int(proj_size[1])],
        "solve_config": {
            "min_views": int(min_views),
            "min_improvement_px": float(min_improvement_px),
            "max_prune_steps": int(max_prune_steps),
        },
        **inputs_summary,
        "usable_view_ids": [str(v.get("view_id")) for v in inputs],
    }
    (solve_dir / "inputs_summary.json").write_text(json.dumps(_json_safe(inputs_summary_payload), indent=2))

    (solve_dir / "stereo_raw.json").write_text(json.dumps(_json_safe(raw_result), indent=2))
    (solve_dir / "stereo_pruned.json").write_text(json.dumps(_json_safe(pruned_result), indent=2))

    solve_report = build_solve_report(
        raw_result=raw_result,
        pruned_result=pruned_result,
        selected_model=selected_model,
        prune_steps=prune_steps,
        suggestions=suggestions,
    )
    (solve_dir / "solve_report.json").write_text(json.dumps(_json_safe(solve_report), indent=2))

    coverage = ((session.get("state", {}) or {}).get("coverage", {}) or {})
    save_coverage_plot(coverage, plots_dir / "coverage.png")

    plot_source = pruned_result if selected_model == "pruned" else raw_result
    save_reprojection_plot(
        plot_source.get("per_view_errors", []) or [],
        key="camera_reproj_error_px",
        title="Camera Reprojection Error by View",
        out_path=plots_dir / "reproj_cam.png",
    )
    save_reprojection_plot(
        plot_source.get("per_view_errors", []) or [],
        key="projector_reproj_error_px",
        title="Projector Reprojection Error by View",
        out_path=plots_dir / "reproj_proj.png",
    )

    hist_cam = pruned_cam_resid if selected_model == "pruned" else raw_cam_resid
    hist_proj = pruned_proj_resid if selected_model == "pruned" else raw_proj_resid
    save_residual_histogram(hist_cam, hist_proj, plots_dir / "residual_hist.png")

    selected_result = pruned_result if selected_model == "pruned" else raw_result
    export_path = export_dir / "stereo.json"
    export_path.write_text(json.dumps(_json_safe(selected_result), indent=2))

    np.savez(solve_dir / "stereo_raw.npz", **raw_mats)
    np.savez(solve_dir / "stereo_pruned.npz", **pruned_mats)

    session_state = (session.get("state", {}) or {})
    session_state["solve"] = {
        "solved_at": datetime.now().isoformat(),
        "raw_rms_stereo": raw_rms,
        "pruned_rms_stereo": pruned_rms,
        "selected_model": selected_model,
        "views_used": int(selected_result.get("views_used", 0)),
        "export_stereo_path": "export/stereo.json",
        "solve_report_path": "solve/solve_report.json",
    }
    session["state"] = session_state
    (session_dir / "session.json").write_text(json.dumps(_json_safe(session), indent=2))

    result_summary = {
        "session_id": sid,
        "raw_rms_stereo": raw_rms,
        "pruned_rms_stereo": pruned_rms,
        "selected_model": selected_model,
        "views_used": int(selected_result.get("views_used", 0)),
        "export_stereo_path": str(export_path),
        "solve_report_path": str(solve_dir / "solve_report.json"),
        "inputs_summary_path": str(solve_dir / "inputs_summary.json"),
    }
    return _json_safe(result_summary)
