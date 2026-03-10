"""Projector calibration v2 workflow."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fringe_app.calibration.projector_stereo import run_uv_scan

from .charuco_detect import (
    CharucoDetection,
    charuco_object_points,
    detect_charuco,
)
from .session_store import (
    allocate_view_id,
    build_session_payload,
    build_session_zip,
    create_session,
    delete_view,
    get_session,
    list_sessions,
    load_session,
    register_view,
    reset_session,
    save_session,
)
from .stereo_solve import solve_session
from .uv_refine import UvRefineConfig, sample_and_refine_uv
from .view_gating import evaluate_view


def _save_image(path: Path, image: np.ndarray) -> None:
    arr = np.asarray(image)
    if arr.ndim == 2:
        out = arr.astype(np.uint8)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        out = arr[:, :, :3].astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image array shape: {arr.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(path)


def _capture_checkerboard_frame(controller, cfg: dict[str, Any]) -> np.ndarray:
    pcal = cfg.get("projector_calibration", {}) or {}
    cap_cfg = pcal.get("capture", {}) or {}
    cam_cfg = pcal.get("camera", {}) or {}

    dn = int(cap_cfg.get("checkerboard_white_dn", 230))
    white_settle_ms = int(cap_cfg.get("checkerboard_white_settle_ms", 200))
    off_settle_ms = int(cap_cfg.get("checkerboard_off_settle_ms", 60))

    flush_frames = int(cap_cfg.get("flush_frames", 1))
    settle_ms = int(cam_cfg.get("settle_ms", 120))

    try:
        controller.set_projector_calibration_light(enabled=True, dn=dn)
        time.sleep(max(0.0, white_settle_ms / 1000.0))

        frame = controller.capture_single_frame(
            flush_frames=flush_frames,
            exposure_us=cam_cfg.get("exposure_us"),
            analogue_gain=cam_cfg.get("analogue_gain"),
            awb_enable=cam_cfg.get("awb_enable"),
            settle_ms=settle_ms,
        )
    finally:
        try:
            controller.set_projector_calibration_light(enabled=False, dn=0)
        except Exception:
            pass
        if off_settle_ms > 0:
            time.sleep(max(0.0, off_settle_ms / 1000.0))

    return frame


def _load_camera_intrinsics_for_pose(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not path.exists():
        return None, None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None, None
    K = np.asarray(payload.get("camera_matrix", []), dtype=np.float64)
    D = np.asarray(payload.get("dist_coeffs", []), dtype=np.float64).reshape(-1, 1)
    if K.shape != (3, 3) or D.size == 0:
        return None, None
    return K, D


def _projector_size_from_uv_meta(uv_meta: dict[str, Any], fallback: tuple[int, int]) -> tuple[int, int]:
    proj = uv_meta.get("projector_resolution", {}) or {}
    surf = proj.get("pattern_surface")
    if isinstance(surf, (list, tuple)) and len(surf) == 2:
        try:
            return int(surf[0]), int(surf[1])
        except Exception:
            return fallback
    w = uv_meta.get("projector_width")
    h = uv_meta.get("projector_height")
    if w is not None and h is not None:
        try:
            return int(w), int(h)
        except Exception:
            return fallback
    return fallback


def _view_files(view_id: str) -> dict[str, str | None]:
    base = Path("views") / view_id
    return {
        "camera": str(base / "camera.png"),
        "overlay": str(base / "overlay.png"),
        "charuco": str(base / "charuco.json"),
        "view_report": str(base / "view_report.json"),
        "uv_overlay": str(base / "uv" / "uv_overlay.png"),
    }


def create_projector_v2_session(cfg: dict[str, Any]) -> dict[str, Any]:
    sdir, _ = create_session(cfg)
    return build_session_payload(sdir)


def list_projector_v2_sessions(cfg: dict[str, Any]) -> dict[str, Any]:
    return {"sessions": list_sessions(cfg)}


def get_projector_v2_session(cfg: dict[str, Any], session_id: str) -> dict[str, Any]:
    sdir, _ = get_session(cfg, session_id)
    return build_session_payload(sdir)


def delete_projector_v2_view(cfg: dict[str, Any], session_id: str, view_id: str) -> dict[str, Any]:
    sdir, _ = get_session(cfg, session_id)
    delete_view(sdir, view_id)
    return build_session_payload(sdir)


def reset_projector_v2_session(cfg: dict[str, Any], session_id: str) -> dict[str, Any]:
    sdir, _ = get_session(cfg, session_id)
    reset_session(sdir)
    return build_session_payload(sdir)


def download_projector_v2_session_zip(cfg: dict[str, Any], session_id: str) -> Path:
    sdir, _ = get_session(cfg, session_id)
    return build_session_zip(sdir)


def _capture_cfg_for_uv(cfg: dict[str, Any]) -> dict[str, Any]:
    # Reuse existing scan/phase/unwrap pipeline as-is, only capture products needed for v2.
    out = dict(cfg)
    pcal = dict((out.get("projector_calibration", {}) or {}))
    proj = dict((pcal.get("projector", {}) or {}))
    scan = out.get("scan", {}) or {}
    proj.setdefault("width", int(scan.get("width", 1024)))
    proj.setdefault("height", int(scan.get("height", 768)))
    pcal["projector"] = proj
    out["projector_calibration"] = pcal
    return out


def capture_projector_v2_view(
    cfg: dict[str, Any],
    controller,
    session_id: str,
) -> dict[str, Any]:
    session_dir, session = get_session(cfg, session_id)

    v2_cfg = (((session.get("config_snapshot", {}) or {}).get("calibration_v2", {}) or {}))
    gating_cfg = (v2_cfg.get("gating", {}) or {})
    uv_cfg = UvRefineConfig.from_dict(v2_cfg.get("uv_refine", {}) or {})

    view_id, session = allocate_view_id(session_dir, session)
    view_dir = session_dir / "views" / view_id
    view_dir.mkdir(parents=True, exist_ok=False)

    frame = _capture_checkerboard_frame(controller, cfg)
    _save_image(view_dir / "camera.png", frame)

    board = None
    try:
        detection, overlay, board = detect_charuco(frame, v2_cfg.get("charuco", {}) or {})
    except Exception as exc:
        detection = CharucoDetection(
            found=False,
            marker_count=0,
            corner_count=0,
            image_size=(int(frame.shape[1]), int(frame.shape[0])),
            ids=np.zeros((0,), dtype=np.int32),
            corners=np.zeros((0, 2), dtype=np.float32),
            reject_reason="charuco_detection_exception",
            hint=str(exc),
        )
        overlay = np.asarray(frame).copy()
    _save_image(view_dir / "overlay.png", overlay)

    object_points = np.zeros((0, 3), dtype=np.float32)
    if board is not None and detection.corners.shape[0] > 0 and detection.ids.shape[0] == detection.corners.shape[0]:
        object_points = charuco_object_points(board, detection.ids)

    intr_path = Path(str((session.get("config_snapshot", {}) or {}).get("camera_intrinsics_path", "")))
    K, D = _load_camera_intrinsics_for_pose(intr_path)

    gate = evaluate_view(
        marker_count=int(detection.marker_count),
        corner_count=int(detection.corner_count),
        corners_px=detection.corners,
        image_shape_hw=frame.shape[:2],
        gating_cfg=gating_cfg,
        object_points=object_points if object_points.shape[0] > 0 else None,
        camera_matrix=K,
        dist_coeffs=D,
    )

    reasons = list(gate.get("reasons", []) or [])
    hints = list(gate.get("hints", []) or [])
    if detection.reject_reason:
        reasons.insert(0, str(detection.reject_reason))
    if detection.hint:
        hints.append(str(detection.hint))
    accepted = bool(gate.get("accepted", False)) and bool(detection.found)

    uv_payload: dict[str, Any] = {
        "available": False,
        "projector_size": None,
        "valid_count": 0,
        "valid_ratio": 0.0,
        "projector_corners_px": [],
        "valid_mask": [],
        "refine": {},
    }

    if accepted:
        try:
            uv_dir = view_dir / "uv"
            uv_cfg_run = _capture_cfg_for_uv(cfg)
            u, v, mask_uv, uv_meta, run_info = run_uv_scan(controller, uv_cfg_run, uv_dir)

            np.save(uv_dir / "u.npy", u.astype(np.float32))
            np.save(uv_dir / "v.npy", v.astype(np.float32))
            np.save(uv_dir / "mask.npy", np.asarray(mask_uv, dtype=bool))
            if (uv_dir / "mask_uv.npy").exists():
                shutil.copy2(uv_dir / "mask_uv.npy", uv_dir / "mask.npy")

            proj_size = _projector_size_from_uv_meta(
                uv_meta,
                fallback=(int((cfg.get("scan", {}) or {}).get("width", 1024)), int((cfg.get("scan", {}) or {}).get("height", 768))),
            )
            sampled_uv, valid_mask, refine_diag = sample_and_refine_uv(
                u_map=u,
                v_map=v,
                mask=mask_uv,
                corners_px=detection.corners,
                projector_size=proj_size,
                cfg=uv_cfg,
            )

            valid_count = int(np.count_nonzero(valid_mask))
            valid_ratio = float(valid_count / max(1, valid_mask.size))

            min_corners = int(gating_cfg.get("min_corners", 16))
            if valid_count < min_corners:
                accepted = False
                reasons.append(f"UV-valid corners {valid_count} below minimum {min_corners}.")
                hints.append("Re-capture with better phase signal on the board area.")

            state = (session.get("state", {}) or {})
            expected_size_raw = state.get("projector_size")
            if expected_size_raw is not None:
                if isinstance(expected_size_raw, (list, tuple)) and len(expected_size_raw) == 2:
                    expected_size = (int(expected_size_raw[0]), int(expected_size_raw[1]))
                    if expected_size != proj_size:
                        accepted = False
                        reasons.append(f"Projector size mismatch for view ({proj_size}) expected ({expected_size}).")
                        hints.append("Use the same projector mode/resolution across all captures.")

            uv_payload = {
                "available": True,
                "projector_size": [int(proj_size[0]), int(proj_size[1])],
                "valid_count": valid_count,
                "valid_ratio": valid_ratio,
                "projector_corners_px": sampled_uv.astype(float).tolist(),
                "valid_mask": valid_mask.astype(bool).tolist(),
                "refine": refine_diag,
                "uv_meta": uv_meta,
                "uv_source_runs": run_info,
            }
        except Exception as exc:
            accepted = False
            reasons.append(f"UV capture failed: {exc}")
            hints.append("Ensure scan/phase/unwrap succeeds and recapture this view.")

    status = "accepted" if accepted else "rejected"
    reason = reasons[0] if len(reasons) > 0 else None

    charuco_payload = detection.to_json_dict()
    charuco_payload["object_points_m"] = object_points.astype(float).tolist()
    (view_dir / "charuco.json").write_text(json.dumps(charuco_payload, indent=2))

    metrics = dict((gate.get("metrics", {}) or {}))
    metrics["uv_valid_count"] = int(uv_payload.get("valid_count", 0))
    metrics["uv_valid_ratio"] = float(uv_payload.get("valid_ratio", 0.0))

    view_report = {
        "schema_version": 1,
        "view_id": view_id,
        "status": status,
        "accepted": bool(accepted),
        "reason": reason,
        "reasons": reasons,
        "hints": sorted(set(hints)),
        "gating": {
            "thresholds": gate.get("thresholds", {}),
            "metrics": gate.get("metrics", {}),
        },
        "charuco": {
            "marker_count": int(detection.marker_count),
            "corner_count": int(detection.corner_count),
            "ids": [int(v) for v in detection.ids.reshape(-1).tolist()],
            "corners_px": [[float(x), float(y)] for x, y in detection.corners.reshape(-1, 2)],
            "object_points_m": object_points.astype(float).tolist(),
        },
        "uv": uv_payload,
    }
    (view_dir / "view_report.json").write_text(json.dumps(view_report, indent=2))

    record = {
        "view_id": view_id,
        "status": status,
        "reason": reason,
        "hints": sorted(set(hints)),
        "metrics": metrics,
        "files": _view_files(view_id),
    }

    updated = register_view(session_dir, record)

    if accepted and uv_payload.get("projector_size") is not None:
        st = (updated.get("state", {}) or {})
        st["projector_size"] = list(uv_payload["projector_size"])
        updated["state"] = st
        save_session(session_dir, updated)

    payload = build_session_payload(session_dir)
    payload["last_capture"] = {
        "view_id": view_id,
        "accepted": bool(accepted),
        "reason": reason,
        "reasons": reasons,
        "hints": sorted(set(hints)),
    }
    return payload


def solve_projector_v2_session(cfg: dict[str, Any], session_id: str) -> dict[str, Any]:
    session_dir, _ = get_session(cfg, session_id)
    solve_result = solve_session(session_dir)
    payload = build_session_payload(session_dir)
    payload["solve_result"] = solve_result
    return payload
