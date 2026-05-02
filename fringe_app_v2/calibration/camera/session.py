"""
Camera intrinsics calibration session management and solve.

Filesystem layout:
    data/calibration/camera/
        sessions/
            20240101_120000/
                session.json
                captures/
                    000001/
                        image.png
                        charuco.json
                        overlay.png
        intrinsics_latest.json
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fringe_app_v2.calibration.camera.charuco import detect_charuco, CharucoDetection


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


def _camera_root(config: dict[str, Any]) -> Path:
    calib = config.get("calibration", {}) or {}
    root = Path(str(calib.get("root", "data/calibration")))
    cam_root = calib.get("camera_root")
    return Path(str(cam_root)) if cam_root else root / "camera"


# ── Session CRUD ──────────────────────────────────────────────────────────────

def create_session(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    root = _camera_root(config)
    sessions_root = root / "sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sid = stamp
    i = 0
    while (sessions_root / sid).exists():
        i += 1
        sid = f"{stamp}_{i:02d}"

    session_dir = sessions_root / sid
    (session_dir / "captures").mkdir(parents=True, exist_ok=True)

    charuco_cfg = (config.get("calibration", {}) or {}).get("charuco") or {
        "dict": "DICT_4X4_50",
        "squares_x": 9,
        "squares_y": 7,
        "square_length_m": 0.010,
        "marker_length_m": 0.0075,
    }
    min_detections = int((config.get("calibration", {}) or {}).get("min_valid_detections", 10))

    payload: dict[str, Any] = {
        "session_id": sid,
        "created_at": datetime.now().isoformat(),
        "charuco": charuco_cfg,
        "min_valid_detections": min_detections,
        "captures": [],
        "calibration": None,
    }
    _save_session(session_dir, payload)
    return session_dir, payload


def load_session(session_dir: Path) -> dict[str, Any]:
    p = session_dir / "session.json"
    if not p.exists():
        raise FileNotFoundError(f"Camera calibration session not found: {session_dir}")
    return json.loads(p.read_text())


def _save_session(session_dir: Path, payload: dict[str, Any]) -> None:
    (session_dir / "session.json").write_text(json.dumps(_json_safe(payload), indent=2))


def list_sessions(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = _camera_root(config)
    sessions_root = root / "sessions"
    if not sessions_root.exists():
        return []
    result: list[dict[str, Any]] = []
    for p in sorted(sessions_root.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        sf = p / "session.json"
        if not sf.exists():
            continue
        try:
            sess = json.loads(sf.read_text())
        except Exception:
            continue
        captures = sess.get("captures", []) or []
        good = [c for c in captures if c.get("found")]
        result.append({
            "session_id": str(sess.get("session_id", p.name)),
            "created_at": sess.get("created_at"),
            "n_captures": len(captures),
            "n_good": len(good),
            "solved": sess.get("calibration") is not None,
        })
    return result


def get_session_dir(config: dict[str, Any], session_id: str) -> Path:
    root = _camera_root(config)
    return root / "sessions" / session_id


# ── Capture ──────────────────────────────────────────────────────────────────

def add_capture(
    session_dir: Path,
    image: np.ndarray,
    charuco_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Detect ChArUco in image, save results, append to session.

    Returns the capture record dict.
    """
    session = load_session(session_dir)
    captures: list[dict[str, Any]] = list(session.get("captures", []) or [])

    idx = len(captures) + 1
    cap_dir = session_dir / "captures" / f"{idx:06d}"
    cap_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(cap_dir / "image.png"), image)

    det, overlay, _ = detect_charuco(image, charuco_cfg)
    cv2.imwrite(str(cap_dir / "overlay.png"), overlay)
    charuco_json = det.to_json_dict()
    (cap_dir / "charuco.json").write_text(json.dumps(charuco_json, indent=2))

    record: dict[str, Any] = {
        "capture_id": f"{idx:06d}",
        "found": bool(det.found),
        "marker_count": int(det.marker_count),
        "corner_count": int(det.corner_count),
        "image_size": list(det.image_size),
        "reject_reason": det.reject_reason,
        "files": {
            "image": str(cap_dir / "image.png"),
            "overlay": str(cap_dir / "overlay.png"),
            "charuco": str(cap_dir / "charuco.json"),
        },
    }
    captures.append(record)
    session["captures"] = captures
    _save_session(session_dir, session)
    return record


def delete_capture(session_dir: Path, capture_id: str) -> dict[str, Any]:
    import shutil
    session = load_session(session_dir)
    captures = [c for c in (session.get("captures", []) or []) if str(c.get("capture_id")) != capture_id]
    cap_dir = session_dir / "captures" / capture_id
    if cap_dir.exists():
        shutil.rmtree(cap_dir)
    session["captures"] = captures
    _save_session(session_dir, session)
    return session


# ── Solve ─────────────────────────────────────────────────────────────────────

def solve_session(session_dir: Path, output_root: Path | None = None) -> dict[str, Any]:
    """
    Run cv2.calibrateCamera on all accepted captures and save intrinsics.

    Writes:
        session_dir/session.json  (updated with calibration result)
        output_root/intrinsics_latest.json   (compatible with load_camera_intrinsics)

    Returns:
        The calibration result dict.
    """
    session = load_session(session_dir)
    charuco_cfg = session.get("charuco", {}) or {}

    obj_pts: list[np.ndarray] = []
    img_pts: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    used_ids: list[str] = []

    for cap in (session.get("captures", []) or []):
        if not cap.get("found"):
            continue
        cap_dir = session_dir / "captures" / str(cap["capture_id"])
        charuco_path = cap_dir / "charuco.json"
        if not charuco_path.exists():
            continue
        charuco = json.loads(charuco_path.read_text())

        corners = np.asarray(charuco.get("corners_px", []), dtype=np.float32).reshape(-1, 1, 2)
        obj = np.asarray(charuco.get("object_points_m", []), dtype=np.float32).reshape(-1, 1, 3)
        if corners.shape[0] < 4 or obj.shape[0] != corners.shape[0]:
            continue

        obj_pts.append(obj)
        img_pts.append(corners)
        used_ids.append(str(cap["capture_id"]))

        if image_size is None:
            sz = charuco.get("image_size")
            if isinstance(sz, (list, tuple)) and len(sz) == 2:
                image_size = (int(sz[0]), int(sz[1]))

    min_det = int(session.get("min_valid_detections", 10))
    if len(obj_pts) < min_det:
        raise ValueError(
            f"Need at least {min_det} valid captures; found {len(obj_pts)}. "
            "Capture more board positions."
        )
    if image_size is None:
        raise ValueError("No image_size found in captures.")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7)
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, image_size, None, None, criteria=criteria
    )

    # Per-view reprojection errors
    per_view: list[dict[str, Any]] = []
    for i, (obj, img, rvec, tvec, vid) in enumerate(zip(obj_pts, img_pts, rvecs, tvecs, used_ids)):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        errs = np.linalg.norm(projected.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
        per_view.append({
            "capture_id": vid,
            "rms_px": float(np.sqrt(np.mean(np.square(errs)))),
            "n_corners": int(obj.shape[0]),
        })

    calibration: dict[str, Any] = {
        "solved_at": datetime.now().isoformat(),
        "rms_px": float(rms),
        "n_views": int(len(obj_pts)),
        "capture_ids_used": used_ids,
        "camera_matrix": K.tolist(),
        "dist_coeffs": D.reshape(-1).tolist(),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "per_view_errors": per_view,
    }

    session["calibration"] = calibration
    _save_session(session_dir, session)

    # Write intrinsics_latest.json in the camera root
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        out_path = output_root / "intrinsics_latest.json"
        out_path.write_text(json.dumps(_json_safe(calibration), indent=2))

    return calibration
