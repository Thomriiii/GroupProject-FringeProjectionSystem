"""Checkerboard detection and camera intrinsics calibration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import cv2
except Exception as exc:  # pragma: no cover - hard fail on systems without OpenCV
    cv2 = None
    _cv2_import_error = exc
else:
    _cv2_import_error = None

_log = logging.getLogger(__name__)


def _require_cv2() -> Any:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for checkerboard calibration. "
            f"Import error: {_cv2_import_error}"
        )
    return cv2


def to_gray_u8(image: np.ndarray) -> np.ndarray:
    """Convert RGB/gray image to uint8 gray."""
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.ndim == 3 and image.shape[2] >= 3:
        f = image.astype(np.float32)
        gray = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
        return np.clip(np.rint(gray), 0, 255).astype(np.uint8)
    raise ValueError("Unsupported image shape for grayscale conversion")


@dataclass(slots=True)
class CheckerboardDetection:
    found: bool
    corners_px: list[list[float]]
    image_size: tuple[int, int]  # width, height
    corner_count: int
    method: str
    detection_mode: str = "checkerboard"
    rejection_reason: str | None = None
    rejection_hint: str | None = None
    diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "corners_px": self.corners_px,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "corner_count": int(self.corner_count),
            "method": self.method,
            "detection_mode": self.detection_mode,
            "rejection_reason": self.rejection_reason,
            "rejection_hint": self.rejection_hint,
            "diagnostics": self.diagnostics or {},
        }


def detect_checkerboard(
    image: np.ndarray,
    cols: int,
    rows: int,
    refine_subpix: bool = True,
    board_type: str = "checkerboard",
    charuco_cfg: dict[str, Any] | None = None,
) -> tuple[CheckerboardDetection, np.ndarray]:
    """Detect checkerboard corners and return detection + overlay image."""
    cv = _require_cv2()
    gray = to_gray_u8(image)
    pattern_size = (int(cols), int(rows))

    found = False
    corners: np.ndarray | None = None
    method = "none"
    reject_reason: str | None = None
    reject_hint: str | None = None
    diag: dict[str, Any] = {}
    charuco_ids_arr: np.ndarray | None = None

    board_mode = str(board_type or "checkerboard").strip().lower()
    diag["detection_mode"] = board_mode

    if board_mode == "charuco":
        cfg = dict(charuco_cfg or {})
        aruco = getattr(cv, "aruco", None)
        dict_name = str(cfg.get("dictionary", "DICT_4X4_50"))
        squares_x = int(cfg.get("squares_x", cols + 1))
        squares_y = int(cfg.get("squares_y", rows + 1))
        square_len = float(cfg.get("square_length_m", cfg.get("square_length", 0.012)))
        marker_len = float(cfg.get("marker_length_m", cfg.get("marker_length", 0.009)))
        expected = int(cols * rows)
        diag["image_size"] = [int(gray.shape[1]), int(gray.shape[0])]
        diag["charuco_cfg"] = {
            "dictionary": dict_name,
            "squares_x": squares_x,
            "squares_y": squares_y,
            "square_length_m": square_len,
            "marker_length_m": marker_len,
            "expected_corners": expected,
        }
        _log.info("Charuco detection dictionary=%s", dict_name)

        config_valid = True
        config_error: str | None = None
        if aruco is None:
            config_valid = False
            config_error = "OpenCV ArUco module unavailable."
        elif not hasattr(aruco, dict_name):
            config_valid = False
            config_error = f"Unknown ArUco dictionary: {dict_name}"
        elif squares_x < 4 or squares_y < 4:
            config_valid = False
            config_error = "Charuco squares_x/squares_y must be >= 4."
        elif marker_len >= square_len:
            config_valid = False
            config_error = "marker_length must be smaller than square_length."

        if not config_valid:
            reject_reason = "invalid_board_configuration"
            reject_hint = config_error
            diag["invalid_board_configuration"] = {
                "message": config_error,
                "config": {
                    "dictionary": dict_name,
                    "squares_x": squares_x,
                    "squares_y": squares_y,
                    "square_length_m": square_len,
                    "marker_length_m": marker_len,
                },
            }

        if config_valid:
            try:
                dictionary_id = getattr(aruco, dict_name)
                dictionary = aruco.getPredefinedDictionary(dictionary_id)
                board = aruco.CharucoBoard(
                    (int(squares_x), int(squares_y)),
                    float(square_len),
                    float(marker_len),
                    dictionary,
                )
                marker_corners, marker_ids, _ = aruco.detectMarkers(gray, dictionary)
                method = "aruco.detectMarkers"
                marker_count = 0 if marker_ids is None else int(len(marker_ids))
                diag["aruco_markers_detected"] = marker_count
                diag["aruco_ids"] = [] if marker_ids is None else [int(v) for v in marker_ids.reshape(-1).tolist()]
                if marker_count > 0 and marker_corners is not None:
                    pts = np.concatenate([np.asarray(mc, dtype=np.float32).reshape(-1, 2) for mc in marker_corners], axis=0)
                    if pts.shape[0] >= 3:
                        hull = cv.convexHull(pts.reshape(-1, 1, 2))
                        area = float(cv.contourArea(hull))
                        img_area = float(max(1, gray.shape[0] * gray.shape[1]))
                        diag["board_visible_fraction"] = float(area / img_area)
                    else:
                        diag["board_visible_fraction"] = 0.0
                else:
                    diag["board_visible_fraction"] = 0.0

                if marker_ids is None or len(marker_ids) == 0:
                    reject_reason = "no_aruco_markers"
                    reject_hint = "Check lighting, dictionary, or marker visibility."
                elif len(marker_ids) < 4:
                    reject_reason = "insufficient_markers"
                    reject_hint = "Board too oblique or partially visible."
                else:
                    ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                        marker_corners, marker_ids, gray, board
                    )
                    charuco_count = 0 if charuco_ids is None else int(len(charuco_ids))
                    diag["charuco_corners_detected"] = charuco_count
                    method = "aruco.interpolateCornersCharuco"
                    if charuco_corners is None or charuco_ids is None or ret is None or float(ret) < 6.0:
                        reject_reason = "charuco_interpolation_failed"
                        reject_hint = "Board partially visible or incorrect board config."
                    else:
                        ids = charuco_ids.reshape(-1).astype(np.int32)
                        order = np.argsort(ids)
                        corners = charuco_corners[order].astype(np.float32)
                        charuco_ids_arr = ids[order].reshape(-1, 1).astype(np.int32)
                        diag["charuco_ids"] = [int(v) for v in charuco_ids_arr.reshape(-1).tolist()]
                        if refine_subpix and corners is not None and corners.shape[0] > 0:
                            criteria = (
                                cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                                30,
                                1e-3,
                            )
                            cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                        found = bool(corners is not None and corners.shape[0] >= 6)
                        if not found:
                            reject_reason = "charuco_interpolation_failed"
                            reject_hint = "Board partially visible or incorrect board config."
                        else:
                            reject_reason = None
                            reject_hint = None
            except Exception as exc:
                reject_reason = "detection_exception"
                reject_hint = str(exc)
                diag["exception"] = str(exc)
                found = False
                corners = None
    else:
        try:
            sb_flags = cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY
            found, corners = cv.findChessboardCornersSB(gray, pattern_size, flags=sb_flags)
            method = "findChessboardCornersSB"
        except Exception:
            found = False
            corners = None

        if not found or corners is None:
            flags = cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
            found, corners = cv.findChessboardCorners(gray, pattern_size, flags=flags)
            method = "findChessboardCorners"
            if found and corners is not None and refine_subpix:
                criteria = (
                    cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                    30,
                    1e-3,
                )
                cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if not found:
            reject_reason = "checkerboard_not_found"
            reject_hint = "Ensure full checkerboard is visible and in focus."

    if image.ndim == 2:
        overlay = np.stack([image, image, image], axis=2).astype(np.uint8)
    elif image.ndim == 3 and image.shape[2] >= 3:
        overlay = image[:, :, :3].astype(np.uint8).copy()
    else:
        raise ValueError("Unsupported image shape for overlay")

    corners_list: list[list[float]] = []
    if found and corners is not None:
        c = corners.reshape(-1, 2).astype(np.float32)
        corners_list = [[float(x), float(y)] for x, y in c]
        cv.drawChessboardCorners(overlay, pattern_size, corners, found)
    else:
        msg = reject_reason or "checkerboard_not_found"
        cv.putText(overlay, msg, (24, 42), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 64, 64), 2, cv.LINE_AA)
        if reject_hint:
            cv.putText(overlay, reject_hint[:90], (24, 72), cv.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv.LINE_AA)

    if board_mode == "charuco":
        aruco = getattr(cv, "aruco", None)
        if aruco is not None:
            try:
                dict_name = str((charuco_cfg or {}).get("dictionary", "DICT_4X4_50"))
                if hasattr(aruco, dict_name):
                    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
                    marker_corners, marker_ids, _ = aruco.detectMarkers(gray, dictionary)
                    if marker_ids is not None and len(marker_ids) > 0:
                        aruco.drawDetectedMarkers(overlay, marker_corners, marker_ids)
            except Exception:
                pass
        if corners is not None and corners.shape[0] > 0:
            try:
                aruco = getattr(cv, "aruco", None)
                if aruco is not None:
                    if charuco_ids_arr is None or charuco_ids_arr.shape[0] != corners.shape[0]:
                        charuco_ids_arr = np.arange(corners.shape[0], dtype=np.int32).reshape(-1, 1)
                    aruco.drawDetectedCornersCharuco(overlay, corners, charuco_ids_arr)
            except Exception:
                pass

    det = CheckerboardDetection(
        found=bool(found),
        corners_px=corners_list,
        image_size=(int(gray.shape[1]), int(gray.shape[0])),
        corner_count=len(corners_list),
        method=method,
        detection_mode=board_mode,
        rejection_reason=reject_reason,
        rejection_hint=reject_hint,
        diagnostics=diag,
    )
    return det, overlay


def save_detection_json(path: Path, detection: CheckerboardDetection, extra: dict[str, Any] | None = None) -> None:
    payload = detection.to_dict()
    payload["saved_at"] = datetime.now().isoformat()
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2))


def save_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image.astype(np.uint8)).save(path)


def calibrate_intrinsics(
    detections: list[dict[str, Any]],
    cols: int,
    rows: int,
    square_size_mm: float,
) -> dict[str, Any]:
    """Run cv2.calibrateCamera from found detections."""
    cv = _require_cv2()
    if not detections:
        raise ValueError("No detections provided")

    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * float(square_size_mm)

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []

    width = int(detections[0]["image_size"][0])
    height = int(detections[0]["image_size"][1])
    image_size = (width, height)

    for det in detections:
        corners = np.asarray(det["corners_px"], dtype=np.float32).reshape(-1, 1, 2)
        if corners.shape[0] != rows * cols:
            continue
        object_points.append(objp.copy())
        image_points.append(corners)

    if not image_points:
        raise ValueError("No valid found checkerboard detections for calibration")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    per_view_errors: list[float] = []
    for i, (obj, img, rv, tv) in enumerate(zip(object_points, image_points, rvecs, tvecs)):
        proj, _ = cv.projectPoints(obj, rv, tv, camera_matrix, dist_coeffs)
        err = cv.norm(img, proj, cv.NORM_L2) / max(len(proj), 1)
        per_view_errors.append(float(err))

    return {
        "rms": float(rms),
        "camera_matrix": camera_matrix.astype(float).tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).astype(float).tolist(),
        "image_size": [width, height],
        "checkerboard": {
            "cols": int(cols),
            "rows": int(rows),
            "square_size_mm": float(square_size_mm),
        },
        "views_used": int(len(image_points)),
        "per_view_errors": per_view_errors,
    }
