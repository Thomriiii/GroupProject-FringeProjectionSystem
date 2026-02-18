"""Checkerboard detection and camera intrinsics calibration."""

from __future__ import annotations

import json
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "corners_px": self.corners_px,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "corner_count": int(self.corner_count),
            "method": self.method,
        }


def detect_checkerboard(
    image: np.ndarray,
    cols: int,
    rows: int,
    refine_subpix: bool = True,
) -> tuple[CheckerboardDetection, np.ndarray]:
    """Detect checkerboard corners and return detection + overlay image."""
    cv = _require_cv2()
    gray = to_gray_u8(image)
    pattern_size = (int(cols), int(rows))

    found = False
    corners: np.ndarray | None = None
    method = "none"

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
        cv.putText(
            overlay,
            "checkerboard not found",
            (24, 42),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 64, 64),
            2,
            cv.LINE_AA,
        )

    det = CheckerboardDetection(
        found=bool(found),
        corners_px=corners_list,
        image_size=(int(gray.shape[1]), int(gray.shape[0])),
        corner_count=len(corners_list),
        method=method,
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
