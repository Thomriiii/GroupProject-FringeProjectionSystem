"""Charuco detection utilities for projector calibration v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    _cv2_import_error = exc
else:
    _cv2_import_error = None


def _require_cv2():
    if cv2 is None:
        raise RuntimeError(f"OpenCV is required for Charuco detection: {_cv2_import_error}")
    return cv2


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.ndim == 3 and image.shape[2] >= 3:
        f = image.astype(np.float32)
        g = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
        return np.clip(np.rint(g), 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported image shape for Charuco detection: {image.shape}")


def _dictionary(dict_name: str):
    cv = _require_cv2()
    aruco = getattr(cv, "aruco", None)
    if aruco is None:
        raise RuntimeError("OpenCV ArUco module is unavailable")
    if not hasattr(aruco, dict_name):
        raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
    return aruco.getPredefinedDictionary(getattr(aruco, dict_name))


def build_charuco_board(charuco_cfg: dict[str, Any]):
    cv = _require_cv2()
    aruco = getattr(cv, "aruco", None)
    if aruco is None:
        raise RuntimeError("OpenCV ArUco module is unavailable")

    dict_name = str(charuco_cfg.get("dict", "DICT_4X4_50"))
    squares_x = int(charuco_cfg.get("squares_x", 9))
    squares_y = int(charuco_cfg.get("squares_y", 7))
    square_length_m = float(charuco_cfg.get("square_length_m", 0.015))
    marker_length_m = float(charuco_cfg.get("marker_length_m", 0.011))

    if squares_x < 4 or squares_y < 4:
        raise ValueError("Charuco board requires squares_x >= 4 and squares_y >= 4")
    if marker_length_m >= square_length_m:
        raise ValueError("marker_length_m must be smaller than square_length_m")

    dictionary = _dictionary(dict_name)
    board = aruco.CharucoBoard(
        (int(squares_x), int(squares_y)),
        float(square_length_m),
        float(marker_length_m),
        dictionary,
    )
    return dictionary, board


def charuco_object_points(board: Any, ids: np.ndarray) -> np.ndarray:
    ids_i = np.asarray(ids, dtype=np.int32).reshape(-1)
    if hasattr(board, "getChessboardCorners"):
        corners = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    else:
        corners = np.asarray(board.chessboardCorners, dtype=np.float64)
    if corners.ndim != 2 or corners.shape[1] != 3:
        raise ValueError("Unexpected Charuco board corner array shape")
    if ids_i.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if int(np.max(ids_i)) >= int(corners.shape[0]) or int(np.min(ids_i)) < 0:
        raise ValueError("Charuco ids out of board corner range")
    return corners[ids_i, :].astype(np.float32)


@dataclass(slots=True)
class CharucoDetection:
    found: bool
    marker_count: int
    corner_count: int
    image_size: tuple[int, int]
    ids: np.ndarray
    corners: np.ndarray
    reject_reason: str | None
    hint: str | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "found": bool(self.found),
            "marker_count": int(self.marker_count),
            "corner_count": int(self.corner_count),
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "ids": [int(v) for v in self.ids.reshape(-1).tolist()],
            "corners_px": [[float(x), float(y)] for x, y in self.corners.reshape(-1, 2)],
            "reject_reason": self.reject_reason,
            "hint": self.hint,
        }


def detect_charuco(
    image: np.ndarray,
    charuco_cfg: dict[str, Any],
) -> tuple[CharucoDetection, np.ndarray, Any]:
    cv = _require_cv2()
    aruco = getattr(cv, "aruco", None)
    if aruco is None:
        raise RuntimeError("OpenCV ArUco module is unavailable")

    gray = _to_gray_u8(image)
    dictionary, board = build_charuco_board(charuco_cfg)

    marker_corners, marker_ids, _ = aruco.detectMarkers(gray, dictionary)
    marker_count = 0 if marker_ids is None else int(marker_ids.shape[0])

    reject_reason: str | None = None
    hint: str | None = None
    ids = np.zeros((0,), dtype=np.int32)
    corners = np.zeros((0, 2), dtype=np.float32)

    if marker_count <= 0:
        reject_reason = "no_markers"
        hint = "Ensure the Charuco board is visible and well lit."
    else:
        ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
        )
        ret_count = int(round(float(ret))) if ret is not None else 0
        if charuco_corners is None or charuco_ids is None or ret_count <= 0:
            reject_reason = "charuco_interpolation_failed"
            hint = "Adjust pose or focus so more board markers are visible."
        else:
            ids_raw = charuco_ids.reshape(-1).astype(np.int32)
            corners_raw = charuco_corners.reshape(-1, 2).astype(np.float32)
            order = np.argsort(ids_raw)
            ids = ids_raw[order]
            corners = corners_raw[order]

    found = bool(corners.shape[0] > 0)

    if image.ndim == 2:
        overlay = np.stack([image, image, image], axis=2).astype(np.uint8)
    else:
        overlay = image[:, :, :3].astype(np.uint8).copy()

    if marker_count > 0 and marker_corners is not None and marker_ids is not None:
        try:
            aruco.drawDetectedMarkers(overlay, marker_corners, marker_ids)
        except Exception:
            pass
    if corners.shape[0] > 0:
        try:
            aruco.drawDetectedCornersCharuco(
                overlay,
                corners.reshape(-1, 1, 2),
                ids.reshape(-1, 1),
            )
        except Exception:
            pass

    status = f"markers={marker_count} corners={corners.shape[0]}"
    cv.putText(
        overlay,
        status,
        (16, 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 64),
        2,
        cv.LINE_AA,
    )
    if reject_reason:
        cv.putText(
            overlay,
            reject_reason,
            (16, 56),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 80, 80),
            2,
            cv.LINE_AA,
        )

    det = CharucoDetection(
        found=found,
        marker_count=int(marker_count),
        corner_count=int(corners.shape[0]),
        image_size=(int(gray.shape[1]), int(gray.shape[0])),
        ids=ids.astype(np.int32),
        corners=corners.astype(np.float32),
        reject_reason=reject_reason,
        hint=hint,
    )
    return det, overlay, board
