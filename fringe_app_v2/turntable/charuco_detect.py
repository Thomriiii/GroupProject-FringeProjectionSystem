"""
ChArUco detection utilities for turntable calibration.
Self-contained — no dependency on src/fringe_app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.ndim == 3 and image.shape[2] >= 3:
        f = image.astype(np.float32)
        g = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
        return np.clip(np.rint(g), 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def build_charuco_board(charuco_cfg: dict[str, Any]):
    """Return (dictionary, board)."""
    dict_name = str(charuco_cfg.get("dict", charuco_cfg.get("dict_name", "DICT_4X4_50")))
    squares_x = int(charuco_cfg.get("squares_x", 9))
    squares_y = int(charuco_cfg.get("squares_y", 7))
    sq_len = float(charuco_cfg.get("square_length_m", 0.015))
    mk_len = float(charuco_cfg.get("marker_length_m", 0.011))

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), sq_len, mk_len, dictionary)
    return dictionary, board


def charuco_object_points(board, ids: np.ndarray) -> np.ndarray:
    ids_i = np.asarray(ids, dtype=np.int32).reshape(-1)
    if hasattr(board, "getChessboardCorners"):
        corners = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    else:
        corners = np.asarray(board.chessboardCorners, dtype=np.float64)
    if ids_i.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return corners[ids_i, :].astype(np.float32)


@dataclass
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
    """Detect ChArUco corners. Returns (detection, overlay_bgr, board)."""
    gray = _to_gray_u8(image)
    dictionary, board = build_charuco_board(charuco_cfg)

    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    marker_count = 0 if marker_ids is None else int(marker_ids.shape[0])

    reject_reason: str | None = None
    hint: str | None = None
    ids = np.zeros((0,), dtype=np.int32)
    corners = np.zeros((0, 2), dtype=np.float32)

    if marker_count <= 0:
        reject_reason = "no_markers"
        hint = "Ensure the ChArUco board is visible and well lit."
    else:
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board,
        )
        ret_count = int(round(float(ret))) if ret is not None else 0
        if charuco_corners is None or charuco_ids is None or ret_count <= 0:
            reject_reason = "charuco_interpolation_failed"
            hint = "Adjust board pose so more markers are visible."
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
            cv2.aruco.drawDetectedMarkers(overlay, marker_corners, marker_ids)
        except Exception:
            pass
    if corners.shape[0] > 0:
        try:
            cv2.aruco.drawDetectedCornersCharuco(
                overlay, corners.reshape(-1, 1, 2), ids.reshape(-1, 1),
            )
        except Exception:
            pass

    status = f"markers={marker_count} corners={corners.shape[0]}"
    cv2.putText(overlay, status, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 64), 2, cv2.LINE_AA)
    if reject_reason:
        cv2.putText(overlay, reject_reason, (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 80, 80), 2, cv2.LINE_AA)

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
