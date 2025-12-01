#!/usr/bin/env python3
"""
calibration/checkerboard.py

Checkerboard-based camera calibration helpers used by the Flask server.

Implements:
    - capture_frame(camera): grab an RGB frame and convert to grayscale
    - detect_checkerboard(gray_frame, pattern_size): find + refine chessboard corners
    - add_view(objpoints, imgpoints, corners, pattern_size, square_size): append a view
    - run_calibration(objpoints, imgpoints, image_size): run cv2.calibrateCamera
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


# -------------------------------------------------------------------------
# Frame capture
# -------------------------------------------------------------------------

def capture_frame(camera) -> np.ndarray:
    """
    Capture a single RGB frame from the camera and return grayscale.

    Parameters
    ----------
    camera : CameraController
        Must provide capture_rgb() -> np.ndarray (H x W x 3, RGB888).

    Returns
    -------
    gray : np.ndarray
        Grayscale image, uint8.
    """
    frame_rgb = camera.capture_rgb()
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    return gray


# -------------------------------------------------------------------------
# Checkerboard detection
# -------------------------------------------------------------------------

def detect_checkerboard(gray_frame: np.ndarray, pattern_size: Tuple[int, int]):
    """
    Detect an inner-corner chessboard pattern.

    Parameters
    ----------
    gray_frame : np.ndarray
        Grayscale frame.
    pattern_size : tuple[int, int]
        (cols, rows) inner-corner pattern size, e.g. (8, 6).

    Returns
    -------
    found : bool
        True if detection succeeded.
    corners_sub : np.ndarray | None
        Refined corner positions (N x 1 x 2) if found, else None.
    """
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )

    found, corners = cv2.findChessboardCorners(gray_frame, pattern_size, flags=flags)
    if not found:
        return False, None

    # Refine corner positions to sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_sub = cv2.cornerSubPix(
        gray_frame,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria,
    )
    return True, corners_sub


# -------------------------------------------------------------------------
# Add a detected view
# -------------------------------------------------------------------------

def add_view(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    corners: np.ndarray,
    pattern_size: Tuple[int, int],
    square_size: float,
):
    """
    Append one detected checkerboard view to the calibration lists.

    Parameters
    ----------
    objpoints : list[np.ndarray]
        List of 3D object point arrays (will be appended).
    imgpoints : list[np.ndarray]
        List of 2D image point arrays (will be appended).
    corners : np.ndarray
        Refined corner coordinates from detect_checkerboard.
    pattern_size : tuple[int, int]
        (cols, rows) inner corners.
    square_size : float
        Square size in metres.
    """
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)

    objpoints.append(objp)
    imgpoints.append(corners)


# -------------------------------------------------------------------------
# Run calibration
# -------------------------------------------------------------------------

def run_calibration(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    image_size: Tuple[int, int],
):
    """
    Run cv2.calibrateCamera on accumulated views.

    Parameters
    ----------
    objpoints, imgpoints : lists
        Accumulated calibration points.
    image_size : tuple[int, int]
        (width, height) of images.

    Returns
    -------
    rms : float
        RMS reprojection error.
    K : np.ndarray
        Camera intrinsic matrix.
    dist : np.ndarray
        Distortion coefficients.
    rvecs, tvecs : list
        Extrinsic parameters per view.
    """
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    return rms, K, dist, rvecs, tvecs
