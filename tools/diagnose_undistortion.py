#!/usr/bin/env python3
"""
diagnose_undistortion.py

Diagnostic helper to visually validate the distortion model on a real captured image.

It:
  - Loads camera intrinsics (K, dist) from camera_intrinsics.npz
  - Undistorts the image
  - Overlays a guide grid and detected straight lines on both views
  - Writes a side-by-side PNG for inspection
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import cv2
except Exception as e:  # pragma: no cover
    raise SystemExit(f"OpenCV is required for this diagnostic: {e}")


def _load_intrinsics(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Load intrinsics and image size from an NPZ file.
    """
    data = np.load(path)
    K = data["K"]
    dist = data["dist"]
    image_size = data["image_size"]
    return K, dist, (int(image_size[0]), int(image_size[1]))  # (W, H)


def _overlay_grid(img_bgr: np.ndarray, color=(0, 255, 0), thickness: int = 1) -> np.ndarray:
    """
    Draw a simple rule-of-thirds grid over an image.
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]
    for frac in (0.25, 0.5, 0.75):
        cv2.line(out, (int(w * frac), 0), (int(w * frac), h - 1), color, thickness)
        cv2.line(out, (0, int(h * frac)), (w - 1, int(h * frac)), color, thickness)
    return out


def _overlay_hough_lines(img_bgr: np.ndarray, *, max_lines: int = 80) -> np.ndarray:
    """
    Draw a subset of Hough-detected lines for visual inspection.
    """
    out = img_bgr.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=90, minLineLength=80, maxLineGap=10)
    if lines is None:
        return out

    # Draw strongest/earliest lines. This is a qualitative check, not a metric.
    for i, ln in enumerate(lines[:max_lines]):
        x1, y1, x2, y2 = (int(v) for v in ln[0])
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if i >= max_lines:
            break
    return out


def main() -> int:
    """
    CLI entry point for undistortion diagnostics.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=Path, default=Path("camera_intrinsics.npz"), help="camera intrinsics npz")
    ap.add_argument("--image", type=Path, required=True, help="input image (scan/calibration frame)")
    ap.add_argument("--out", type=Path, default=Path("undistort_diagnostic.png"), help="output PNG path")
    ap.add_argument("--alpha", type=float, default=0.0, help="alpha for getOptimalNewCameraMatrix (0=crop, 1=keep all)")
    args = ap.parse_args()

    if not args.camera.exists():
        raise SystemExit(f"Missing camera intrinsics: {args.camera}")
    if not args.image.exists():
        raise SystemExit(f"Missing image: {args.image}")

    K, dist, calib_size = _load_intrinsics(args.camera)
    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cv2.imread failed: {args.image}")

    h, w = img.shape[:2]
    if (w, h) != calib_size:
        print(
            f"[WARN] Image size {w}x{h} differs from calibration size {calib_size[0]}x{calib_size[1]}. "
            "Undistortion will be wrong unless intrinsics are rescaled first."
        )

    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), args.alpha)
    und = cv2.undistort(img, K, dist, None, new_K)

    left = _overlay_hough_lines(_overlay_grid(img))
    right = _overlay_hough_lines(_overlay_grid(und))

    combined = np.hstack([left, right])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), combined)

    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    print(f"[OK] Wrote {args.out}")
    print(f"[CALIB] image_size={calib_size[0]}x{calib_size[1]} K: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"[CALIB] dist: {dist.ravel()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

