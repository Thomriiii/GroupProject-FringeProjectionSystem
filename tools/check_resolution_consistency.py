#!/usr/bin/env python3
"""
check_resolution_consistency.py

Verify that camera calibration resolution matches the images used for scanning or reconstruction.

This check is intentionally strict: geometric reconstruction assumes camera intrinsics correspond
to the pixel grid being used. Cropping, rotation, or resizing without rescaling intrinsics will
produce warped or streaky point clouds even if ray intersection errors look small.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def _load_camera_intrinsics(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Load camera intrinsics from an NPZ file.
    """
    data = np.load(path)
    K = data["K"]
    dist = data["dist"]
    image_size = data["image_size"]
    return K, dist, (int(image_size[0]), int(image_size[1]))  # (W, H)


def _find_first_image(path: Path) -> Path | None:
    """
    Return the first image file found in a directory, if any.
    """
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    for p in sorted(path.glob("*")):
        if p.suffix.lower() in exts:
            return p
    return None


def _shape_of_npy(path: Path) -> Tuple[int, int] | None:
    """
    Return (width, height) for an array stored in an NPY file.
    """
    try:
        arr = np.load(path)
    except Exception:
        return None
    if arr.ndim < 2:
        return None
    h, w = arr.shape[:2]
    return int(w), int(h)


def _shape_of_image(path: Path) -> Tuple[int, int] | None:
    """
    Return (width, height) for an image file if OpenCV is available.
    """
    try:
        import cv2
    except Exception:
        cv2 = None
    if cv2 is None:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    h, w = img.shape[:2]
    return int(w), int(h)


def _collect_shapes_calib(calib_dir: Path) -> Iterable[Tuple[Path, Tuple[int, int] | None]]:
    """
    Yield image files and their shapes from a calibration directory.
    """
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    for p in sorted(calib_dir.glob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p, _shape_of_image(p)


def main() -> int:
    """
    CLI entry point for resolution consistency checks.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=Path, default=Path("camera_intrinsics.npz"), help="camera intrinsics npz path")
    ap.add_argument("--scan-dir", type=Path, default=None, help="scan output directory (contains proj_u.npy/proj_v.npy)")
    ap.add_argument("--calib-dir", type=Path, default=None, help="camera calibration session directory (checkerboard images)")
    args = ap.parse_args()

    if not args.camera.exists():
        raise SystemExit(f"Missing camera intrinsics: {args.camera}")

    K, dist, calib_size = _load_camera_intrinsics(args.camera)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    print(f"[CAM] {args.camera}: image_size={calib_size[0]}x{calib_size[1]}")
    print(f"[CAM] K: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"[CAM] dist: {dist.ravel()}")

    if args.calib_dir is not None:
        if not args.calib_dir.exists():
            raise SystemExit(f"Calibration directory not found: {args.calib_dir}")
        shapes = list(_collect_shapes_calib(args.calib_dir))
        if not shapes:
            print(f"[CALIB] No images found in {args.calib_dir}")
        else:
            uniq = {s for _, s in shapes if s is not None}
            print(f"[CALIB] {args.calib_dir}: {len(shapes)} images, unique shapes={sorted(uniq)}")
            bad = [p for p, s in shapes if s is not None and s != calib_size]
            if bad:
                print("[CALIB][WARN] Images that do not match camera_intrinsics image_size:")
                for p in bad[:20]:
                    s = _shape_of_image(p)
                    print(f"  - {p.name}: {s[0]}x{s[1] if s else '???'}")

    if args.scan_dir is not None:
        if not args.scan_dir.exists():
            raise SystemExit(f"Scan directory not found: {args.scan_dir}")
        pu = args.scan_dir / "proj_u.npy"
        pv = args.scan_dir / "proj_v.npy"
        if pu.exists():
            scan_size = _shape_of_npy(pu)
            if scan_size is None:
                print(f"[SCAN] {pu}: shape=???")
            else:
                print(f"[SCAN] {pu}: shape={scan_size[0]}x{scan_size[1]}")
            if scan_size is not None and scan_size != calib_size:
                print(f"[SCAN][FATAL] Size mismatch: scan={scan_size[0]}x{scan_size[1]} vs calib={calib_size[0]}x{calib_size[1]}")
        else:
            img = _find_first_image(args.scan_dir)
            if img is None:
                print(f"[SCAN] No proj_u.npy and no images found in {args.scan_dir}")
            else:
                scan_size = _shape_of_image(img)
                if scan_size is None:
                    print(f"[SCAN] {img}: shape=???")
                else:
                    print(f"[SCAN] {img}: shape={scan_size[0]}x{scan_size[1]}")
                if scan_size is not None and scan_size != calib_size:
                    print(f"[SCAN][FATAL] Size mismatch: scan={scan_size[0]}x{scan_size[1]} vs calib={calib_size[0]}x{calib_size[1]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
