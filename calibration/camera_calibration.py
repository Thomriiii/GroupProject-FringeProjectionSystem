"""
camera_calibration.py

Fresh camera intrinsic calibration pipeline using a printed checkerboard.

Features:
  - Loads a folder of calibration images.
  - Image quality checks (mean/min/max/std) with rejection.
  - Robust checkerboard detection with SB + legacy fallbacks.
  - Builds 3D–2D correspondences and runs cv2.calibrateCamera with no
    forced principal point or distortion hacks.
  - Reports global/per-image reprojection error and dataset coverage.
  - Saves/loads calibration results and provides an undistort preview.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Types
ImageSize = Tuple[int, int]          # (width, height)
PatternSize = Tuple[int, int]        # (cols, rows) inner corners
Corners = np.ndarray                 # Nx1x2 float32

# Default thresholds
MIN_MEAN = 30.0
MAX_MEAN = 220.0
MIN_STD = 10.0
MIN_BOARD_AREA_FRAC = 0.02           # reject if board is tiny in frame
MIN_BOARD_ASPECT = 0.18              # reject if board is an extreme sliver (tilted hard)
DEFAULT_MIN_IMAGES = 12


@dataclass
class LoadedImage:
    path: Path
    gray: np.ndarray
    mean: float
    std: float
    min_val: float
    max_val: float


@dataclass
class DetectionResult:
    corners: Optional[Corners]
    method: str
    area_frac: float
    aspect_ratio: float
    reason: Optional[str]


@dataclass
class CalibrationResult:
    rms: float
    K: np.ndarray
    dist: np.ndarray
    rvecs: List[np.ndarray]
    tvecs: List[np.ndarray]
    per_image_errors: List[float]
    used_images: List[Path]
    image_size: ImageSize


def list_image_files(folder: Path, extensions: Sequence[str] = ("png", "jpg", "jpeg", "bmp", "tif", "tiff")) -> List[Path]:
    files: List[Path] = []
    for ext in extensions:
        files.extend(folder.glob(f"*.{ext}"))
        files.extend(folder.glob(f"*.{ext.upper()}"))
    return sorted(set(files))


def compute_image_stats(gray: np.ndarray) -> Tuple[float, float, float, float]:
    mean, std = cv2.meanStdDev(gray)
    return float(mean[0][0]), float(std[0][0]), float(gray.min()), float(gray.max())


def load_calibration_images(
    image_dir: Path,
    max_images: Optional[int] = None,
    downscale_if_large: bool = False,
    max_side: int = 3000,
) -> Tuple[List[LoadedImage], List[Dict[str, str]], ImageSize]:
    """
    Load calibration images, convert to grayscale, enforce consistent resolution,
    and apply basic quality checks.
    """
    files = list_image_files(image_dir)
    if not files:
        raise FileNotFoundError(f"No calibration images found in {image_dir}")

    loaded: List[LoadedImage] = []
    rejected: List[Dict[str, str]] = []
    image_size: Optional[ImageSize] = None

    for path in files:
        if max_images is not None and len(loaded) >= max_images:
            break

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            rejected.append({"path": str(path), "reason": "cv2.imread failed"})
            continue

        if downscale_if_large:
            h, w = img.shape[:2]
            scale = min(max_side / max(h, w), 1.0)
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if image_size is None:
            image_size = (w, h)
        elif (w, h) != image_size:
            rejected.append({"path": str(path), "reason": f"resolution mismatch {(w, h)} != {image_size}"})
            continue

        mean, std, min_val, max_val = compute_image_stats(gray)
        reason = None
        if mean < MIN_MEAN:
            reason = f"too dark (mean={mean:.1f})"
        elif mean > MAX_MEAN:
            reason = f"too bright (mean={mean:.1f})"
        elif std < MIN_STD:
            reason = f"low contrast (std={std:.1f})"

        if reason:
            rejected.append({"path": str(path), "reason": reason})
            continue

        loaded.append(LoadedImage(path=path, gray=gray, mean=mean, std=std, min_val=min_val, max_val=max_val))

    if image_size is None:
        raise ValueError("No valid calibration images after loading.")

    return loaded, rejected, image_size


def create_object_points(pattern_size: PatternSize, square_size: float) -> np.ndarray:
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = [[i * square_size, j * square_size] for j in range(rows) for i in range(cols)]
    return objp


def detect_checkerboard(
    gray: np.ndarray,
    pattern_size: PatternSize,
    debug_path: Optional[Path] = None,
) -> DetectionResult:
    expected = pattern_size[0] * pattern_size[1]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)

    found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE)
    method = "findChessboardCornersSB"
    if not found:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        method = "findChessboardCorners"

    if not found or corners is None:
        return DetectionResult(corners=None, method=method, area_frac=0.0, aspect_ratio=0.0, reason="no checkerboard found")

    if corners.shape[0] != expected:
        return DetectionResult(corners=None, method=method, area_frac=0.0, aspect_ratio=0.0, reason=f"corner count mismatch ({corners.shape[0]} != {expected})")

    # Refine to subpixel accuracy
    corners_refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    h, w = gray.shape[:2]
    xs = corners_refined[:, 0, 0]
    ys = corners_refined[:, 0, 1]
    area = (xs.max() - xs.min()) * (ys.max() - ys.min())
    area_frac = float(area) / float(w * h)

    rect = cv2.minAreaRect(corners_refined)
    width, height = rect[1]
    aspect_ratio = 0.0
    if width > 1e-6 and height > 1e-6:
        aspect_ratio = float(min(width, height) / max(width, height))

    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, pattern_size, corners_refined, True)
        cv2.imwrite(str(debug_path), vis)

    return DetectionResult(
        corners=corners_refined,
        method=method,
        area_frac=area_frac,
        aspect_ratio=aspect_ratio,
        reason=None,
    )


def collect_corners(
    images: List[LoadedImage],
    pattern_size: PatternSize,
    square_size: float,
    debug_dir: Path,
    min_board_area_frac: float = MIN_BOARD_AREA_FRAC,
    min_board_aspect: float = MIN_BOARD_ASPECT,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Path], List[Dict[str, str]]]:
    objp = create_object_points(pattern_size, square_size)
    object_points: List[np.ndarray] = []
    image_points: List[np.ndarray] = []
    used_images: List[Path] = []
    rejected: List[Dict[str, str]] = []

    for img in images:
        dbg_path = debug_dir / f"{img.path.stem}_corners.png"
        detection = detect_checkerboard(img.gray, pattern_size, debug_path=dbg_path)

        if detection.corners is None:
            rejected.append({"path": str(img.path), "reason": detection.reason or "detection failed"})
            continue

        if detection.area_frac < min_board_area_frac:
            rejected.append({"path": str(img.path), "reason": f"board too small in frame (area_frac={detection.area_frac:.3f})"})
            continue

        if detection.aspect_ratio < min_board_aspect:
            rejected.append({"path": str(img.path), "reason": f"board extremely tilted (aspect_ratio={detection.aspect_ratio:.3f})"})
            continue

        object_points.append(objp.copy())
        image_points.append(detection.corners)
        used_images.append(img.path)

    return object_points, image_points, used_images, rejected


def calibrate_camera(
    object_points: List[np.ndarray],
    image_points: List[np.ndarray],
    image_size: ImageSize,
    flags: int = 0,
    used_images: Optional[List[Path]] = None,
) -> CalibrationResult:
    if len(object_points) == 0 or len(image_points) == 0:
        raise ValueError("No valid object/image point pairs for calibration.")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=flags,
    )

    per_image_errors = compute_reprojection_errors(object_points, image_points, rvecs, tvecs, K, dist)

    return CalibrationResult(
        rms=float(ret),
        K=K,
        dist=dist,
        rvecs=rvecs,
        tvecs=tvecs,
        per_image_errors=per_image_errors,
        used_images=list(used_images) if used_images else [],
        image_size=image_size,
    )


def compute_reprojection_errors(
    object_points: List[np.ndarray],
    image_points: List[np.ndarray],
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
) -> List[float]:
    errors: List[float] = []
    for objp, imgp, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        projected = projected.reshape(-1, 2)
        err = np.linalg.norm(imgp.reshape(-1, 2) - projected, axis=1)
        rms = float(np.sqrt(np.mean(err * err)))
        errors.append(rms)
    return errors


def report_dataset_coverage(
    image_points: List[np.ndarray],
    image_size: ImageSize,
    pattern_size: PatternSize,
) -> Dict[str, float]:
    """
    Compute simple coverage metrics: center distribution and board scale spread.
    """
    if not image_points:
        return {}

    w, h = image_size
    centers = np.array([pts.reshape(-1, 2).mean(axis=0) for pts in image_points])
    x_norm = centers[:, 0] / w
    y_norm = centers[:, 1] / h

    scales = [estimate_board_scale(pts, pattern_size) for pts in image_points]
    scales = np.array(scales)

    coverage = {
        "center_x_min": float(x_norm.min()),
        "center_x_max": float(x_norm.max()),
        "center_y_min": float(y_norm.min()),
        "center_y_max": float(y_norm.max()),
        "scale_min": float(scales.min()),
        "scale_max": float(scales.max()),
        "scale_mean": float(scales.mean()),
    }

    print("[COVERAGE] Board center x-range (normalized): "
          f"{coverage['center_x_min']:.3f} – {coverage['center_x_max']:.3f}")
    print("[COVERAGE] Board center y-range (normalized): "
          f"{coverage['center_y_min']:.3f} – {coverage['center_y_max']:.3f}")
    print("[COVERAGE] Board scale (px per square) min/mean/max: "
          f"{coverage['scale_min']:.1f} / {coverage['scale_mean']:.1f} / {coverage['scale_max']:.1f}")

    return coverage


def estimate_board_scale(corners: np.ndarray, pattern_size: PatternSize) -> float:
    cols, rows = pattern_size
    grid = corners.reshape(rows, cols, 2)
    dx = np.linalg.norm(grid[:, 1:, :] - grid[:, :-1, :], axis=2)
    dy = np.linalg.norm(grid[1:, :, :] - grid[:-1, :, :], axis=2)
    vals = []
    if dx.size > 0:
        vals.append(dx.mean())
    if dy.size > 0:
        vals.append(dy.mean())
    if not vals:
        return 0.0
    return float(np.mean(vals))


def maybe_filter_outliers(
    calib: CalibrationResult,
    object_points: List[np.ndarray],
    image_points: List[np.ndarray],
    used_images: List[Path],
    image_size: ImageSize,
    flags: int,
    min_images: int,
    max_per_view_error: Optional[float],
) -> Tuple[CalibrationResult, List[np.ndarray], List[np.ndarray], List[Path]]:
    """
    Optionally drop high-error views and recalibrate.
    """
    if max_per_view_error is None:
        return calib, object_points, image_points, used_images

    keep_mask = [err <= max_per_view_error for err in calib.per_image_errors]
    if all(keep_mask):
        return calib, object_points, image_points, used_images

    filtered_obj = [op for op, k in zip(object_points, keep_mask) if k]
    filtered_img = [ip for ip, k in zip(image_points, keep_mask) if k]
    filtered_names = [p for p, k in zip(used_images, keep_mask) if k]
    dropped = [p for p, k in zip(used_images, keep_mask) if not k]

    print(f"[CALIB] Dropping {len(dropped)} views with RMS > {max_per_view_error}:")
    for p, err, keep in zip(used_images, calib.per_image_errors, keep_mask):
        if not keep:
            print(f"  - {p.name}: {err:.4f} px")

    if len(filtered_obj) < min_images:
        print(f"[CALIB] Not enough views after dropping ({len(filtered_obj)}/{min_images}); keeping all.")
        return calib, object_points, image_points, used_images

    recalib = calibrate_camera(filtered_obj, filtered_img, image_size=image_size, flags=flags, used_images=filtered_names)
    return recalib, filtered_obj, filtered_img, filtered_names


def save_calibration(path: Path, K: np.ndarray, dist: np.ndarray, image_size: ImageSize, rms: float) -> None:
    np.savez_compressed(path, K=K, dist=dist, image_size=np.array(image_size, dtype=np.int32), rms=np.array([rms], dtype=np.float32))
    print(f"[SAVE] Calibration saved to {path}")


def load_calibration(path: Path) -> Tuple[np.ndarray, np.ndarray, ImageSize, float]:
    data = np.load(path)
    K = data["K"]
    dist = data["dist"]
    image_size_arr = data["image_size"]
    image_size = (int(image_size_arr[0]), int(image_size_arr[1]))
    rms = float(data["rms"][0]) if "rms" in data else float("nan")
    return K, dist, image_size, rms


def undistort_preview(
    image_path: Path,
    K: np.ndarray,
    dist: np.ndarray,
    output_path: Path,
    alpha: float = 0.0,
) -> None:
    """
    Create a side-by-side original vs undistorted preview to visually inspect straight lines.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"undistort_preview: cannot read {image_path}")

    h, w = img.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha)
    undistorted = cv2.undistort(img, K, dist, None, new_K)

    # Overlay simple guide lines
    overlay = undistorted.copy()
    color = (0, 255, 0)
    thickness = 1
    for frac in [0.25, 0.5, 0.75]:
        cv2.line(overlay, (int(w * frac), 0), (int(w * frac), h), color, thickness)
        cv2.line(overlay, (0, int(h * frac)), (w, int(h * frac)), color, thickness)

    combined = np.hstack([img, overlay])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)
    print(f"[UNDISTORT] Preview saved to {output_path}")


def write_calibration_report(
    path: Path,
    calib: CalibrationResult,
    quality_rejects: List[Dict[str, str]],
    detect_rejects: List[Dict[str, str]],
) -> None:
    lines = []
    lines.append("Camera Intrinsic Calibration Report\n")
    lines.append(f"Images used: {len(calib.per_image_errors)}\n")
    lines.append(f"Image size: {calib.image_size[0]} x {calib.image_size[1]}\n")
    lines.append(f"RMS reprojection error: {calib.rms:.4f} px\n")
    lines.append("\nIntrinsics (K):\n")
    lines.append(str(calib.K))
    lines.append("\n\nDistortion coefficients (k1, k2, p1, p2, k3[, ...]):\n")
    lines.append(str(calib.dist.ravel()))
    lines.append("\n\nPer-image RMS errors:\n")
    for idx, err in enumerate(calib.per_image_errors):
        fname = calib.used_images[idx].name if idx < len(calib.used_images) else f"img_{idx}"
        lines.append(f"  {fname}: {err:.4f}")
    lines.append("\nRejected images (quality):\n")
    if quality_rejects:
        for r in quality_rejects:
            lines.append(f"  {Path(r['path']).name}: {r['reason']}")
    else:
        lines.append("  None")
    lines.append("\nRejected images (detection/geometry):\n")
    if detect_rejects:
        for r in detect_rejects:
            lines.append(f"  {Path(r['path']).name}: {r['reason']}")
    else:
        lines.append("  None")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"[REPORT] Calibration report written to {path}")


def run_calibration_from_folder(
    image_dir: Path,
    pattern_size: PatternSize,
    square_size: float,
    debug_dir: Path = Path("debug_corners"),
    min_images: int = DEFAULT_MIN_IMAGES,
    flags: int = 0,
    max_images: Optional[int] = None,
    max_per_view_error: Optional[float] = None,
) -> CalibrationResult:
    """
    High-level convenience: load images, detect corners, calibrate, and report.
    """
    loaded, quality_rejects, image_size = load_calibration_images(image_dir, max_images=max_images)
    object_points, image_points, used_images, detection_rejects = collect_corners(
        loaded,
        pattern_size=pattern_size,
        square_size=square_size,
        debug_dir=debug_dir,
    )

    if len(object_points) < min_images:
        raise ValueError(f"Not enough valid calibration frames ({len(object_points)}/{min_images}).")

    calib = calibrate_camera(object_points, image_points, image_size=image_size, flags=flags, used_images=used_images)
    calib, object_points, image_points, used_images = maybe_filter_outliers(
        calib,
        object_points,
        image_points,
        used_images,
        image_size,
        flags,
        min_images,
        max_per_view_error,
    )

    # Coverage printout
    report_dataset_coverage(image_points, image_size, pattern_size)

    # Save defaults
    save_calibration(Path("camera_intrinsics.npz"), calib.K, calib.dist, image_size, calib.rms)
    write_calibration_report(Path("calibration_report.txt"), calib, quality_rejects, detection_rejects)

    print("\n[CALIBRATION RESULTS]")
    fx, fy = calib.K[0, 0], calib.K[1, 1]
    cx, cy = calib.K[0, 2], calib.K[1, 2]
    print(f"  rms={calib.rms:.4f} px")
    print(f"  fx={fx:.2f}, fy={fy:.2f}")
    print(f"  cx={cx:.2f}, cy={cy:.2f}")
    print(f"  dist={calib.dist.ravel()}")
    print(f"  images used={len(calib.per_image_errors)} / {len(loaded)} (min required: {min_images})")
    # Sanity checks:
    # Calibration is only meaningful if it matches the scan capture resolution and
    # the estimated intrinsics are physically plausible. Extremely narrow implied FOV
    # or a principal point far from the image center often indicates:
    #   - cropping/ROI/rotation between calibration and scan
    #   - a weak calibration dataset (board too small / too fronto-parallel / not enough tilt)
    # These issues can produce streak-like point clouds even if ray intersection errors look low.
    W, H = calib.image_size
    fov_x = float(2.0 * np.degrees(np.arctan((W / 2.0) / (fx + 1e-9))))
    fov_y = float(2.0 * np.degrees(np.arctan((H / 2.0) / (fy + 1e-9))))
    off_x = abs(float(cx) - (W / 2.0)) / float(W)
    off_y = abs(float(cy) - (H / 2.0)) / float(H)
    if off_x > 0.10 or off_y > 0.10:
        print(f"[CALIB][WARN] Principal point far from image center (dx={off_x*100:.1f}%, dy={off_y*100:.1f}%).")
    if fov_x < 25.0 or fov_y < 20.0:
        print(f"[CALIB][WARN] Implied FOV is very narrow (fov_x={fov_x:.1f}°, fov_y={fov_y:.1f}°).")
        print("[CALIB][WARN] This usually means the dataset did not sufficiently constrain focal length.")
        print("[CALIB][WARN] Re-capture calibration with the board closer and with strong tilt across the frame.")

    if calib.per_image_errors:
        worst = max(calib.per_image_errors)
        print(f"  per-image RMS: min={min(calib.per_image_errors):.4f}, mean={np.mean(calib.per_image_errors):.4f}, max={worst:.4f}")

    return calib
