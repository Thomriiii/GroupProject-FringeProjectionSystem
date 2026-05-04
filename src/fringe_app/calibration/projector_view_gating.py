"""Per-view quality gating for projector calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
        raise RuntimeError(f"OpenCV is required for projector view gating: {_cv2_import_error}")
    return cv2


@dataclass
class ProjectorViewDiagnostics:
    corners_found: bool
    valid_corner_ratio: float
    hull_area_ratio: float
    board_tilt_deg: float
    board_center_norm: tuple[float, float]
    board_depth_proxy: float
    b_median_board: float
    unwrap_residual_p95: float
    unwrap_residual_p95_board: float
    unwrap_residual_vertical_p95_board: float
    unwrap_residual_horizontal_p95_board: float
    unwrap_residual_gt_1rad_pct_board: float
    board_mask_area_ratio: float
    clipping_detected: bool
    edge_corner_pct: float
    duplicate_pose: bool
    accept: bool
    reject_reasons: list[str]
    hints: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def checkerboard_object_points(corners_x: int, corners_y: int, square_size_m: float) -> np.ndarray:
    obj = np.zeros((corners_x * corners_y, 3), dtype=np.float32)
    grid = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1, 2)
    obj[:, :2] = grid * float(square_size_m)
    return obj


def board_hull_area_ratio(corners_cam: np.ndarray, image_shape_hw: tuple[int, int]) -> float:
    cv = _require_cv2()
    h, w = int(image_shape_hw[0]), int(image_shape_hw[1])
    if corners_cam.shape[0] < 3 or h <= 0 or w <= 0:
        return 0.0
    hull = cv.convexHull(corners_cam.reshape(-1, 1, 2).astype(np.float32))
    area = float(cv.contourArea(hull))
    return float(area / float(h * w))


def board_center_norm(corners_cam: np.ndarray, image_shape_hw: tuple[int, int]) -> tuple[float, float]:
    h, w = int(image_shape_hw[0]), int(image_shape_hw[1])
    if corners_cam.size == 0 or h <= 0 or w <= 0:
        return (0.0, 0.0)
    cx = float(np.mean(corners_cam[:, 0])) / float(w)
    cy = float(np.mean(corners_cam[:, 1])) / float(h)
    return (float(np.clip(cx, 0.0, 1.0)), float(np.clip(cy, 0.0, 1.0)))


def estimate_board_pose(
    corners_cam: np.ndarray,
    corners_x: int,
    corners_y: int,
    square_size_m: float,
    camera_matrix: np.ndarray | None,
    dist_coeffs: np.ndarray | None,
) -> tuple[float, float, dict[str, Any] | None]:
    """Return (tilt_deg, depth_proxy_z, pose_entry)."""
    cv = _require_cv2()
    expected = int(corners_x * corners_y)
    if (
        camera_matrix is None
        or dist_coeffs is None
        or corners_cam.shape[0] != expected
    ):
        return float("nan"), float("nan"), None
    obj = checkerboard_object_points(corners_x, corners_y, square_size_m)
    img = corners_cam.reshape(-1, 1, 2).astype(np.float32)
    try:
        ok, rvec, tvec = cv.solvePnP(obj, img, camera_matrix, dist_coeffs)
    except Exception:
        return float("nan"), float("nan"), None
    if not ok:
        return float("nan"), float("nan"), None

    R, _ = cv.Rodrigues(rvec)
    # Board normal in camera frame: board +Z transformed by R.
    n = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    nz = float(np.clip(n[2], -1.0, 1.0))
    tilt_deg = float(np.degrees(np.arccos(abs(nz))))
    depth_proxy = float(tvec.reshape(-1)[2])
    pose_entry = {
        "R": R.astype(float).tolist(),
        "tvec": tvec.reshape(-1).astype(float).tolist(),
        "tilt_deg": tilt_deg,
        "depth_proxy": depth_proxy,
    }
    return tilt_deg, depth_proxy, pose_entry


def is_duplicate_pose(
    pose_entry: dict[str, Any] | None,
    board_center_norm_xy: tuple[float, float],
    recent_poses: list[dict[str, Any]],
    rot_thresh_deg: float = 5.0,
    center_shift_thresh_norm: float = 0.05,
) -> bool:
    if pose_entry is None:
        return False
    Rc = np.asarray(pose_entry.get("R", []), dtype=np.float64)
    if Rc.shape != (3, 3):
        return False

    cx, cy = board_center_norm_xy
    for prev in recent_poses[-3:]:
        Rp = np.asarray(prev.get("R", []), dtype=np.float64)
        pcenter = prev.get("center_norm", [None, None])
        if Rp.shape != (3, 3):
            continue
        try:
            pcx = float(pcenter[0])
            pcy = float(pcenter[1])
        except Exception:
            continue
        Rrel = Rp.T @ Rc
        tr = float(np.trace(Rrel))
        ang = float(np.degrees(np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0))))
        shift = float(np.hypot(cx - pcx, cy - pcy))
        if ang < rot_thresh_deg and shift < center_shift_thresh_norm:
            return True
    return False


def evaluate_projector_view(
    *,
    corners_found: bool,
    expected_corner_count: int,
    corners_cam: np.ndarray,
    valid_corner_mask: np.ndarray,
    corner_reasons: list[str],
    image_shape_hw: tuple[int, int],
    corners_x: int,
    corners_y: int,
    square_size_m: float,
    camera_matrix: np.ndarray | None,
    dist_coeffs: np.ndarray | None,
    recent_accepted_poses: list[dict[str, Any]],
    b_median_board: float,
    unwrap_residual_p95: float,
    unwrap_residual_vertical_p95_board: float,
    unwrap_residual_horizontal_p95_board: float,
    unwrap_residual_gt_1rad_pct_board: float,
    board_mask_area_ratio: float,
    clipping_detected: bool,
    min_valid_corner_ratio: float = 0.90,
    min_hull_area_ratio: float = 0.08,
    min_tilt_deg: float = 10.0,
    max_unwrap_residual_p95: float = 1.2,
    max_unwrap_gt_1rad_pct: float = 0.15,
    min_b_median_board: float = 20.0,
    require_full_corner_count: bool = True,
) -> tuple[ProjectorViewDiagnostics, dict[str, Any] | None]:
    total = int(expected_corner_count)
    found_count = int(corners_cam.shape[0])
    valid_count = int(np.count_nonzero(valid_corner_mask))
    valid_ratio = float(valid_count / max(total, 1))
    hull_ratio = board_hull_area_ratio(corners_cam, image_shape_hw) if found_count >= 3 else 0.0
    center_norm = board_center_norm(corners_cam, image_shape_hw)

    tilt_deg, depth_proxy, pose_entry = estimate_board_pose(
        corners_cam=corners_cam,
        corners_x=corners_x,
        corners_y=corners_y,
        square_size_m=square_size_m,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    if pose_entry is not None:
        pose_entry["center_norm"] = [float(center_norm[0]), float(center_norm[1])]

    near_edge_count = sum(1 for r in corner_reasons if r in ("near_edge", "oob"))
    edge_corner_pct = float(near_edge_count / max(total, 1))

    duplicate = is_duplicate_pose(pose_entry, center_norm, recent_accepted_poses)

    reasons: list[str] = []
    hints: list[str] = []

    # 1) Checkerboard gate.
    if (not corners_found) or (bool(require_full_corner_count) and (found_count != total)):
        reasons.append("Checkerboard not fully detected")
        hints.append("Ensure full checkerboard is visible and in focus.")

    # 2) UV corner validity gate.
    if valid_ratio < float(min_valid_corner_ratio):
        reasons.append("Only {:.1f}% UV-valid corners".format(100.0 * valid_ratio))
        hints.append("Move board fully inside projector coverage.")

    # 3) Board size gate.
    if hull_ratio < float(min_hull_area_ratio):
        reasons.append("Board footprint too small in image")
        hints.append("Move board closer — it is too small in the image.")

    # 4) Pose diversity gate.
    if np.isfinite(tilt_deg) and tilt_deg < float(min_tilt_deg) and duplicate:
        reasons.append("Pose duplicate / too fronto-parallel")
        hints.append("Tilt board more or change position.")

    # 5) Modulation gate.
    if not np.isfinite(b_median_board) or b_median_board < float(min_b_median_board):
        reasons.append("Low board modulation")
        hints.append("Low modulation — reduce ambient light or increase projector brightness.")

    # 6) Residual gate.
    if np.isfinite(unwrap_residual_p95) and unwrap_residual_p95 > float(max_unwrap_residual_p95):
        reasons.append("Board unwrap residual too high")
        hints.append("Phase quality too low — re-capture view.")
    if (
        np.isfinite(unwrap_residual_gt_1rad_pct_board)
        and unwrap_residual_gt_1rad_pct_board > float(max_unwrap_gt_1rad_pct)
    ):
        reasons.append("Board unwrap unstable region")
        hints.append("Large unstable phase region on board — re-capture view.")

    # 7) Clipping gate.
    if bool(clipping_detected):
        reasons.append("Clipping detected on board")
        hints.append("Clipping detected — reduce exposure or projector intensity.")

    accept = len(reasons) == 0 and (not duplicate)
    diag = ProjectorViewDiagnostics(
        corners_found=bool(corners_found),
        valid_corner_ratio=float(valid_ratio),
        hull_area_ratio=float(hull_ratio),
        board_tilt_deg=float(tilt_deg) if np.isfinite(tilt_deg) else float("nan"),
        board_center_norm=(float(center_norm[0]), float(center_norm[1])),
        board_depth_proxy=float(depth_proxy) if np.isfinite(depth_proxy) else float("nan"),
        b_median_board=float(b_median_board) if np.isfinite(b_median_board) else float("nan"),
        unwrap_residual_p95=float(unwrap_residual_p95) if np.isfinite(unwrap_residual_p95) else float("nan"),
        unwrap_residual_p95_board=float(unwrap_residual_p95) if np.isfinite(unwrap_residual_p95) else float("nan"),
        unwrap_residual_vertical_p95_board=float(unwrap_residual_vertical_p95_board) if np.isfinite(unwrap_residual_vertical_p95_board) else float("nan"),
        unwrap_residual_horizontal_p95_board=float(unwrap_residual_horizontal_p95_board) if np.isfinite(unwrap_residual_horizontal_p95_board) else float("nan"),
        unwrap_residual_gt_1rad_pct_board=float(unwrap_residual_gt_1rad_pct_board) if np.isfinite(unwrap_residual_gt_1rad_pct_board) else float("nan"),
        board_mask_area_ratio=float(board_mask_area_ratio) if np.isfinite(board_mask_area_ratio) else float("nan"),
        clipping_detected=bool(clipping_detected),
        edge_corner_pct=float(edge_corner_pct),
        duplicate_pose=bool(duplicate),
        accept=bool(accept),
        reject_reasons=reasons,
        hints=sorted(set(hints)),
    )
    return diag, pose_entry
