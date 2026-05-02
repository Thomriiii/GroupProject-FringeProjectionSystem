"""Calibration consistency checks before reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fringe_app_v2.utils.io import load_image, write_json


def check_resolution(scan_res: tuple[int, int], calib_res: tuple[int, int]) -> None:
    if tuple(scan_res) != tuple(calib_res):
        raise AssertionError(f"Resolution mismatch: scan={scan_res}, calibration={calib_res}")


def run_calibration_check(
    run_dir: Path,
    config: dict[str, Any],
    calibration: Any,
    *,
    fail_on_error: bool | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    quality_dir = run_dir / "phase_quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    if fail_on_error is None:
        pq = config.get("phase_quality", {}) or {}
        fail_on_error = bool(pq.get("calibration_fail_on_mismatch", False))

    scan = config.get("scan", {}) or {}
    requested_projector = (int(scan.get("width", 1024)), int(scan.get("height", 768)))
    camera_observed = _observed_camera_resolution(run_dir)
    camera_calib = tuple(int(v) for v in calibration.camera.image_size)
    projector_calib = tuple(int(v) for v in calibration.projector.projector_size)

    checks = [
        _resolution_check("camera", camera_observed, camera_calib),
        _resolution_check("projector", requested_projector, projector_calib),
    ]
    intrinsic_checks = {
        "camera": _intrinsic_report(calibration.camera.matrix, camera_calib),
        "projector": _intrinsic_report(calibration.projector.matrix, projector_calib),
    }
    errors = [check["message"] for check in checks if not check["ok"]]
    errors.extend(_intrinsic_errors(intrinsic_checks))
    report = {
        "mode": "calibration_check",
        "ok": len(errors) == 0,
        "errors": errors,
        "checks": checks,
        "intrinsics": intrinsic_checks,
        "paths": {
            "camera_intrinsics": str(calibration.camera.path),
            "projector_stereo": str(calibration.projector.path),
        },
        "fail_on_error": bool(fail_on_error),
    }
    write_json(quality_dir / "calibration_report.json", report)
    if fail_on_error and errors:
        raise RuntimeError("; ".join(errors))
    return report


def _resolution_check(name: str, scan_res: tuple[int, int], calib_res: tuple[int, int]) -> dict[str, Any]:
    try:
        check_resolution(scan_res, calib_res)
        return {"name": name, "ok": True, "scan_resolution": list(scan_res), "calibration_resolution": list(calib_res)}
    except AssertionError as exc:
        return {
            "name": name,
            "ok": False,
            "scan_resolution": list(scan_res),
            "calibration_resolution": list(calib_res),
            "message": str(exc),
        }


def _intrinsic_report(matrix: np.ndarray, resolution: tuple[int, int]) -> dict[str, Any]:
    k = np.asarray(matrix, dtype=np.float64)
    width, height = resolution
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "focal_positive": bool(np.isfinite(fx) and np.isfinite(fy) and fx > 0 and fy > 0),
        "focal_ratio": float(fx / fy) if fy else None,
        "principal_point_inside_image": bool(0 <= cx <= width and 0 <= cy <= height),
        "principal_offset_fraction": [
            float((cx - width / 2.0) / max(width, 1)),
            float((cy - height / 2.0) / max(height, 1)),
        ],
    }


def _intrinsic_errors(reports: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for name, report in reports.items():
        if not report["focal_positive"]:
            errors.append(f"{name} focal lengths are invalid")
        if not report["principal_point_inside_image"]:
            errors.append(f"{name} principal point is outside the calibrated image")
    return errors


def _observed_camera_resolution(run_dir: Path) -> tuple[int, int]:
    raw_path = run_dir / "raw" / "roi_capture.png"
    if raw_path.exists():
        image = load_image(raw_path)
        return int(image.shape[1]), int(image.shape[0])
    for phase_file in sorted((run_dir / "phase").glob("*/*/phi_wrapped.npy")):
        phase = np.load(phase_file)
        return int(phase.shape[1]), int(phase.shape[0])
    raise FileNotFoundError(f"No captured image or phase map found under {run_dir}")
