"""Projector calibration (camera <-> projector stereo) workflow."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    _cv2_import_error = exc
else:
    _cv2_import_error = None

from PIL import Image
from PIL import ImageDraw

from fringe_app.calibration.checkerboard import CheckerboardDetection, detect_checkerboard, save_image
from fringe_app.calibration.projector_session import (
    add_view_if_valid,
    init_coverage_map,
    load_session as load_projector_session,
    next_view_id,
    recent_accepted_poses,
    save_session as save_projector_session,
    update_coverage_map as update_session_coverage_map,
)
from fringe_app.calibration.projector_view_gating import (
    ProjectorViewDiagnostics,
    evaluate_projector_view,
)
from fringe_app.calibration.gamma import calibrate_gamma_lut, load_gamma_lut
from fringe_app.calibration.dense_plane_calibration import calibrate_dense_plane_session
from fringe_app.calibration.view_quality import (
    ViewQualityReport,
    evaluate_view as evaluate_view_quality,
)
from fringe_app.calibration.uv_refine import UvRefineConfig, refine_uv_at_corners
from fringe_app.cli import cmd_phase, cmd_unwrap
from fringe_app.core.models import ScanParams
from fringe_app.phase.masking_post import build_unwrap_mask
from fringe_app.unwrap.temporal import unwrap_multi_frequency
from fringe_app.uv import phase_to_uv, save_uv_outputs


SCHEMA_VERSION = 1
_log = logging.getLogger(__name__)


def _require_cv2():
    if cv2 is None:
        raise RuntimeError(f"OpenCV is required for projector calibration: {_cv2_import_error}")
    return cv2


def _projector_root(root_dir: Path) -> Path:
    return root_dir / "projector"


def _sessions_root(root_dir: Path) -> Path:
    return _projector_root(root_dir) / "sessions"


def _session_dir(root_dir: Path, session_id: str) -> Path:
    return _sessions_root(root_dir) / session_id


def _session_json_path(session_dir: Path) -> Path:
    return session_dir / "session.json"


def _load_session(session_dir: Path) -> dict[str, Any]:
    return load_projector_session(session_dir)


def _save_session(session_dir: Path, data: dict[str, Any]) -> None:
    save_projector_session(session_dir, data)


def _optional_finite_float(value: Any) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return float(v)


def _safe_float_or_nan(value: Any) -> float:
    try:
        v = float(value)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _projector_uv_span(projector_corners: np.ndarray, valid_mask: np.ndarray) -> tuple[float | None, float | None]:
    pts = np.asarray(projector_corners, dtype=np.float64).reshape(-1, 2)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if pts.shape[0] != valid.shape[0]:
        return (None, None)
    pts = pts[valid]
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.size == 0:
        return (None, None)
    u_span = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
    v_span = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
    return (u_span, v_span)


def _calibration_thresholds_used(
    *,
    min_corner_valid_ratio: float,
    residual_p95_board_threshold: float,
    residual_gt_1rad_pct_board_max: float,
    strict_resolution_match: bool,
    min_hull_area_ratio: float = 0.08,
    min_tilt_deg: float = 10.0,
    min_b_median_board: float = 20.0,
    max_unwrap_residual_p95: float | None = None,
    max_unwrap_gt_1rad_pct: float | None = None,
) -> dict[str, Any]:
    return {
        "min_corner_valid_ratio": float(min_corner_valid_ratio),
        "min_hull_area_ratio": float(min_hull_area_ratio),
        "min_tilt_deg": float(min_tilt_deg),
        "min_b_median_board": float(min_b_median_board),
        "residual_p95_board_threshold": float(residual_p95_board_threshold),
        "residual_gt_1rad_pct_board_max": float(residual_gt_1rad_pct_board_max),
        "max_unwrap_residual_p95": float(
            residual_p95_board_threshold if max_unwrap_residual_p95 is None else max_unwrap_residual_p95
        ),
        "max_unwrap_gt_1rad_pct": float(
            residual_gt_1rad_pct_board_max if max_unwrap_gt_1rad_pct is None else max_unwrap_gt_1rad_pct
        ),
        "strict_resolution_match": bool(strict_resolution_match),
    }


def _save_coverage_png(coverage: dict[str, Any], out_path: Path) -> None:
    bins_x = max(1, int(coverage.get("grid_size", [16, 16])[0]))
    bins_y = max(1, int(coverage.get("grid_size", [16, 16])[1]))
    grid = np.asarray(coverage.get("grid", []), dtype=np.int32)
    if grid.ndim != 2 or grid.shape != (bins_y, bins_x):
        grid = np.zeros((bins_y, bins_x), dtype=np.int32)
    cell = 24
    pad = 20
    w = pad * 2 + bins_x * cell
    h = pad * 2 + bins_y * cell + 30
    img = Image.new("RGB", (w, h), (20, 22, 26))
    draw = ImageDraw.Draw(img)
    for by in range(bins_y):
        for bx in range(bins_x):
            x0 = pad + bx * cell
            y0 = pad + by * cell
            x1 = x0 + cell - 2
            y1 = y0 + cell - 2
            filled = bool(grid[by, bx] > 0)
            fill = (78, 170, 96) if filled else (64, 66, 72)
            edge = (30, 115, 55) if filled else (95, 95, 95)
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=edge, width=1)
    covered = int(coverage.get("bins_covered_count", int(np.count_nonzero(grid))))
    total = int(coverage.get("bins_total", int(grid.size)))
    ratio = float(coverage.get("coverage_ratio", covered / max(total, 1)))
    draw.text((pad, h - 24), f"Coverage {covered}/{total} ({100.0 * ratio:.1f}%)", fill=(235, 235, 235))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _make_scan_params(cfg: dict, orientation: str) -> ScanParams:
    scan = cfg.get("scan", {}) or {}
    pat = cfg.get("patterns", {}) or {}
    pcal = cfg.get("projector_calibration", {}) or {}
    cam = (pcal.get("camera", {}) or {})

    freqs = scan.get("frequencies")
    if not freqs:
        freqs = [float(scan.get("frequency", 8.0))]
    frequencies = [float(f) for f in freqs]
    mode = str(scan.get("mode", "stable"))
    n_steps = int(
        scan.get("n_steps_stable", scan.get("n_steps", 8))
        if mode == "stable"
        else scan.get("n_steps_fast", scan.get("n_steps", 4))
    )
    return ScanParams(
        n_steps=n_steps,
        frequency=float(frequencies[0]),
        frequencies=frequencies,
        orientation=orientation,  # type: ignore[arg-type]
        brightness=float(scan.get("brightness", 1.0)),
        resolution=(int(scan.get("width", 1024)), int(scan.get("height", 768))),
        settle_ms=int((pcal.get("capture", {}) or {}).get("settle_ms", scan.get("settle_ms", 150))),
        save_patterns=False,
        preview_fps=float(scan.get("preview_fps", 10.0)),
        projector_screen_index=scan.get("projector_screen_index"),
        phase_convention=str(scan.get("phase_convention", "atan2(-S,C)")),
        exposure_us=int(cam.get("exposure_us", scan.get("exposure_us", 2000))),
        analogue_gain=float(cam.get("analogue_gain", scan.get("analogue_gain", 1.0))),
        awb_mode=scan.get("awb_mode"),
        awb_enable=bool(cam.get("awb_enable", False)),
        ae_enable=bool(cam.get("ae_enable", False)),
        iso=scan.get("iso"),
        brightness_offset=float(cam.get("brightness_offset", pat.get("brightness_offset", 0.45))),
        contrast=float(cam.get("contrast", pat.get("contrast", 0.6))),
        min_intensity=float(cam.get("min_intensity", pat.get("min_intensity", 0.10))),
        frequency_semantics=str(scan.get("frequency_semantics", "cycles_across_dimension")),
        phase_origin_rad=float(scan.get("phase_origin_rad", 0.0)),
        auto_normalise=False,
        expert_mode=True,
    )


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _freq_tag(freq: float) -> str:
    if float(freq).is_integer():
        return f"f_{int(freq):03d}"
    return f"f_{str(freq).replace('.', 'p')}"


def _select_high_phase_dir(run_dir: Path, f_high: float) -> Path:
    return run_dir / "phase" / _freq_tag(f_high)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _find_camera_intrinsics_latest(cfg: dict) -> Path | None:
    calib = cfg.get("calibration", {}) or {}
    candidates = [
        Path(calib.get("camera_root", "data/calibration/camera")) / "intrinsics_latest.json",
        Path(calib.get("root", "data/calibration")) / "camera" / "intrinsics_latest.json",
        Path("data/calibration/camera/intrinsics_latest.json"),
        Path("data/calibration/camera_intrinsics/intrinsics_latest.json"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _calibration_root_from_cfg(cfg: dict) -> Path:
    calib = cfg.get("calibration", {}) or {}
    return Path(calib.get("root", "data/calibration"))


def capture_projector_view(
    session_id: str,
    cfg: dict,
    controller,
    *,
    root_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Capture-mode orchestration (fast per-pose):
      capture -> unwrap/uv -> checkerboard correspondences -> per-view diagnostics.

    Does NOT run stereo solve and does NOT write session-level solve results.
    """
    base_root = root_dir if root_dir is not None else _calibration_root_from_cfg(cfg)
    sdir = _session_dir(base_root, session_id)
    if not sdir.exists():
        raise FileNotFoundError(f"Projector calibration session not found: {sdir}")
    return capture_view(sdir, controller, cfg)


def solve_projector_session(
    session_id: str,
    cfg: dict,
    *,
    root_dir: Path | None = None,
    camera_intrinsics_path: Path | None = None,
    prune: bool = False,
) -> dict[str, Any]:
    """
    Solve-mode orchestration (batch):
      load accepted views -> run stereo solve -> optional pruning -> write results/.

    Does NOT recapture views and does NOT rewrite per-view UV artifacts.
    """
    base_root = root_dir if root_dir is not None else _calibration_root_from_cfg(cfg)
    sdir = _session_dir(base_root, session_id)
    if not sdir.exists():
        raise FileNotFoundError(f"Projector calibration session not found: {sdir}")
    intr_path = camera_intrinsics_path or _find_camera_intrinsics_latest(cfg)
    if intr_path is None or not intr_path.exists():
        raise FileNotFoundError("Camera intrinsics latest file not found.")
    return stereo_calibrate(sdir, cfg, intr_path, prune=prune)


def calibrate_projector_gamma(
    session_id: str,
    cfg: dict,
    controller,
    *,
    root_dir: Path | None = None,
) -> dict[str, Any]:
    base_root = root_dir if root_dir is not None else _calibration_root_from_cfg(cfg)
    sdir = _session_dir(base_root, session_id)
    if not sdir.exists():
        raise FileNotFoundError(f"Projector calibration session not found: {sdir}")
    return calibrate_gamma_lut(controller, sdir, cfg)


def create_session(root_dir: Path, cfg: dict) -> str:
    sessions_dir = _sessions_root(root_dir)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (_projector_root(root_dir) / "results").mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = stamp
    i = 0
    while (sessions_dir / session_id).exists():
        i += 1
        session_id = f"{stamp}_{i:02d}"
    session_dir = sessions_dir / session_id
    (session_dir / "views").mkdir(parents=True, exist_ok=True)
    (session_dir / "results").mkdir(parents=True, exist_ok=True)

    pcal = cfg.get("projector_calibration", {}) or {}
    calib_cfg = cfg.get("calibration", {}) or {}
    coverage_cfg = (pcal.get("coverage", {}) or {})
    cond_cfg = {
        "grid_w": int(coverage_cfg.get("grid_size_x", 64)),
        "grid_h": int(coverage_cfg.get("grid_size_y", 36)),
        "min_projector_coverage_ratio": float(calib_cfg.get("min_projector_coverage_ratio", 0.6)),
        "min_uniformity_metric": float(coverage_cfg.get("min_uniformity_metric", 0.40)),
        "min_edge_coverage_ratio": float(coverage_cfg.get("min_edge_coverage_ratio", 0.25)),
        "stop_when_sufficient": bool(coverage_cfg.get("stop_when_sufficient", True)),
    }
    session_cfg = {
        "checkerboard": pcal.get("checkerboard", {}),
        "projector": pcal.get("projector", {}),
        "capture": pcal.get("capture", {}),
        "uv_mask_policy": pcal.get("uv_mask_policy", {}),
        "uv_sampling": pcal.get("uv_sampling", {}),
        "uv_gate": pcal.get("uv_gate", {}),
        "uv_refinement": pcal.get("uv_refinement", {}),
        "conditioning": cond_cfg,
    }
    proj_w = int((session_cfg.get("projector", {}) or {}).get("width", 1024))
    proj_h = int((session_cfg.get("projector", {}) or {}).get("height", 768))
    coverage_map = init_coverage_map((proj_w, proj_h), conditioning_cfg=cond_cfg)
    (session_dir / "config.json").write_text(json.dumps(session_cfg, indent=2))
    session = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "config": session_cfg,
        "views": [],
        "accepted_pose_history": [],
        "coverage_map": coverage_map,
        "capture_attempts": 0,
        "last_capture_result": None,
        "results": None,
    }
    _save_session(session_dir, session)
    try:
        (session_dir / "session_coverage.json").write_text(
            json.dumps(session["coverage_map"], indent=2)
        )
    except Exception:
        pass
    return session_id


def list_views(session_dir: Path) -> list[dict[str, Any]]:
    session = _load_session(session_dir)
    accepted = {str(v.get("view_id")): v for v in (session.get("views", []) or []) if isinstance(v, dict)}
    views: list[dict[str, Any]] = []
    for vdir in _session_view_folders(session_dir):
        vid = vdir.name
        diag_path = vdir / "view_diag.json"
        diag = _load_json(diag_path)
        item = dict(accepted.get(vid, {}))
        item["view_id"] = vid
        is_accepted = vid in accepted
        diag_status = str(diag.get("status", "PASS" if is_accepted else "FAIL")).upper()
        item["accept"] = bool(diag.get("accept", is_accepted))
        item["status"] = "valid" if item["accept"] else "rejected"
        if not item["accept"] and diag_status == "PASS":
            item["status"] = "valid"
        reject_reasons = [str(r) for r in (diag.get("reject_reasons", []) or [])]
        item["reject_reasons"] = reject_reasons
        item["reason"] = reject_reasons[0] if reject_reasons else item.get("reason")
        item["hints"] = [str(h) for h in (diag.get("hints", []) or item.get("hints", []))]
        item["diag"] = {
            "schema_version": int(diag.get("schema_version", SCHEMA_VERSION)),
            "status": diag_status,
            "accept": bool(diag.get("accept", item["accept"])),
            "thresholds": diag.get("thresholds", {}),
            "measured": diag.get("measured", {}),
            "reject_reasons": reject_reasons,
            "hints": item["hints"],
            "corner_validity_breakdown": diag.get("corner_validity_breakdown", {}),
            "uv": diag.get("uv", {}),
            "diagnostics": diag.get("diagnostics", {}),
        }
        if "valid_corner_ratio" not in item:
            vc = _safe_float_or_nan((diag.get("measured", {}) or {}).get("valid_corner_ratio"))
            item["valid_corner_ratio"] = float(vc) if np.isfinite(vc) else 0.0
        views.append(_to_json_safe(item))
    return views


def delete_view(session_dir: Path, view_id: str) -> None:
    session = _load_session(session_dir)
    vdir = session_dir / "views" / view_id
    if vdir.exists():
        shutil.rmtree(vdir)
    session["views"] = [v for v in session.get("views", []) if v.get("view_id") != view_id]
    # Rebuild coverage map from remaining accepted views.
    session_cfg = session.get("config", {}) or {}
    proj_cfg = (session_cfg.get("projector", {}) or {})
    cond_cfg = (session_cfg.get("conditioning", {}) or {})
    pw = int(proj_cfg.get("width", 1024))
    ph = int(proj_cfg.get("height", 768))
    cov = init_coverage_map((pw, ph), conditioning_cfg=cond_cfg)
    for view in session.get("views", []) or []:
        vid = str(view.get("view_id", ""))
        if not vid:
            continue
        corr_path = session_dir / "views" / vid / "correspondences.json"
        if not corr_path.exists():
            continue
        try:
            corr = json.loads(corr_path.read_text())
            pts = np.asarray(corr.get("projector_corners_px", []), dtype=np.float32).reshape(-1, 2)
            vm = np.asarray(corr.get("valid_mask", []), dtype=bool).reshape(-1)
            pts = pts[vm]
            if pts.size == 0:
                continue
            cov = update_session_coverage_map(cov, pts, conditioning_cfg=cond_cfg)
        except Exception:
            continue
    session["coverage_map"] = cov
    # Conservative reset of pose history after deletion to avoid stale duplicate checks.
    session["accepted_pose_history"] = []
    try:
        (session_dir / "session_coverage.json").write_text(json.dumps(cov, indent=2))
    except Exception:
        pass
    _save_session(session_dir, session)


def _capture_single_camera_frame(controller, cfg: dict) -> np.ndarray:
    pcal = cfg.get("projector_calibration", {}) or {}
    cam_cfg = (pcal.get("camera", {}) or {})
    flush = int((pcal.get("capture", {}) or {}).get("flush_frames", 1))
    frame = controller.capture_single_frame(
        flush_frames=flush,
        exposure_us=cam_cfg.get("exposure_us"),
        analogue_gain=cam_cfg.get("analogue_gain"),
        awb_enable=cam_cfg.get("awb_enable"),
        settle_ms=int(cam_cfg.get("settle_ms", 120)),
    )
    return frame


def _set_projector_calibration_light(controller, cfg: dict, enabled: bool) -> None:
    pcal = cfg.get("projector_calibration", {}) or {}
    cap_cfg = pcal.get("capture", {}) or {}
    dn = int(cap_cfg.get("checkerboard_white_dn", 230))
    try:
        controller.set_projector_calibration_light(enabled=enabled, dn=dn)
    except Exception:
        pass


def _capture_checkerboard_frame_with_projector(controller, cfg: dict) -> np.ndarray:
    """
    Project a bright white frame for checkerboard detection, capture one frame,
    then switch projector to black before UV scan.
    """
    pcal = cfg.get("projector_calibration", {}) or {}
    cap_cfg = pcal.get("capture", {}) or {}
    white_settle_ms = int(cap_cfg.get("checkerboard_white_settle_ms", 200))
    off_settle_ms = int(cap_cfg.get("checkerboard_off_settle_ms", 60))
    _set_projector_calibration_light(controller, cfg, enabled=True)
    time.sleep(max(0.0, white_settle_ms / 1000.0))

    frame = _capture_single_camera_frame(controller, cfg)
    _set_projector_calibration_light(controller, cfg, enabled=False)
    time.sleep(max(0.0, off_settle_ms / 1000.0))
    return frame


def _wait_for_controller_idle(controller, timeout_s: float = 600.0) -> tuple[str, dict[str, Any]]:
    t0 = time.time()
    while True:
        st = controller.get_status()
        state = str(st.get("state", "IDLE"))
        if state in ("IDLE", "ERROR"):
            return state, st
        if (time.time() - t0) > timeout_s:
            raise TimeoutError("Timed out waiting for scan to finish")
        time.sleep(0.2)


def _run_orientation_scan(controller, params: ScanParams) -> str:
    run_id = controller.start_scan(params)
    state, st = _wait_for_controller_idle(controller)
    if state == "ERROR":
        raise RuntimeError(st.get("last_error") or "orientation scan failed")
    rc_phase = cmd_phase(argparse.Namespace(run=run_id))
    if rc_phase != 0:
        raise RuntimeError(f"phase failed for run {run_id}")
    rc_unwrap = cmd_unwrap(argparse.Namespace(run=run_id, use_roi="auto"))
    if rc_unwrap != 0:
        raise RuntimeError(f"unwrap failed for run {run_id}")
    return run_id


def _load_mask_for_unwrap(phase_dir: Path) -> np.ndarray:
    for name in ("mask_for_unwrap.npy", "mask_clean.npy", "mask_raw.npy", "mask.npy"):
        p = phase_dir / name
        if p.exists():
            return np.load(p).astype(bool)
    raise FileNotFoundError(f"No unwrap mask found in {phase_dir}")


def _load_phase_artifacts(run_dir: Path, freq: float) -> dict[str, np.ndarray]:
    pdir = _select_high_phase_dir(run_dir, freq)
    return {
        "phi_wrapped": np.load(pdir / "phi_wrapped.npy").astype(np.float32),
        "A": np.load(pdir / "A.npy").astype(np.float32),
        "B": np.load(pdir / "B.npy").astype(np.float32),
        "clipped_any": np.load(pdir / "clipped_any.npy").astype(bool),
    }


def _load_roi_mask(run_dir: Path) -> np.ndarray | None:
    p = run_dir / "roi" / "roi_mask.png"
    if not p.exists():
        return None
    return np.array(Image.open(p)) > 0


def _projector_size_from_run_meta(run_dir: Path) -> tuple[int, int] | None:
    meta = _load_json(run_dir / "meta.json")
    dev = meta.get("device_info", {}) or {}
    size = dev.get("projector_surface_size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        try:
            return int(size[0]), int(size[1])
        except Exception:
            return None
    return None


def _calibration_uv_policy(cfg: dict) -> dict[str, Any]:
    pcal = cfg.get("projector_calibration", {}) or {}
    policy = (pcal.get("uv_mask_policy", {}) or {})
    unwrap_policy = (policy.get("unwrap_mask", {}) or {})
    return {
        "use_roi": bool(policy.get("use_roi", False)),
        "a_min_unwrap": float(unwrap_policy.get("a_min_unwrap", 10.0)),
        "b_thresh_unwrap": float(unwrap_policy.get("b_thresh_unwrap", 5.0)),
        "closing_radius_px": int(unwrap_policy.get("closing_radius_px", 3)),
        "erosion_radius_px": int(unwrap_policy.get("erosion_radius_px", 0)),
        "keep_largest_component": bool(unwrap_policy.get("keep_largest_component", True)),
        "min_area_px": int(unwrap_policy.get("min_area_px", 2000)),
    }


def _uv_refine_config(cfg: dict) -> UvRefineConfig:
    pcal = cfg.get("projector_calibration", {}) or {}
    return UvRefineConfig.from_dict((pcal.get("uv_refinement", {}) or {}))


def _build_unwrap_masks_for_run(
    run_dir: Path,
    freqs: list[float],
    roi_mask: np.ndarray | None,
    policy: dict[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    phases: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    per_freq: dict[str, Any] = {}
    use_roi = bool(policy.get("use_roi", False))
    for f in freqs:
        art = _load_phase_artifacts(run_dir, f)
        base_mask = (
            (~art["clipped_any"])
            & (art["A"] >= float(policy.get("a_min_unwrap", 10.0)))
            & (art["B"] >= float(policy.get("b_thresh_unwrap", 5.0)))
        )
        unwrap_cfg = {
            "closing_radius_px": int(policy.get("closing_radius_px", 3)),
            "erosion_radius_px": int(policy.get("erosion_radius_px", 0)),
            "keep_largest_component": bool(policy.get("keep_largest_component", True)),
            "min_area_px": int(policy.get("min_area_px", 2000)),
        }
        mask = build_unwrap_mask(
            base_mask,
            roi_mask if use_roi else None,
            art["clipped_any"],
            unwrap_cfg,
        )
        phases.append(art["phi_wrapped"])
        masks.append(mask.astype(bool))
        per_freq[_freq_tag(f)] = {
            "valid_ratio": float(np.count_nonzero(mask) / mask.size),
            "a_median": float(np.nanmedian(art["A"][mask])) if np.any(mask) else 0.0,
            "b_median": float(np.nanmedian(art["B"][mask])) if np.any(mask) else 0.0,
            "clipped_any_pct": float(np.mean(art["clipped_any"])),
            "clipped_any_pct_roi": float(np.mean(art["clipped_any"][roi_mask])) if (roi_mask is not None and np.any(roi_mask)) else float(np.mean(art["clipped_any"])),
        }
    return phases, masks, per_freq


def _compute_phase_quality_for_run(run_dir: Path, f_high: float) -> dict[str, Any]:
    phase_meta = _load_json(run_dir / "phase" / _freq_tag(f_high) / "phase_meta.json")
    unwrap_meta = _load_json(run_dir / "unwrap" / "unwrap_meta.json")
    return {
        "roi_valid_ratio": float(phase_meta.get("roi_valid_ratio", 0.0)),
        "clipped_roi": float(phase_meta.get("clipped_any_pct_roi", 1.0)),
        "residual_p95": float(unwrap_meta.get("residual_p95", 999.0)),
        "residual_ok": bool(unwrap_meta.get("residual_ok", False)),
        "b_median": float(phase_meta.get("roi_b_median", phase_meta.get("B_median", 0.0))),
    }


def run_uv_scan(
    controller,
    cfg: dict,
    out_dir: Path,
    overrides: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """
    Run vertical+horizontal UV scan using controller (no shell-out).
    Saves UV artifacts in out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    run_root = Path((cfg.get("storage", {}) or {}).get("run_root", "data/runs"))

    cfg_scan = _deep_update(cfg, overrides or {})
    p_v = _make_scan_params(cfg_scan, "vertical")
    p_h = _make_scan_params(cfg_scan, "horizontal")
    run_v = _run_orientation_scan(controller, p_v)
    run_h = _run_orientation_scan(controller, p_h)

    freqs = p_v.get_frequencies()
    f_high = float(max(freqs))
    tag = _freq_tag(f_high)
    v_dir = run_root / run_v
    h_dir = run_root / run_h

    roi_v = _load_roi_mask(v_dir)
    roi_h = _load_roi_mask(h_dir)
    roi_intersection = None
    if roi_v is not None and roi_h is not None:
        roi_intersection = roi_v & roi_h
    elif roi_v is not None:
        roi_intersection = roi_v
    elif roi_h is not None:
        roi_intersection = roi_h

    policy = _calibration_uv_policy(cfg_scan)
    phases_v, masks_v, mask_stats_v = _build_unwrap_masks_for_run(v_dir, freqs, roi_v, policy)
    phases_h, masks_h, mask_stats_h = _build_unwrap_masks_for_run(h_dir, freqs, roi_h, policy)
    if len(freqs) >= 2:
        phi_v, mask_v_unwrap, unwrap_meta_v, _ = unwrap_multi_frequency(
            phases=phases_v,
            masks=masks_v,
            freqs=freqs,
            roi_mask=roi_v,
            use_roi=bool(policy.get("use_roi", False)),
        )
        phi_h, mask_h_unwrap, unwrap_meta_h, _ = unwrap_multi_frequency(
            phases=phases_h,
            masks=masks_h,
            freqs=freqs,
            roi_mask=roi_h,
            use_roi=bool(policy.get("use_roi", False)),
        )
        mask_v = masks_v[freqs.index(f_high)] & mask_v_unwrap
        mask_h = masks_h[freqs.index(f_high)] & mask_h_unwrap
    else:
        phi_v = np.load(v_dir / "unwrap" / "phi_abs.npy").astype(np.float32)
        phi_h = np.load(h_dir / "unwrap" / "phi_abs.npy").astype(np.float32)
        mask_v = masks_v[0] & np.isfinite(phi_v)
        mask_h = masks_h[0] & np.isfinite(phi_h)
        unwrap_meta_v = _load_json(v_dir / "unwrap" / "unwrap_meta.json")
        unwrap_meta_h = _load_json(h_dir / "unwrap" / "unwrap_meta.json")

    pcal = cfg_scan.get("projector_calibration", {}) or {}
    proj = pcal.get("projector", {}) or {}
    proj_w_cfg = int(proj.get("width", p_v.resolution[0]))
    proj_h_cfg = int(proj.get("height", p_v.resolution[1]))
    surf_v = _projector_size_from_run_meta(v_dir)
    surf_h = _projector_size_from_run_meta(h_dir)
    actual_surface = surf_v or surf_h
    if surf_v is not None and surf_h is not None and surf_v != surf_h:
        actual_surface = surf_v
    proj_w = int(actual_surface[0]) if actual_surface is not None else proj_w_cfg
    proj_h = int(actual_surface[1]) if actual_surface is not None else proj_h_cfg
    projector_res_mismatch = bool(actual_surface is not None and (proj_w_cfg != proj_w or proj_h_cfg != proj_h))

    uv_res = phase_to_uv(
        phi_abs_vertical=phi_v,
        phi_abs_horizontal=phi_h,
        freq_u=f_high,
        freq_v=f_high,
        proj_width=proj_w,
        proj_height=proj_h,
        mask_u=mask_v,
        mask_v=mask_h,
        roi_mask=roi_intersection if bool(policy.get("use_roi", False)) else None,
        frequency_semantics=p_v.frequency_semantics,
        phase_origin_u_rad=float(p_v.phase_origin_rad),
        phase_origin_v_rad=float(p_h.phase_origin_rad),
        gate_cfg=((pcal.get("uv_gate", {}) or {}) or (cfg_scan.get("uv_gate", {}) or {})),
    )

    # Use latest camera frame of vertical scan as overlay base.
    base = None
    cap_dir = v_dir / "captures" / tag
    frames = sorted(cap_dir.glob("step_*.png"))
    if frames:
        base = np.array(Image.open(frames[0]))
    save_uv_outputs(out_dir, uv_res, base_frame=base)

    meta = dict(uv_res.meta)
    meta["projector_resolution"] = {
        "configured": [proj_w_cfg, proj_h_cfg],
        "pattern_surface": [proj_w, proj_h],
        "mismatch": projector_res_mismatch,
    }
    meta["mask_policy"] = policy
    meta["phase_quality"] = {
        "vertical": _compute_phase_quality_for_run(v_dir, f_high),
        "horizontal": _compute_phase_quality_for_run(h_dir, f_high),
    }
    meta["unwrap_meta_policy"] = {
        "vertical": unwrap_meta_v,
        "horizontal": unwrap_meta_h,
    }
    meta["per_frequency_mask_stats"] = {
        "vertical": mask_stats_v,
        "horizontal": mask_stats_h,
    }
    meta["source_runs"] = {"vertical": run_v, "horizontal": run_h}
    (out_dir / "uv_meta.json").write_text(json.dumps(meta, indent=2))
    return uv_res.u, uv_res.v, uv_res.mask_uv, meta, {
        "vertical_run_id": run_v,
        "horizontal_run_id": run_h,
        "projector_surface_size": [proj_w, proj_h],
        "configured_projector_size": [proj_w_cfg, proj_h_cfg],
        "projector_res_mismatch": projector_res_mismatch,
    }


def _bilinear(values: np.ndarray, x: float, y: float) -> float:
    h, w = values.shape
    if x < 0 or y < 0 or x > (w - 1) or y > (h - 1):
        return float("nan")
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    q11 = values[y0, x0]
    q21 = values[y0, x1]
    q12 = values[y1, x0]
    q22 = values[y1, x1]
    if not (np.isfinite(q11) and np.isfinite(q21) and np.isfinite(q12) and np.isfinite(q22)):
        return float("nan")
    wx = x - x0
    wy = y - y0
    return float(
        q11 * (1 - wx) * (1 - wy)
        + q21 * wx * (1 - wy)
        + q12 * (1 - wx) * wy
        + q22 * wx * wy
    )


def _sample_uv(
    u: np.ndarray,
    v: np.ndarray,
    mask_uv: np.ndarray,
    corners: np.ndarray,
    cfg: dict,
    projector_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[int], dict[str, Any]]:
    pcal = cfg.get("projector_calibration", {}) or {}
    cap_cfg = pcal.get("capture", {}) or {}
    uv_sampling_cfg = (pcal.get("uv_sampling", {}) or {})
    legacy_uv_cfg = cap_cfg.get("uv_sample", {}) or {}
    method = str(uv_sampling_cfg.get("method", "median_patch"))
    patch_radius = int(uv_sampling_cfg.get("patch_radius", legacy_uv_cfg.get("patch_radius_px", 2)))
    patch_radius = max(1, patch_radius)
    min_valid_points = int(
        uv_sampling_cfg.get("min_valid_points", legacy_uv_cfg.get("min_finite_samples", 8))
    )
    min_valid_points = max(1, min_valid_points)
    edge_margin = max(0, int(cap_cfg.get("max_corner_uv_edge_px", 2)))
    # Optional future refinement stays available but median patch is baseline.
    refine_cfg = uv_sampling_cfg.get("refinement", {}) or {}
    refine_enabled = bool(refine_cfg.get("enabled", False))
    refine_min_points = max(3, int(refine_cfg.get("min_valid_points", 12)))
    proj = pcal.get("projector", {}) or {}
    if projector_size is not None:
        pw = int(projector_size[0])
        ph = int(projector_size[1])
    else:
        pw = int(proj.get("width", 1024))
        ph = int(proj.get("height", 768))

    cam = corners.reshape(-1, 2).astype(np.float32)
    proj_pts = np.full((cam.shape[0], 2), np.nan, dtype=np.float32)
    valid = np.zeros((cam.shape[0],), dtype=bool)
    reasons: list[str] = []
    finite_counts: list[int] = []
    refine_used = 0
    refine_patch_points: list[int] = []
    refine_residuals: list[float] = []

    h, w = u.shape
    for i, (x, y) in enumerate(cam):
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        finite_count = 0
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            reasons.append("oob")
            finite_counts.append(finite_count)
            continue

        x0 = max(0, xi - patch_radius)
        x1 = min(w, xi + patch_radius + 1)
        y0 = max(0, yi - patch_radius)
        y1 = min(h, yi + patch_radius + 1)
        patch_mask = mask_uv[y0:y1, x0:x1]
        pu = u[y0:y1, x0:x1]
        pv = v[y0:y1, x0:x1]
        local_valid = patch_mask & np.isfinite(pu) & np.isfinite(pv)
        vals_u = pu[local_valid].astype(np.float64)
        vals_v = pv[local_valid].astype(np.float64)
        finite_count = int(min(vals_u.size, vals_v.size))
        if finite_count < min_valid_points:
            reasons.append("mask_uv_false" if not bool(mask_uv[yi, xi]) else "nan_uv")
            finite_counts.append(finite_count)
            continue
        us = float(np.median(vals_u))
        vs = float(np.median(vals_v))

        if refine_enabled and finite_count >= refine_min_points:
            ys_local, xs_local = np.where(local_valid)
            xs_abs = xs_local.astype(np.float64) + float(x0)
            ys_abs = ys_local.astype(np.float64) + float(y0)
            A = np.column_stack([xs_abs, ys_abs, np.ones(xs_abs.shape[0], dtype=np.float64)])
            try:
                coeff_u, *_ = np.linalg.lstsq(A, vals_u, rcond=None)
                coeff_v, *_ = np.linalg.lstsq(A, vals_v, rcond=None)
                us_ref = float(coeff_u[0] * float(x) + coeff_u[1] * float(y) + coeff_u[2])
                vs_ref = float(coeff_v[0] * float(x) + coeff_v[1] * float(y) + coeff_v[2])
                if np.isfinite(us_ref) and np.isfinite(vs_ref):
                    us = us_ref
                    vs = vs_ref
                    refine_used += 1
                    refine_patch_points.append(finite_count)
                    u_fit = A @ coeff_u
                    v_fit = A @ coeff_v
                    r_u = float(np.mean(np.abs(vals_u - u_fit)))
                    r_v = float(np.mean(np.abs(vals_v - v_fit)))
                    refine_residuals.append(0.5 * (r_u + r_v))
            except Exception:
                pass

        if not (np.isfinite(us) and np.isfinite(vs)):
            reasons.append("nan_uv")
            finite_counts.append(finite_count)
            continue
        if us < 0.0 or us >= float(pw) or vs < 0.0 or vs >= float(ph):
            reasons.append("oob")
            finite_counts.append(finite_count)
            continue
        if us < edge_margin or us > (pw - 1 - edge_margin):
            reasons.append("near_edge")
            finite_counts.append(finite_count)
            continue
        if vs < edge_margin or vs > (ph - 1 - edge_margin):
            reasons.append("near_edge")
            finite_counts.append(finite_count)
            continue

        proj_pts[i, 0] = us
        proj_pts[i, 1] = vs
        valid[i] = True
        reasons.append("ok")
        finite_counts.append(finite_count)

    sampling_meta = {
        "method": method,
        "uv_sampling_method": method,
        "patch_radius": int(patch_radius),
        "min_valid_points": int(min_valid_points),
        "refinement": {
            "enabled": bool(refine_enabled),
            "min_valid_points": int(refine_min_points),
            "refined_corners": int(refine_used),
            "mean_patch_size": float(np.mean(refine_patch_points)) if refine_patch_points else 0.0,
            "plane_fit_residual_mean": float(np.mean(refine_residuals)) if refine_residuals else 0.0,
        },
        "max_corner_uv_edge_px": edge_margin,
        "projector_size_for_sampling": [int(pw), int(ph)],
    }
    return cam, proj_pts, valid, reasons, finite_counts, sampling_meta


def _draw_correspondence_overlay(image: np.ndarray, corners_cam: np.ndarray, corners_proj: np.ndarray, valid: np.ndarray) -> np.ndarray:
    cv = _require_cv2()
    if image.ndim == 2:
        out = np.stack([image, image, image], axis=2).astype(np.uint8)
    else:
        out = image[:, :, :3].astype(np.uint8).copy()
    for i, ((x, y), (u, v), ok) in enumerate(zip(corners_cam, corners_proj, valid)):
        color = (64, 220, 64) if ok else (220, 64, 64)
        cv.circle(out, (int(round(x)), int(round(y))), 4, color, 1, cv.LINE_AA)
        if ok and (i % max(1, len(corners_cam) // 12) == 0):
            cv.putText(
                out,
                f"{u:.0f},{v:.0f}",
                (int(round(x)) + 6, int(round(y)) - 6),
                cv.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 64),
                1,
                cv.LINE_AA,
            )
    return out


def _draw_corner_validity_overlay(
    image: np.ndarray,
    corners_cam: np.ndarray,
    reasons: list[str],
    counts: dict[str, int],
) -> np.ndarray:
    cv = _require_cv2()
    if image.ndim == 2:
        out = np.stack([image, image, image], axis=2).astype(np.uint8)
    else:
        out = image[:, :, :3].astype(np.uint8).copy()
    color_map = {
        "ok": (64, 220, 64),            # green
        "nan_uv": (220, 64, 64),        # red
        "mask_uv_false": (255, 170, 48),# orange
        "near_edge": (64, 128, 255),    # blue
        "oob": (64, 128, 255),          # blue
    }
    for (x, y), reason in zip(corners_cam, reasons):
        color = color_map.get(reason, (220, 64, 64))
        cv.circle(out, (int(round(float(x))), int(round(float(y)))), 4, color, 1, cv.LINE_AA)
    text = (
        f"ok={counts.get('ok', 0)} "
        f"nan={counts.get('nan_uv', 0)} "
        f"mask={counts.get('mask_uv_false', 0)} "
        f"edge={counts.get('near_edge', 0)} "
        f"oob={counts.get('oob', 0)}"
    )
    cv.putText(
        out,
        text,
        (16, 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 64),
        2,
        cv.LINE_AA,
    )
    return out


def _save_uv_refine_delta_overlay(
    image: np.ndarray,
    corners_cam: np.ndarray,
    uv_raw: np.ndarray,
    uv_refined: np.ndarray,
    valid_refined: np.ndarray,
    out_path: Path,
) -> dict[str, float]:
    if image.ndim == 2:
        rgb = np.stack([image, image, image], axis=2).astype(np.uint8)
    else:
        rgb = image[:, :, :3].astype(np.uint8).copy()
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)

    corners = np.asarray(corners_cam, dtype=np.float32).reshape(-1, 2)
    raw = np.asarray(uv_raw, dtype=np.float32).reshape(-1, 2)
    refined = np.asarray(uv_refined, dtype=np.float32).reshape(-1, 2)
    valid = np.asarray(valid_refined, dtype=bool).reshape(-1)
    if corners.shape[0] != raw.shape[0] or raw.shape[0] != refined.shape[0] or refined.shape[0] != valid.shape[0]:
        canvas.save(out_path)
        return {"mean_delta_px": float("nan"), "p95_delta_px": float("nan"), "max_delta_px": float("nan")}

    delta = np.linalg.norm(refined - raw, axis=1)
    finite = np.isfinite(delta) & np.isfinite(raw[:, 0]) & np.isfinite(raw[:, 1]) & np.isfinite(refined[:, 0]) & np.isfinite(refined[:, 1])
    delta_valid = delta[valid & finite]
    dmax = float(np.max(delta_valid)) if delta_valid.size else 1.0
    if dmax <= 1e-6:
        dmax = 1.0

    for (x, y), d, ok in zip(corners, delta, valid):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if not ok or not np.isfinite(d):
            color = (255, 64, 64)
        else:
            t = float(np.clip(d / dmax, 0.0, 1.0))
            color = (int(255 * t), int(255 * (1.0 - t)), 80)
        r = 4
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=1)

    mean_delta = float(np.mean(delta_valid)) if delta_valid.size else float("nan")
    p95_delta = float(np.percentile(delta_valid, 95)) if delta_valid.size else float("nan")
    max_delta = float(np.max(delta_valid)) if delta_valid.size else float("nan")
    label = f"mean={mean_delta:.3f}px p95={p95_delta:.3f}px"
    draw.rectangle((8, 8, 360, 36), fill=(0, 0, 0))
    draw.text((14, 12), label, fill=(255, 255, 64))
    canvas.save(out_path)
    return {"mean_delta_px": mean_delta, "p95_delta_px": p95_delta, "max_delta_px": max_delta}


def _write_view_uv_refinement_artifacts(
    *,
    view_dir: Path,
    uv_dir: Path,
    camera_image: np.ndarray,
    corners_cam: np.ndarray,
    raw_uv: np.ndarray,
    raw_valid: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    refine_cfg = _uv_refine_config(cfg)
    raw_uv_f32 = np.asarray(raw_uv, dtype=np.float32).reshape(-1, 2)
    raw_valid_b = np.asarray(raw_valid, dtype=bool).reshape(-1)
    corners = np.asarray(corners_cam, dtype=np.float32).reshape(-1, 2)
    np.save(view_dir / "uv_corners_raw.npy", raw_uv_f32)

    if not refine_cfg.enabled:
        np.save(view_dir / "uv_corners_refined.npy", raw_uv_f32)
        np.save(view_dir / "uv_corners_valid.npy", raw_valid_b)
        diag = {
            "config": _to_json_safe(asdict(refine_cfg)),
            "summary": {
                "enabled": False,
                "corners_total": int(raw_valid_b.size),
                "corners_valid": int(np.count_nonzero(raw_valid_b)),
                "valid_ratio": float(np.count_nonzero(raw_valid_b) / max(1, raw_valid_b.size)),
            },
        }
        (view_dir / "uv_refine_diag.json").write_text(json.dumps(diag, indent=2))
        _save_uv_refine_delta_overlay(
            image=camera_image,
            corners_cam=corners,
            uv_raw=raw_uv_f32,
            uv_refined=raw_uv_f32,
            valid_refined=raw_valid_b,
            out_path=view_dir / "uv_refine_delta.png",
        )
        return diag

    u_path = uv_dir / "u.npy"
    v_path = uv_dir / "v.npy"
    m_path = uv_dir / "mask_uv.npy"
    if not (u_path.exists() and v_path.exists() and m_path.exists()):
        diag = {
            "config": _to_json_safe(asdict(refine_cfg)),
            "summary": {"enabled": True, "error": "missing_uv_maps"},
        }
        (view_dir / "uv_refine_diag.json").write_text(json.dumps(diag, indent=2))
        np.save(view_dir / "uv_corners_refined.npy", raw_uv_f32)
        np.save(view_dir / "uv_corners_valid.npy", raw_valid_b)
        return diag

    U = np.load(u_path).astype(np.float32)
    V = np.load(v_path).astype(np.float32)
    M = np.load(m_path).astype(bool)
    refined_uv, refined_valid, refine_diag = refine_uv_at_corners(U, V, M, corners, refine_cfg)
    np.save(view_dir / "uv_corners_refined.npy", refined_uv.astype(np.float32))
    np.save(view_dir / "uv_corners_valid.npy", refined_valid.astype(bool))
    delta_stats = _save_uv_refine_delta_overlay(
        image=camera_image,
        corners_cam=corners,
        uv_raw=raw_uv_f32,
        uv_refined=refined_uv,
        valid_refined=refined_valid,
        out_path=view_dir / "uv_refine_delta.png",
    )
    payload = {
        "config": _to_json_safe(asdict(refine_cfg)),
        "summary": _to_json_safe(refine_diag.get("summary", {})),
        "per_corner": _to_json_safe(refine_diag.get("per_corner", [])),
        "delta_stats": _to_json_safe(delta_stats),
    }
    (view_dir / "uv_refine_diag.json").write_text(json.dumps(payload, indent=2))
    return payload


def _draw_simple_bar_plot(
    values: list[float],
    labels: list[str],
    out_path: Path,
    title: str,
) -> None:
    w, h = 900, 520
    m_l, m_r, m_t, m_b = 70, 30, 60, 80
    img = Image.new("RGB", (w, h), (20, 20, 24))
    draw = ImageDraw.Draw(img)
    draw.text((20, 18), title, fill=(235, 235, 235))

    finite_vals = [v for v in values if np.isfinite(v)]
    ymax = max(finite_vals) if finite_vals else 1.0
    if ymax <= 0:
        ymax = 1.0
    n = max(1, len(values))
    plot_w = w - m_l - m_r
    plot_h = h - m_t - m_b
    bar_w = max(12, int(plot_w / (n * 1.8)))
    gap = (plot_w - n * bar_w) / max(1, n + 1)
    draw.line((m_l, h - m_b, w - m_r, h - m_b), fill=(120, 120, 120), width=1)
    draw.line((m_l, m_t, m_l, h - m_b), fill=(120, 120, 120), width=1)

    for i, (v, lbl) in enumerate(zip(values, labels)):
        x0 = int(m_l + gap * (i + 1) + bar_w * i)
        x1 = x0 + bar_w
        if np.isfinite(v):
            bh = int(plot_h * (v / ymax))
            y0 = h - m_b - bh
            draw.rectangle((x0, y0, x1, h - m_b), fill=(74, 154, 255))
            draw.text((x0 - 4, y0 - 18), f"{v:.3f}", fill=(220, 220, 220))
        draw.text((x0 - 4, h - m_b + 8), lbl, fill=(220, 220, 220))
    img.save(out_path)


def _draw_group_bar_plot(
    categories: list[str],
    before_vals: list[float],
    after_vals: list[float],
    out_path: Path,
    title: str,
) -> None:
    w, h = 1000, 560
    m_l, m_r, m_t, m_b = 70, 30, 60, 110
    img = Image.new("RGB", (w, h), (20, 20, 24))
    draw = ImageDraw.Draw(img)
    draw.text((20, 18), title, fill=(235, 235, 235))

    vals = [v for v in (before_vals + after_vals) if np.isfinite(v)]
    ymax = max(vals) if vals else 1.0
    if ymax <= 0:
        ymax = 1.0
    n = max(1, len(categories))
    plot_w = w - m_l - m_r
    plot_h = h - m_t - m_b
    group_w = plot_w / n
    bw = max(6, int(group_w * 0.28))
    draw.line((m_l, h - m_b, w - m_r, h - m_b), fill=(120, 120, 120), width=1)
    draw.line((m_l, m_t, m_l, h - m_b), fill=(120, 120, 120), width=1)
    for i, cat in enumerate(categories):
        gx = m_l + int(i * group_w + group_w * 0.5)
        b = before_vals[i] if i < len(before_vals) else float("nan")
        a = after_vals[i] if i < len(after_vals) else float("nan")
        if np.isfinite(b):
            bh = int(plot_h * (b / ymax))
            draw.rectangle((gx - bw - 2, h - m_b - bh, gx - 2, h - m_b), fill=(255, 146, 76))
        if np.isfinite(a):
            ah = int(plot_h * (a / ymax))
            draw.rectangle((gx + 2, h - m_b - ah, gx + bw + 2, h - m_b), fill=(76, 175, 80))
        draw.text((gx - 24, h - m_b + 8), cat, fill=(220, 220, 220))
    draw.rectangle((w - 240, 16, w - 28, 52), fill=(0, 0, 0))
    draw.rectangle((w - 230, 22, w - 212, 36), fill=(255, 146, 76))
    draw.text((w - 206, 20), "before", fill=(220, 220, 220))
    draw.rectangle((w - 140, 22, w - 122, 36), fill=(76, 175, 80))
    draw.text((w - 116, 20), "after", fill=(220, 220, 220))
    img.save(out_path)


def _board_mask_from_corners(shape: tuple[int, int], corners: np.ndarray) -> np.ndarray:
    cv = _require_cv2()
    h, w = shape
    if corners.shape[0] < 3:
        return np.zeros((h, w), dtype=bool)
    hull = cv.convexHull(corners.reshape(-1, 1, 2).astype(np.float32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv.fillConvexPoly(mask, hull.astype(np.int32), 255)
    return mask > 0


def _sample_b_on_board(run_dir: Path, freq: float, board_mask: np.ndarray) -> float:
    b_path = run_dir / "phase" / _freq_tag(freq) / "B.npy"
    if not b_path.exists() or not np.any(board_mask):
        return 0.0
    B = np.load(b_path).astype(np.float32)
    vals = B[board_mask & np.isfinite(B)]
    if vals.size == 0:
        return 0.0
    return float(np.median(vals))


def _board_residual_stats(
    run_dir_vertical: Path,
    run_dir_horizontal: Path,
    board_mask: np.ndarray,
) -> dict[str, float]:
    def _load_residual(run_dir: Path) -> np.ndarray | None:
        p = run_dir / "unwrap" / "residual.npy"
        if not p.exists():
            return None
        try:
            return np.load(p).astype(np.float32)
        except Exception:
            return None

    rv = _load_residual(run_dir_vertical)
    rh = _load_residual(run_dir_horizontal)
    area_ratio = float(np.mean(board_mask)) if board_mask.size else 0.0

    def _p95_on_board(arr: np.ndarray | None) -> float:
        if arr is None or arr.shape != board_mask.shape:
            return float("nan")
        vals = arr[board_mask & np.isfinite(arr)]
        if vals.size == 0:
            return float("nan")
        return float(np.percentile(vals, 95))

    p95_v = _p95_on_board(rv)
    p95_h = _p95_on_board(rh)
    if np.isfinite(p95_v) and np.isfinite(p95_h):
        p95 = float(max(p95_v, p95_h))
    elif np.isfinite(p95_v):
        p95 = float(p95_v)
    elif np.isfinite(p95_h):
        p95 = float(p95_h)
    else:
        p95 = float("nan")

    gt_vals: list[np.ndarray] = []
    for arr in (rv, rh):
        if arr is None or arr.shape != board_mask.shape:
            continue
        vals = arr[board_mask & np.isfinite(arr)]
        if vals.size:
            gt_vals.append(vals)
    if gt_vals:
        stacked = np.concatenate(gt_vals)
        gt_1 = float(np.mean(np.abs(stacked) > 1.0))
    else:
        gt_1 = float("nan")

    return {
        "unwrap_residual_p95_board": p95,
        "unwrap_residual_vertical_p95_board": p95_v,
        "unwrap_residual_horizontal_p95_board": p95_h,
        "unwrap_residual_gt_1rad_pct_board": gt_1,
        "board_mask_area_ratio": area_ratio,
    }


def _build_view_hints(
    mismatch: bool,
    uv_meta: dict[str, Any],
    breakdown: dict[str, int],
    phase_quality: dict[str, Any],
) -> list[str]:
    hints: list[str] = []
    if mismatch:
        hints.append("PROJECTOR_RES_MISMATCH: configured size does not match actual projector surface size.")
    total_invalid = (
        breakdown.get("nan_uv", 0)
        + breakdown.get("mask_uv_false", 0)
        + breakdown.get("near_edge", 0)
        + breakdown.get("oob", 0)
    )
    if total_invalid > 0 and breakdown.get("mask_uv_false", 0) >= int(0.45 * total_invalid):
        hints.append(
            "Most corners are outside UV mask (mask_uv_false). Likely ROI/mask too tight or board partially outside projected area."
        )
    if total_invalid > 0 and breakdown.get("nan_uv", 0) >= int(0.45 * total_invalid):
        hints.append("Most corners have NaN UV. Check unwrap mask overlap, board illumination, and phase quality.")
    if breakdown.get("near_edge", 0) > 0 or breakdown.get("oob", 0) > 0:
        hints.append("Several corners are near projector edges/out-of-bounds. Keep board slightly away from projection borders.")
    if not bool(uv_meta.get("uv_gate_ok", False)):
        hints.extend([str(h) for h in uv_meta.get("uv_gate_hints", [])])
    pv = phase_quality.get("vertical", {}) or {}
    ph = phase_quality.get("horizontal", {}) or {}
    if float(pv.get("roi_valid_ratio", 0.0)) < 0.80 or float(ph.get("roi_valid_ratio", 0.0)) < 0.80:
        hints.append("ROI validity is low in one orientation. Increase exposure slightly or reduce mask strictness for calibration.")
    return sorted(set(hints))


def _view_quality_hint(reason: str | None) -> str | None:
    mapping = {
        "uv_valid_ratio_low": "UV validity too low. Move board fully inside projector coverage.",
        "u_span_too_small": "Increase horizontal board span in projector space.",
        "v_span_too_small": "Increase vertical board span in projector space.",
        "board_area_too_small": "Move board closer; board area in image is too small.",
        "tilt_too_low": "Tilt board more to improve conditioning.",
        "tilt_too_high": "Tilt board less; current pose is too extreme.",
    }
    if reason is None:
        return None
    return mapping.get(str(reason))


def _save_view_quality_overlay(
    image: np.ndarray,
    report: ViewQualityReport,
    out_path: Path,
) -> None:
    if image.ndim == 2:
        rgb = np.stack([image, image, image], axis=2).astype(np.uint8)
    else:
        rgb = image[:, :, :3].astype(np.uint8).copy()
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    status = "PASS" if bool(report.accepted) else "FAIL"
    color = (70, 220, 70) if bool(report.accepted) else (230, 80, 80)
    lines = [
        f"View quality: {status}",
        f"uv_valid_ratio={report.uv_valid_ratio:.3f}",
        f"u_span_px={report.u_span_px:.1f} v_span_px={report.v_span_px:.1f}",
        f"board_area_ratio={report.board_area_ratio:.4f}",
        f"tilt_angle_deg={report.tilt_angle_deg:.2f}",
        f"conditioning_score={report.conditioning_score:.3f}",
    ]
    if report.rejection_reason:
        lines.append(f"reason={report.rejection_reason}")
    box_w = 540
    box_h = 22 + 18 * len(lines)
    draw.rectangle((10, 10, 10 + box_w, 10 + box_h), fill=(0, 0, 0))
    draw.rectangle((10, 10, 10 + box_w, 10 + box_h), outline=color, width=2)
    y = 18
    for line in lines:
        draw.text((18, y), line, fill=(245, 245, 245))
        y += 18
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _build_view_diag_payload(
    *,
    view_id: str,
    checkerboard_found: bool,
    corners_total: int,
    accept: bool,
    reject_reasons: list[str],
    thresholds: dict[str, Any],
    measured: dict[str, Any],
    corner_breakdown: dict[str, Any],
    corner_sampling: dict[str, Any],
    uv: dict[str, Any],
    phase_quality: dict[str, Any],
    projector_resolution: dict[str, Any],
    diagnostics: ProjectorViewDiagnostics | None,
    view_quality: ViewQualityReport | None,
    hints: list[str],
) -> dict[str, Any]:
    diag_dict = diagnostics.to_dict() if diagnostics is not None else {}
    measured_out = dict(measured)
    measured_out.setdefault("B_median_board_available", measured_out.get("B_median_board") is not None)
    return {
        "schema_version": SCHEMA_VERSION,
        "view_id": view_id,
        "status": "PASS" if bool(accept) else "FAIL",
        "accept": bool(accept),
        "checkerboard": {"found": bool(checkerboard_found), "corners_total": int(corners_total)},
        "thresholds": thresholds,
        "measured": measured_out,
        "reject_reasons": list(reject_reasons),
        "hints": sorted(set(hints)),
        # Backward-compatible blocks retained for existing UI/tools.
        "uv": uv,
        "corner_sampling": corner_sampling,
        "corner_validity_breakdown": corner_breakdown,
        "phase_quality": phase_quality,
        "projector_resolution": projector_resolution,
        "diagnostics": diag_dict,
        "view_quality": view_quality.to_dict() if view_quality is not None else {},
    }


def _detection_quality_payload(
    *,
    detection: CheckerboardDetection,
    image_shape_hw: tuple[int, int],
    rejection_reason: str | None = None,
    rejection_hint: str | None = None,
    exception: str | None = None,
) -> dict[str, Any]:
    diag = dict(detection.diagnostics or {})
    payload = {
        "detection_mode": str(detection.detection_mode),
        "aruco_markers_detected": int(diag.get("aruco_markers_detected", 0)),
        "aruco_ids": [int(v) for v in (diag.get("aruco_ids", []) or [])],
        "charuco_corners_detected": int(diag.get("charuco_corners_detected", detection.corner_count)),
        "image_size": [int(detection.image_size[0]), int(detection.image_size[1])],
        "board_visible_fraction": _optional_finite_float(diag.get("board_visible_fraction")),
        "rejection_reason": rejection_reason,
        "rejection_hint": rejection_hint,
        "exception": exception or diag.get("exception"),
        "diagnostics": diag,
    }
    if payload["image_size"] == [0, 0]:
        payload["image_size"] = [int(image_shape_hw[1]), int(image_shape_hw[0])]
    return _to_json_safe(payload)


def _is_complete_view_folder(view_dir: Path) -> bool:
    return (
        (view_dir / "camera.png").exists()
        and (view_dir / "camera_corners.json").exists()
        and ((view_dir / "view_diag.json").exists() or (view_dir / "correspondences.json").exists())
    )


def _session_view_folders(session_dir: Path, include_partial: bool = False) -> list[Path]:
    views_dir = session_dir / "views"
    if not views_dir.exists():
        return []
    out = sorted([p for p in views_dir.glob("view_*") if p.is_dir()], key=lambda p: p.name)
    if include_partial:
        return out
    return [p for p in out if _is_complete_view_folder(p)]


def _accepted_view_ids_from_session(session: dict[str, Any]) -> set[str]:
    accepted: set[str] = set()
    for v in session.get("views", []) or []:
        if not isinstance(v, dict):
            continue
        vid = str(v.get("view_id", "")).strip()
        if not vid:
            continue
        accepted.add(vid)
    return accepted


def _coverage_from_session(
    session_dir: Path,
    session: dict[str, Any],
    grid_size: tuple[int, int] = (16, 16),
) -> dict[str, Any]:
    proj_cfg = (session.get("config", {}) or {}).get("projector", {}) or {}
    pw = max(1, int(proj_cfg.get("width", 1024)))
    ph = max(1, int(proj_cfg.get("height", 768)))
    bins_x = max(1, int(grid_size[0]))
    bins_y = max(1, int(grid_size[1]))
    grid = np.zeros((bins_y, bins_x), dtype=np.int32)
    view_ids: list[str] = []

    accepted_ids = _accepted_view_ids_from_session(session)
    for vdir in _session_view_folders(session_dir):
        vid = vdir.name
        if vid not in accepted_ids:
            continue
        corr_path = vdir / "correspondences.json"
        if not corr_path.exists():
            continue
        try:
            corr = json.loads(corr_path.read_text())
        except Exception:
            continue
        proj_pts = np.asarray(corr.get("projector_corners_px", []), dtype=np.float64).reshape(-1, 2)
        valid = np.asarray(corr.get("valid_mask", []), dtype=bool).reshape(-1)
        if proj_pts.shape[0] != valid.shape[0]:
            continue
        pts = proj_pts[valid]
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if pts.shape[0] < 3:
            continue
        cv = None
        try:
            cv = _require_cv2()
            hull = cv.convexHull(pts.astype(np.float32).reshape(-1, 1, 2))
            hull_pts = hull.reshape(-1, 2).astype(np.float64)
        except Exception:
            hull_pts = pts
        if hull_pts.shape[0] < 3:
            continue

        u = np.clip(hull_pts[:, 0], 0.0, float(pw - 1))
        vv = np.clip(hull_pts[:, 1], 0.0, float(ph - 1))
        bx = np.clip((u / float(pw)) * bins_x, 0.0, bins_x - 1e-6).astype(np.int32)
        by = np.clip((vv / float(ph)) * bins_y, 0.0, bins_y - 1e-6).astype(np.int32)
        corners = np.stack([bx, by], axis=1)
        min_x = int(np.min(corners[:, 0]))
        max_x = int(np.max(corners[:, 0]))
        min_y = int(np.min(corners[:, 1]))
        max_y = int(np.max(corners[:, 1]))
        poly = np.stack([bx, by], axis=1).astype(np.int32)
        if poly.shape[0] >= 3 and cv is not None:
            for gy in range(min_y, max_y + 1):
                for gx in range(min_x, max_x + 1):
                    inside = cv.pointPolygonTest(poly.astype(np.float32), (float(gx), float(gy)), False)
                    if inside >= 0:
                        grid[gy, gx] = 1
        else:
            grid[by, bx] = 1
        view_ids.append(vid)

    covered = int(np.count_nonzero(grid))
    total = int(grid.size)
    return {
        "schema_version": SCHEMA_VERSION,
        "grid_size": [bins_x, bins_y],
        "projector_size": [pw, ph],
        "bins_covered_count": covered,
        "bins_total": total,
        "coverage_ratio": float(covered / max(total, 1)),
        "view_ids_used": sorted(view_ids),
        "grid": grid.astype(int).tolist(),
    }


def _build_session_report(session_dir: Path, session: dict[str, Any]) -> dict[str, Any]:
    per_view: list[dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()
    residual_vals: list[float] = []
    valid_ratio_vals: list[float] = []
    u_span_vals: list[float] = []
    v_span_vals: list[float] = []
    accepted_ids = _accepted_view_ids_from_session(session)

    for vdir in _session_view_folders(session_dir):
        diag_path = vdir / "view_diag.json"
        if not diag_path.exists():
            continue
        try:
            diag = json.loads(diag_path.read_text())
        except Exception:
            continue
        measured = diag.get("measured", {}) or {}
        reject_reasons = [str(r) for r in (diag.get("reject_reasons", []) or [])]
        for r in reject_reasons:
            if r:
                reject_counts[r] += 1
        vid = vdir.name
        status = "accepted" if vid in accepted_ids else "rejected"

        def _append_if_finite(values: list[float], key: str) -> None:
            raw = measured.get(key)
            if raw is None:
                return
            v = _safe_float_or_nan(raw)
            if np.isfinite(v):
                values.append(float(v))

        _append_if_finite(residual_vals, "residual_p95_board")
        _append_if_finite(valid_ratio_vals, "valid_corner_ratio")
        _append_if_finite(u_span_vals, "u_span_px")
        _append_if_finite(v_span_vals, "v_span_px")
        per_view.append(
            {
                "view_id": vid,
                "status": status,
                "reject_reasons": reject_reasons,
                "measured": measured,
            }
        )

    def _distribution(values: list[float]) -> dict[str, float | int]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return {"count": 0, "p50": float("nan"), "p95": float("nan"), "min": float("nan"), "max": float("nan")}
        return {
            "count": int(arr.size),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": str(session.get("session_id", session_dir.name)),
        "generated_at": datetime.now().isoformat(),
        "views_total": int(len(per_view)),
        "views_accepted": int(len(accepted_ids)),
        "views_rejected": int(max(0, len(per_view) - len(accepted_ids))),
        "reject_reason_counts": dict(sorted(reject_counts.items(), key=lambda kv: kv[0])),
        "distributions": {
            "residual_p95_board": _distribution(residual_vals),
            "valid_corner_ratio": _distribution(valid_ratio_vals),
            "u_span_px": _distribution(u_span_vals),
            "v_span_px": _distribution(v_span_vals),
        },
        "per_view": per_view,
    }


def finalize_session_reports(session_dir: Path, cfg: dict) -> dict[str, Any]:
    session = _load_session(session_dir)
    coverage_cfg = (cfg.get("projector_calibration", {}) or {}).get("coverage", {}) or {}
    gx = int(coverage_cfg.get("grid_size_x", 16))
    gy = int(coverage_cfg.get("grid_size_y", 16))
    coverage = _coverage_from_session(session_dir, session, grid_size=(gx, gy))
    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = results_dir / "coverage.json"
    coverage_png_path = results_dir / "coverage.png"
    coverage_path.write_text(json.dumps(_to_json_safe(coverage), indent=2))
    _save_coverage_png(coverage, coverage_png_path)

    session_report = _build_session_report(session_dir, session)
    session_report["coverage"] = {
        "coverage_ratio": float(coverage.get("coverage_ratio", 0.0)),
        "bins_covered_count": int(coverage.get("bins_covered_count", 0)),
        "bins_total": int(coverage.get("bins_total", 0)),
        "coverage_json": "results/coverage.json",
        "coverage_png": "results/coverage.png",
    }
    session_report_path = results_dir / "session_report.json"
    session_report_path.write_text(json.dumps(_to_json_safe(session_report), indent=2))
    return {
        "coverage": coverage,
        "coverage_json_path": str(coverage_path),
        "coverage_png_path": str(coverage_png_path),
        "session_report": session_report,
        "session_report_path": str(session_report_path),
    }


def capture_view(session_dir: Path, controller, cfg: dict) -> dict[str, Any]:
    """
    Capture one projector calibration view:
    camera image + checkerboard + UV scan + sampled correspondences.
    """
    _require_cv2()
    session = _load_session(session_dir)
    proj_cfg0 = ((session.get("config", {}) or {}).get("projector", {}) or {})
    proj_size0 = (int(proj_cfg0.get("width", 1024)), int(proj_cfg0.get("height", 768)))
    view_id = next_view_id(session_dir, proj_size0)
    view_dir = session_dir / "views" / view_id
    view_dir.mkdir(parents=True, exist_ok=False)

    # Always use session-stored projector calibration settings so captures are
    # consistent across retries and independent of later global config edits.
    session_pcal = (session.get("config", {}) or {})
    cfg_session = _deep_update(cfg, {"projector_calibration": session_pcal})
    pcal = cfg_session.get("projector_calibration", {}) or {}
    chk = pcal.get("checkerboard", {}) or {}
    corners_x = int(chk.get("corners_x", 8))
    corners_y = int(chk.get("corners_y", 6))
    cap_cfg = pcal.get("capture", {}) or {}
    proj_cfg = pcal.get("projector", {}) or {}
    min_corner_valid_ratio = float(cap_cfg.get("min_corner_valid_ratio", 0.90))
    max_retries = int(cap_cfg.get("max_retries", 2))
    low_b_threshold = float(cap_cfg.get("low_b_median_threshold", 8.0))
    retry_exposure_bump = float(cap_cfg.get("retry_exposure_bump_factor", 1.25))
    strict_resolution_match = bool(proj_cfg.get("strict_resolution_match", False))
    calib_cfg = cfg_session.get("calibration", {}) or {}
    residual_p95_board_threshold = float(calib_cfg.get("residual_p95_board_threshold", 1.2))
    residual_gt_1rad_pct_board_max = float(calib_cfg.get("residual_gt_1rad_pct_board_max", 0.15))
    uv_sampling_cfg = (pcal.get("uv_sampling", {}) or {})
    thresholds_used = _calibration_thresholds_used(
        min_corner_valid_ratio=min_corner_valid_ratio,
        residual_p95_board_threshold=residual_p95_board_threshold,
        residual_gt_1rad_pct_board_max=residual_gt_1rad_pct_board_max,
        strict_resolution_match=strict_resolution_match,
    )
    thresholds_used["uv_sampling"] = {
        "method": str(uv_sampling_cfg.get("method", "median_patch")),
        "patch_radius": int(uv_sampling_cfg.get("patch_radius", 2)),
        "min_valid_points": int(uv_sampling_cfg.get("min_valid_points", 8)),
    }

    summary: dict[str, Any] = {
        "view_id": view_id,
        "created_at": datetime.now().isoformat(),
        "status": "rejected",
        "accept": False,
        "reason": None,
        "reject_reasons": [],
        "hints": [],
        "checkerboard_found": False,
        "valid_corner_ratio": 0.0,
        "valid_corners": 0,
        "total_corners": int(corners_x * corners_y),
        "paths": {
            "camera": str(Path("views") / view_id / "camera.png"),
            "camera_overlay": str(Path("views") / view_id / "camera_overlay.png"),
            "charuco_overlay": str(Path("views") / view_id / "view_charuco_overlay.png"),
            "corners": str(Path("views") / view_id / "camera_corners.json"),
            "correspondences": str(Path("views") / view_id / "correspondences.json"),
            "overlay": str(Path("views") / view_id / "overlay.png"),
            "corner_validity_overlay": str(Path("views") / view_id / "corner_validity_overlay.png"),
            "uv_meta": str(Path("views") / view_id / "uv" / "uv_meta.json"),
            "view_diag": str(Path("views") / view_id / "view_diag.json"),
        },
    }

    frame = _capture_checkerboard_frame_with_projector(controller, cfg_session)
    save_image(view_dir / "camera.png", frame)
    calib_cfg_local = cfg_session.get("calibration", {}) or {}
    board_type = str(calib_cfg_local.get("board_type", "checkerboard"))
    charuco_cfg = (calib_cfg_local.get("charuco", {}) or {})
    detection_exception: str | None = None
    try:
        det, overlay = detect_checkerboard(
            frame,
            cols=corners_x,
            rows=corners_y,
            refine_subpix=True,
            board_type=board_type,
            charuco_cfg=charuco_cfg,
        )
    except Exception as exc:
        detection_exception = str(exc)
        _log.exception("Projector calibration detection exception for %s: %s", view_id, exc)
        det = CheckerboardDetection(
            found=False,
            corners_px=[],
            image_size=(int(frame.shape[1]), int(frame.shape[0])),
            corner_count=0,
            method="detection_exception",
            detection_mode=str(board_type).strip().lower(),
            rejection_reason="detection_exception",
            rejection_hint=detection_exception,
            diagnostics={"exception": detection_exception},
        )
        overlay = frame.copy()
    save_image(view_dir / "camera_overlay.png", overlay)
    if str(board_type).strip().lower() == "charuco":
        save_image(view_dir / "view_charuco_overlay.png", overlay)
    summary["checkerboard_found"] = bool(det.found)

    corners_payload = det.to_dict()
    corners_payload["checkerboard"] = {"corners_x": corners_x, "corners_y": corners_y}
    (view_dir / "camera_corners.json").write_text(json.dumps(corners_payload, indent=2))

    if not det.found:
        reject_reason = str(det.rejection_reason or "checkerboard_not_found")
        reject_hint = str(det.rejection_hint or "Ensure full checkerboard is visible and in focus.")
        summary["reason"] = reject_reason
        summary["reject_reasons"] = [reject_reason]
        summary["hints"] = [reject_hint] if reject_hint else []
        (view_dir / "correspondences.json").write_text(
            json.dumps({"valid_ratio": 0.0, "error": reject_reason}, indent=2)
        )
        diag_obj, _ = evaluate_projector_view(
            corners_found=False,
            expected_corner_count=int(corners_x * corners_y),
            corners_cam=np.zeros((0, 2), dtype=np.float32),
            valid_corner_mask=np.zeros((0,), dtype=bool),
            corner_reasons=[],
            image_shape_hw=frame.shape[:2],
            corners_x=corners_x,
            corners_y=corners_y,
            square_size_m=float(chk.get("square_size_m", 0.01)),
            camera_matrix=None,
            dist_coeffs=None,
            recent_accepted_poses=recent_accepted_poses(session, 3),
            b_median_board=0.0,
            unwrap_residual_p95=float("inf"),
            unwrap_residual_vertical_p95_board=float("inf"),
            unwrap_residual_horizontal_p95_board=float("inf"),
            unwrap_residual_gt_1rad_pct_board=1.0,
            board_mask_area_ratio=0.0,
            clipping_detected=False,
            require_full_corner_count=(str(board_type).strip().lower() != "charuco"),
        )
        diag_obj.accept = False
        diag_obj.reject_reasons = [reject_reason]
        diag_obj.hints = [reject_hint] if reject_hint else []
        vq_not_found = ViewQualityReport(
            n_camera_corners=0,
            n_uv_valid=0,
            uv_valid_ratio=0.0,
            u_span_px=0.0,
            v_span_px=0.0,
            board_area_ratio=0.0,
            tilt_angle_deg=float("nan"),
            reproj_estimate_proxy=float("nan"),
            conditioning_score=0.0,
            accepted=False,
            rejection_reason=reject_reason,
        )
        measured = {
            "valid_corner_ratio": 0.0,
            "board_area_ratio": _optional_finite_float(diag_obj.hull_area_ratio),
            "u_span_px": None,
            "v_span_px": None,
            "B_median_board": None,
            "B_median_board_available": False,
            "residual_p95_global": None,
            "residual_p95_board": None,
            "residual_p95_vertical_board": None,
            "residual_p95_horizontal_board": None,
            "residual_gt_1rad_pct_board": None,
            "board_mask_area_ratio": _optional_finite_float(diag_obj.board_mask_area_ratio),
            "uv_sampling_method": str((thresholds_used.get("uv_sampling", {}) or {}).get("method", "median_patch")),
            "clipping_detected": bool(diag_obj.clipping_detected),
        }
        diag = _build_view_diag_payload(
            view_id=view_id,
            checkerboard_found=False,
            corners_total=int(corners_x * corners_y),
            accept=False,
            reject_reasons=list(diag_obj.reject_reasons),
            thresholds=thresholds_used,
            measured=measured,
            corner_breakdown={"ok": 0, "nan_uv": 0, "mask_uv_false": 0, "oob": 0, "near_edge": 0},
            corner_sampling={
                "uv_sampling_method": str((thresholds_used.get("uv_sampling", {}) or {}).get("method", "median_patch")),
                "patch_radius": int((thresholds_used.get("uv_sampling", {}) or {}).get("patch_radius", 2)),
                "min_valid_points": int((thresholds_used.get("uv_sampling", {}) or {}).get("min_valid_points", 8)),
            },
            uv={},
            phase_quality={},
            projector_resolution={},
            diagnostics=diag_obj,
            view_quality=vq_not_found,
            hints=list(diag_obj.hints),
        )
        (view_dir / "view_diag.json").write_text(json.dumps(diag, indent=2))
        (view_dir / "view_quality.json").write_text(
            json.dumps(
                _detection_quality_payload(
                    detection=det,
                    image_shape_hw=frame.shape[:2],
                    rejection_reason=reject_reason,
                    rejection_hint=reject_hint,
                    exception=detection_exception,
                )
                | {"view_quality": _to_json_safe(vq_not_found.to_dict())},
                indent=2,
            )
        )
        _save_view_quality_overlay(frame, vq_not_found, view_dir / "view_quality_overlay.png")
        add_view_if_valid(
            session_dir=session_dir,
            view_data=summary,
            diagnostics=diag_obj,
            projector_points=None,
            projector_size=proj_size0,
            pose_entry=None,
        )
        _set_projector_calibration_light(controller, cfg_session, enabled=True)
        return summary

    corners = np.asarray(det.corners_px, dtype=np.float32).reshape(-1, 2)
    board_mask = _board_mask_from_corners(frame.shape[:2], corners)
    current_exposure = int((pcal.get("camera", {}) or {}).get("exposure_us", 2000))
    exposure_max = int((cfg_session.get("normalise", {}) or {}).get("exposure_max_extended_us", 12000))
    base_mask_policy = _calibration_uv_policy(cfg_session)
    cam_intr_path = _find_camera_intrinsics_latest(cfg_session)
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None
    if cam_intr_path is not None:
        try:
            camera_matrix, dist_coeffs = _load_camera_intrinsics(cam_intr_path)
        except Exception:
            camera_matrix, dist_coeffs = None, None
    pose_history = recent_accepted_poses(session, 3)

    best_attempt: dict[str, Any] | None = None
    exposure_retry_used = False
    mask_retry_used = False
    final_reason = "uv_scan_failed"
    for attempt in range(max_retries + 1):
        overrides: dict[str, Any] = {}
        if attempt > 0:
            overrides = {"projector_calibration": {"camera": {"exposure_us": current_exposure}}}
            cap_retry = overrides.setdefault("projector_calibration", {}).setdefault("capture", {})
            base_settle = int(cap_cfg.get("settle_ms", 150))
            base_flush = int(cap_cfg.get("flush_frames", 1))
            cap_retry["settle_ms"] = max(base_settle, 180)
            cap_retry["flush_frames"] = max(base_flush, 2)
            base_uv_sampling = (pcal.get("uv_sampling", {}) or {})
            uv_sampling_retry = overrides.setdefault("projector_calibration", {}).setdefault("uv_sampling", {})
            uv_sampling_retry["method"] = str(base_uv_sampling.get("method", "median_patch"))
            uv_sampling_retry["patch_radius"] = max(int(base_uv_sampling.get("patch_radius", 2)), 3)
            uv_sampling_retry["min_valid_points"] = max(6, int(base_uv_sampling.get("min_valid_points", 8)) - 2)
            if mask_retry_used:
                looser = dict(base_mask_policy)
                looser["erosion_radius_px"] = 0
                looser["b_thresh_unwrap"] = max(1.0, float(base_mask_policy.get("b_thresh_unwrap", 5.0)) - 1.0)
                overrides.setdefault("projector_calibration", {})["uv_mask_policy"] = {
                    "use_roi": False,
                    "unwrap_mask": {
                        "closing_radius_px": int(looser.get("closing_radius_px", 3)),
                        "erosion_radius_px": int(looser.get("erosion_radius_px", 0)),
                        "keep_largest_component": bool(looser.get("keep_largest_component", True)),
                        "min_area_px": int(looser.get("min_area_px", 2000)),
                        "b_thresh_unwrap": float(looser.get("b_thresh_unwrap", 4.0)),
                        "a_min_unwrap": float(looser.get("a_min_unwrap", 10.0)),
                    },
                }

        attempt_uv_dir = view_dir / ("uv" if attempt == 0 else f"uv_retry_{attempt}")
        gamma_cfg = (cfg_session.get("calibration", {}) or {}).get("gamma", {}) or {}
        gamma_enabled = bool(gamma_cfg.get("enabled", False))
        generator = getattr(controller, "generator", None)
        gamma_prev_lut: np.ndarray | None = None
        gamma_prev_enabled: bool = True
        gamma_applied = False
        try:
            if gamma_enabled and generator is not None and hasattr(generator, "set_gamma_lut"):
                gamma_prev_lut = getattr(generator, "_gamma_lut", None)
                gamma_prev_enabled = bool(getattr(generator, "_gamma_enabled", True))
                lut = load_gamma_lut(session_dir)
                if lut is not None:
                    generator.set_gamma_lut(lut, enabled=True)
                    gamma_applied = True
            u, v, mask_uv, uv_meta, run_info = run_uv_scan(
                controller,
                cfg_session,
                attempt_uv_dir,
                overrides=overrides,
            )
        except Exception as exc:
            final_reason = f"uv_scan_failed: {exc}"
            if attempt >= max_retries:
                break
            continue
        finally:
            if gamma_applied and generator is not None and hasattr(generator, "set_gamma_lut"):
                generator.set_gamma_lut(gamma_prev_lut, enabled=gamma_prev_enabled)

        proj_res = uv_meta.get("projector_resolution", {}) or {}
        pattern_surface = proj_res.get("pattern_surface")
        sampling_size = None
        if isinstance(pattern_surface, (list, tuple)) and len(pattern_surface) == 2:
            sampling_size = (int(pattern_surface[0]), int(pattern_surface[1]))
        cam_pts, proj_pts, valid, reasons, finite_counts, sampling_meta = _sample_uv(
            u,
            v,
            mask_uv,
            corners,
            _deep_update(cfg_session, overrides),
            projector_size=sampling_size,
        )
        projector_eval_size = (
            sampling_size
            if sampling_size is not None
            else (int(proj_cfg.get("width", 1024)), int(proj_cfg.get("height", 768)))
        )
        view_quality_cfg = (cfg_session.get("calibration", {}) or {})
        vq_report = evaluate_view_quality(
            camera_corners=cam_pts,
            uv_corners=proj_pts,
            mask_uv=mask_uv,
            projector_size=projector_eval_size,
            config=view_quality_cfg,
            image_shape_hw=frame.shape[:2],
        )
        valid_ratio = float(np.count_nonzero(valid) / max(len(valid), 1))
        counts = Counter(reasons)
        breakdown = {
            "ok": int(counts.get("ok", 0)),
            "nan_uv": int(counts.get("nan_uv", 0)),
            "mask_uv_false": int(counts.get("mask_uv_false", 0)),
            "oob": int(counts.get("oob", 0)),
            "near_edge": int(counts.get("near_edge", 0)),
            "finite_samples_median": float(np.median(finite_counts)) if finite_counts else 0.0,
        }

        v_run_dir = Path(cfg.get("storage", {}).get("run_root", "data/runs")) / run_info["vertical_run_id"]
        h_run_dir = Path(cfg.get("storage", {}).get("run_root", "data/runs")) / run_info["horizontal_run_id"]
        f_high = float(max(_make_scan_params(_deep_update(cfg, overrides), "vertical").get_frequencies()))
        b_board_v = _sample_b_on_board(v_run_dir, f_high, board_mask)
        b_board_h = _sample_b_on_board(h_run_dir, f_high, board_mask)
        b_board = float(0.5 * (b_board_v + b_board_h))
        b_path_v = v_run_dir / "phase" / _freq_tag(f_high) / "B.npy"
        b_path_h = h_run_dir / "phase" / _freq_tag(f_high) / "B.npy"
        b_available = bool(np.any(board_mask) and (b_path_v.exists() or b_path_h.exists()))

        phase_quality = uv_meta.get("phase_quality", {}) or {}
        proj_res = uv_meta.get("projector_resolution", {}) or {}
        mismatch = bool(proj_res.get("mismatch", False))

        clipped_any_v = _load_phase_artifacts(v_run_dir, f_high).get("clipped_any")
        clipped_any_h = _load_phase_artifacts(h_run_dir, f_high).get("clipped_any")
        clipping_detected = False
        if clipped_any_v is not None and clipped_any_h is not None and board_mask.shape == clipped_any_v.shape == clipped_any_h.shape:
            clipping_detected = bool(np.any(clipped_any_v[board_mask]) or np.any(clipped_any_h[board_mask]))
        elif clipped_any_v is not None and board_mask.shape == clipped_any_v.shape:
            clipping_detected = bool(np.any(clipped_any_v[board_mask]))

        rv_global = _safe_float_or_nan((phase_quality.get("vertical", {}) or {}).get("residual_p95"))
        rh_global = _safe_float_or_nan((phase_quality.get("horizontal", {}) or {}).get("residual_p95"))
        if np.isfinite(rv_global) and np.isfinite(rh_global):
            unwrap_residual_p95_global = float(max(rv_global, rh_global))
        elif np.isfinite(rv_global):
            unwrap_residual_p95_global = float(rv_global)
        elif np.isfinite(rh_global):
            unwrap_residual_p95_global = float(rh_global)
        else:
            unwrap_residual_p95_global = float("nan")
        board_residual_stats = _board_residual_stats(v_run_dir, h_run_dir, board_mask)
        expected_corner_count_for_gate = int(corners.shape[0]) if str(board_type).strip().lower() == "charuco" else int(corners_x * corners_y)
        diag_obj, pose_entry = evaluate_projector_view(
            corners_found=True,
            expected_corner_count=expected_corner_count_for_gate,
            corners_cam=cam_pts.astype(np.float32),
            valid_corner_mask=valid.astype(bool),
            corner_reasons=reasons,
            image_shape_hw=frame.shape[:2],
            corners_x=corners_x,
            corners_y=corners_y,
            square_size_m=float(chk.get("square_size_m", 0.01)),
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            recent_accepted_poses=pose_history,
            b_median_board=float(b_board),
            unwrap_residual_p95=float(board_residual_stats.get("unwrap_residual_p95_board", float("nan"))),
            unwrap_residual_vertical_p95_board=float(board_residual_stats.get("unwrap_residual_vertical_p95_board", float("nan"))),
            unwrap_residual_horizontal_p95_board=float(board_residual_stats.get("unwrap_residual_horizontal_p95_board", float("nan"))),
            unwrap_residual_gt_1rad_pct_board=float(board_residual_stats.get("unwrap_residual_gt_1rad_pct_board", float("nan"))),
            board_mask_area_ratio=float(board_residual_stats.get("board_mask_area_ratio", 0.0)),
            clipping_detected=bool(clipping_detected),
            min_valid_corner_ratio=min_corner_valid_ratio,
            max_unwrap_residual_p95=residual_p95_board_threshold,
            max_unwrap_gt_1rad_pct=residual_gt_1rad_pct_board_max,
            require_full_corner_count=(str(board_type).strip().lower() != "charuco"),
        )
        if bool(view_quality_cfg.get("auto_reject", True)) and not bool(vq_report.accepted):
            diag_obj.accept = False
            tag = f"view_quality:{vq_report.rejection_reason or 'rejected'}"
            if tag not in diag_obj.reject_reasons:
                diag_obj.reject_reasons.append(tag)
            hint = _view_quality_hint(vq_report.rejection_reason)
            if hint:
                diag_obj.hints.append(hint)
        # Keep strict resolution mismatch as a hard reject when requested.
        if mismatch and strict_resolution_match:
            diag_obj.accept = False
            if "PROJECTOR_RES_MISMATCH" not in diag_obj.reject_reasons:
                diag_obj.reject_reasons.append("PROJECTOR_RES_MISMATCH")
            diag_obj.hints.append("PROJECTOR_RES_MISMATCH: configured size does not match actual projector surface size.")
        # Preserve UV gate diagnostics in reject reasons/hints.
        if not bool(uv_meta.get("uv_gate_ok", False)):
            diag_obj.accept = False
            if "uv_gate_failed" not in diag_obj.reject_reasons:
                diag_obj.reject_reasons.append("uv_gate_failed")
            for hint in uv_meta.get("uv_gate_hints", []) or []:
                diag_obj.hints.append(str(hint))
        diag_obj.hints = sorted(set(diag_obj.hints))
        u_span_px, v_span_px = _projector_uv_span(proj_pts, valid)
        thresholds_attempt = _to_json_safe(dict(thresholds_used))
        thresholds_attempt["uv_sampling"] = {
            "method": str(sampling_meta.get("uv_sampling_method", "median_patch")),
            "patch_radius": int(sampling_meta.get("patch_radius", 2)),
            "min_valid_points": int(sampling_meta.get("min_valid_points", 8)),
        }
        thresholds_attempt["view_quality"] = {
            "auto_reject": bool(view_quality_cfg.get("auto_reject", True)),
            "min_uv_valid_ratio": float(view_quality_cfg.get("min_uv_valid_ratio", 0.85)),
            "min_u_span_px": float(view_quality_cfg.get("min_u_span_px", 120.0)),
            "min_v_span_px": float(view_quality_cfg.get("min_v_span_px", 80.0)),
            "min_board_area_ratio": float(view_quality_cfg.get("min_board_area_ratio", 0.02)),
            "min_tilt_deg": float(view_quality_cfg.get("min_tilt_deg", 8.0)),
            "max_tilt_deg": float(view_quality_cfg.get("max_tilt_deg", 75.0)),
        }
        measured = {
            "valid_corner_ratio": float(diag_obj.valid_corner_ratio),
            "board_area_ratio": _optional_finite_float(diag_obj.hull_area_ratio),
            "u_span_px": _optional_finite_float(u_span_px),
            "v_span_px": _optional_finite_float(v_span_px),
            "view_quality_uv_valid_ratio": _optional_finite_float(vq_report.uv_valid_ratio),
            "view_quality_u_span_px": _optional_finite_float(vq_report.u_span_px),
            "view_quality_v_span_px": _optional_finite_float(vq_report.v_span_px),
            "view_quality_board_area_ratio": _optional_finite_float(vq_report.board_area_ratio),
            "view_quality_tilt_deg": _optional_finite_float(vq_report.tilt_angle_deg),
            "view_quality_conditioning_score": _optional_finite_float(vq_report.conditioning_score),
            "B_median_board": _optional_finite_float(b_board) if b_available else None,
            "B_median_board_available": bool(b_available and (_optional_finite_float(b_board) is not None)),
            "residual_p95_global": _optional_finite_float(unwrap_residual_p95_global),
            "residual_p95_board": _optional_finite_float(
                board_residual_stats.get("unwrap_residual_p95_board", float("nan"))
            ),
            "residual_p95_vertical_board": _optional_finite_float(
                board_residual_stats.get("unwrap_residual_vertical_p95_board", float("nan"))
            ),
            "residual_p95_horizontal_board": _optional_finite_float(
                board_residual_stats.get("unwrap_residual_horizontal_p95_board", float("nan"))
            ),
            "residual_gt_1rad_pct_board": _optional_finite_float(
                board_residual_stats.get("unwrap_residual_gt_1rad_pct_board", float("nan"))
            ),
            "board_mask_area_ratio": _optional_finite_float(
                board_residual_stats.get("board_mask_area_ratio", float("nan"))
            ),
            "uv_sampling_method": str(sampling_meta.get("uv_sampling_method", "median_patch")),
            "clipping_detected": bool(clipping_detected),
        }
        diag = _build_view_diag_payload(
            view_id=view_id,
            checkerboard_found=True,
            corners_total=int(corners.shape[0]),
            accept=bool(diag_obj.accept),
            reject_reasons=list(diag_obj.reject_reasons),
            thresholds=thresholds_attempt,
            measured=measured,
            corner_breakdown=breakdown,
            corner_sampling=sampling_meta,
            uv={
                "uv_gate_ok": bool(uv_meta.get("uv_gate_ok", False)),
                "valid_ratio": float(uv_meta.get("valid_ratio", 0.0)),
                "u_range": float(uv_meta.get("u_range", 0.0)),
                "v_range": float(uv_meta.get("v_range", 0.0)),
                "u_edge_pct": float(uv_meta.get("u_edge_pct", 0.0)),
                "v_edge_pct": float(uv_meta.get("v_edge_pct", 0.0)),
                "u_zero_pct": float(uv_meta.get("u_zero_pct", 0.0)),
                "v_zero_pct": float(uv_meta.get("v_zero_pct", 0.0)),
            },
            phase_quality=phase_quality,
            projector_resolution={
                "configured": proj_res.get("configured", []),
                "pattern_surface": proj_res.get("pattern_surface", []),
                "mismatch": mismatch,
            },
            diagnostics=diag_obj,
            view_quality=vq_report,
            hints=list(diag_obj.hints),
        )

        diag_name = "view_diag.json" if attempt == 0 else f"view_diag_retry_{attempt}.json"
        (view_dir / diag_name).write_text(json.dumps(_to_json_safe(diag), indent=2))
        vq_name = "view_quality.json" if attempt == 0 else f"view_quality_retry_{attempt}.json"
        (view_dir / vq_name).write_text(
            json.dumps(
                _detection_quality_payload(
                    detection=det,
                    image_shape_hw=frame.shape[:2],
                    rejection_reason=(str(diag_obj.reject_reasons[0]) if diag_obj.reject_reasons else None),
                    rejection_hint=(str(diag_obj.hints[0]) if diag_obj.hints else None),
                    exception=detection_exception,
                )
                | {"view_quality": _to_json_safe(vq_report.to_dict())},
                indent=2,
            )
        )
        ov_name = "view_quality_overlay.png" if attempt == 0 else f"view_quality_overlay_retry_{attempt}.png"
        _save_view_quality_overlay(frame, vq_report, view_dir / ov_name)

        corr = {
            "schema_version": SCHEMA_VERSION,
            "camera_corners_px": cam_pts.astype(float).tolist(),
            "projector_corners_px": proj_pts.astype(float).tolist(),
            "valid_mask": valid.astype(bool).tolist(),
            "valid_ratio": valid_ratio,
            "min_corner_valid_ratio": min_corner_valid_ratio,
            "uv_source_runs": run_info,
            "corner_reasons": reasons,
            "finite_samples_count": finite_counts,
            "projector_resolution": uv_meta.get("projector_resolution", {}),
            "uv_sampling_method": str(sampling_meta.get("uv_sampling_method", "median_patch")),
            "uv_sampling_params": {
                "patch_radius": int(sampling_meta.get("patch_radius", 2)),
                "min_valid_points": int(sampling_meta.get("min_valid_points", 8)),
            },
        }
        corr_name = "correspondences.json" if attempt == 0 else f"correspondences_retry_{attempt}.json"
        (view_dir / corr_name).write_text(json.dumps(corr, indent=2))

        ov = _draw_correspondence_overlay(frame, cam_pts, proj_pts, valid)
        save_image(view_dir / ("overlay.png" if attempt == 0 else f"overlay_retry_{attempt}.png"), ov)
        corner_ov = _draw_corner_validity_overlay(frame, cam_pts, reasons, breakdown)
        save_image(view_dir / ("corner_validity_overlay.png" if attempt == 0 else f"corner_validity_overlay_retry_{attempt}.png"), corner_ov)

        attempt_payload = {
            "attempt": attempt,
            "valid_ratio": valid_ratio,
            "diag": diag,
            "diag_obj": diag_obj,
            "view_quality": vq_report,
            "pose_entry": pose_entry,
            "uv_meta": uv_meta,
            "run_info": run_info,
            "corr": corr,
            "uv_dir": attempt_uv_dir,
            "accept": bool(diag_obj.accept),
            "reason": "; ".join(diag_obj.reject_reasons) if diag_obj.reject_reasons else None,
        }

        if best_attempt is None or attempt_payload["valid_ratio"] > best_attempt["valid_ratio"]:
            best_attempt = attempt_payload
        if bool(attempt_payload["accept"]):
            best_attempt = attempt_payload
            break

        final_reason = str(attempt_payload["reason"] or "view_rejected")
        clipped_ok = not bool(clipping_detected)
        mask_heavy = breakdown["mask_uv_false"] >= max(8, int(0.35 * len(reasons)))
        low_b = b_board < low_b_threshold

        if attempt >= max_retries:
            break
        if mismatch and strict_resolution_match:
            break
        if (low_b or mask_heavy) and clipped_ok and not exposure_retry_used:
            exposure_retry_used = True
            current_exposure = int(min(exposure_max, max(current_exposure + 1, round(current_exposure * retry_exposure_bump))))
            continue
        if not mask_retry_used:
            mask_retry_used = True
            continue
        break

    if best_attempt is None:
        summary["reason"] = final_reason
        summary["reject_reasons"] = [final_reason]
        summary["hints"] = []
        diag_obj = ProjectorViewDiagnostics(
            corners_found=True,
            valid_corner_ratio=0.0,
            hull_area_ratio=0.0,
            board_tilt_deg=float("nan"),
            board_center_norm=(0.0, 0.0),
            board_depth_proxy=float("nan"),
            b_median_board=0.0,
            unwrap_residual_p95=float("inf"),
            unwrap_residual_p95_board=float("inf"),
            unwrap_residual_vertical_p95_board=float("inf"),
            unwrap_residual_horizontal_p95_board=float("inf"),
            unwrap_residual_gt_1rad_pct_board=1.0,
            board_mask_area_ratio=0.0,
            clipping_detected=False,
            edge_corner_pct=0.0,
            duplicate_pose=False,
            accept=False,
            reject_reasons=[final_reason],
            hints=[],
        )
        vq_fail = ViewQualityReport(
            n_camera_corners=int(corners_x * corners_y),
            n_uv_valid=0,
            uv_valid_ratio=0.0,
            u_span_px=0.0,
            v_span_px=0.0,
            board_area_ratio=0.0,
            tilt_angle_deg=float("nan"),
            reproj_estimate_proxy=float("nan"),
            conditioning_score=0.0,
            accepted=False,
            rejection_reason=str(final_reason),
        )
        measured = {
            "valid_corner_ratio": 0.0,
            "board_area_ratio": _optional_finite_float(diag_obj.hull_area_ratio),
            "u_span_px": None,
            "v_span_px": None,
            "B_median_board": None,
            "B_median_board_available": False,
            "residual_p95_global": None,
            "residual_p95_board": None,
            "residual_p95_vertical_board": None,
            "residual_p95_horizontal_board": None,
            "residual_gt_1rad_pct_board": None,
            "board_mask_area_ratio": _optional_finite_float(diag_obj.board_mask_area_ratio),
            "uv_sampling_method": str((thresholds_used.get("uv_sampling", {}) or {}).get("method", "median_patch")),
            "clipping_detected": False,
        }
        diag = _build_view_diag_payload(
            view_id=view_id,
            checkerboard_found=True,
            corners_total=int(corners_x * corners_y),
            accept=False,
            reject_reasons=[final_reason],
            thresholds=thresholds_used,
            measured=measured,
            corner_breakdown={"ok": 0, "nan_uv": 0, "mask_uv_false": 0, "oob": 0, "near_edge": 0},
            corner_sampling={
                "uv_sampling_method": str((thresholds_used.get("uv_sampling", {}) or {}).get("method", "median_patch")),
                "patch_radius": int((thresholds_used.get("uv_sampling", {}) or {}).get("patch_radius", 2)),
                "min_valid_points": int((thresholds_used.get("uv_sampling", {}) or {}).get("min_valid_points", 8)),
            },
            uv={},
            phase_quality={},
            projector_resolution={},
            diagnostics=diag_obj,
            view_quality=vq_fail,
            hints=[],
        )
        (view_dir / "view_diag.json").write_text(json.dumps(_to_json_safe(diag), indent=2))
        (view_dir / "view_quality.json").write_text(
            json.dumps(
                _detection_quality_payload(
                    detection=det,
                    image_shape_hw=frame.shape[:2],
                    rejection_reason=str(final_reason),
                    rejection_hint=None,
                    exception=detection_exception,
                )
                | {"view_quality": _to_json_safe(vq_fail.to_dict())},
                indent=2,
            )
        )
        _save_view_quality_overlay(frame, vq_fail, view_dir / "view_quality_overlay.png")
        corr_stub = {
            "schema_version": SCHEMA_VERSION,
            "valid_ratio": 0.0,
            "error": final_reason,
            "uv_sampling_method": str((thresholds_used.get("uv_sampling", {}) or {}).get("method", "median_patch")),
        }
        (view_dir / "correspondences.json").write_text(json.dumps(corr_stub, indent=2))
        add_view_if_valid(
            session_dir=session_dir,
            view_data=summary,
            diagnostics=diag_obj,
            projector_points=None,
            projector_size=proj_size0,
            pose_entry=None,
        )
        _set_projector_calibration_light(controller, cfg_session, enabled=True)
        return summary

    selected = best_attempt
    selected_uv_dir: Path = selected["uv_dir"]
    final_uv_dir = view_dir / "uv"
    if selected_uv_dir != final_uv_dir:
        if final_uv_dir.exists():
            shutil.rmtree(final_uv_dir)
        shutil.copytree(selected_uv_dir, final_uv_dir, dirs_exist_ok=True)

    (view_dir / "correspondences.json").write_text(json.dumps(_to_json_safe(selected["corr"]), indent=2))
    save_image(view_dir / "overlay.png", _draw_correspondence_overlay(frame, np.asarray(selected["corr"]["camera_corners_px"]), np.asarray(selected["corr"]["projector_corners_px"]), np.asarray(selected["corr"]["valid_mask"], dtype=bool)))
    save_image(view_dir / "corner_validity_overlay.png", _draw_corner_validity_overlay(frame, np.asarray(selected["corr"]["camera_corners_px"]), selected["corr"]["corner_reasons"], selected["diag"]["corner_validity_breakdown"]))
    (view_dir / "view_diag.json").write_text(json.dumps(_to_json_safe(selected["diag"]), indent=2))
    if selected.get("view_quality") is not None:
        final_reasons = selected["diag"].get("reject_reasons", []) or []
        final_hints = selected["diag"].get("hints", []) or []
        (view_dir / "view_quality.json").write_text(
            json.dumps(
                _detection_quality_payload(
                    detection=det,
                    image_shape_hw=frame.shape[:2],
                    rejection_reason=(str(final_reasons[0]) if final_reasons else None),
                    rejection_hint=(str(final_hints[0]) if final_hints else None),
                    exception=detection_exception,
                )
                | {"view_quality": _to_json_safe(selected["view_quality"].to_dict())},
                indent=2,
            )
        )
        _save_view_quality_overlay(frame, selected["view_quality"], view_dir / "view_quality_overlay.png")
    uv_refine_diag = _write_view_uv_refinement_artifacts(
        view_dir=view_dir,
        uv_dir=final_uv_dir,
        camera_image=frame,
        corners_cam=np.asarray(selected["corr"]["camera_corners_px"], dtype=np.float32),
        raw_uv=np.asarray(selected["corr"]["projector_corners_px"], dtype=np.float32),
        raw_valid=np.asarray(selected["corr"]["valid_mask"], dtype=bool),
        cfg=cfg_session,
    )

    summary["valid_corner_ratio"] = float(selected["valid_ratio"])
    summary["valid_corners"] = int(np.count_nonzero(np.asarray(selected["corr"]["valid_mask"], dtype=bool)))
    summary["total_corners"] = int(len(selected["corr"]["valid_mask"]))
    summary["uv_meta"] = selected["uv_meta"]
    summary["uv_source_runs"] = selected["run_info"]
    summary["accept"] = bool(selected.get("accept", False))
    summary["diag"] = {
        "schema_version": SCHEMA_VERSION,
        "corner_validity_breakdown": selected["diag"].get("corner_validity_breakdown", {}),
        "uv_gate_failed_checks": selected["uv_meta"].get("uv_gate_failed_checks", []),
        "hints": selected["diag"].get("hints", []),
        "diagnostics": selected.get("diag_obj").to_dict() if selected.get("diag_obj") is not None else {},
        "view_quality": selected.get("view_quality").to_dict() if selected.get("view_quality") is not None else {},
        "uv_refinement": uv_refine_diag,
        "thresholds": selected["diag"].get("thresholds", {}),
        "measured": selected["diag"].get("measured", {}),
        "reject_reasons": selected["diag"].get("reject_reasons", []),
        "residual_thresholds": {
            "residual_p95_board_threshold": residual_p95_board_threshold,
            "residual_gt_1rad_pct_board_max": residual_gt_1rad_pct_board_max,
        },
        "residual_p95_board_pass": bool(
            selected.get("diag_obj") is not None
            and np.isfinite(float(selected.get("diag_obj").unwrap_residual_p95_board))
            and float(selected.get("diag_obj").unwrap_residual_p95_board) <= residual_p95_board_threshold
        ),
        "residual_gt_1rad_pct_board_pass": bool(
            selected.get("diag_obj") is not None
            and np.isfinite(float(selected.get("diag_obj").unwrap_residual_gt_1rad_pct_board))
            and float(selected.get("diag_obj").unwrap_residual_gt_1rad_pct_board) <= residual_gt_1rad_pct_board_max
        ),
    }
    proj_res_sel = (selected["uv_meta"].get("projector_resolution", {}) or {})
    pattern_surface = proj_res_sel.get("pattern_surface")
    if isinstance(pattern_surface, (list, tuple)) and len(pattern_surface) == 2:
        summary["effective_projector_size"] = [int(pattern_surface[0]), int(pattern_surface[1])]
    diag_obj = selected.get("diag_obj")
    if diag_obj is not None:
        summary["reject_reasons"] = list(diag_obj.reject_reasons)
        summary["hints"] = list(diag_obj.hints)
        summary["reason"] = "; ".join(diag_obj.reject_reasons) if diag_obj.reject_reasons else None
    else:
        summary["reject_reasons"] = []
        summary["hints"] = []
        summary["reason"] = selected["reason"]
    summary["status"] = "valid" if bool(selected.get("accept", False)) else "rejected"

    # Self-heal session projector resolution from actual pattern surface size.
    if not strict_resolution_match and isinstance(pattern_surface, (list, tuple)) and len(pattern_surface) == 2:
        try:
            pw, ph = int(pattern_surface[0]), int(pattern_surface[1])
            session_cfg = session.get("config", {}) or {}
            proj_cfg_s = session_cfg.get("projector", {}) or {}
            if int(proj_cfg_s.get("width", 0)) != pw or int(proj_cfg_s.get("height", 0)) != ph:
                proj_cfg_s["width"] = pw
                proj_cfg_s["height"] = ph
                session_cfg["projector"] = proj_cfg_s
                session["config"] = session_cfg
                _save_session(session_dir, session)
                try:
                    (session_dir / "config.json").write_text(json.dumps(session_cfg, indent=2))
                except Exception:
                    pass
        except Exception:
            pass

    proj_points = np.asarray(selected["corr"]["projector_corners_px"], dtype=np.float32)
    proj_valid = np.asarray(selected["corr"]["valid_mask"], dtype=bool)
    proj_points_valid = proj_points[proj_valid]
    proj_size_for_cov = proj_size0
    if isinstance(pattern_surface, (list, tuple)) and len(pattern_surface) == 2:
        proj_size_for_cov = (int(pattern_surface[0]), int(pattern_surface[1]))

    accepted, session_after = add_view_if_valid(
        session_dir=session_dir,
        view_data=summary,
        diagnostics=diag_obj,
        projector_points=proj_points_valid,
        projector_size=proj_size_for_cov,
        pose_entry=selected.get("pose_entry"),
    )
    summary["accept"] = bool(accepted)
    summary["status"] = "valid" if accepted else "rejected"
    if accepted and isinstance(session_after.get("coverage_map"), dict):
        summary["coverage_map"] = session_after.get("coverage_map")
    _set_projector_calibration_light(controller, cfg_session, enabled=True)
    return summary


def _load_camera_intrinsics(camera_intrinsics_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not camera_intrinsics_path.exists():
        raise FileNotFoundError(f"Camera intrinsics not found: {camera_intrinsics_path}")
    data = json.loads(camera_intrinsics_path.read_text())
    k = np.asarray(data.get("camera_matrix"), dtype=np.float64)
    d = np.asarray(data.get("dist_coeffs"), dtype=np.float64).reshape(-1, 1)
    if k.shape != (3, 3):
        raise ValueError("Invalid camera intrinsics matrix shape")
    return k, d


def _checkerboard_object_points(corners_x: int, corners_y: int, square_size_m: float) -> np.ndarray:
    obj = np.zeros((corners_x * corners_y, 3), dtype=np.float32)
    grid = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1, 2)
    obj[:, :2] = grid * float(square_size_m)
    return obj


def _collect_projector_calibration_views(
    session_dir: Path,
    session: dict[str, Any],
    *,
    corners_x: int,
    corners_y: int,
    square_size_m: float,
    min_corner_valid_ratio: float,
    proj_size_cfg: tuple[int, int],
    allowed_view_ids: set[str] | None = None,
    use_refined_uv: bool = False,
    min_refined_corner_ratio: float = 0.85,
    dropped_views: list[dict[str, Any]] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str], tuple[int, int]]:
    obj_template = _checkerboard_object_points(corners_x, corners_y, square_size_m)
    object_points: list[np.ndarray] = []
    cam_points: list[np.ndarray] = []
    proj_points: list[np.ndarray] = []
    view_ids: list[str] = []
    proj_size: tuple[int, int] | None = None

    valid_views = [v for v in session.get("views", []) if v.get("status") == "valid"]
    for view in valid_views:
        view_id = str(view.get("view_id", ""))
        if not view_id:
            continue
        if allowed_view_ids is not None and view_id not in allowed_view_ids:
            continue
        vdir = session_dir / "views" / view_id
        corr_path = vdir / "correspondences.json"
        if not corr_path.exists():
            if dropped_views is not None:
                dropped_views.append({"view_id": view_id, "reason": "missing_correspondences"})
            continue
        corr = json.loads(corr_path.read_text())
        corr_res = corr.get("projector_resolution", {}) or {}
        patt = corr_res.get("pattern_surface")
        if isinstance(patt, (list, tuple)) and len(patt) == 2:
            cand = (int(patt[0]), int(patt[1]))
            if proj_size is None:
                proj_size = cand
            elif proj_size != cand:
                if dropped_views is not None:
                    dropped_views.append({"view_id": view_id, "reason": "projector_size_mismatch", "size": list(cand)})
                continue

        if use_refined_uv:
            cam_corner_path = vdir / "camera_corners.json"
            uv_ref_path = vdir / "uv_corners_refined.npy"
            uv_valid_path = vdir / "uv_corners_valid.npy"
            if not cam_corner_path.exists() or not uv_ref_path.exists() or not uv_valid_path.exists():
                if dropped_views is not None:
                    dropped_views.append({"view_id": view_id, "reason": "missing_refined_uv_files"})
                continue
            cam_data = json.loads(cam_corner_path.read_text())
            cam = np.asarray(cam_data.get("corners_px", []), dtype=np.float32).reshape(-1, 2)
            proj = np.load(uv_ref_path).astype(np.float32).reshape(-1, 2)
            valid = np.load(uv_valid_path).astype(bool).reshape(-1)
            min_ratio_for_view = float(min_refined_corner_ratio)
        else:
            if float(corr.get("valid_ratio", 0.0)) < min_corner_valid_ratio:
                if dropped_views is not None:
                    dropped_views.append({"view_id": view_id, "reason": "below_min_corner_ratio_raw"})
                continue
            cam = np.asarray(corr.get("camera_corners_px", []), dtype=np.float32).reshape(-1, 2)
            proj = np.asarray(corr.get("projector_corners_px", []), dtype=np.float32).reshape(-1, 2)
            valid = np.asarray(corr.get("valid_mask", []), dtype=bool).reshape(-1)
            min_ratio_for_view = float(min_corner_valid_ratio)

        if (
            cam.shape[0] != obj_template.shape[0]
            or proj.shape[0] != obj_template.shape[0]
            or valid.shape[0] != obj_template.shape[0]
        ):
            if dropped_views is not None:
                dropped_views.append({"view_id": view_id, "reason": "shape_mismatch"})
            continue
        idx = np.where(valid)[0]
        min_required = int(np.ceil(min_ratio_for_view * obj_template.shape[0]))
        if idx.size < min_required:
            if dropped_views is not None:
                dropped_views.append(
                    {
                        "view_id": view_id,
                        "reason": "insufficient_refined_corners" if use_refined_uv else "insufficient_corners",
                        "valid_corners": int(idx.size),
                        "required": int(min_required),
                    }
                )
            continue
        object_points.append(obj_template[idx].astype(np.float32))
        cam_points.append(cam[idx].reshape(-1, 1, 2).astype(np.float32))
        proj_points.append(proj[idx].reshape(-1, 1, 2).astype(np.float32))
        view_ids.append(view_id)

    if proj_size is None:
        proj_size = proj_size_cfg
    return object_points, cam_points, proj_points, view_ids, proj_size


def _stereo_calibrate_once(
    session_dir: Path,
    cfg: dict,
    camera_intrinsics_path: Path,
    *,
    min_views_required: int,
    min_corner_valid_ratio: float,
    allowed_view_ids: set[str] | None = None,
    save_reproj_overlays: bool = True,
    use_refined_uv: bool = False,
    min_refined_corner_ratio: float = 0.85,
    dropped_views: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cv = _require_cv2()
    session = _load_session(session_dir)
    pcal = cfg.get("projector_calibration", {}) or {}
    chk = pcal.get("checkerboard", {}) or {}
    proj_cfg = pcal.get("projector", {}) or {}

    corners_x = int(chk.get("corners_x", 8))
    corners_y = int(chk.get("corners_y", 6))
    square_size_m = float(chk.get("square_size_m", 0.01))
    proj_size_cfg = (int(proj_cfg.get("width", 1024)), int(proj_cfg.get("height", 768)))
    object_points, cam_points, proj_points, view_ids, proj_size = _collect_projector_calibration_views(
        session_dir,
        session,
        corners_x=corners_x,
        corners_y=corners_y,
        square_size_m=square_size_m,
        min_corner_valid_ratio=min_corner_valid_ratio,
        proj_size_cfg=proj_size_cfg,
        allowed_view_ids=allowed_view_ids,
        use_refined_uv=use_refined_uv,
        min_refined_corner_ratio=min_refined_corner_ratio,
        dropped_views=dropped_views,
    )

    if len(object_points) < int(min_views_required):
        raise ValueError(f"Need at least {min_views_required} valid projector-calibration views.")

    k_cam, d_cam = _load_camera_intrinsics(camera_intrinsics_path)

    rms_proj, k_proj, d_proj, _, _ = cv.calibrateCamera(
        object_points,
        proj_points,
        proj_size,
        None,
        None,
    )

    # Then calibrate stereo extrinsics with both intrinsics fixed.
    flags = cv.CALIB_FIX_INTRINSIC
    rms_stereo, k1, d1, k2, d2, r, t, e, f = cv.stereoCalibrate(
        object_points,
        cam_points,
        proj_points,
        k_cam.copy(),
        d_cam.copy(),
        k_proj.copy(),
        d_proj.copy(),
        proj_size,
        flags=flags,
    )

    r1, r2, p1, p2, q, _, _ = cv.stereoRectify(
        k1, d1, k2, d2, proj_size, r, t
    )

    per_view: list[dict[str, Any]] = []
    reproj_dir = session_dir / "results" / "reprojection_debug"
    if save_reproj_overlays:
        reproj_dir.mkdir(parents=True, exist_ok=True)
    for obj, cam_obs, proj_obs, view_id in zip(object_points, cam_points, proj_points, view_ids):
        ok1, rv_cam, tv_cam = cv.solvePnP(obj, cam_obs, k1, d1)
        ok2, rv_proj, tv_proj = cv.solvePnP(obj, proj_obs, k2, d2)
        if not ok1 or not ok2:
            continue
        cam_reproj, _ = cv.projectPoints(obj, rv_cam, tv_cam, k1, d1)
        proj_reproj, _ = cv.projectPoints(obj, rv_proj, tv_proj, k2, d2)
        cam_err = float(np.linalg.norm(cam_obs.reshape(-1, 2) - cam_reproj.reshape(-1, 2), axis=1).mean())
        proj_err = float(np.linalg.norm(proj_obs.reshape(-1, 2) - proj_reproj.reshape(-1, 2), axis=1).mean())

        if save_reproj_overlays:
            camera_img_path = session_dir / "views" / view_id / "camera.png"
            if camera_img_path.exists():
                im = np.array(Image.open(camera_img_path))
                if im.ndim == 2:
                    im = np.stack([im, im, im], axis=2)
                out = im[:, :, :3].copy()
                for p_obs, p_rep in zip(cam_obs.reshape(-1, 2), cam_reproj.reshape(-1, 2)):
                    cv.circle(out, (int(round(float(p_obs[0]))), int(round(float(p_obs[1])))), 3, (255, 64, 64), 1, cv.LINE_AA)
                    cv.circle(out, (int(round(float(p_rep[0]))), int(round(float(p_rep[1])))), 2, (64, 255, 64), 1, cv.LINE_AA)
                cv.putText(
                    out,
                    f"cam_err={cam_err:.3f}px proj_err={proj_err:.3f}px",
                    (16, 28),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 64),
                    2,
                    cv.LINE_AA,
                )
                Image.fromarray(out.astype(np.uint8)).save(reproj_dir / f"{view_id}_overlay.png")

        per_view.append(
            {
                "view_id": view_id,
                "camera_reproj_error_px": cam_err,
                "projector_reproj_error_px": proj_err,
                "corners_used": int(obj.shape[0]),
            }
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session.get("session_id"),
        "created_at": datetime.now().isoformat(),
        "views_used": int(len(object_points)),
        "view_ids": list(view_ids),
        "rms_projector_intrinsics": float(rms_proj),
        "rms_stereo": float(rms_stereo),
        "camera_matrix": k1.astype(float).tolist(),
        "camera_dist_coeffs": d1.reshape(-1).astype(float).tolist(),
        "projector_matrix": k2.astype(float).tolist(),
        "projector_dist_coeffs": d2.reshape(-1).astype(float).tolist(),
        "R": r.astype(float).tolist(),
        "T": t.reshape(-1).astype(float).tolist(),
        "E": e.astype(float).tolist(),
        "F": f.astype(float).tolist(),
        "rectification": {
            "R1": r1.astype(float).tolist(),
            "R2": r2.astype(float).tolist(),
            "P1": p1.astype(float).tolist(),
            "P2": p2.astype(float).tolist(),
            "Q": q.astype(float).tolist(),
        },
        "per_view_errors": per_view,
        "checkerboard": {
            "corners_x": corners_x,
            "corners_y": corners_y,
            "square_size_m": square_size_m,
        },
        "projector": {"width": proj_size[0], "height": proj_size[1]},
        "points_source": "refined_uv" if use_refined_uv else "raw_uv",
        "uv_sampling": {
            "method": str((pcal.get("uv_sampling", {}) or {}).get("method", "median_patch")),
            "patch_radius": int((pcal.get("uv_sampling", {}) or {}).get("patch_radius", 2)),
            "min_valid_points": int((pcal.get("uv_sampling", {}) or {}).get("min_valid_points", 8)),
        },
    }
    mats = {
        "camera_matrix": k1,
        "camera_dist": d1,
        "projector_matrix": k2,
        "projector_dist": d2,
        "R": r,
        "T": t,
        "E": e,
        "F": f,
        "R1": r1,
        "R2": r2,
        "P1": p1,
        "P2": p2,
        "Q": q,
    }
    return result, mats


def _write_stereo_outputs(
    session_dir: Path,
    session: dict[str, Any],
    result: dict[str, Any],
    mats: dict[str, np.ndarray],
    *,
    json_name: str = "stereo.json",
    npz_name: str = "stereo.npz",
    update_latest: bool = True,
    session_result_path: str | None = None,
) -> None:
    res_dir = session_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / json_name).write_text(json.dumps(result, indent=2))
    np.savez(
        res_dir / npz_name,
        **mats,
    )
    if update_latest:
        proj_root = session_dir.parent.parent
        (proj_root / "stereo_latest.json").write_text(json.dumps(result, indent=2))
    if session_result_path is None:
        session_result_path = f"results/{json_name}"
    coverage_ratio = _safe_float_or_nan(result.get("coverage_ratio"))
    bins_covered = result.get("bins_covered_count")
    session["results"] = {
        "rms_stereo": float(result.get("rms_stereo", 0.0)),
        "views_used": int(result.get("views_used", 0)),
        "updated_at": datetime.now().isoformat(),
        "path": session_result_path,
        "coverage_ratio": float(coverage_ratio) if np.isfinite(coverage_ratio) else None,
        "bins_covered_count": int(bins_covered) if isinstance(bins_covered, (int, np.integer)) else None,
    }
    _save_session(session_dir, session)


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _result_error_summary(result: dict[str, Any]) -> dict[str, float]:
    per = result.get("per_view_errors", []) or []
    cam = np.asarray([float(v.get("camera_reproj_error_px", np.nan)) for v in per], dtype=np.float64)
    proj = np.asarray([float(v.get("projector_reproj_error_px", np.nan)) for v in per], dtype=np.float64)
    cam = cam[np.isfinite(cam)]
    proj = proj[np.isfinite(proj)]
    return {
        "camera_reproj_p50": float(np.percentile(cam, 50)) if cam.size else float("nan"),
        "camera_reproj_p95": float(np.percentile(cam, 95)) if cam.size else float("nan"),
        "projector_reproj_p50": float(np.percentile(proj, 50)) if proj.size else float("nan"),
        "projector_reproj_p95": float(np.percentile(proj, 95)) if proj.size else float("nan"),
    }


def _find_existing_stereo_result(session_dir: Path, filename: str = "stereo.json") -> dict[str, Any] | None:
    p = session_dir / "results" / filename
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def refine_uv_corners_for_session(session_dir: Path, cfg: dict) -> dict[str, Any]:
    """Offline refinement over existing projector-calibration views."""
    session = _load_session(session_dir)
    pcal = cfg.get("projector_calibration", {}) or {}
    refine_cfg = _uv_refine_config(cfg)
    if not refine_cfg.enabled:
        return {
            "enabled": False,
            "session_id": session.get("session_id"),
            "views_total": 0,
            "views_processed": 0,
            "views_failed": 0,
            "per_view": [],
        }

    proj_cfg = (session.get("config", {}) or {}).get("projector", {}) or {}
    fallback_proj = (int(proj_cfg.get("width", 1024)), int(proj_cfg.get("height", 768)))
    views_dir = session_dir / "views"
    view_dirs = sorted([p for p in views_dir.glob("view_*") if p.is_dir()])
    per_view: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    delta_values: list[float] = []
    invalid_corners_total = 0

    for vdir in view_dirs:
        view_id = vdir.name
        corners_path = vdir / "camera_corners.json"
        uv_dir = vdir / "uv"
        u_path = uv_dir / "u.npy"
        v_path = uv_dir / "v.npy"
        m_path = uv_dir / "mask_uv.npy"
        uv_meta_path = uv_dir / "uv_meta.json"
        corr_path = vdir / "correspondences.json"
        cam_img_path = vdir / "camera.png"

        missing = [
            str(p.name)
            for p in (corners_path, u_path, v_path, m_path, uv_meta_path, corr_path, cam_img_path)
            if not p.exists()
        ]
        if missing:
            failed.append({"view_id": view_id, "reason": "missing_files", "missing": missing})
            continue

        try:
            corners_data = json.loads(corners_path.read_text())
            corners = np.asarray(corners_data.get("corners_px", []), dtype=np.float32).reshape(-1, 2)
            corr = json.loads(corr_path.read_text())
            uv_meta = json.loads(uv_meta_path.read_text())
            U = np.load(u_path).astype(np.float32)
            V = np.load(v_path).astype(np.float32)
            mask_uv = np.load(m_path).astype(bool)
            if corners.size == 0:
                failed.append({"view_id": view_id, "reason": "no_corners"})
                continue
            raw_uv = np.asarray(corr.get("projector_corners_px", []), dtype=np.float32).reshape(-1, 2)
            if raw_uv.shape[0] != corners.shape[0]:
                # Fallback: recover raw samples from saved correspondences shape if possible.
                raw_uv = np.full((corners.shape[0], 2), np.nan, dtype=np.float32)
                vm = np.asarray(corr.get("valid_mask", []), dtype=bool).reshape(-1)
                pv = np.asarray(corr.get("projector_corners_px", []), dtype=np.float32).reshape(-1, 2)
                ncopy = min(raw_uv.shape[0], pv.shape[0], vm.shape[0])
                raw_uv[:ncopy] = pv[:ncopy]

            refined_uv, valid_refined, refine_diag = refine_uv_at_corners(U, V, mask_uv, corners, refine_cfg)
            np.save(vdir / "uv_corners_raw.npy", raw_uv.astype(np.float32))
            np.save(vdir / "uv_corners_refined.npy", refined_uv.astype(np.float32))
            np.save(vdir / "uv_corners_valid.npy", valid_refined.astype(bool))

            valid_ratio = float(np.count_nonzero(valid_refined) / max(1, valid_refined.size))
            corr["projector_corners_px"] = refined_uv.astype(float).tolist()
            corr["valid_mask"] = valid_refined.astype(bool).tolist()
            corr["valid_ratio"] = valid_ratio
            corr["uv_refinement_applied"] = True
            corr["uv_refinement_config"] = _to_json_safe(asdict(refine_cfg))
            corr_path.write_text(json.dumps(_to_json_safe(corr), indent=2))

            breakdown = Counter(corr.get("corner_reasons", []))
            # Rebuild reasons from valid mask when not present.
            if sum(breakdown.values()) <= 0:
                breakdown = Counter(["ok" if bool(v) else "nan_uv" for v in valid_refined])
            breakdown_dict = {
                "ok": int(np.count_nonzero(valid_refined)),
                "nan_uv": int(valid_refined.size - np.count_nonzero(valid_refined)),
                "mask_uv_false": int(breakdown.get("mask_uv_false", 0)),
                "oob": int(breakdown.get("oob", 0)),
                "near_edge": int(breakdown.get("near_edge", 0)),
            }

            diag_path = vdir / "view_diag.json"
            diag = _load_json(diag_path)
            diag["schema_version"] = SCHEMA_VERSION
            diag.setdefault(
                "checkerboard",
                {"found": bool(corners_data.get("found", True)), "corners_total": int(corners.shape[0])},
            )
            diag.setdefault("uv", {})
            for key in (
                "uv_gate_ok",
                "valid_ratio",
                "u_range",
                "v_range",
                "u_edge_pct",
                "v_edge_pct",
                "u_zero_pct",
                "v_zero_pct",
            ):
                if key in uv_meta:
                    diag["uv"][key] = uv_meta[key]
            uv_sampling_cfg = (pcal.get("uv_sampling", {}) or {})
            diag["corner_sampling"] = {
                "uv_sampling_method": str(uv_sampling_cfg.get("method", "median_patch")),
                "patch_radius": int(uv_sampling_cfg.get("patch_radius", 2)),
                "min_valid_points": int(uv_sampling_cfg.get("min_valid_points", 8)),
                "uv_refinement": _to_json_safe(refine_diag.get("summary", {})),
            }
            diag["corner_validity_breakdown"] = {
                **breakdown_dict,
                "finite_samples_median": float(np.median([pc.get("n_points", 0) for pc in refine_diag.get("per_corner", [])]))
                if refine_diag.get("per_corner")
                else 0.0,
            }
            measured = diag.get("measured", {}) or {}
            measured["valid_corner_ratio"] = float(valid_ratio)
            measured["uv_sampling_method"] = str(uv_sampling_cfg.get("method", "median_patch"))
            diag["measured"] = measured
            diag.setdefault("reject_reasons", [])
            diag_path.write_text(json.dumps(_to_json_safe(diag), indent=2))

            cam_img = np.array(Image.open(cam_img_path))
            overlay = _draw_correspondence_overlay(cam_img, corners, refined_uv, valid_refined)
            save_image(vdir / "overlay.png", overlay)
            validity_overlay = _draw_corner_validity_overlay(cam_img, corners, ["ok" if bool(v) else "nan_uv" for v in valid_refined], breakdown_dict)
            save_image(vdir / "corner_validity_overlay.png", validity_overlay)
            delta_stats = _save_uv_refine_delta_overlay(
                image=cam_img,
                corners_cam=corners,
                uv_raw=raw_uv,
                uv_refined=refined_uv,
                valid_refined=valid_refined,
                out_path=vdir / "uv_refine_delta.png",
            )

            refine_diag_out = {
                "config": _to_json_safe(asdict(refine_cfg)),
                "summary": _to_json_safe(refine_diag.get("summary", {})),
                "per_corner": _to_json_safe(refine_diag.get("per_corner", [])),
                "delta_stats": _to_json_safe(delta_stats),
                "valid_ratio": valid_ratio,
            }
            (vdir / "uv_refine_diag.json").write_text(json.dumps(refine_diag_out, indent=2))

            if np.isfinite(delta_stats.get("mean_delta_px", np.nan)):
                delta_values.append(float(delta_stats["mean_delta_px"]))
            invalid_corners_total += int(valid_refined.size - np.count_nonzero(valid_refined))
            per_view.append(
                {
                    "view_id": view_id,
                    "corners_total": int(valid_refined.size),
                    "corners_valid": int(np.count_nonzero(valid_refined)),
                    "valid_ratio": valid_ratio,
                    "delta_stats": _to_json_safe(delta_stats),
                }
            )
        except Exception as exc:
            failed.append({"view_id": view_id, "reason": f"exception: {exc}"})

    # Sync session view-level counters with refined correspondences.
    session_changed = False
    by_id = {str(v.get("view_id")): v for v in (session.get("views", []) or []) if isinstance(v, dict)}
    for rec in per_view:
        item = by_id.get(rec["view_id"])
        if item is None:
            continue
        item["valid_corner_ratio"] = float(rec["valid_ratio"])
        item["valid_corners"] = int(rec["corners_valid"])
        item["total_corners"] = int(rec["corners_total"])
        session_changed = True
    if session_changed:
        _save_session(session_dir, session)

    report = {
        "enabled": True,
        "session_id": session.get("session_id"),
        "views_total": int(len(view_dirs)),
        "views_processed": int(len(per_view)),
        "views_failed": int(len(failed)),
        "failed_views": failed,
        "mean_corner_delta_px": float(np.mean(delta_values)) if delta_values else float("nan"),
        "p95_corner_delta_px": float(np.percentile(delta_values, 95)) if delta_values else float("nan"),
        "invalidated_corners_total": int(invalid_corners_total),
        "per_view": per_view,
        "config": _to_json_safe(asdict(refine_cfg)),
    }
    (session_dir / "results").mkdir(parents=True, exist_ok=True)
    (session_dir / "results" / "uv_refine_session_report.json").write_text(
        json.dumps(_to_json_safe(report), indent=2)
    )
    return report


def _build_refine_compare_outputs(
    session_dir: Path,
    before_result: dict[str, Any] | None,
    refined_result: dict[str, Any] | None,
    refined_pruned_result: dict[str, Any] | None,
    refine_report: dict[str, Any],
) -> dict[str, Any]:
    reports_dir = session_dir / "reports" / "refine_compare"
    reports_dir.mkdir(parents=True, exist_ok=True)

    before_rms = float(before_result.get("rms_stereo", np.nan)) if before_result else float("nan")
    refined_rms = float(refined_result.get("rms_stereo", np.nan)) if refined_result else float("nan")
    pruned_rms = float(refined_pruned_result.get("rms_stereo", np.nan)) if refined_pruned_result else float("nan")

    _draw_simple_bar_plot(
        values=[before_rms, refined_rms, pruned_rms],
        labels=["before", "refined", "ref+prune"],
        out_path=reports_dir / "rms_before_after.png",
        title="Stereo RMS Comparison",
    )

    before_per = {str(v.get("view_id")): float(v.get("projector_reproj_error_px", np.nan)) for v in (before_result or {}).get("per_view_errors", [])}
    after_per = {str(v.get("view_id")): float(v.get("projector_reproj_error_px", np.nan)) for v in (refined_pruned_result or refined_result or {}).get("per_view_errors", [])}
    cats = sorted(set(before_per.keys()) | set(after_per.keys()))
    _draw_group_bar_plot(
        categories=cats[:20],  # keep readable if many views
        before_vals=[before_per.get(c, np.nan) for c in cats[:20]],
        after_vals=[after_per.get(c, np.nan) for c in cats[:20]],
        out_path=reports_dir / "proj_reproj_per_view_before_after.png",
        title="Projector Reprojection Error Per View (first 20 views)",
    )

    distort = {
        "before": (before_result or {}).get("projector_dist_coeffs"),
        "refined": (refined_result or {}).get("projector_dist_coeffs"),
        "refined_pruned": (refined_pruned_result or {}).get("projector_dist_coeffs"),
    }
    (reports_dir / "distortion_coeffs_before_after.json").write_text(
        json.dumps(_to_json_safe(distort), indent=2)
    )

    def _warn_extreme_k3(dist: Any) -> bool:
        if not isinstance(dist, list) or len(dist) < 5:
            return False
        k3 = float(dist[4])
        return bool(np.isfinite(k3) and abs(k3) > 200.0)

    warnings: list[str] = []
    if _warn_extreme_k3(distort.get("before")):
        warnings.append("before: extreme |k3| > 200")
    if _warn_extreme_k3(distort.get("refined")):
        warnings.append("refined: extreme |k3| > 200")
    if _warn_extreme_k3(distort.get("refined_pruned")):
        warnings.append("refined_pruned: extreme |k3| > 200")

    compare_summary = {
        "before": {
            "rms_stereo": before_rms,
            **(_result_error_summary(before_result or {})),
        },
        "refined": {
            "rms_stereo": refined_rms,
            **(_result_error_summary(refined_result or {})),
        },
        "refined_pruned": {
            "rms_stereo": pruned_rms,
            **(_result_error_summary(refined_pruned_result or {})),
        },
        "refine_report": _to_json_safe(refine_report),
        "warnings": warnings,
    }
    (reports_dir / "compare_summary.json").write_text(json.dumps(_to_json_safe(compare_summary), indent=2))
    return compare_summary


def stereo_calibrate(session_dir: Path, cfg: dict, camera_intrinsics_path: Path, prune: bool = False) -> dict[str, Any]:
    session = _load_session(session_dir)
    pcal = cfg.get("projector_calibration", {}) or {}
    cap_cfg = pcal.get("capture", {}) or {}
    calib_cfg = cfg.get("calibration", {}) or {}
    uv_ref_cfg = pcal.get("uv_refinement", {}) or {}
    min_views_cfg = int(cap_cfg.get("min_views", 10))
    min_corner_valid_ratio = float(cap_cfg.get("min_corner_valid_ratio", 0.90))
    use_refined_for_stereo = bool(uv_ref_cfg.get("enabled", False) and uv_ref_cfg.get("use_for_stereo", True))
    min_refined_corner_ratio = float(uv_ref_cfg.get("min_view_valid_ratio", 0.85))
    projector_mode = str(calib_cfg.get("projector_mode", "corner")).strip().lower()
    dense_cfg = (calib_cfg.get("dense_plane", {}) or {})
    use_dense_mode = projector_mode == "dense_plane" and bool(dense_cfg.get("enabled", False))

    if use_refined_for_stereo:
        refine_uv_corners_for_session(session_dir, cfg)

    if use_dense_mode:
        if not prune:
            result, mats = calibrate_dense_plane_session(
                session_dir=session_dir,
                cfg=cfg,
                camera_intrinsics_path=camera_intrinsics_path,
                min_views_required=min_views_cfg,
                min_corner_valid_ratio=min_corner_valid_ratio,
                allowed_view_ids=None,
            )
            reports = finalize_session_reports(session_dir, cfg)
            coverage = reports.get("coverage", {}) or {}
            result["coverage_ratio"] = float(coverage.get("coverage_ratio", 0.0))
            result["bins_covered_count"] = int(coverage.get("bins_covered_count", 0))
            result["bins_total"] = int(coverage.get("bins_total", 0))
            result["coverage"] = {
                "coverage_json": "results/coverage.json",
                "coverage_png": "results/coverage.png",
                "coverage_ratio": result["coverage_ratio"],
                "bins_covered_count": result["bins_covered_count"],
                "bins_total": result["bins_total"],
            }
            res_dir = session_dir / "results"
            res_dir.mkdir(parents=True, exist_ok=True)
            (res_dir / "stereo_dense.json").write_text(json.dumps(_to_json_safe(result), indent=2))
            np.savez(res_dir / "stereo_dense.npz", **mats)
            session["results"] = {
                "rms_stereo": float(result.get("rms_stereo", 0.0)),
                "views_used": int(result.get("views_used", 0)),
                "updated_at": datetime.now().isoformat(),
                "path": "results/stereo_dense.json",
                "coverage_ratio": float(result.get("coverage_ratio", 0.0)),
                "bins_covered_count": int(result.get("bins_covered_count", 0)),
                "points_source": "dense_plane",
            }
            _save_session(session_dir, session)
            return result

        valid_views = [v for v in session.get("views", []) if v.get("status") == "valid" and v.get("view_id")]
        current_view_ids = [str(v["view_id"]) for v in valid_views]
        min_views_floor = 8
        min_improvement = 0.05
        best_result: dict[str, Any] | None = None
        best_mats: dict[str, np.ndarray] | None = None
        best_view_ids: list[str] = []
        best_rms = float("inf")
        removed_views: list[str] = []
        history: list[dict[str, Any]] = []
        while len(current_view_ids) >= min_views_floor:
            result_i, mats_i = calibrate_dense_plane_session(
                session_dir=session_dir,
                cfg=cfg,
                camera_intrinsics_path=camera_intrinsics_path,
                min_views_required=min_views_floor,
                min_corner_valid_ratio=min_corner_valid_ratio,
                allowed_view_ids=set(current_view_ids),
            )
            rms = float(result_i.get("rms_stereo", float("inf")))
            prev_best = best_rms
            improved = bool(np.isfinite(rms) and (rms < best_rms))
            if improved:
                best_rms = rms
                best_result = result_i
                best_mats = mats_i
                best_view_ids = list(current_view_ids)
            hist = {"views_used": int(len(current_view_ids)), "rms": rms, "improved": improved}
            if np.isfinite(prev_best):
                hist["improvement_px"] = float(prev_best - rms)
            history.append(hist)
            if not improved:
                hist["stop_reason"] = "rms_not_improving"
                break
            if np.isfinite(prev_best) and (prev_best - rms) < min_improvement:
                hist["stop_reason"] = "improvement_below_threshold"
                break
            if len(current_view_ids) <= min_views_floor:
                hist["stop_reason"] = "min_view_floor_reached"
                break
            removable = [
                pv for pv in (result_i.get("per_view_errors", []) or [])
                if int(pv.get("corners_used", 0)) >= 48 and str(pv.get("view_id", "")) in current_view_ids
            ]
            if not removable:
                hist["stop_reason"] = "no_removable_view_with_48_corners"
                break
            worst = max(removable, key=lambda v: float(v.get("projector_reproj_error_px", -1.0)))
            worst_id = str(worst.get("view_id", ""))
            if not worst_id or len(current_view_ids) - 1 < min_views_floor:
                hist["stop_reason"] = "cannot_remove_worst"
                break
            hist["removed_view"] = worst_id
            hist["removed_projector_reproj_error_px"] = float(worst.get("projector_reproj_error_px", float("nan")))
            current_view_ids = [vid for vid in current_view_ids if vid != worst_id]
            removed_views.append(worst_id)

        if best_result is None or best_mats is None:
            raise ValueError("Dense-plane pruning could not produce a valid calibration result.")
        final_result, final_mats = calibrate_dense_plane_session(
            session_dir=session_dir,
            cfg=cfg,
            camera_intrinsics_path=camera_intrinsics_path,
            min_views_required=min_views_floor,
            min_corner_valid_ratio=min_corner_valid_ratio,
            allowed_view_ids=set(best_view_ids),
        )
        reports = finalize_session_reports(session_dir, cfg)
        coverage = reports.get("coverage", {}) or {}
        final_result["coverage_ratio"] = float(coverage.get("coverage_ratio", 0.0))
        final_result["bins_covered_count"] = int(coverage.get("bins_covered_count", 0))
        final_result["bins_total"] = int(coverage.get("bins_total", 0))
        final_result["coverage"] = {
            "coverage_json": "results/coverage.json",
            "coverage_png": "results/coverage.png",
            "coverage_ratio": final_result["coverage_ratio"],
            "bins_covered_count": final_result["bins_covered_count"],
            "bins_total": final_result["bins_total"],
        }
        prune_report = {
            "initial_rms": float(history[0]["rms"]) if history else None,
            "final_rms": float(final_result.get("rms_stereo", 0.0)),
            "views_removed": removed_views,
            "history": history,
            "min_views_floor": min_views_floor,
            "min_improvement_px": min_improvement,
            "best_view_ids": best_view_ids,
        }
        final_result["prune_report"] = prune_report
        res_dir = session_dir / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "stereo_dense_pruned.json").write_text(json.dumps(_to_json_safe(final_result), indent=2))
        np.savez(res_dir / "stereo_dense_pruned.npz", **final_mats)
        (res_dir / "prune_dense_report.json").write_text(json.dumps(_to_json_safe(prune_report), indent=2))
        session["results"] = {
            "rms_stereo": float(final_result.get("rms_stereo", 0.0)),
            "views_used": int(final_result.get("views_used", 0)),
            "updated_at": datetime.now().isoformat(),
            "path": "results/stereo_dense_pruned.json",
            "coverage_ratio": float(final_result.get("coverage_ratio", 0.0)),
            "bins_covered_count": int(final_result.get("bins_covered_count", 0)),
            "points_source": "dense_plane",
        }
        _save_session(session_dir, session)
        return final_result

    if not prune:
        result, mats = _stereo_calibrate_once(
            session_dir,
            cfg,
            camera_intrinsics_path,
            min_views_required=min_views_cfg,
            min_corner_valid_ratio=min_corner_valid_ratio,
            allowed_view_ids=None,
            save_reproj_overlays=True,
            use_refined_uv=use_refined_for_stereo,
            min_refined_corner_ratio=min_refined_corner_ratio,
        )
        result["points_source"] = "refined_uv" if use_refined_for_stereo else "raw_uv"
        reports = finalize_session_reports(session_dir, cfg)
        coverage = reports.get("coverage", {}) or {}
        result["coverage_ratio"] = float(coverage.get("coverage_ratio", 0.0))
        result["bins_covered_count"] = int(coverage.get("bins_covered_count", 0))
        result["bins_total"] = int(coverage.get("bins_total", 0))
        result["coverage"] = {
            "coverage_json": "results/coverage.json",
            "coverage_png": "results/coverage.png",
            "coverage_ratio": result["coverage_ratio"],
            "bins_covered_count": result["bins_covered_count"],
            "bins_total": result["bins_total"],
        }
        if use_refined_for_stereo:
            (session_dir / "results" / "stereo_refined.json").write_text(
                json.dumps(_to_json_safe(result), indent=2)
            )
            np.savez(session_dir / "results" / "stereo_refined.npz", **mats)
            session["results"] = {
                "rms_stereo": float(result.get("rms_stereo", 0.0)),
                "views_used": int(result.get("views_used", 0)),
                "updated_at": datetime.now().isoformat(),
                "path": "results/stereo_refined.json",
                "coverage_ratio": float(result.get("coverage_ratio", 0.0)),
                "bins_covered_count": int(result.get("bins_covered_count", 0)),
                "points_source": "refined_uv",
            }
            _save_session(session_dir, session)
        else:
            _write_stereo_outputs(
                session_dir,
                session,
                result,
                mats,
                json_name="stereo.json",
                npz_name="stereo.npz",
                update_latest=True,
                session_result_path="results/stereo.json",
            )
        return result

    # Iterative pruning mode.
    valid_views = [v for v in session.get("views", []) if v.get("status") == "valid" and v.get("view_id")]
    current_view_ids = [str(v["view_id"]) for v in valid_views]
    min_views_floor = 8
    min_improvement = 0.05
    if len(current_view_ids) < min_views_floor:
        raise ValueError(f"Need at least {min_views_floor} valid views for pruning mode.")

    best_result: dict[str, Any] | None = None
    best_mats: dict[str, np.ndarray] | None = None
    best_view_ids: list[str] = []
    best_rms = float("inf")
    removed_views: list[str] = []
    history: list[dict[str, Any]] = []

    while len(current_view_ids) >= min_views_floor:
        result, mats = _stereo_calibrate_once(
            session_dir,
            cfg,
            camera_intrinsics_path,
            min_views_required=min_views_floor,
            min_corner_valid_ratio=min_corner_valid_ratio,
            allowed_view_ids=set(current_view_ids),
            save_reproj_overlays=False,
            use_refined_uv=use_refined_for_stereo,
            min_refined_corner_ratio=min_refined_corner_ratio,
        )
        rms = float(result.get("rms_stereo", float("inf")))
        prev_best = best_rms
        improved = bool(np.isfinite(rms) and (rms < best_rms))
        if improved:
            best_rms = rms
            best_result = result
            best_mats = mats
            best_view_ids = list(current_view_ids)

        entry: dict[str, Any] = {
            "views_used": int(len(current_view_ids)),
            "rms": rms,
            "improved": improved,
        }
        if np.isfinite(prev_best):
            entry["improvement_px"] = float(prev_best - rms)
        history.append(entry)

        if not improved:
            entry["stop_reason"] = "rms_not_improving"
            break
        if np.isfinite(prev_best) and (prev_best - rms) < min_improvement:
            entry["stop_reason"] = "improvement_below_threshold"
            break

        if len(current_view_ids) <= min_views_floor:
            entry["stop_reason"] = "min_view_floor_reached"
            break

        removable = [
            pv for pv in (result.get("per_view_errors", []) or [])
            if int(pv.get("corners_used", 0)) >= 48 and str(pv.get("view_id", "")) in current_view_ids
        ]
        if not removable:
            entry["stop_reason"] = "no_removable_view_with_48_corners"
            break

        worst = max(removable, key=lambda v: float(v.get("projector_reproj_error_px", -1.0)))
        worst_id = str(worst.get("view_id", ""))
        if not worst_id:
            entry["stop_reason"] = "invalid_worst_view_id"
            break
        if len(current_view_ids) - 1 < min_views_floor:
            entry["stop_reason"] = "would_go_below_min_view_floor"
            break
        current_view_ids = [vid for vid in current_view_ids if vid != worst_id]
        removed_views.append(worst_id)
        entry["removed_view"] = worst_id

    if best_result is None or best_mats is None:
        raise ValueError("Pruning could not produce a valid calibration result.")

    # Re-run once for best subset to generate overlays and consistent output payload.
    final_result, final_mats = _stereo_calibrate_once(
        session_dir,
        cfg,
        camera_intrinsics_path,
        min_views_required=min_views_floor,
        min_corner_valid_ratio=min_corner_valid_ratio,
        allowed_view_ids=set(best_view_ids),
        save_reproj_overlays=True,
        use_refined_uv=use_refined_for_stereo,
        min_refined_corner_ratio=min_refined_corner_ratio,
    )
    reports = finalize_session_reports(session_dir, cfg)
    coverage = reports.get("coverage", {}) or {}
    final_result["coverage_ratio"] = float(coverage.get("coverage_ratio", 0.0))
    final_result["bins_covered_count"] = int(coverage.get("bins_covered_count", 0))
    final_result["bins_total"] = int(coverage.get("bins_total", 0))
    final_result["coverage"] = {
        "coverage_json": "results/coverage.json",
        "coverage_png": "results/coverage.png",
        "coverage_ratio": final_result["coverage_ratio"],
        "bins_covered_count": final_result["bins_covered_count"],
        "bins_total": final_result["bins_total"],
    }
    res_dir = session_dir / "results"
    if use_refined_for_stereo:
        (res_dir / "stereo_refined_pruned.json").write_text(json.dumps(_to_json_safe(final_result), indent=2))
        np.savez(res_dir / "stereo_refined_pruned.npz", **final_mats)
        session["results"] = {
            "rms_stereo": float(final_result.get("rms_stereo", 0.0)),
            "views_used": int(final_result.get("views_used", 0)),
            "updated_at": datetime.now().isoformat(),
            "path": "results/stereo_refined_pruned.json",
            "coverage_ratio": float(final_result.get("coverage_ratio", 0.0)),
            "bins_covered_count": int(final_result.get("bins_covered_count", 0)),
            "points_source": "refined_uv",
        }
        _save_session(session_dir, session)
    else:
        _write_stereo_outputs(
            session_dir,
            session,
            final_result,
            final_mats,
            json_name="stereo.json",
            npz_name="stereo.npz",
            update_latest=True,
            session_result_path="results/stereo_pruned.json",
        )
        (res_dir / "stereo_pruned.json").write_text(json.dumps(final_result, indent=2))
    # Additional prune-specific outputs.
    prune_report = {
        "initial_rms": float(history[0]["rms"]) if history else None,
        "final_rms": float(final_result.get("rms_stereo", 0.0)),
        "views_removed": removed_views,
        "history": history,
        "min_views_floor": min_views_floor,
        "min_improvement_px": min_improvement,
        "best_view_ids": best_view_ids,
    }
    (res_dir / "prune_report.json").write_text(json.dumps(prune_report, indent=2))
    final_result["prune_report"] = prune_report
    return final_result


def stereo_calibrate_refined(
    session_dir: Path,
    cfg: dict,
    camera_intrinsics_path: Path,
    *,
    recalibrate: bool = True,
    prune: bool = False,
) -> dict[str, Any]:
    """
    Offline refined projector calibration:
    - Refine UV corners in existing session views
    - Recalibrate with refined correspondences
    - Optional pruning loop
    - Save compare reports without overwriting baseline stereo.json
    """
    session = _load_session(session_dir)
    pcal = cfg.get("projector_calibration", {}) or {}
    cap_cfg = pcal.get("capture", {}) or {}
    min_views_cfg = int(cap_cfg.get("min_views", 10))
    min_corner_valid_ratio = float(cap_cfg.get("min_corner_valid_ratio", 0.90))
    refine_cfg = _uv_refine_config(cfg)
    min_refined_corner_ratio = float((pcal.get("uv_refinement", {}) or {}).get("min_view_valid_ratio", 0.85))

    refine_report = refine_uv_corners_for_session(session_dir, cfg)
    baseline = _find_existing_stereo_result(session_dir, "stereo.json")
    if baseline is None:
        baseline, _ = _stereo_calibrate_once(
            session_dir,
            cfg,
            camera_intrinsics_path,
            min_views_required=min_views_cfg,
            min_corner_valid_ratio=min_corner_valid_ratio,
            allowed_view_ids=None,
            save_reproj_overlays=False,
            use_refined_uv=False,
        )

    refined_result: dict[str, Any] | None = None
    refined_mats: dict[str, np.ndarray] | None = None
    refined_pruned: dict[str, Any] | None = None

    if recalibrate:
        dropped_refined: list[dict[str, Any]] = []
        refined_result, refined_mats = _stereo_calibrate_once(
            session_dir,
            cfg,
            camera_intrinsics_path,
            min_views_required=min_views_cfg,
            min_corner_valid_ratio=min_corner_valid_ratio,
            allowed_view_ids=None,
            save_reproj_overlays=True,
            use_refined_uv=True,
            min_refined_corner_ratio=min_refined_corner_ratio,
            dropped_views=dropped_refined,
        )
        refined_result["uv_refinement"] = {
            "config": _to_json_safe(asdict(refine_cfg)),
            "min_view_valid_ratio": float(min_refined_corner_ratio),
            "views_auto_dropped": _to_json_safe(dropped_refined),
            "report_path": "results/uv_refine_session_report.json",
        }
        (session_dir / "results" / "stereo_refined.json").write_text(
            json.dumps(_to_json_safe(refined_result), indent=2)
        )
        if refined_mats is not None:
            np.savez(session_dir / "results" / "stereo_refined.npz", **refined_mats)

        if prune:
            # Iterative pruning over refined correspondences only.
            valid_views = [v for v in session.get("views", []) if v.get("status") == "valid" and v.get("view_id")]
            current_view_ids = [str(v["view_id"]) for v in valid_views]
            min_views_floor = 8
            min_improvement = 0.05
            best_result: dict[str, Any] | None = None
            best_mats: dict[str, np.ndarray] | None = None
            best_view_ids: list[str] = []
            best_rms = float("inf")
            removed_views: list[str] = []
            history: list[dict[str, Any]] = []

            while len(current_view_ids) >= min_views_floor:
                result_i, mats_i = _stereo_calibrate_once(
                    session_dir,
                    cfg,
                    camera_intrinsics_path,
                    min_views_required=min_views_floor,
                    min_corner_valid_ratio=min_corner_valid_ratio,
                    allowed_view_ids=set(current_view_ids),
                    save_reproj_overlays=False,
                    use_refined_uv=True,
                    min_refined_corner_ratio=min_refined_corner_ratio,
                    dropped_views=None,
                )
                rms = float(result_i.get("rms_stereo", float("inf")))
                prev_best = best_rms
                improved = bool(np.isfinite(rms) and (rms < best_rms))
                if improved:
                    best_rms = rms
                    best_result = result_i
                    best_mats = mats_i
                    best_view_ids = list(current_view_ids)
                hist = {
                    "views_used": int(len(current_view_ids)),
                    "rms": rms,
                    "improved": improved,
                }
                if np.isfinite(prev_best):
                    hist["improvement_px"] = float(prev_best - rms)
                history.append(hist)
                if not improved:
                    hist["stop_reason"] = "rms_not_improving"
                    break
                if np.isfinite(prev_best) and (prev_best - rms) < min_improvement:
                    hist["stop_reason"] = "improvement_below_threshold"
                    break
                if len(current_view_ids) <= min_views_floor:
                    hist["stop_reason"] = "min_view_floor_reached"
                    break

                removable = [
                    pv for pv in (result_i.get("per_view_errors", []) or [])
                    if int(pv.get("corners_used", 0)) >= 48 and str(pv.get("view_id", "")) in current_view_ids
                ]
                if not removable:
                    hist["stop_reason"] = "no_removable_view_with_48_corners"
                    break
                worst = max(removable, key=lambda v: float(v.get("projector_reproj_error_px", -1.0)))
                worst_id = str(worst.get("view_id", ""))
                if not worst_id:
                    hist["stop_reason"] = "invalid_worst_view_id"
                    break
                if len(current_view_ids) - 1 < min_views_floor:
                    hist["stop_reason"] = "would_go_below_min_view_floor"
                    break
                hist["removed_view"] = worst_id
                hist["removed_projector_reproj_error_px"] = float(worst.get("projector_reproj_error_px", float("nan")))
                current_view_ids = [vid for vid in current_view_ids if vid != worst_id]
                removed_views.append(worst_id)

            if best_result is not None and best_mats is not None:
                final_refined_pruned, final_refined_pruned_mats = _stereo_calibrate_once(
                    session_dir,
                    cfg,
                    camera_intrinsics_path,
                    min_views_required=min_views_floor,
                    min_corner_valid_ratio=min_corner_valid_ratio,
                    allowed_view_ids=set(best_view_ids),
                    save_reproj_overlays=True,
                    use_refined_uv=True,
                    min_refined_corner_ratio=min_refined_corner_ratio,
                    dropped_views=None,
                )
                prune_refined_report = {
                    "initial_rms": float(history[0]["rms"]) if history else None,
                    "final_rms": float(final_refined_pruned.get("rms_stereo", 0.0)),
                    "views_removed": removed_views,
                    "history": history,
                    "min_views_floor": min_views_floor,
                    "min_improvement_px": min_improvement,
                    "final_view_ids": list(final_refined_pruned.get("view_ids", [])),
                }
                final_refined_pruned["uv_refinement"] = {
                    "config": _to_json_safe(asdict(refine_cfg)),
                    "min_view_valid_ratio": float(min_refined_corner_ratio),
                    "report_path": "results/uv_refine_session_report.json",
                }
                final_refined_pruned["prune_report"] = prune_refined_report
                (session_dir / "results" / "stereo_refined_pruned.json").write_text(
                    json.dumps(_to_json_safe(final_refined_pruned), indent=2)
                )
                (session_dir / "results" / "prune_refined_report.json").write_text(
                    json.dumps(_to_json_safe(prune_refined_report), indent=2)
                )
                np.savez(session_dir / "results" / "stereo_refined_pruned.npz", **final_refined_pruned_mats)
                refined_pruned = final_refined_pruned

    compare_summary = _build_refine_compare_outputs(
        session_dir=session_dir,
        before_result=baseline,
        refined_result=refined_result,
        refined_pruned_result=refined_pruned,
        refine_report=refine_report,
    )
    coverage_reports = finalize_session_reports(session_dir, cfg)

    session_report = {
        "before": _to_json_safe(baseline or {}),
        "after": _to_json_safe(refined_result or {}),
        "after_pruned": _to_json_safe(refined_pruned or {}),
        "refine_summary": _to_json_safe(refine_report),
        "compare_summary_path": "reports/refine_compare/compare_summary.json",
        "coverage_summary_path": "results/session_report.json",
    }
    (session_dir / "results" / "refine_session_report.json").write_text(
        json.dumps(_to_json_safe(session_report), indent=2)
    )

    response = {
        "session_id": session.get("session_id"),
        "recalibrated": bool(recalibrate),
        "pruned": bool(prune and refined_pruned is not None),
        "baseline_rms": float((baseline or {}).get("rms_stereo", float("nan"))),
        "refined_rms": float((refined_result or {}).get("rms_stereo", float("nan"))),
        "refined_pruned_rms": float((refined_pruned or {}).get("rms_stereo", float("nan"))),
        "best_view_ids": (refined_pruned or refined_result or baseline or {}).get("view_ids", []),
        "reports": {
            "uv_refine_session_report": "results/uv_refine_session_report.json",
            "stereo_refined": "results/stereo_refined.json" if refined_result is not None else None,
            "stereo_refined_pruned": "results/stereo_refined_pruned.json" if refined_pruned is not None else None,
            "prune_refined_report": "results/prune_refined_report.json" if refined_pruned is not None else None,
            "refine_session_report": "results/refine_session_report.json",
            "compare_summary": "reports/refine_compare/compare_summary.json",
            "coverage_json": "results/coverage.json",
            "coverage_png": "results/coverage.png",
            "session_report": "results/session_report.json",
        },
        "compare_summary": _to_json_safe(compare_summary),
        "coverage": {
            "coverage_ratio": float((coverage_reports.get("coverage", {}) or {}).get("coverage_ratio", 0.0)),
            "bins_covered_count": int((coverage_reports.get("coverage", {}) or {}).get("bins_covered_count", 0)),
            "bins_total": int((coverage_reports.get("coverage", {}) or {}).get("bins_total", 0)),
        },
    }
    return response


def _legacy_stereo_calibrate_write_back_compat(session_dir: Path, result: dict[str, Any]) -> None:
    """No-op placeholder to keep import stability if needed."""
    return None
