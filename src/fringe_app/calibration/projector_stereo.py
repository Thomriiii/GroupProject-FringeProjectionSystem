"""Projector calibration (camera <-> projector stereo) workflow."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
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

from fringe_app.calibration.checkerboard import detect_checkerboard, save_image
from fringe_app.cli import cmd_phase, cmd_unwrap
from fringe_app.core.models import ScanParams
from fringe_app.phase.masking_post import build_unwrap_mask
from fringe_app.unwrap.temporal import unwrap_multi_frequency
from fringe_app.uv import phase_to_uv, save_uv_outputs


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
    p = _session_json_path(session_dir)
    if not p.exists():
        raise FileNotFoundError(f"Projector calibration session not found: {session_dir.name}")
    return json.loads(p.read_text())


def _save_session(session_dir: Path, data: dict[str, Any]) -> None:
    _session_json_path(session_dir).write_text(json.dumps(data, indent=2))


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
    session_cfg = {
        "checkerboard": pcal.get("checkerboard", {}),
        "projector": pcal.get("projector", {}),
        "capture": pcal.get("capture", {}),
        "uv_mask_policy": pcal.get("uv_mask_policy", {}),
        "uv_gate": pcal.get("uv_gate", {}),
    }
    (session_dir / "config.json").write_text(json.dumps(session_cfg, indent=2))
    session = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "config": session_cfg,
        "views": [],
        "results": None,
    }
    _save_session(session_dir, session)
    return session_id


def list_views(session_dir: Path) -> list[dict[str, Any]]:
    session = _load_session(session_dir)
    return list(session.get("views", []))


def delete_view(session_dir: Path, view_id: str) -> None:
    session = _load_session(session_dir)
    vdir = session_dir / "views" / view_id
    if vdir.exists():
        shutil.rmtree(vdir)
    session["views"] = [v for v in session.get("views", []) if v.get("view_id") != view_id]
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
    uv_cfg = cap_cfg.get("uv_sample", {}) or {}
    method = str(uv_cfg.get("method", "bilinear"))
    patch_radius = int(uv_cfg.get("patch_radius_px", 2))
    use_patch_median = bool(uv_cfg.get("use_patch_median", True))
    min_finite_samples = int(uv_cfg.get("min_finite_samples", 8))
    edge_margin = int(cap_cfg.get("max_corner_uv_edge_px", 2))
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

    h, w = u.shape
    for i, (x, y) in enumerate(cam):
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        finite_count = 0
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            reasons.append("oob")
            finite_counts.append(finite_count)
            continue

        us = float("nan")
        vs = float("nan")
        if use_patch_median and patch_radius > 0:
            x0 = max(0, xi - patch_radius)
            x1 = min(w, xi + patch_radius + 1)
            y0 = max(0, yi - patch_radius)
            y1 = min(h, yi + patch_radius + 1)
            patch_mask = mask_uv[y0:y1, x0:x1]
            pu = u[y0:y1, x0:x1]
            pv = v[y0:y1, x0:x1]
            vals_u = pu[patch_mask & np.isfinite(pu)]
            vals_v = pv[patch_mask & np.isfinite(pv)]
            finite_count = int(min(vals_u.size, vals_v.size))
            if finite_count >= max(1, min_finite_samples):
                us = float(np.median(vals_u))
                vs = float(np.median(vals_v))
        if not (np.isfinite(us) and np.isfinite(vs)):
            if method == "nearest":
                us = float(u[yi, xi])
                vs = float(v[yi, xi])
            else:
                us = _bilinear(u, float(x), float(y))
                vs = _bilinear(v, float(x), float(y))
            if finite_count <= 0 and bool(mask_uv[yi, xi]):
                finite_count = 1

        if not (np.isfinite(us) and np.isfinite(vs)):
            reasons.append("nan_uv")
            finite_counts.append(finite_count)
            continue
        if not bool(mask_uv[yi, xi]) and finite_count < max(1, min_finite_samples):
            reasons.append("mask_uv_false")
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
        "use_patch_median": use_patch_median,
        "patch_radius_px": patch_radius,
        "min_finite_samples": min_finite_samples,
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


def capture_view(session_dir: Path, controller, cfg: dict) -> dict[str, Any]:
    """
    Capture one projector calibration view:
    camera image + checkerboard + UV scan + sampled correspondences.
    """
    _require_cv2()
    session = _load_session(session_dir)
    existing_ids: set[str] = {
        str(v.get("view_id"))
        for v in session.get("views", [])
        if v.get("view_id")
    }
    for p in (session_dir / "views").glob("view_*"):
        if p.is_dir():
            existing_ids.add(p.name)
    view_idx = 1
    view_id = f"view_{view_idx:04d}"
    while view_id in existing_ids:
        view_idx += 1
        view_id = f"view_{view_idx:04d}"
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

    summary: dict[str, Any] = {
        "view_id": view_id,
        "created_at": datetime.now().isoformat(),
        "status": "failed",
        "reason": None,
        "checkerboard_found": False,
        "valid_corner_ratio": 0.0,
        "valid_corners": 0,
        "total_corners": int(corners_x * corners_y),
        "paths": {
            "camera": str(Path("views") / view_id / "camera.png"),
            "camera_overlay": str(Path("views") / view_id / "camera_overlay.png"),
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
    det, overlay = detect_checkerboard(frame, cols=corners_x, rows=corners_y, refine_subpix=True)
    save_image(view_dir / "camera_overlay.png", overlay)
    summary["checkerboard_found"] = bool(det.found)

    corners_payload = det.to_dict()
    corners_payload["checkerboard"] = {"corners_x": corners_x, "corners_y": corners_y}
    (view_dir / "camera_corners.json").write_text(json.dumps(corners_payload, indent=2))

    if not det.found:
        summary["reason"] = "checkerboard_not_found"
        (view_dir / "correspondences.json").write_text(
            json.dumps({"valid_ratio": 0.0, "error": "checkerboard_not_found"}, indent=2)
        )
        diag = {
            "view_id": view_id,
            "checkerboard": {"found": False, "corners_total": int(corners_x * corners_y)},
            "uv": {},
            "corner_sampling": {},
            "corner_validity_breakdown": {"ok": 0, "nan_uv": 0, "mask_uv_false": 0, "oob": 0, "near_edge": 0},
            "phase_quality": {},
            "projector_resolution": {},
            "hints": ["Checkerboard not detected. Move board, improve focus, and ensure full board is visible."],
        }
        (view_dir / "view_diag.json").write_text(json.dumps(diag, indent=2))
        session.setdefault("views", []).append(summary)
        _save_session(session_dir, session)
        _set_projector_calibration_light(controller, cfg_session, enabled=True)
        return summary

    corners = np.asarray(det.corners_px, dtype=np.float32).reshape(-1, 2)
    board_mask = _board_mask_from_corners(frame.shape[:2], corners)
    current_exposure = int((pcal.get("camera", {}) or {}).get("exposure_us", 2000))
    exposure_max = int((cfg_session.get("normalise", {}) or {}).get("exposure_max_extended_us", 12000))
    base_mask_policy = _calibration_uv_policy(cfg_session)

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
            uv_retry = cap_retry.setdefault("uv_sample", {})
            uv_retry["patch_radius_px"] = max(int((cap_cfg.get("uv_sample", {}) or {}).get("patch_radius_px", 2)), 3)
            uv_retry["min_finite_samples"] = max(6, int((cap_cfg.get("uv_sample", {}) or {}).get("min_finite_samples", 8)) - 2)
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
        try:
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

        phase_quality = uv_meta.get("phase_quality", {})
        proj_res = uv_meta.get("projector_resolution", {})
        mismatch = bool(proj_res.get("mismatch", False))
        hints = _build_view_hints(mismatch, uv_meta, breakdown, phase_quality)
        if b_board < low_b_threshold:
            hints.append(
                "Low modulation on board. Increase projector brightness/coverage, adjust board angle, or reduce ambient light."
            )
        hints = sorted(set(hints))

        diag = {
            "view_id": view_id,
            "checkerboard": {"found": True, "corners_total": int(corners.shape[0])},
            "uv": {
                "uv_gate_ok": bool(uv_meta.get("uv_gate_ok", False)),
                "valid_ratio": float(uv_meta.get("valid_ratio", 0.0)),
                "u_range": float(uv_meta.get("u_range", 0.0)),
                "v_range": float(uv_meta.get("v_range", 0.0)),
                "u_edge_pct": float(uv_meta.get("u_edge_pct", 0.0)),
                "v_edge_pct": float(uv_meta.get("v_edge_pct", 0.0)),
                "u_zero_pct": float(uv_meta.get("u_zero_pct", 0.0)),
                "v_zero_pct": float(uv_meta.get("v_zero_pct", 0.0)),
            },
            "corner_sampling": sampling_meta,
            "corner_validity_breakdown": breakdown,
            "phase_quality": phase_quality,
            "projector_resolution": {
                "configured": proj_res.get("configured", []),
                "pattern_surface": proj_res.get("pattern_surface", []),
                "mismatch": mismatch,
            },
            "board_quality": {
                "b_median_on_board": b_board,
                "b_median_on_board_vertical": b_board_v,
                "b_median_on_board_horizontal": b_board_h,
            },
            "hints": hints,
        }

        diag_name = "view_diag.json" if attempt == 0 else f"view_diag_retry_{attempt}.json"
        (view_dir / diag_name).write_text(json.dumps(diag, indent=2))

        corr = {
            "camera_corners_px": cam_pts.astype(float).tolist(),
            "projector_corners_px": proj_pts.astype(float).tolist(),
            "valid_mask": valid.astype(bool).tolist(),
            "valid_ratio": valid_ratio,
            "min_corner_valid_ratio": min_corner_valid_ratio,
            "uv_source_runs": run_info,
            "corner_reasons": reasons,
            "finite_samples_count": finite_counts,
            "projector_resolution": uv_meta.get("projector_resolution", {}),
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
            "uv_meta": uv_meta,
            "run_info": run_info,
            "corr": corr,
            "uv_dir": attempt_uv_dir,
            "reason": None,
        }
        if mismatch and strict_resolution_match:
            attempt_payload["reason"] = "PROJECTOR_RES_MISMATCH"
        elif not bool(uv_meta.get("uv_gate_ok", False)):
            attempt_payload["reason"] = "uv_gate_failed"
        elif valid_ratio < min_corner_valid_ratio:
            attempt_payload["reason"] = "insufficient_valid_uv_corners"

        if best_attempt is None or attempt_payload["valid_ratio"] > best_attempt["valid_ratio"]:
            best_attempt = attempt_payload

        if attempt_payload["reason"] is None:
            best_attempt = attempt_payload
            break

        final_reason = str(attempt_payload["reason"])
        clipped_v = float((phase_quality.get("vertical", {}) or {}).get("clipped_roi", 1.0))
        clipped_h = float((phase_quality.get("horizontal", {}) or {}).get("clipped_roi", 1.0))
        clipped_ok = max(clipped_v, clipped_h) <= 0.0
        mask_heavy = breakdown["mask_uv_false"] >= max(8, int(0.35 * len(reasons)))
        low_b = b_board < low_b_threshold

        if attempt >= max_retries:
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
        session.setdefault("views", []).append(summary)
        _save_session(session_dir, session)
        _set_projector_calibration_light(controller, cfg_session, enabled=True)
        return summary

    selected = best_attempt
    selected_uv_dir: Path = selected["uv_dir"]
    final_uv_dir = view_dir / "uv"
    if selected_uv_dir != final_uv_dir:
        if final_uv_dir.exists():
            shutil.rmtree(final_uv_dir)
        shutil.copytree(selected_uv_dir, final_uv_dir, dirs_exist_ok=True)

    (view_dir / "correspondences.json").write_text(json.dumps(selected["corr"], indent=2))
    save_image(view_dir / "overlay.png", _draw_correspondence_overlay(frame, np.asarray(selected["corr"]["camera_corners_px"]), np.asarray(selected["corr"]["projector_corners_px"]), np.asarray(selected["corr"]["valid_mask"], dtype=bool)))
    save_image(view_dir / "corner_validity_overlay.png", _draw_corner_validity_overlay(frame, np.asarray(selected["corr"]["camera_corners_px"]), selected["corr"]["corner_reasons"], selected["diag"]["corner_validity_breakdown"]))
    (view_dir / "view_diag.json").write_text(json.dumps(selected["diag"], indent=2))

    summary["valid_corner_ratio"] = float(selected["valid_ratio"])
    summary["valid_corners"] = int(np.count_nonzero(np.asarray(selected["corr"]["valid_mask"], dtype=bool)))
    summary["total_corners"] = int(len(selected["corr"]["valid_mask"]))
    summary["uv_meta"] = selected["uv_meta"]
    summary["uv_source_runs"] = selected["run_info"]
    summary["diag"] = {
        "corner_validity_breakdown": selected["diag"].get("corner_validity_breakdown", {}),
        "uv_gate_failed_checks": selected["uv_meta"].get("uv_gate_failed_checks", []),
        "hints": selected["diag"].get("hints", []),
    }
    proj_res_sel = (selected["uv_meta"].get("projector_resolution", {}) or {})
    pattern_surface = proj_res_sel.get("pattern_surface")
    if isinstance(pattern_surface, (list, tuple)) and len(pattern_surface) == 2:
        summary["effective_projector_size"] = [int(pattern_surface[0]), int(pattern_surface[1])]
    summary["status"] = "valid" if selected["reason"] is None else "invalid"
    summary["reason"] = selected["reason"]
    if selected["reason"] == "PROJECTOR_RES_MISMATCH":
        summary["status"] = "failed"
    if summary["reason"] and selected["diag"].get("hints"):
        hint0 = str(selected["diag"]["hints"][0])
        reason = str(summary["reason"])
        if reason in hint0:
            summary["reason"] = hint0
        elif hint0 in reason:
            summary["reason"] = reason
        else:
            summary["reason"] = f"{reason}: {hint0}"

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
                try:
                    (session_dir / "config.json").write_text(json.dumps(session_cfg, indent=2))
                except Exception:
                    pass
        except Exception:
            pass

    session.setdefault("views", []).append(summary)
    _save_session(session_dir, session)
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


def stereo_calibrate(session_dir: Path, cfg: dict, camera_intrinsics_path: Path) -> dict[str, Any]:
    cv = _require_cv2()
    session = _load_session(session_dir)
    pcal = cfg.get("projector_calibration", {}) or {}
    chk = pcal.get("checkerboard", {}) or {}
    proj_cfg = pcal.get("projector", {}) or {}
    cap_cfg = pcal.get("capture", {}) or {}
    min_views = int(cap_cfg.get("min_views", 10))
    min_corner_valid_ratio = float(cap_cfg.get("min_corner_valid_ratio", 0.90))

    corners_x = int(chk.get("corners_x", 8))
    corners_y = int(chk.get("corners_y", 6))
    square_size_m = float(chk.get("square_size_m", 0.01))
    proj_size_cfg = (int(proj_cfg.get("width", 1024)), int(proj_cfg.get("height", 768)))
    proj_size: tuple[int, int] | None = None

    valid_views = [v for v in session.get("views", []) if v.get("status") == "valid"]
    if len(valid_views) < min_views:
        raise ValueError(f"Need at least {min_views} valid projector-calibration views.")

    obj_template = _checkerboard_object_points(corners_x, corners_y, square_size_m)
    object_points: list[np.ndarray] = []
    cam_points: list[np.ndarray] = []
    proj_points: list[np.ndarray] = []
    view_ids: list[str] = []

    for view in valid_views:
        view_id = str(view["view_id"])
        corr_path = session_dir / "views" / view_id / "correspondences.json"
        if not corr_path.exists():
            continue
        corr = json.loads(corr_path.read_text())
        if float(corr.get("valid_ratio", 0.0)) < min_corner_valid_ratio:
            continue
        corr_res = corr.get("projector_resolution", {}) or {}
        patt = corr_res.get("pattern_surface")
        if isinstance(patt, (list, tuple)) and len(patt) == 2:
            cand = (int(patt[0]), int(patt[1]))
            if proj_size is None:
                proj_size = cand
            elif proj_size != cand:
                # Mixed projector sizes are not calibratable together.
                continue
        cam = np.asarray(corr.get("camera_corners_px", []), dtype=np.float32).reshape(-1, 2)
        proj = np.asarray(corr.get("projector_corners_px", []), dtype=np.float32).reshape(-1, 2)
        valid = np.asarray(corr.get("valid_mask", []), dtype=bool).reshape(-1)
        if cam.shape[0] != obj_template.shape[0] or proj.shape[0] != obj_template.shape[0] or valid.shape[0] != obj_template.shape[0]:
            continue
        idx = np.where(valid)[0]
        if idx.size < int(np.ceil(min_corner_valid_ratio * obj_template.shape[0])):
            continue
        object_points.append(obj_template[idx].astype(np.float32))
        cam_points.append(cam[idx].reshape(-1, 1, 2).astype(np.float32))
        proj_points.append(proj[idx].reshape(-1, 1, 2).astype(np.float32))
        view_ids.append(view_id)

    if len(object_points) < min_views:
        raise ValueError(f"Need at least {min_views} valid projector-calibration views.")

    k_cam, d_cam = _load_camera_intrinsics(camera_intrinsics_path)

    # Estimate projector intrinsics from board correspondences.
    if proj_size is None:
        proj_size = proj_size_cfg

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

        # Save camera-side reprojection overlay.
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
        "session_id": session.get("session_id"),
        "created_at": datetime.now().isoformat(),
        "views_used": int(len(object_points)),
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
    }

    res_dir = session_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "stereo.json").write_text(json.dumps(result, indent=2))
    np.savez(
        res_dir / "stereo.npz",
        camera_matrix=k1,
        camera_dist=d1,
        projector_matrix=k2,
        projector_dist=d2,
        R=r,
        T=t,
        E=e,
        F=f,
        R1=r1,
        R2=r2,
        P1=p1,
        P2=p2,
        Q=q,
    )

    # session_dir = .../calibration/projector/sessions/<id>; publish latest at
    # .../calibration/projector/stereo_latest.json
    proj_root = session_dir.parent.parent
    (proj_root / "stereo_latest.json").write_text(json.dumps(result, indent=2))
    session["results"] = {
        "rms_stereo": float(rms_stereo),
        "views_used": int(len(object_points)),
        "updated_at": datetime.now().isoformat(),
        "path": "results/stereo.json",
    }
    _save_session(session_dir, session)
    return result
