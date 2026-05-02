"""
Projector calibration view capture.

For each calibration view:
  1. Lock camera to calibration exposure settings
  2. Run structured-light capture (phase-shifting patterns)
  3. Compute wrapped phase for each orientation × frequency
  4. Temporally unwrap to absolute phase
  5. Convert to projector UV coordinates
  6. Capture a white-light image for ChArUco detection
  7. Detect ChArUco corners
  8. Sample + refine UV at corner positions
  9. Gate the view (quality checks)
 10. Save all results to the view directory
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fringe_app_v2.calibration.camera.charuco import detect_charuco
from fringe_app_v2.calibration.projector.uv_refine import UvRefineConfig, sample_and_refine_uv
from fringe_app_v2.calibration.projector.view_gating import evaluate_view
from fringe_app_v2.pipeline.phase_to_projector import phase_to_projector_coords


def _phase_from_images(
    images: list[np.ndarray],
    params: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run phase shift processing on a list of captured frames.

    Returns:
        (phi_wrapped, mask) arrays
    """
    from fringe_app_v2.core.psp import PhaseShiftProcessor, PhaseThresholds

    thresholds = PhaseThresholds(
        sat_low=0,
        sat_high=250,
        B_thresh=7,
        A_min=15,
        debug_percentiles=(1.0, 99.0),
    )
    processor = PhaseShiftProcessor()
    result = processor.process(images, params, thresholds)
    return result.phi_wrapped, result.mask_for_unwrap


def _unwrap_phases(
    phases: list[np.ndarray],
    masks: list[np.ndarray],
    freqs: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Temporal multi-frequency unwrap. Returns (phi_abs, mask_unwrap)."""
    from fringe_app_v2.core.temporal_unwrap import unwrap_multi_frequency
    phi_abs, mask, _, _ = unwrap_multi_frequency(phases, masks, freqs)
    return phi_abs, mask


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.ndim == 3 and image.shape[2] >= 3:
        f = image.astype(np.float32)
        g = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
        return np.clip(np.rint(g), 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _draw_uv_overlay(image: np.ndarray, u_map: np.ndarray, v_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Colorise the UV map and blend over the image for a debug overlay."""
    if image.ndim == 2:
        bgr = np.stack([image, image, image], axis=2).astype(np.uint8)
    else:
        bgr = image[:, :, :3].astype(np.uint8).copy()

    h, w = u_map.shape
    u_norm = np.zeros_like(u_map, dtype=np.float32)
    v_norm = np.zeros_like(v_map, dtype=np.float32)

    if np.any(mask):
        u_vals = u_map[mask]
        v_vals = v_map[mask]
        u_min, u_max = float(np.nanmin(u_vals)), float(np.nanmax(u_vals))
        v_min, v_max = float(np.nanmin(v_vals)), float(np.nanmax(v_vals))
        if u_max > u_min:
            u_norm[mask] = (u_vals - u_min) / (u_max - u_min)
        if v_max > v_min:
            v_norm[mask] = (v_vals - v_min) / (v_max - v_min)

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = (u_norm * 120).astype(np.uint8)   # hue = U
    hsv[:, :, 1] = np.where(mask, 220, 0).astype(np.uint8)
    hsv[:, :, 2] = np.where(mask, (v_norm * 200 + 55).clip(0, 255), 0).astype(np.uint8)
    coloured = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    overlay = bgr.copy()
    overlay[mask] = (0.5 * bgr[mask].astype(np.float32) + 0.5 * coloured[mask].astype(np.float32)).clip(0, 255).astype(np.uint8)
    return overlay


def capture_calibration_view(
    view_dir: Path,
    camera: Any,
    projector: Any,
    patterns: Any,
    params: Any,
    config: dict[str, Any],
    charuco_cfg: dict[str, Any],
    K: np.ndarray | None,
    D: np.ndarray | None,
    projector_size: tuple[int, int],
    uv_refine_cfg: dict[str, Any] | None = None,
    gating_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Capture one projector calibration view.

    Runs structured-light capture, computes UV maps, detects ChArUco corners,
    samples UV at corner positions, and gates the result.

    Args:
        view_dir:       Directory to save all results (will be created)
        camera:         CameraService (already started)
        projector:      ProjectorService
        patterns:       PatternService
        params:         ScanParams (baseline scan parameters)
        config:         Full app config dict
        charuco_cfg:    ChArUco board configuration
        K, D:           Camera intrinsics (None → skip gating tilt estimate)
        projector_size: (width, height) of projector in pixels
        uv_refine_cfg:  UV refinement config dict
        gating_cfg:     View gating config dict

    Returns:
        view_record dict with {view_id, status, reason, hints, metrics, files, uv}
    """
    from fringe_app_v2.phase_quality.gamma import apply_gamma, gamma_from_config

    view_dir.mkdir(parents=True, exist_ok=True)
    uv_dir = view_dir / "uv"
    uv_dir.mkdir(parents=True, exist_ok=True)

    scan_cfg = config.get("scan", {}) or {}
    freqs = [float(f) for f in params.get_frequencies()]
    if len(freqs) < 2:
        raise ValueError("At least 2 frequencies are required for projector calibration")

    orientations = ["vertical", "horizontal"]
    settle_ms = int(scan_cfg.get("settle_ms", getattr(params, "settle_ms", 120)))
    first_ms = int(scan_cfg.get("settle_ms_first_step", 300))
    flush_step = int(scan_cfg.get("flush_frames_per_step", 1))
    gamma = gamma_from_config(config)

    # Apply calibration camera settings (brighter than scan defaults)
    cal_cam_cfg = (config.get("calibration", {}) or {}).get("camera", {}) or {}
    if cal_cam_cfg:
        camera.apply_controls({
            "ExposureTime": int(cal_cam_cfg.get("exposure_us", 12000)),
            "AnalogueGain": float(cal_cam_cfg.get("analogue_gain", 3.0)),
            "AwbEnable": bool(cal_cam_cfg.get("awb_enable", True)),
            "AeEnable": False,
        })
        time.sleep(0.3)

    # ── 1. Structured-light capture ───────────────────────────────────────────
    captured_images: dict[str, dict[str, list[np.ndarray]]] = {
        orient: {} for orient in orientations
    }

    for orientation in orientations:
        orient_params = replace(params, orientation=orientation)  # type: ignore[arg-type]
        for freq in freqs:
            sequence = patterns.sequence(orient_params, frequency=freq)
            frames: list[np.ndarray] = []
            for step, pattern in enumerate(sequence):
                projector.show_gray(apply_gamma(pattern, gamma))
                time.sleep((first_ms if step == 0 else settle_ms) / 1000.0)
                frame = camera.capture(flush_frames=flush_step)
                frames.append(frame)
            captured_images[orientation][freq] = frames

    # Save step images for provenance
    for orientation in orientations:
        orient_params = replace(params, orientation=orientation)  # type: ignore[arg-type]
        for freq in freqs:
            frames = captured_images[orientation][freq]
            step_dir = view_dir / "structured" / orientation / f"freq_{freq:.4f}"
            step_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(frames):
                cv2.imwrite(str(step_dir / f"step_{i:03d}.png"), frame)

    # ── 2. Phase computation ──────────────────────────────────────────────────
    phi_abs: dict[str, np.ndarray] = {}
    mask_abs: dict[str, np.ndarray] = {}

    for orientation in orientations:
        orient_params = replace(params, orientation=orientation)  # type: ignore[arg-type]
        phases: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for freq in freqs:
            imgs = captured_images[orientation][freq]
            phi_w, mask_w = _phase_from_images(imgs, orient_params)
            phases.append(phi_w)
            masks.append(mask_w)

        phi_u, mask_u = _unwrap_phases(phases, masks, freqs)
        phi_abs[orientation] = phi_u
        mask_abs[orientation] = mask_u

        # Save phase maps for debugging
        phase_dir = view_dir / "phase" / orientation
        phase_dir.mkdir(parents=True, exist_ok=True)
        np.save(phase_dir / "phi_abs.npy", phi_u.astype(np.float32))
        np.save(phase_dir / "mask.npy", mask_u.astype(bool))

    # ── 3. UV maps ────────────────────────────────────────────────────────────
    freq_semantics = str(scan_cfg.get("frequency_semantics", "cycles_across_dimension"))

    # Highest frequency used for UV (most accurate)
    freq_u = float(max(freqs))
    freq_v = float(max(freqs))

    uv_map = phase_to_projector_coords(
        phi_horizontal=phi_abs["horizontal"],
        phi_vertical=phi_abs["vertical"],
        mask_horizontal=mask_abs["horizontal"],
        mask_vertical=mask_abs["vertical"],
        projector_width=projector_size[0],
        projector_height=projector_size[1],
        frequency_u=freq_u,
        frequency_v=freq_v,
        frequency_semantics=freq_semantics,
    )

    np.save(uv_dir / "u_map.npy", uv_map.u.astype(np.float32))
    np.save(uv_dir / "v_map.npy", uv_map.v.astype(np.float32))
    np.save(uv_dir / "mask_uv.npy", uv_map.mask.astype(bool))

    # ── 4. White-light capture for ChArUco ───────────────────────────────────
    projector.show_level(200)   # near-white uniform illumination
    time.sleep(first_ms / 1000.0)
    white_img = camera.capture(flush_frames=flush_step)
    projector.show_level(0)     # turn off

    cv2.imwrite(str(view_dir / "image.png"), white_img)

    # ── 5. ChArUco detection ──────────────────────────────────────────────────
    det, overlay, _ = detect_charuco(white_img, charuco_cfg)
    cv2.imwrite(str(view_dir / "overlay.png"), overlay)
    charuco_dict = det.to_json_dict()
    (view_dir / "charuco.json").write_text(json.dumps(charuco_dict, indent=2))

    # ── 6. UV sampling at corners ─────────────────────────────────────────────
    uv_report: dict[str, Any] = {"projector_size": list(projector_size)}
    projector_corners: np.ndarray | None = None
    valid_mask: np.ndarray | None = None

    if det.found and det.corner_count > 0:
        refine_cfg = UvRefineConfig.from_dict(uv_refine_cfg)
        uv_out, valid, diag = sample_and_refine_uv(
            u_map=uv_map.u,
            v_map=uv_map.v,
            mask=uv_map.mask,
            corners_px=det.corners,
            projector_size=projector_size,
            cfg=refine_cfg,
        )
        projector_corners = uv_out
        valid_mask = valid

        uv_report.update({
            "projector_corners_px": uv_out.tolist(),
            "valid_mask": valid.tolist(),
            "n_valid": int(np.count_nonzero(valid)),
            "n_corners": int(det.corner_count),
            "uv_refine_diag": diag,
        })

        # UV overlay for visualisation
        uv_overlay = _draw_uv_overlay(white_img, uv_map.u, uv_map.v, uv_map.mask)
        if det.corners.shape[0] > 0:
            for (cx, cy), is_valid in zip(det.corners.reshape(-1, 2), valid):
                colour = (0, 255, 0) if is_valid else (0, 64, 255)
                cv2.circle(uv_overlay, (int(round(cx)), int(round(cy))), 4, colour, -1)
        cv2.imwrite(str(uv_dir / "uv_overlay.png"), uv_overlay)
    else:
        uv_report.update({"n_valid": 0, "n_corners": 0})

    # ── 7. View gating ────────────────────────────────────────────────────────
    gate = evaluate_view(
        marker_count=det.marker_count,
        corner_count=det.corner_count,
        corners_px=det.corners,
        image_shape_hw=(white_img.shape[0], white_img.shape[1]),
        gating_cfg=gating_cfg or {},
        object_points=det.object_points if det.found else None,
        camera_matrix=K,
        dist_coeffs=D,
    )

    # Also gate on minimum valid UV corners
    min_corners_gating = int((gating_cfg or {}).get("min_corners", 16))
    n_valid_uv = int(uv_report.get("n_valid", 0))
    if gate["accepted"] and n_valid_uv < min_corners_gating:
        gate["accepted"] = False
        gate["reasons"].append(
            f"Only {n_valid_uv} ChArUco corners have valid UV samples; need {min_corners_gating}."
        )
        gate["hints"].append("Ensure the structured-light pattern illuminates the ChArUco board.")

    view_report: dict[str, Any] = {
        "gate": gate,
        "uv": uv_report,
        "charuco": charuco_dict,
        "uv_map_meta": uv_map.meta,
    }
    (view_dir / "view_report.json").write_text(json.dumps(view_report, indent=2))

    view_record: dict[str, Any] = {
        "view_id": view_dir.name,
        "status": "accepted" if gate["accepted"] else "rejected",
        "reason": "; ".join(gate["reasons"]) if gate["reasons"] else None,
        "hints": gate["hints"],
        "metrics": gate["metrics"],
        "files": {
            "image": str(view_dir / "image.png"),
            "overlay": str(view_dir / "overlay.png"),
            "uv_overlay": str(uv_dir / "uv_overlay.png"),
            "charuco": str(view_dir / "charuco.json"),
            "view_report": str(view_dir / "view_report.json"),
        },
    }
    return view_record
