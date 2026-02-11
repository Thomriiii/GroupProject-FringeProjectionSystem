"""CLI commands for scan/phase/score/autotune."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from fringe_app.core.models import ScanParams
from fringe_app.patterns.generator import FringePatternGenerator
from fringe_app.display.pygame_display import PygameProjectorDisplay
from fringe_app.camera.picamera2_impl import Picamera2Camera
from fringe_app.camera.mock import MockCamera
from fringe_app.io.run_store import RunStore
from fringe_app.web.preview import PreviewBroadcaster
from fringe_app.core.controller import ScanController
from fringe_app.phase.psp import PhaseShiftProcessor, PhaseThresholds
from fringe_app.phase.metrics import score_mask
from fringe_app.vision.object_roi import detect_object_roi, ObjectRoiConfig
from fringe_app.overlays.generate import overlay_masks


def _load_config() -> dict:
    cfg_path = Path("config/default.yaml")
    if not cfg_path.exists():
        return {}
    import yaml
    return yaml.safe_load(cfg_path.read_text()) or {}


def _camera_from_cfg(cfg: dict):
    cam_cfg = cfg.get("camera", {})
    cam_type = cam_cfg.get("type", "picamera2")
    if cam_type == "mock":
        return MockCamera(data_dir=cam_cfg.get("mock_data", "mock_data"))
    return Picamera2Camera(
        lores_yuv_format=str(cam_cfg.get("lores_yuv_format", "nv12")),
        lores_uv_swap=bool(cam_cfg.get("lores_uv_swap", False)),
    )


def _build_params(args, cfg: dict) -> ScanParams:
    scan_cfg = cfg.get("scan", {})
    width = getattr(args, "width", None)
    height = getattr(args, "height", None)
    width = int(scan_cfg.get("width", 1024) if width is None else width)
    height = int(scan_cfg.get("height", 768) if height is None else height)
    return ScanParams(
        n_steps=int(getattr(args, "n", scan_cfg.get("n_steps", 4))),
        frequency=float(getattr(args, "frequency", scan_cfg.get("frequency", 8.0))),
        orientation=str(getattr(args, "orientation", scan_cfg.get("orientation", "vertical"))),
        brightness=float(getattr(args, "brightness", scan_cfg.get("brightness", 1.0))),
        resolution=(width, height),
        settle_ms=int(getattr(args, "settle_ms", scan_cfg.get("settle_ms", 50))),
        save_patterns=bool(getattr(args, "save_patterns", scan_cfg.get("save_patterns", False))),
        preview_fps=float(getattr(args, "preview_fps", scan_cfg.get("preview_fps", 10.0))),
        phase_convention=str(scan_cfg.get("phase_convention", "atan2(-S,C)")),
        exposure_us=getattr(args, "exposure_us", None),
        analogue_gain=getattr(args, "gain", None),
        awb_mode=getattr(args, "awb_mode", None),
        awb_enable=getattr(args, "awb_enable", None),
        iso=getattr(args, "iso", None),
        brightness_offset=float(getattr(args, "brightness_offset", scan_cfg.get("brightness_offset", 0.5))),
        contrast=float(getattr(args, "contrast", scan_cfg.get("contrast", 1.0))),
    )


def _phase_thresholds(cfg: dict, overrides: Dict[str, Any] | None = None) -> PhaseThresholds:
    phase_cfg = cfg.get("phase", {})
    perc = phase_cfg.get("debug_percentiles", [1, 99])
    base = PhaseThresholds(
        sat_low=float(phase_cfg.get("sat_low", 5)),
        sat_high=float(phase_cfg.get("sat_high", 250)),
        B_thresh=float(phase_cfg.get("B_thresh", 10)),
        A_min=float(phase_cfg.get("A_min", 0)),
        debug_percentiles=(float(perc[0]), float(perc[1])),
    )
    if overrides:
        for k, v in overrides.items():
            setattr(base, k, v)
    return base


def cmd_scan(args) -> int:
    cfg = _load_config()
    params = _build_params(args, cfg)
    display = PygameProjectorDisplay()
    camera = _camera_from_cfg(cfg)
    generator = FringePatternGenerator()
    store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
    preview = PreviewBroadcaster()

    controller = ScanController(
        display=display,
        camera=camera,
        generator=generator,
        store=store,
        preview=preview,
        config=cfg,
    )

    run_id = controller.start_scan(params)
    print(f"run_id={run_id}")

    while True:
        status = controller.get_status()
        if status["state"] in ("IDLE", "ERROR"):
            print(json.dumps(status))
            break
        time.sleep(0.2)

    return 0


def cmd_phase(args) -> int:
    cfg = _load_config()
    store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
    images = store.load_captures(args.run)
    params = _params_from_meta(args.run, store)
    thresholds = _phase_thresholds(cfg)
    processor = PhaseShiftProcessor()
    result = processor.compute_phase(images, params, thresholds)
    ref = store.load_reference_image(args.run)
    roi_cfg = ObjectRoiConfig()
    roi_res = detect_object_roi(ref, roi_cfg)
    store.save_roi(args.run, roi_res.roi_mask, roi_res.bbox, {"cfg": asdict(roi_cfg), "debug": roi_res.debug})
    roi_mask_for_score = None if roi_res.debug.get("roi_fallback") else roi_res.roi_mask
    score = score_mask(result.mask, result.B, roi_mask=roi_mask_for_score)
    result.debug.update({
        "roi_valid_ratio": score.roi_valid_ratio,
        "roi_largest_component_ratio": score.roi_largest_component_ratio,
        "roi_edge_noise_ratio": score.roi_edge_noise_ratio,
        "roi_b_median": score.roi_b_median,
        "roi_score": score.roi_score,
        "roi_fallback": roi_res.debug.get("roi_fallback", False),
    })
    store.save_phase_outputs(args.run, result)
    phase_dir = Path(store.root) / args.run / "phase"
    try:
        from fringe_app.phase import visualize as _viz
        valid_in_roi = result.mask & roi_res.roi_mask
        _viz.save_mask_png(valid_in_roi, str(phase_dir / "valid_in_roi.png"))
    except Exception:
        pass
    overlays_dir = Path(store.root) / args.run / "overlays"
    overlays_dir.mkdir(exist_ok=True)
    try:
        empty = np.zeros_like(roi_res.roi_mask, dtype=bool)
        overlay_masks(ref, roi_res.roi_mask, empty, overlays_dir / "roi_overlay.png")
        overlay_masks(ref, empty, result.mask, overlays_dir / "mask_overlay.png")
        overlay_masks(ref, roi_res.roi_mask, valid_in_roi, overlays_dir / "valid_in_roi_overlay.png")
    except Exception:
        pass
    print(json.dumps(result.debug, indent=2))
    return 0


def cmd_score(args) -> int:
    cfg = _load_config()
    store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
    phase_dir = Path(store.root) / args.run / "phase"
    if not phase_dir.exists():
        raise SystemExit("phase outputs not found; run phase first")
    mask = np.load(phase_dir / "mask.npy") if (phase_dir / "mask.npy").exists() else None
    if mask is None:
        mask_png = phase_dir / "mask.png"
        if not mask_png.exists():
            raise SystemExit("mask not found")
        from PIL import Image
        mask = np.array(Image.open(mask_png)) > 0
    run_dir = Path(store.root) / args.run
    roi_mask = None
    roi_path = run_dir / "roi" / "roi_mask.png"
    if roi_path.exists():
        from PIL import Image
        roi_mask = np.array(Image.open(roi_path)) > 0
        roi_meta_path = run_dir / "roi" / "roi_meta.json"
        if roi_meta_path.exists():
            roi_meta = json.loads(roi_meta_path.read_text())
            if roi_meta.get("debug", {}).get("roi_fallback"):
                roi_mask = None
    B = np.load(phase_dir / "B.npy")
    score = score_mask(mask, B, roi_mask=roi_mask)
    print(json.dumps(asdict(score), indent=2))
    _print_score(score)
    return 0


def cmd_overlay(args) -> int:
    cfg = _load_config()
    store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
    run_dir = Path(store.root) / args.run
    ref = store.load_reference_image(args.run)
    roi_path = run_dir / "roi" / "roi_mask.png"
    phase_dir = run_dir / "phase"
    mask_path = phase_dir / "mask.npy"
    if not roi_path.exists() or not mask_path.exists():
        raise SystemExit("roi or phase mask missing; run phase first")
    from PIL import Image
    roi_mask = np.array(Image.open(roi_path)) > 0
    mask = np.load(mask_path)
    valid_in_roi = mask & roi_mask
    overlays_dir = run_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)
    empty = np.zeros_like(roi_mask, dtype=bool)
    overlay_masks(ref, roi_mask, empty, overlays_dir / "roi_overlay.png")
    overlay_masks(ref, empty, mask, overlays_dir / "mask_overlay.png")
    overlay_masks(ref, roi_mask, valid_in_roi, overlays_dir / "valid_in_roi_overlay.png")
    print(f"overlays saved to {overlays_dir}")
    return 0


def _print_score(score):
    print(
        f"valid_ratio={score.valid_ratio:.3f} | largest_component_ratio={score.largest_component_ratio:.3f} | "
        f"edge_noise_ratio={score.edge_noise_ratio:.3f} | b_median={score.b_median:.2f} | score={score.score:.2f}"
    )
    print(
        f"roi_valid_ratio={score.roi_valid_ratio:.3f} | roi_largest_component_ratio={score.roi_largest_component_ratio:.3f} | "
        f"roi_edge_noise_ratio={score.roi_edge_noise_ratio:.3f} | roi_b_median={score.roi_b_median:.2f} | roi_score={score.roi_score:.2f}"
    )


def _params_from_meta(run_id: str, store: RunStore) -> ScanParams:
    run_dir = Path(store.root) / run_id
    meta = json.loads((run_dir / "meta.json").read_text())
    p = meta.get("params", {})
    width, height = p.get("resolution", (1024, 768))
    return ScanParams(
        n_steps=int(p.get("n_steps", p.get("N", 4))),
        frequency=float(p.get("frequency", 8.0)),
        orientation=str(p.get("orientation", "vertical")),
        brightness=float(p.get("brightness", 1.0)),
        resolution=(int(width), int(height)),
        settle_ms=int(p.get("settle_ms", 50)),
        save_patterns=bool(p.get("save_patterns", False)),
        preview_fps=float(p.get("preview_fps", 10.0)),
        phase_convention=str(p.get("phase_convention", "atan2(-S,C)")),
        exposure_us=p.get("exposure_us"),
        analogue_gain=p.get("analogue_gain"),
        awb_mode=p.get("awb_mode"),
        awb_enable=p.get("awb_enable"),
        iso=p.get("iso"),
        brightness_offset=float(p.get("brightness_offset", 0.5)),
        contrast=float(p.get("contrast", 1.0)),
    )


def _score_thresholds_from_images(images: List[np.ndarray], params: ScanParams, thresholds: PhaseThresholds) -> tuple[float, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    proc = PhaseShiftProcessor()
    res = proc.compute_phase(images, params, thresholds)
    score = score_mask(res.mask, res.B)
    return score.score, score.__dict__, res.mask, res.A, res.B


def cmd_autotune(args) -> int:
    cfg = _load_config()
    exp_root = Path("data/experiments") / datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root.mkdir(parents=True, exist_ok=True)

    target_valid = float(args.target_valid)
    min_largest = float(args.min_largest_component)
    max_edge = float(args.max_edge_noise)

    use_roi = getattr(args, "use_roi", "auto")

    def passes(score, roi_fallback: bool):
        if use_roi != "off" and not roi_fallback:
            cond1 = score.roi_valid_ratio >= 0.60 and score.roi_largest_component_ratio >= 0.85 and score.roi_edge_noise_ratio <= 0.20 and score.roi_b_median >= 20
            cond2 = score.roi_largest_component_ratio >= 0.75 and score.roi_valid_ratio >= 0.15
            return cond1 or cond2
        cond1 = score.valid_ratio >= target_valid and score.largest_component_ratio >= min_largest and score.edge_noise_ratio <= max_edge
        cond2 = score.largest_component_ratio >= 0.75 and score.valid_ratio >= 0.15
        return cond1 or cond2

    params = _build_params(args, cfg)

    best = None
    best_info = None
    attempts = 0
    attempt_records = []

    # Baseline scan
    run_id = _run_scan_for_autotune(params, cfg, exp_root, 0)
    if run_id is None:
        print("Baseline scan failed; aborting autotune.")
        return 1

    score, roi_info = _compute_score_for_run(run_id, cfg)
    _record_attempt(exp_root, 0, params, score, run_id=run_id, roi_meta=roi_info.get("roi_meta"))
    attempt_records.append((0, run_id, score))
    best = score
    best_info = {"run_id": run_id, "params": params.to_dict(), "score": asdict(score)}
    attempts += 1

    if passes(score, roi_info.get("roi_fallback", False)):
        _write_best(exp_root, best_info)
        _print_summary(exp_root, best_info, attempt_records)
        print("PASS")
        return 0

    # Threshold sweep
    thresholds_list = []
    for b in [2, 3, 4, 5, 7, 10, 12, 15, 20]:
        for s_lo in [0, 2, 5]:
            for s_hi in [245, 250, 253, 255]:
                thresholds_list.append({"B_thresh": b, "sat_low": s_lo, "sat_high": s_hi})

    store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
    images = store.load_captures(run_id)
    params_from_meta = _params_from_meta(run_id, store)

    for i, th in enumerate(thresholds_list, start=1):
        if attempts >= args.max_iters:
            _write_best(exp_root, best_info)
            _print_summary(exp_root, best_info, attempt_records)
            print("MAX_ITERS")
            return 0
        thresholds = _phase_thresholds(cfg, overrides=th)
        proc = PhaseShiftProcessor()
        res = proc.compute_phase(images, params_from_meta, thresholds)
        score = score_mask(res.mask, res.B, roi_mask=roi_info.get("roi_mask"))
        _record_attempt(exp_root, i, params_from_meta, score, run_id=run_id, thresholds=th, roi_meta=roi_info.get("roi_meta"))
        attempt_records.append((i, run_id, score))
        if best is None or score.score > best.score:
            best = score
            best_info = {"run_id": run_id, "params": params_from_meta.to_dict(), "score": asdict(score), "thresholds": th}
        if passes(score, roi_info.get("roi_fallback", False)):
            _write_best(exp_root, best_info)
            _print_summary(exp_root, best_info, attempt_records)
            print("PASS")
            return 0
        attempts += 1

    # Exposure/gain ramp
    exposures = [4000, 6000, 8000, 12000, 16000, 24000]
    gains = [1.0, 1.5, 2.0, 3.0, 4.0]

    attempt = len(thresholds_list) + 1
    for exp in exposures:
        for gain in gains:
            if attempts >= args.max_iters:
                _write_best(exp_root, best_info)
                _print_summary(exp_root, best_info, attempt_records)
                print("MAX_ITERS")
                return 0
            params.exposure_us = exp
            params.analogue_gain = gain
            run_id = _run_scan_for_autotune(params, cfg, exp_root, attempt)
            if run_id is None:
                attempt += 1
                attempts += 1
                continue
            score, roi_info = _compute_score_for_run(run_id, cfg)
            _record_attempt(exp_root, attempt, params, score, run_id=run_id, roi_meta=roi_info.get("roi_meta"))
            attempt_records.append((attempt, run_id, score))
            if score.score > best.score:
                best = score
                best_info = {"run_id": run_id, "params": params.to_dict(), "score": asdict(score)}
            if passes(score, roi_info.get("roi_fallback", False)):
                _write_best(exp_root, best_info)
                _print_summary(exp_root, best_info, attempt_records)
                print("PASS")
                return 0
            attempt += 1
            attempts += 1

    # Frequency fallback
    base_freq = params.frequency
    for freq in [base_freq / 2.0, base_freq / 4.0]:
        if freq < 1.0 or attempts >= args.max_iters:
            break
        params.frequency = freq
        run_id = _run_scan_for_autotune(params, cfg, exp_root, attempt)
        if run_id is None:
            attempt += 1
            attempts += 1
            continue
        score, roi_info = _compute_score_for_run(run_id, cfg)
        _record_attempt(exp_root, attempt, params, score, run_id=run_id, roi_meta=roi_info.get("roi_meta"))
        attempt_records.append((attempt, run_id, score))
        if score.score > best.score:
            best = score
            best_info = {"run_id": run_id, "params": params.to_dict(), "score": asdict(score)}
        if passes(score, roi_info.get("roi_fallback", False)):
            _write_best(exp_root, best_info)
            _print_summary(exp_root, best_info, attempt_records)
            print("PASS")
            return 0
        attempt += 1
        attempts += 1

    # Projector brightness/contrast sweep
    for contrast in [0.6, 0.8, 1.0]:
        for brightness in [0.2, 0.35, 0.5]:
            if attempts >= args.max_iters:
                _write_best(exp_root, best_info)
                _print_summary(exp_root, best_info, attempt_records)
                print("MAX_ITERS")
                return 0
            params.contrast = contrast
            params.brightness_offset = brightness
            run_id = _run_scan_for_autotune(params, cfg, exp_root, attempt)
            if run_id is None:
                attempt += 1
                attempts += 1
                continue
            score, roi_info = _compute_score_for_run(run_id, cfg)
            _record_attempt(exp_root, attempt, params, score, run_id=run_id, roi_meta=roi_info.get("roi_meta"))
            attempt_records.append((attempt, run_id, score))
            if score.score > best.score:
                best = score
                best_info = {"run_id": run_id, "params": params.to_dict(), "score": asdict(score)}
            if passes(score, roi_info.get("roi_fallback", False)):
                _write_best(exp_root, best_info)
                _print_summary(exp_root, best_info, attempt_records)
                print("PASS")
                return 0
            attempt += 1
            attempts += 1

    _write_best(exp_root, best_info)
    _print_summary(exp_root, best_info, attempt_records)
    print("MAX_ITERS")
    return 0


def _run_scan_for_autotune(params: ScanParams, cfg: dict, exp_root: Path, attempt: int) -> str | None:
    try:
        display = PygameProjectorDisplay()
        camera = _camera_from_cfg(cfg)
        generator = FringePatternGenerator()
        store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
        preview = PreviewBroadcaster()
        controller = ScanController(display=display, camera=camera, generator=generator, store=store, preview=preview, config=cfg)
        run_id = controller.start_scan(params)
        while True:
            status = controller.get_status()
            if status["state"] in ("IDLE", "ERROR"):
                break
            time.sleep(0.2)
        if status["state"] == "ERROR":
            print("scan error", status.get("last_error"))
            return None
        return run_id
    except Exception as exc:
        print(f"scan failed: {exc}")
        return None


def _compute_score_for_run(run_id: str, cfg: dict):
    store = RunStore(root=cfg.get("storage", {}).get("run_root", "data/runs"))
    images = store.load_captures(run_id)
    params = _params_from_meta(run_id, store)
    thresholds = _phase_thresholds(cfg)
    proc = PhaseShiftProcessor()
    res = proc.compute_phase(images, params, thresholds)
    ref = store.load_reference_image(run_id)
    roi_cfg = ObjectRoiConfig()
    roi_res = detect_object_roi(ref, roi_cfg)
    roi_meta = {"cfg": asdict(roi_cfg), "debug": roi_res.debug, "bbox": roi_res.bbox}
    store.save_roi(run_id, roi_res.roi_mask, roi_res.bbox, {"cfg": asdict(roi_cfg), "debug": roi_res.debug})
    roi_mask_for_score = None if roi_res.debug.get("roi_fallback") else roi_res.roi_mask
    score = score_mask(res.mask, res.B, roi_mask=roi_mask_for_score)
    res.debug.update({
        "roi_valid_ratio": score.roi_valid_ratio,
        "roi_largest_component_ratio": score.roi_largest_component_ratio,
        "roi_edge_noise_ratio": score.roi_edge_noise_ratio,
        "roi_b_median": score.roi_b_median,
        "roi_score": score.roi_score,
        "roi_fallback": roi_res.debug.get("roi_fallback", False),
    })
    store.save_phase_outputs(run_id, res)
    phase_dir = Path(store.root) / run_id / "phase"
    try:
        from fringe_app.phase import visualize as _viz
        valid_in_roi = res.mask & roi_res.roi_mask
        _viz.save_mask_png(valid_in_roi, str(phase_dir / "valid_in_roi.png"))
    except Exception:
        pass
    return score, {
        "roi_mask": roi_res.roi_mask,
        "roi_fallback": roi_res.debug.get("roi_fallback", False),
        "roi_meta": roi_meta,
    }


def _record_attempt(exp_root: Path, attempt: int, params: ScanParams, score, run_id: str | None = None, thresholds: Dict[str, Any] | None = None, roi_meta: Dict[str, Any] | None = None) -> None:
    attempt_dir = exp_root / f"attempt_{attempt:02d}"
    attempt_dir.mkdir(exist_ok=True)
    data = {
        "params": params.to_dict(),
        "score": asdict(score),
        "thresholds": thresholds,
        "run_id": run_id,
    }
    (attempt_dir / "attempt.json").write_text(json.dumps(data, indent=2))
    if run_id:
        (attempt_dir / "run_id.txt").write_text(str(run_id))
    if roi_meta is not None:
        (attempt_dir / "roi_meta.json").write_text(json.dumps(roi_meta, indent=2))


def _write_best(exp_root: Path, best_info: Dict[str, Any] | None) -> None:
    if best_info is None:
        return
    (exp_root / "best.json").write_text(json.dumps(best_info, indent=2))


def _print_summary(exp_root: Path, best_info: Dict[str, Any] | None, records: List[tuple]) -> None:
    print("attempt | run_id | score | valid_ratio | largest_component_ratio | edge_noise_ratio | b_median | roi_valid | roi_largest | roi_edge | roi_b")
    for attempt, run_id, score in records:
        print(
            f"{attempt:>7} | {run_id} | {score.score:>5.2f} | {score.valid_ratio:>10.3f} | "
            f"{score.largest_component_ratio:>23.3f} | {score.edge_noise_ratio:>15.3f} | {score.b_median:>7.2f} | "
            f"{score.roi_valid_ratio:>9.3f} | {score.roi_largest_component_ratio:>10.3f} | "
            f"{score.roi_edge_noise_ratio:>7.3f} | {score.roi_b_median:>6.2f}"
        )
    if best_info is not None:
        print(f"best.json: {exp_root / 'best.json'}")
        print(json.dumps(best_info, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fringe_app")
    sub = p.add_subparsers(dest="cmd")

    scan = sub.add_parser("scan")
    scan.add_argument("--n", type=int, default=4)
    scan.add_argument("--frequency", type=float, default=8.0)
    scan.add_argument("--orientation", type=str, default="vertical")
    scan.add_argument("--settle-ms", type=int, default=150)
    scan.add_argument("--exposure-us", type=int, default=None)
    scan.add_argument("--gain", type=float, default=None)
    scan.add_argument("--width", type=int, default=None)
    scan.add_argument("--height", type=int, default=None)
    scan.add_argument("--brightness", type=float, default=1.0)
    scan.add_argument("--brightness-offset", type=float, default=0.5)
    scan.add_argument("--contrast", type=float, default=1.0)

    phase = sub.add_parser("phase")
    phase.add_argument("--run", required=True)

    score = sub.add_parser("score")
    score.add_argument("--run", required=True)

    overlay = sub.add_parser("overlay")
    overlay.add_argument("--run", required=True)

    autotune = sub.add_parser("autotune-mask")
    autotune.add_argument("--n", type=int, default=4)
    autotune.add_argument("--frequency", type=float, default=8.0)
    autotune.add_argument("--orientation", type=str, default="vertical")
    autotune.add_argument("--settle-ms", type=int, default=150)
    autotune.add_argument("--target-valid", type=float, default=0.30)
    autotune.add_argument("--min-largest-component", type=float, default=0.60)
    autotune.add_argument("--max-edge-noise", type=float, default=0.35)
    autotune.add_argument("--max-iters", type=int, default=25)
    autotune.add_argument("--use-roi", type=str, default="auto", choices=["auto", "on", "off"])

    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "scan":
        return cmd_scan(args)
    if args.cmd == "phase":
        return cmd_phase(args)
    if args.cmd == "score":
        return cmd_score(args)
    if args.cmd == "overlay":
        return cmd_overlay(args)
    if args.cmd == "autotune-mask":
        return cmd_autotune(args)
    parser.print_help()
    return 0
