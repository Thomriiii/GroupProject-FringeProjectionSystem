"""Entrypoint for the clean v2 application."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import fringe_app_v2
from fringe_app_v2.core.calibration import CalibrationService
from fringe_app_v2.core.camera import CameraService, CameraSettings, build_scan_params
from fringe_app_v2.core.patterns import PatternService
from fringe_app_v2.core.projector import ProjectorService, ProjectorSettings
from fringe_app_v2.pipeline.defect import run_defect_for_run
from fringe_app_v2.pipeline.flatten import run_flatten_for_run
from fringe_app_v2.phase_quality.diagnostics import run_phase_debug_for_run
from fringe_app_v2.utils.io import load_yaml
from fringe_app_v2.utils.math_utils import json_safe
from fringe_app_v2.web.server import create_app


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (Path(__file__).resolve().parent / "config" / "default.yaml")
    config = load_yaml(cfg_path)
    config["_config_path"] = str(cfg_path)
    return config


def build_app(config: dict[str, Any]):
    params = build_scan_params(config)
    camera = CameraService(CameraSettings.from_config(config), params)
    projector = ProjectorService(ProjectorSettings.from_config(config))
    patterns = PatternService()
    calibration = CalibrationService(config)
    camera.start_preview()
    return create_app(config, camera, projector, patterns, calibration)


_TURNTABLE_CAL_COMMANDS = {
    "turntable-print-board",
    "turntable-new-session",
    "turntable-capture",
    "turntable-capture-sequence",
    "turntable-capture-auto",
    "turntable-analyse",
    "turntable-report",
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {
        "run-defect",
        "run-phase-defect",
        "run-3d-defect",
        "run-height-defect",
        "run-scratch-defect",
        "run-flatten",
        "run-phase-debug",
    }:
        _main_defect_command(argv)
        return

    if argv and argv[0] in _TURNTABLE_CAL_COMMANDS:
        _main_turntable_cal(argv)
        return

    _main_server(argv)


def _main_server(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Run the clean fringe app v2 Flask UI")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    app = build_app(config)
    web = config.get("web", {}) or {}
    host = args.host or str(web.get("host", "0.0.0.0"))
    port = int(args.port or web.get("port", 5000))
    app.run(host=host, port=port, debug=bool(web.get("debug", False)), threaded=True)


def _main_defect_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Run V2 defect detection on an existing run")
    parser.add_argument(
        "command",
        choices=[
            "run-defect",
            "run-phase-defect",
            "run-3d-defect",
            "run-height-defect",
            "run-scratch-defect",
            "run-flatten",
            "run-phase-debug",
        ],
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run", type=Path, required=True, help="Run ID or path under storage.run_root")
    parser.add_argument("--orientation", default=None, help="Accepted for legacy defect commands; ignored")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run_dir = _resolve_run_dir(args.run, config)
    if args.command == "run-flatten":
        summary = run_flatten_for_run(run_dir, config)
    elif args.command == "run-phase-debug":
        summary = run_phase_debug_for_run(run_dir, config, CalibrationService(config))
    else:
        summary = run_defect_for_run(run_dir, config)
    print(json.dumps(json_safe({"run_dir": str(run_dir), "result": summary}), indent=2))


def _main_turntable_cal(argv: list[str]) -> None:
    """CLI dispatcher for turntable-calibration sub-commands."""
    import time as _time

    command = argv[0]
    rest = argv[1:]

    config = load_config(None)
    tt_cfg = config.get("turntable_calibration") or {}
    storage_root = Path(tt_cfg.get("storage_root", "data/turntable/sessions"))
    nominal_step = float(tt_cfg.get("nominal_step_deg", 15.0))

    def _load_intrinsics():
        import numpy as np
        path_str = (config.get("calibration") or {}).get(
            "camera_intrinsics_path",
            "data/calibration/camera/intrinsics_latest.json",
        )
        p = Path(path_str)
        if not p.exists():
            p = Path("data/calibration/camera/intrinsics_latest.json")
        if not p.exists():
            return None, None
        data = json.loads(p.read_text())
        K = np.array(data["camera_matrix"], dtype=np.float64)
        D = np.array(
            data.get("dist_coeffs", data.get("distortion_coefficients", [[0, 0, 0, 0, 0]])),
            dtype=np.float64,
        ).reshape(-1)
        return K, D

    if command == "turntable-print-board":
        p = argparse.ArgumentParser(prog=f"python -m fringe_app_v2 {command}")
        p.add_argument("--output", default="data/turntable/charuco_board.png")
        args = p.parse_args(rest)
        from fringe_app_v2.turntable.board import save_board
        save_board(args.output, tt_cfg.get("board") or {})

    elif command == "turntable-new-session":
        p = argparse.ArgumentParser(prog=f"python -m fringe_app_v2 {command}")
        p.add_argument("--step", type=float, default=nominal_step)
        p.add_argument("--config", type=Path, default=None)
        args = p.parse_args(rest)
        if args.config:
            config = load_config(args.config)
            tt_cfg = config.get("turntable_calibration") or {}
            storage_root = Path(tt_cfg.get("storage_root", "data/turntable/sessions"))
        from fringe_app_v2.turntable.session import new_session
        session = new_session(storage_root, args.step)
        print(f"Session created: {session.session_id}")
        print(f"  Root: {session.root}")

    elif command == "turntable-capture":
        p = argparse.ArgumentParser(prog=f"python -m fringe_app_v2 {command}")
        p.add_argument("--session", required=True)
        p.add_argument("--angle", type=float, required=True)
        p.add_argument("--note", default="")
        args = p.parse_args(rest)

        from fringe_app_v2.turntable.session import load_session, add_frame
        from fringe_app_v2.turntable.charuco_pose import process_frame
        import cv2, numpy as np

        session = load_session(storage_root, args.session)
        settle_ms = int((tt_cfg.get("capture") or {}).get("settle_ms_after_rotation", 300))

        cam_svc = CameraService(
            __import__("fringe_app_v2.core.camera", fromlist=["CameraSettings"]).CameraSettings.from_config(config),
            build_scan_params(config),
        )
        cam_svc.start()
        if settle_ms > 0:
            _time.sleep(settle_ms / 1000.0)
        image = cam_svc.capture(flush_frames=1)

        label = f"angle_{int(round(args.angle)):03d}"
        frame_dir = session.frame_dir(label)
        frame_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(frame_dir / "image.png"), image)

        K, D = _load_intrinsics()
        if K is not None:
            charuco_cfg = tt_cfg.get("charuco") or {}
            pose = process_frame(frame_dir, image, charuco_cfg, K, D,
                                 min_corners=int(charuco_cfg.get("min_corners", 6)))
        else:
            print("[warn] No camera intrinsics — skipping pose estimation")
            pose = {"ok": False, "n_corners": 0, "reprojection_error_px": None}

        rec = add_frame(session, args.angle, str(frame_dir / "image.png"), note=args.note)
        rec.charuco_ok = pose.get("ok", False) or (pose.get("n_corners", 0) > 0)
        rec.n_corners = int(pose.get("n_corners") or 0)
        rec.pose_ok = bool(pose.get("ok"))
        rec.reprojection_error_px = float(pose.get("reprojection_error_px") or 0.0)
        session.save()

        badge = "✅" if rec.status == "good" else ("⚠️" if rec.status == "warning" else "❌")
        print(f"Captured angle={args.angle}°  corners={rec.n_corners}  {badge} {rec.status}")

    elif command == "turntable-capture-sequence":
        p = argparse.ArgumentParser(prog=f"python -m fringe_app_v2 {command}")
        p.add_argument("--session", required=True)
        args = p.parse_args(rest)

        from fringe_app_v2.turntable.session import load_session, add_frame
        from fringe_app_v2.turntable.charuco_pose import process_frame
        import cv2

        session = load_session(storage_root, args.session)
        total = float(tt_cfg.get("total_degrees", 360))
        step = session.nominal_step_deg
        angles = [step * i for i in range(int(round(total / step)))]
        settle_ms = int((tt_cfg.get("capture") or {}).get("settle_ms_after_rotation", 300))

        cam_svc = CameraService(
            __import__("fringe_app_v2.core.camera", fromlist=["CameraSettings"]).CameraSettings.from_config(config),
            build_scan_params(config),
        )
        cam_svc.start()
        K, D = _load_intrinsics()
        charuco_cfg = tt_cfg.get("charuco") or {}

        print(f"Session {session.session_id}  step={step}°  {len(angles)} angles")
        print("Rotate turntable manually and press Enter to capture, or 'q' to quit.\n")

        for angle in angles:
            inp = input(f"  → Rotate to {angle:.1f}° then press Enter (q=quit): ").strip()
            if inp.lower() == "q":
                break

            label = f"angle_{int(round(angle)):03d}"
            frame_dir = session.frame_dir(label)
            frame_dir.mkdir(parents=True, exist_ok=True)

            if settle_ms > 0:
                _time.sleep(settle_ms / 1000.0)
            image = cam_svc.capture(flush_frames=1)
            cv2.imwrite(str(frame_dir / "image.png"), image)

            if K is not None:
                pose = process_frame(frame_dir, image, charuco_cfg, K, D,
                                     min_corners=int(charuco_cfg.get("min_corners", 6)))
            else:
                pose = {"ok": False, "n_corners": 0, "reprojection_error_px": None}

            rec = add_frame(session, angle, str(frame_dir / "image.png"))
            rec.charuco_ok = pose.get("ok", False) or (pose.get("n_corners", 0) > 0)
            rec.n_corners = int(pose.get("n_corners") or 0)
            rec.pose_ok = bool(pose.get("ok"))
            rec.reprojection_error_px = float(pose.get("reprojection_error_px") or 0.0)
            session.save()
            badge = "✅" if rec.status == "good" else ("⚠️" if rec.status == "warning" else "❌")
            print(f"    {badge} corners={rec.n_corners}  {rec.status}\n")

        print(f"\nDone. {len(session.frames)} frames saved.")
        print(f"Run: python -m fringe_app_v2 turntable-analyse --session {session.session_id}")

    elif command == "turntable-analyse":
        p = argparse.ArgumentParser(prog=f"python -m fringe_app_v2 {command}")
        p.add_argument("--session", required=True)
        args = p.parse_args(rest)

        from fringe_app_v2.turntable.session import load_session
        from fringe_app_v2.turntable.axis_fit import run_analysis
        from fringe_app_v2.turntable.alignment import run_alignment
        from fringe_app_v2.turntable.report import write_report

        session = load_session(storage_root, args.session)
        print(f"Analysing {session.session_id} ({len(session.frames)} frames)…")
        axis = run_analysis(session.root, nominal_step_deg=session.nominal_step_deg)
        print(f"  Axis fit: {axis.get('message', 'failed')}")
        align = run_alignment(session.root)
        print(f"  Alignment: {align.get('n_ok', 0)}/{align.get('n_pairs', 0)} pairs OK")
        session.analysed = True
        session.save()
        report_path = write_report(session.root)
        print(f"  Report: {report_path}")

    elif command == "turntable-report":
        p = argparse.ArgumentParser(prog=f"python -m fringe_app_v2 {command}")
        p.add_argument("--session", required=True)
        args = p.parse_args(rest)

        from fringe_app_v2.turntable.session import load_session
        from fringe_app_v2.turntable.report import write_report

        session = load_session(storage_root, args.session)
        report_path = write_report(session.root)
        print(f"Report written: {report_path}")

    elif command == "turntable-capture-auto":
        p = argparse.ArgumentParser(prog=f"python -m fringe_app_v2 {command}")
        p.add_argument("--session", default=None, help="Session ID (creates new if omitted)")
        p.add_argument("--step", type=float, default=None, help="Step in degrees (default from config)")
        p.add_argument("--ip", default=None, help="Turntable IP (overrides config)")
        args = p.parse_args(rest)

        from fringe_app_v2.core.turntable import TurntableClient, discover_turntable, get_local_subnet
        from fringe_app_v2.turntable.session import new_session, load_session
        from fringe_app_v2.turntable.auto_capture import run_auto_capture
        from fringe_app_v2.turntable.axis_fit import run_analysis
        from fringe_app_v2.turntable.alignment import run_alignment
        from fringe_app_v2.turntable.report import write_report
        import numpy as np

        step = args.step or nominal_step
        total = float(tt_cfg.get("total_degrees", 360))
        settle_ms = int((tt_cfg.get("capture") or {}).get("settle_ms_after_rotation", 600))

        # Connect to turntable
        ip = args.ip or (config.get("turntable") or {}).get("ip")
        if not ip:
            print("No turntable IP in config — auto-discovering…")
            ip = discover_turntable(get_local_subnet())
        if not ip:
            print("Error: could not find turntable. Set turntable.ip in config or use --ip.")
            sys.exit(1)

        tt_port = int((config.get("turntable") or {}).get("port", 80))
        tt_timeout = float((config.get("turntable") or {}).get("timeout_s", 10.0))
        turntable = TurntableClient(ip, port=tt_port, timeout_s=tt_timeout)
        if not turntable.ping():
            print(f"Error: turntable at {ip} did not respond.")
            sys.exit(1)
        print(f"Turntable connected: {ip}")

        # Session
        if args.session:
            session = load_session(storage_root, args.session)
        else:
            session = new_session(storage_root, nominal_step_deg=step)
            print(f"Session created: {session.session_id}")

        # Camera
        from fringe_app_v2.core.camera import CameraService
        CameraSettings = __import__("fringe_app_v2.core.camera", fromlist=["CameraSettings"]).CameraSettings
        cam_svc = CameraService(CameraSettings.from_config(config), build_scan_params(config))
        cam_svc.start()

        K, D = _load_intrinsics()
        if K is None:
            print("[warn] No camera intrinsics — pose estimation will be skipped")

        charuco_cfg = tt_cfg.get("charuco") or {}
        n_angles = int(round(total / step))
        print(f"\nStarting auto-capture: {n_angles} frames  step={step}°  total={total}°")
        print(f"Settle time: {settle_ms} ms per rotation\n")

        records = run_auto_capture(
            session=session,
            turntable=turntable,
            camera=cam_svc,
            charuco_cfg=charuco_cfg,
            K=K, D=D,
            step_deg=step,
            total_deg=total,
            settle_ms=settle_ms,
        )

        n_good = sum(1 for r in records if r.status == "good")
        print(f"\nCapture complete: {len(records)} frames  {n_good} good")

        # Auto-analyse
        print("Running analysis…")
        axis = run_analysis(session.root, nominal_step_deg=step)
        align = run_alignment(session.root)
        session.analysed = True
        session.save()
        report_path = write_report(session.root)

        print(f"  {axis.get('message', '')}")
        print(f"  Alignment: {align.get('n_ok', 0)}/{align.get('n_pairs', 0)} pairs OK")
        print(f"  Report: {report_path}")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


def _resolve_run_dir(value: Path, config: dict[str, Any]) -> Path:
    if value.exists():
        return value.resolve()
    run_root = Path(str((config.get("storage", {}) or {}).get("run_root", "fringe_app_v2/runs")))
    if not run_root.is_absolute():
        run_root = fringe_app_v2.REPO_ROOT / run_root
    candidate = run_root / value
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Run does not exist: {value} or {candidate}")


if __name__ == "__main__":
    main()
