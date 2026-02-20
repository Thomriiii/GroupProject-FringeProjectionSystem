#!/usr/bin/env python3
"""Run a minimal mock scan and validate outputs."""

from __future__ import annotations

import time
from pathlib import Path
import os

from fringe_app.core.models import ScanParams
from fringe_app.patterns.generator import FringePatternGenerator
from fringe_app.display.pygame_display import PygameProjectorDisplay
from fringe_app.camera.mock import MockCamera
from fringe_app.io.run_store import RunStore
from fringe_app.web.preview import PreviewBroadcaster
from fringe_app.core.controller import ScanController


def main() -> int:
    data_dir = Path("mock_data")
    data_dir.mkdir(exist_ok=True)
    if not any(data_dir.iterdir()):
        print("[SELFTEST] mock_data is empty; add a few images to run this test.")
        return 1

    params = ScanParams(
        n_steps=4,
        frequency=8.0,
        orientation="vertical",
        brightness=1.0,
        resolution=(640, 480),
        settle_ms=10,
        save_patterns=False,
        preview_fps=5.0,
    )

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    display = PygameProjectorDisplay()
    camera = MockCamera()
    generator = FringePatternGenerator()
    store = RunStore(root="data/runs")
    preview = PreviewBroadcaster()

    controller = ScanController(
        display=display,
        camera=camera,
        generator=generator,
        store=store,
        preview=preview,
        config={},
    )

    run_id = controller.start_scan(params)
    print(f"[SELFTEST] run_id={run_id}")

    while True:
        status = controller.get_status()
        print(f"[SELFTEST] state={status['state']} step={status['step_index']}/{status['total_steps']}")
        if status["state"] in ("IDLE", "ERROR"):
            break
        time.sleep(0.2)

    run_dir = Path("data/runs") / run_id
    meta = run_dir / "meta.json"
    captures = list((run_dir / "captures").glob("*.png"))
    if not meta.exists():
        print("[SELFTEST] meta.json missing")
        return 1
    if len(captures) != params.N:
        print(f"[SELFTEST] expected {params.N} captures, got {len(captures)}")
        return 1

    print("[SELFTEST] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
