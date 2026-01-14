"""
main.py

Entry point for the structured-light fringe projection scanner.
"""

from __future__ import annotations

import threading

# Local modules used to build the scan pipeline and web UI.
from core.projector import Projector
from core.patterns import generate_psp_patterns, generate_midgrey_surface
from core.camera import CameraController
from core.scan import ScanController
from core.graycode import generate_graycode_patterns
from web.server import WebServer
from core.triangulation import load_camera_intrinsics, prepare_intrinsics_for_image, _format_K


# Configuration for scan frequencies, phases, and output locations.

FREQS = [4, 8, 16, 32]
N_PHASE = 4
SCAN_ROOT = "scans"
CALIB_ROOT = "calib"


def main():
    """
    Boot the projector/camera stack, build scan patterns, and run the web server.

    The projector event loop must stay on the main thread, so this function
    initializes everything up front and then blocks inside Projector.run().
    """
    print("[MAIN] Initialising projector (main thread)...")
    projector = Projector(fps=60)

    # Generate pattern stacks at the actual projector resolution.
    W, H = projector.width, projector.height
    print(f"[MAIN] Projector resolution: {W}x{H}")

    print("[MAIN] Generating PSP patterns (vertical)...")
    patterns_vert, raw_patterns_vert = generate_psp_patterns(
        width=W,
        height=H,
        freqs=FREQS,
        n_phase=N_PHASE,
        gamma_proj=None,
        orientation="vertical",
    )

    print("[MAIN] Generating PSP patterns (horizontal)...")
    patterns_horiz, raw_patterns_horiz = generate_psp_patterns(
        width=W,
        height=H,
        freqs=FREQS,
        n_phase=N_PHASE,
        gamma_proj=None,
        orientation="horizontal",
    )

    print("[MAIN] Generating mid-grey calibration surface...")
    midgrey_surface = generate_midgrey_surface(
        width=W,
        height=H,
        level=0.5,
        gamma_proj=None,
    )

    print("[MAIN] Generating GrayCode patterns for projector calibration...")
    graycode_set = generate_graycode_patterns(
        width=W,
        height=H,
        gamma_proj=None,
        brightness_scale=1.0,
    )

    # Start with a neutral frame to avoid sudden brightness jumps.
    projector.set_surface(midgrey_surface)

    # Initialise the camera controller.
    print("[MAIN] Initialising camera...")
    camera = CameraController(
        size_main=(1280, 720),
        size_lores=(640, 480),
        framerate=30,
    )
    # Sanity check: camera calibration must match the capture resolution.
    try:
        from pathlib import Path
        if Path("camera_intrinsics.npz").exists():
            Kc, dist, calib_size = load_camera_intrinsics("camera_intrinsics.npz")
            scan_size = tuple(int(x) for x in camera.size_main)
            allow_rescale = False  # Keep rescaling explicit in reconstruction via env var.
            prepare_intrinsics_for_image(Kc, dist, calib_size, scan_size, allow_rescale=allow_rescale, label="camera")
            print(f"[MAIN] Loaded camera_intrinsics.npz: calib_size={calib_size[0]}x{calib_size[1]} K=({_format_K(Kc)})")
    except Exception as e:
        print(f"[MAIN][WARN] Camera calibration sanity check failed: {e}")

    # Scan controller manages capture/unwrap/UV decode for each scan.
    scan_controller = ScanController(
        camera=camera,
        patterns=patterns_vert,
        patterns_horiz=patterns_horiz,
        midgrey_surface=midgrey_surface,
        set_surface_callback=projector.set_surface,
        freqs=FREQS,
        n_phase=N_PHASE,
        scan_root=SCAN_ROOT,
        calib_root=CALIB_ROOT,
        pattern_settle_time=0.15,
        graycode=graycode_set,
    )

    # Web server runs in a background thread so the projector loop stays on main.
    print("[MAIN] Starting Flask web server thread...")
    web = WebServer(
        camera=camera,
        scan_controller=scan_controller,
        set_surface_callback=projector.set_surface,
        graycode_set=graycode_set,
        midgrey_surface=midgrey_surface,
    )

    web_thread = threading.Thread(
        target=web.run,
        kwargs={"host": "0.0.0.0", "port": 5000},
        daemon=True,
    )
    web_thread.start()

    print("[MAIN] Startup complete. Entering projector loop.")

    # IMPORTANT: Projector loop must run on the main thread with KMSDRM.
    projector.run()   # Blocks forever.


if __name__ == "__main__":
    main()
