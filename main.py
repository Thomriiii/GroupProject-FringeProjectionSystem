"""
main.py

ENTRY POINT for the structured-light fringe projection scanner.
"""

from __future__ import annotations

import threading

# Local modules
from core.projector import Projector
from core.patterns import generate_psp_patterns, generate_midgrey_surface
from core.camera import CameraController
from core.scan import ScanController
from core.graycode import generate_graycode_patterns
from web.server import WebServer


# =====================================================================
# CONFIG
# =====================================================================

FREQS = [4, 8, 16, 32]
N_PHASE = 4
SCAN_ROOT = "scans"
CALIB_ROOT = "calib"


# =====================================================================
# MAIN INITIALISATION
# =====================================================================

def main():
    print("[MAIN] Initialising projector (main thread)...")
    projector = Projector(fps=60)

    # ============================================================
    # Generate patterns using projector resolution
    # ============================================================
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

    # Start by displaying mid-grey (safe & neutral)
    projector.set_surface(midgrey_surface)

    # ============================================================
    # Initialise camera controller
    # ============================================================
    print("[MAIN] Initialising camera...")
    camera = CameraController(
        size_main=(1280, 720),
        size_lores=(640, 480),
        framerate=30,
    )

    # ============================================================
    # Scan controller
    # ============================================================
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
    )

    # ============================================================
    # Flask server (background thread)
    # ============================================================
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

    # ============================================================
    # IMPORTANT: Projector loop must run in main thread
    # ============================================================
    projector.run()   # This never returns


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()
