"""
main.py

ENTRY POINT for the structured-light fringe projection scanner.

This script must be executed directly.

Responsibilities:
  - Initialise projector (main thread only)
  - Generate PSP patterns + mid-grey
  - Initialise camera controller
  - Initialise scan controller
  - Start Flask web server (background thread)
  - Run projector loop forever (blocking)

System startup sequence:
    1. Projector initialises (fullscreen)
    2. Patterns load (using projector resolution)
    3. Camera initialises
    4. ScanController assembled
    5. Flask server starts on a worker thread
    6. Projector loop runs (main thread)
"""

from __future__ import annotations

import threading
import time

# Local modules
from projector import Projector
from patterns import generate_psp_patterns, generate_midgrey_surface
from camera import CameraController
from scan import ScanController
from server import WebServer


# =====================================================================
# CONFIG
# =====================================================================

FREQS = [4, 8, 16, 32]
N_PHASE = 4
SCAN_ROOT = "scans"


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

    print("[MAIN] Generating PSP patterns...")
    patterns, raw_patterns = generate_psp_patterns(
        width=W,
        height=H,
        freqs=FREQS,
        n_phase=N_PHASE,
        gamma_proj=None,     # projector gamma compensation disabled for now
    )

    print("[MAIN] Generating mid-grey calibration surface...")
    midgrey_surface = generate_midgrey_surface(
        width=W,
        height=H,
        level=0.5,
        gamma_proj=None,
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
        patterns=patterns,
        midgrey_surface=midgrey_surface,
        set_surface_callback=projector.set_surface,
        freqs=FREQS,
        n_phase=N_PHASE,
        scan_root=SCAN_ROOT,
        pattern_settle_time=0.15,
    )

    # ============================================================
    # Flask server (background thread)
    # ============================================================
    print("[MAIN] Starting Flask web server thread...")
    web = WebServer(camera=camera, scan_controller=scan_controller)

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
