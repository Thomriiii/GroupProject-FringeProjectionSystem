"""
projector.py

Handles HDMI fullscreen projection via pygame + KMSDRM.

This module:
  - initialises pygame in fullscreen/KMSDRM mode
  - provides a thread-safe interface for updating the *current* surface
  - runs a rendering loop in the main thread (required by KMSDRM)
  - continuously displays whatever surface is set

IMPORTANT:
  - Only the main thread may call pygame.display.flip() under KMS/DRM.
  - Therefore the projector must run its own main loop, and other
    modules communicate via set_surface() calls, which are thread-safe.
"""

from __future__ import annotations

import threading
import pygame
import time


class Projector:
    """
    Wrapper for a fullscreen pygame HDMI projector output.

    Attributes
    ----------
    screen : pygame.Surface
        The fullscreen output surface.
    width, height : int
        Output resolution.
    current_surface : pygame.Surface or None
        Surface currently set by other threads/modules.
    set_surface_lock : threading.Lock
        Ensures thread-safe update of current_surface.
    """

    def __init__(self, fps: int = 60):
        # Required for DRM/KMS operation
        import os
        os.environ.setdefault("SDL_VIDEODRIVER", "kmsdrm")

        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.width, self.height = pygame.display.get_window_size()

        self.fps = fps
        self.clock = pygame.time.Clock()

        self.set_surface_lock = threading.Lock()
        self.current_surface = None

        print(f"[PROJECTOR] Initialised at {self.width}x{self.height} (KMSDRM).")

    # =====================================================================
    # SURFACE SETTER
    # =====================================================================

    def set_surface(self, surface):
        """
        Set the next surface to be displayed.
        Thread-safe: may be called from worker threads (scan, AE, etc.).
        """
        with self.set_surface_lock:
            self.current_surface = surface

    # =====================================================================
    # MAIN LOOP
    # =====================================================================

    def run(self):
        """
        Main projection loop.
        Must run in the *main thread* due to pygame/KMSDRM restrictions.

        Continuously blits whatever surface has been set via set_surface().
        """
        print("[PROJECTOR] Main loop started.")

        while True:
            # Fetch surface safely
            with self.set_surface_lock:
                surf = self.current_surface

            if surf is not None:
                self.screen.blit(surf, (0, 0))
                pygame.display.flip()

            self.clock.tick(self.fps)

    # =====================================================================
    # OPTIONAL UTILITY
    # =====================================================================

    def fill_black(self):
        """
        Convenience: instantly fill screen with black.
        """
        black = pygame.Surface((self.width, self.height))
        black.fill((0, 0, 0))
        self.set_surface(black)
