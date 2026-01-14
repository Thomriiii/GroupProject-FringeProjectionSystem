"""
projector.py

HDMI fullscreen projection via pygame + KMSDRM.

The projector loop must run on the main thread to satisfy KMS/DRM, so this
module exposes a thread-safe surface setter and a blocking render loop.
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
        """
        Initialize pygame in fullscreen mode and prepare the render loop.

        Parameters
        ----------
        fps : int
            Target refresh rate for the projector loop.
        """
        # Required for DRM/KMS operation.
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

    def set_surface(self, surface):
        """
        Set the next surface to be displayed.
        Thread-safe: may be called from worker threads (scan, AE, and so on).
        """
        with self.set_surface_lock:
            self.current_surface = surface

    def run(self):
        """
        Main projection loop.
        Must run on the main thread due to pygame/KMSDRM restrictions.

        Continuously blits whatever surface has been set via set_surface().
        """
        print("[PROJECTOR] Main loop started.")

        while True:
            # Fetch the surface safely.
            with self.set_surface_lock:
                surf = self.current_surface

            if surf is not None:
                self.screen.blit(surf, (0, 0))
                pygame.display.flip()

            self.clock.tick(self.fps)

    def fill_black(self):
        """
        Convenience: instantly fill screen with black.
        """
        black = pygame.Surface((self.width, self.height))
        black.fill((0, 0, 0))
        self.set_surface(black)
