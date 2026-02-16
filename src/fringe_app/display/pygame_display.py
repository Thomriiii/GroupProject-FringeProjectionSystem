"""Pygame-based fullscreen projector display."""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pygame


class PygameProjectorDisplay:
    """
    Fullscreen pygame display for projector output.
    """

    def __init__(self) -> None:
        self.screen: Optional[pygame.Surface] = None
        self._opened = False

    def open(self, fullscreen: bool, screen_index: int | None) -> None:
        if self._opened:
            return
        if screen_index is not None:
            os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] = str(screen_index)
        try:
            pygame.display.init()
            flags = pygame.FULLSCREEN if fullscreen else 0
            try:
                self.screen = pygame.display.set_mode((0, 0), flags, vsync=1)
            except TypeError:
                self.screen = pygame.display.set_mode((0, 0), flags)
        except pygame.error as exc:
            raise RuntimeError(
                "Pygame display init failed. If you see EGL_BAD_ACCESS, try setting "
                "SDL_VIDEODRIVER to 'kmsdrm' (console) or 'wayland'/'x11' (desktop)."
            ) from exc
        pygame.display.set_caption("Fringe Projector")
        self._opened = True

    def show_gray(self, image: np.ndarray) -> None:
        if not self._opened or self.screen is None:
            raise RuntimeError("Display not opened")

        if image.ndim != 2:
            raise ValueError("Expected 2D grayscale image")

        rgb = np.repeat(image[:, :, None], 3, axis=2)
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        if surf.get_size() != self.screen.get_size():
            # Scale pattern to fill the projector display.
            surf = pygame.transform.smoothscale(surf, self.screen.get_size())
        self.present_pattern(surf)

    def present_pattern(self, surface: pygame.Surface) -> None:
        if not self._opened or self.screen is None:
            raise RuntimeError("Display not opened")
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()
        pygame.event.pump()
        # Let compositor/driver settle before capture, improves display-camera sync.
        time.sleep(0.008)

    def pump(self) -> None:
        if not self._opened:
            return
        pygame.event.pump()

    def close(self) -> None:
        if not self._opened:
            return
        pygame.display.quit()
        self._opened = False
        self.screen = None
