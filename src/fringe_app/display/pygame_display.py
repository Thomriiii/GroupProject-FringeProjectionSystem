"""Pygame-based fullscreen projector display."""

from __future__ import annotations

import logging
import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont


_LOG = logging.getLogger("fringe_app")


@dataclass(slots=True)
class DisplaySurfaceInfo:
    requested_mode_w_h: tuple[int, int]
    created_surface_w_h: tuple[int, int]
    display_mode_w_h: tuple[int, int]
    window_w_h: tuple[int, int]
    surface_w_h: tuple[int, int]
    fullscreen: bool
    vsync: bool | None
    sdl_scaling_hints: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_mode_w_h": [int(self.requested_mode_w_h[0]), int(self.requested_mode_w_h[1])],
            "created_surface_w_h": [int(self.created_surface_w_h[0]), int(self.created_surface_w_h[1])],
            "display_mode_w_h": [int(self.display_mode_w_h[0]), int(self.display_mode_w_h[1])],
            "window_w_h": [int(self.window_w_h[0]), int(self.window_w_h[1])],
            "surface_w_h": [int(self.surface_w_h[0]), int(self.surface_w_h[1])],
            "fullscreen": bool(self.fullscreen),
            "vsync": self.vsync,
            "sdl_scaling_hints": dict(self.sdl_scaling_hints),
        }


def generate_projector_geometry_sanity_pattern(width: int, height: int) -> np.ndarray:
    """
    Build a geometry sanity pattern with a 1px border, 10px grid every 100px,
    and expected-resolution text label.
    """
    w = max(16, int(width))
    h = max(16, int(height))
    img = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 10 px grid lines every 100 px.
    grid_step = 100
    line_w = 10
    grid_color = (64, 200, 255)
    for x in range(0, w, grid_step):
        x1 = min(w - 1, x + line_w - 1)
        draw.rectangle((x, 0, x1, h - 1), fill=grid_color)
    for y in range(0, h, grid_step):
        y1 = min(h - 1, y + line_w - 1)
        draw.rectangle((0, y, w - 1, y1), fill=grid_color)

    # Border must stay exactly 1 pixel wide and white.
    draw.rectangle((0, 0, w - 1, h - 1), outline=(255, 255, 255), width=1)

    text = f"EXPECTED {w}x{h}"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.rectangle((8, 8, min(w - 8, 320), 36), fill=(0, 0, 0))
    draw.text((12, 12), text, fill=(255, 255, 255), font=font)
    return np.asarray(img, dtype=np.uint8)


def save_projector_geometry_sanity_pattern(path: Path, width: int, height: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = generate_projector_geometry_sanity_pattern(width=width, height=height)
    Image.fromarray(arr).save(path)
    return path


class PygameProjectorDisplay:
    """
    Fullscreen pygame display for projector output.
    """

    def __init__(self) -> None:
        self.screen: Optional[pygame.Surface] = None
        self._opened = False
        self._owner_thread_id: int | None = None
        self._requested_mode = (0, 0)
        self._fullscreen = False
        self._vsync: bool | None = None
        self._surface_info_logged = False

    def open(
        self,
        fullscreen: bool,
        screen_index: int | None,
        requested_mode: tuple[int, int] | None = None,
        vsync: bool = True,
    ) -> None:
        if self._opened:
            return
        if screen_index is not None:
            os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] = str(screen_index)
        try:
            pygame.display.init()
            flags = pygame.FULLSCREEN if fullscreen else 0
            mode = (0, 0) if requested_mode is None else (int(requested_mode[0]), int(requested_mode[1]))
            self._requested_mode = mode
            self._fullscreen = bool(fullscreen)
            self._vsync = None
            try:
                self.screen = pygame.display.set_mode(mode, flags, vsync=1 if vsync else 0)
                self._vsync = bool(vsync)
            except TypeError:
                self.screen = pygame.display.set_mode(mode, flags)
        except pygame.error as exc:
            raise RuntimeError(
                "Pygame display init failed. If you see EGL_BAD_ACCESS, try setting "
                "SDL_VIDEODRIVER to 'kmsdrm' (console) or 'wayland'/'x11' (desktop)."
            ) from exc
        pygame.display.set_caption("Fringe Projector")
        self._opened = True
        self._owner_thread_id = threading.get_ident()
        self._surface_info_logged = False

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

    def show_rgb(self, image: np.ndarray) -> None:
        if not self._opened or self.screen is None:
            raise RuntimeError("Display not opened")
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Expected HxWx3 RGB image")
        rgb = image[:, :, :3].astype(np.uint8, copy=False)
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        if surf.get_size() != self.screen.get_size():
            surf = pygame.transform.smoothscale(surf, self.screen.get_size())
        self.present_pattern(surf)

    def present_pattern(self, surface: pygame.Surface) -> None:
        if not self._opened or self.screen is None:
            raise RuntimeError("Display not opened")
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()
        pygame.event.pump()
        if not self._surface_info_logged:
            info = self.get_actual_display_surface_info()
            if info is not None:
                _LOG.info(
                    "Projector surface requested=%s surface=%s display_mode=%s window=%s fullscreen=%s vsync=%s hints=%s",
                    info["requested_mode_w_h"],
                    info["surface_w_h"],
                    info["display_mode_w_h"],
                    info["window_w_h"],
                    info["fullscreen"],
                    info["vsync"],
                    info["sdl_scaling_hints"],
                )
            self._surface_info_logged = True
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
        self._owner_thread_id = None
        self._surface_info_logged = False

    def get_projector_surface_size(self) -> tuple[int, int] | None:
        if not self._opened or self.screen is None:
            return None
        return tuple(int(v) for v in self.screen.get_size())

    def is_open(self) -> bool:
        return bool(self._opened and self.screen is not None)

    def owner_thread_id(self) -> int | None:
        return self._owner_thread_id

    def get_actual_display_surface_info(self) -> dict[str, Any] | None:
        if not self._opened or self.screen is None:
            return None
        disp_info = pygame.display.Info()
        display_mode = (
            int(getattr(disp_info, "current_w", 0) or 0),
            int(getattr(disp_info, "current_h", 0) or 0),
        )
        if display_mode[0] <= 0 or display_mode[1] <= 0:
            display_mode = tuple(int(v) for v in self.screen.get_size())
        try:
            window_size = tuple(int(v) for v in pygame.display.get_window_size())  # type: ignore[attr-defined]
        except Exception:
            window_size = tuple(int(v) for v in self.screen.get_size())
        surface_size = tuple(int(v) for v in self.screen.get_size())
        hints = {
            "scale_quality": str(
                os.environ.get("SDL_HINT_RENDER_SCALE_QUALITY")
                or os.environ.get("SDL_RENDER_SCALE_QUALITY")
                or "NOT_SET"
            ),
            "SDL_HINT_RENDER_SCALE_QUALITY": str(os.environ.get("SDL_HINT_RENDER_SCALE_QUALITY", "NOT_SET")),
            "SDL_RENDER_SCALE_QUALITY": str(os.environ.get("SDL_RENDER_SCALE_QUALITY", "NOT_SET")),
            "SDL_VIDEO_ALLOW_SCREENSAVER": str(os.environ.get("SDL_VIDEO_ALLOW_SCREENSAVER", "NOT_SET")),
            "SDL_VIDEO_X11_XRANDR": str(os.environ.get("SDL_VIDEO_X11_XRANDR", "NOT_SET")),
        }
        info = DisplaySurfaceInfo(
            requested_mode_w_h=(int(self._requested_mode[0]), int(self._requested_mode[1])),
            created_surface_w_h=surface_size,
            display_mode_w_h=display_mode,
            window_w_h=window_size,
            surface_w_h=surface_size,
            fullscreen=bool(self._fullscreen),
            vsync=self._vsync,
            sdl_scaling_hints=hints,
        )
        return info.to_dict()
