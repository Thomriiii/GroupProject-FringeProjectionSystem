"""Projector display service."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from fringe_app_v2.core.pygame_display import PygameProjectorDisplay


@dataclass(slots=True)
class ProjectorSettings:
    fullscreen: bool
    screen_index: int | None
    vsync: bool
    resolution: tuple[int, int]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ProjectorSettings":
        display = config.get("display", {}) or {}
        scan = config.get("scan", {}) or {}
        return cls(
            fullscreen=bool(display.get("fullscreen", True)),
            screen_index=None if display.get("screen_index") is None else int(display.get("screen_index")),
            vsync=bool(display.get("vsync", True)),
            resolution=(
                int(scan.get("projector_width", scan.get("width", 1024))),
                int(scan.get("projector_height", scan.get("height", 768))),
            ),
        )


class ProjectorService:
    def __init__(self, settings: ProjectorSettings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        # Queue of display commands processed by the dedicated display thread.
        self._cmd_queue: "queue.Queue[tuple]" = queue.Queue()
        self._display_thread: threading.Thread | None = None
        self._display_stop = threading.Event()
        self._display: PygameProjectorDisplay | None = None
        self._start_display_thread()

    def show_gray(self, image: np.ndarray) -> None:
        # Enqueue a synchronous display command executed on the display thread.
        ev = threading.Event()
        res: dict[str, Any] = {}
        self._cmd_queue.put(("show_gray", (np.asarray(image, dtype=np.uint8),), {}, ev, res))
        ev.wait(timeout=5.0)
        if ev.is_set() and res.get("exc") is None:
            return
        if res.get("exc") is not None:
            raise res["exc"]
        raise RuntimeError("Display show_gray timed out")

    def show_level(self, dn: int) -> None:
        width, height = self.settings.resolution
        frame = np.full((height, width), int(np.clip(dn, 0, 255)), dtype=np.uint8)
        self.show_gray(frame)

    def show_idle(self, config: dict[str, Any]) -> None:
        display = config.get("display", {}) or {}
        if not bool(display.get("idle_white_enabled", True)):
            self.close()
            return
        self.show_level(int(display.get("idle_white_dn", 255)))

    def close(self) -> None:
        # Request display thread to stop and close the display there.
        self._display_stop.set()
        if self._display_thread is not None:
            # enqueue a wake command so thread can exit promptly
            self._cmd_queue.put(("_exit", (), {}, None, None))
            self._display_thread.join(timeout=2.0)
        with self._lock:
            if self._display is not None:
                try:
                    self._display.close()
                except Exception:
                    pass

    def surface_size(self) -> tuple[int, int] | None:
        with self._lock:
            return self._display.get_projector_surface_size()

    def _ensure_open(self) -> None:
        # No-op: opening is managed by the display thread. If display isn't
        # available yet, the display thread will open it on first command.
        return

    def _start_display_thread(self) -> None:
        def _thread_main() -> None:
            disp = PygameProjectorDisplay()
            self._display = disp
            # Open display lazily when first real command arrives.
            while not self._display_stop.is_set():
                try:
                    cmd = self._cmd_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if not cmd:
                    continue
                name, args, kwargs, ev, res = cmd
                if name == "_exit":
                    break
                try:
                    if name == "show_gray":
                        if not disp.is_open():
                            disp.open(
                                fullscreen=self.settings.fullscreen,
                                screen_index=self.settings.screen_index,
                                requested_mode=self.settings.resolution,
                                vsync=self.settings.vsync,
                            )
                        disp.show_gray(*args, **(kwargs or {}))
                        disp.pump()
                    elif name == "show_rgb":
                        if not disp.is_open():
                            disp.open(
                                fullscreen=self.settings.fullscreen,
                                screen_index=self.settings.screen_index,
                                requested_mode=self.settings.resolution,
                                vsync=self.settings.vsync,
                            )
                        disp.show_rgb(*args, **(kwargs or {}))
                        disp.pump()
                    elif name == "close":
                        disp.close()
                except Exception as exc:
                    if res is not None:
                        res["exc"] = exc
                finally:
                    if ev is not None:
                        ev.set()

        self._display_thread = threading.Thread(target=_thread_main, daemon=True)
        self._display_thread.start()
