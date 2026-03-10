#!/usr/bin/env python3
"""Display a low-brightness white background in fullscreen mode."""

from __future__ import annotations

import argparse
import os
import time

import pygame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dn",
        type=int,
        default=180,
        help="Grayscale digital number in [0, 255]. Lower is dimmer.",
    )
    parser.add_argument(
        "--screen-index",
        type=int,
        default=None,
        help="Optional SDL fullscreen display index.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to show frame. 0 keeps it up until Esc/q/Ctrl+C.",
    )
    parser.add_argument(
        "--no-vsync",
        action="store_true",
        help="Disable vsync request when creating the fullscreen surface.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dn = max(0, min(255, int(args.dn)))

    if args.screen_index is not None:
        os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] = str(int(args.screen_index))

    pygame.display.init()
    try:
        try:
            screen = pygame.display.set_mode(
                (0, 0),
                pygame.FULLSCREEN,
                vsync=0 if bool(args.no_vsync) else 1,
            )
        except TypeError:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("Low White Fullscreen")
        screen.fill((dn, dn, dn))
        pygame.display.flip()
        print(f"displaying low-white frame: dn={dn} size={screen.get_size()} fullscreen=True")

        start = time.monotonic()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 0
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return 0

            if float(args.duration) > 0.0 and (time.monotonic() - start) >= float(args.duration):
                return 0
            time.sleep(0.02)
    except KeyboardInterrupt:
        return 0
    finally:
        pygame.display.quit()
        pygame.quit()


if __name__ == "__main__":
    raise SystemExit(main())
