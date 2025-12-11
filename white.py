"""
white.py

Display a bright white fullscreen image using pygame.
Press ESC or close the window to exit.
"""

import os
import sys
import pygame


def main():
    # Prefer KMS/DRM on Pi if available, but let existing env override
    os.environ.setdefault("SDL_VIDEODRIVER", "kmsdrm")

    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    width, height = pygame.display.get_window_size()
    print(f"[WHITE] Fullscreen {width}x{height}, showing white.")

    white = pygame.Surface((width, height))
    white.fill((255, 255, 255))

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        screen.blit(white, (0, 0))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
