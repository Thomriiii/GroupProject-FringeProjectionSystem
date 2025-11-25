import os, pygame, numpy as np

# Prefer Wayland if available; fallback to KMSDRM or framebuffer
for drv in ("wayland", "x11", "kmsdrm", "fbcon"):
    os.environ["SDL_VIDEODRIVER"] = drv
    try:
        import pygame
        pygame.display.init()
        pygame.display.quit()
        print(f"Using SDL video driver: {drv}")
        break
    except pygame.error:
        continue
else:
    raise SystemExit("No suitable SDL video driver found.")


W, H = 1920, 1080   # Projector resolution
FREQ = 50           # Fringe cycles across the width
PHASE = 0.0         # radians
BINARY = True      # set True to show binary stripes

def sine_fringe(width, height, freq, phase):
    x = np.linspace(0, 2*np.pi*freq, width, endpoint=False)
    stripe = 0.5 + 0.5*np.sin(x + phase)  # 0..1
    img = np.tile(stripe[np.newaxis, :], (height, 1))
    img8 = (img*255).astype(np.uint8)
    return np.dstack([img8, img8, img8])  # RGB

def binary_fringe(width, height, freq, phase):
    x = np.linspace(0, 2*np.pi*freq, width, endpoint=False)
    stripe = (np.sin(x + phase) >= 0).astype(np.uint8)*255
    img = np.tile(stripe[np.newaxis, :], (height, 1))
    return np.dstack([img, img, img])

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)

    if BINARY:
        img = binary_fringe(W, H, FREQ, PHASE)
    else:
        img = sine_fringe(W, H, FREQ, PHASE)

    surf = pygame.surfarray.make_surface(np.rot90(img))  # surfarray expects (w,h)
    screen.blit(pygame.transform.rotate(surf, -90), (0, 0))  # undo rot to display correctly
    pygame.display.flip()

    # Wait for ESC to quit
    running = True
    clock = pygame.time.Clock()
    while running:
        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
            if e.type == pygame.QUIT:
                running = False
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
