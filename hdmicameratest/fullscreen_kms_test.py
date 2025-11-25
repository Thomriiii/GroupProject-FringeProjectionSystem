import os, pygame, numpy as np

lines = 10

# Force headless HDMI
os.environ.setdefault("SDL_VIDEODRIVER", "kmsdrm")

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
W, H = pygame.display.get_window_size()
print(f"Display mode: {W}x{H}")

# Simple gradient so you can see intensity variation
x = np.linspace(0, 2*np.pi*lines, W, endpoint=False)
stripe = 0.5 + 0.5*np.sin(x)
img = (np.tile(stripe, (H, 1)) * 255).astype(np.uint8)
img = np.dstack([img, img, img])
surf = pygame.surfarray.make_surface(np.swapaxes(img, 0, 1))

screen.blit(surf, (0, 0))
pygame.display.flip()

input("Press Enter to exit...")
