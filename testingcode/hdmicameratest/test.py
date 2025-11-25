import cv2
import numpy as np
import os

# Force DRM/KMS
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "0"

W = 1920
H = 1080

# Create a sinusoidal pattern
lines = 10
x = np.linspace(0, 2*np.pi*lines, W, endpoint=False)
stripe = 0.5 + 0.5*np.sin(x)
img = (np.tile(stripe, (H, 1)) * 255).astype(np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Create a fullscreen window
cv2.namedWindow("HDMI", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("HDMI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.imshow("HDMI", img)
cv2.waitKey(0)
