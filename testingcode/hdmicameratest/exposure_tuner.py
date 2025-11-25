#!/usr/bin/env python3
import numpy as np
import cv2
import time
from picamera2 import Picamera2

# -------------------------
# CAMERA SETUP
# -------------------------
picam2 = Picamera2()
cfg = picam2.create_video_configuration(
    main={"size": (1280, 720), "format": "RGB888"},
    controls={
        "FrameRate": 30,
        "ExposureTime": 10000,     # start with AE off
        "AnalogueGain": 1.0
    }
)
picam2.configure(cfg)
picam2.set_controls({"AeEnable": False})     # disable auto exposure
picam2.start()
time.sleep(0.3)

print("=== Exposure Tuner ===")
print("Point the object under the projector and run repeatedly")
print("Press Ctrl+C to stop\n")

def analyse_frame():
    frame = picam2.capture_array("main")
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    mn  = float(gray.min())
    mx  = float(gray.max())
    mean = float(gray.mean())

    frac_lt5   = float((gray < 5).mean()*100)
    frac_lt10  = float((gray < 10).mean()*100)
    frac_gt245 = float((gray > 245).mean()*100)
    frac_gt250 = float((gray > 250).mean()*100)

    print("--------------------------------------")
    print(f"Min:   {mn:.1f}")
    print(f"Max:   {mx:.1f}")
    print(f"Mean:  {mean:.1f}")
    print(f"Dark < 5 DN:    {frac_lt5:.2f}%")
    print(f"Dark < 10 DN:   {frac_lt10:.2f}%")
    print(f"Bright > 245 DN:{frac_gt245:.2f}%")
    print(f"Bright > 250 DN:{frac_gt250:.2f}%")

    # ------------------------
    # Simple decision logic
    # ------------------------

    too_dark = frac_lt5 > 6 or mean < 90
    too_bright = frac_gt250 > 0.5 or mean > 130

    if too_dark and not too_bright:
        print("→ Suggestion: OPEN lens slightly (increase brightness)")
    elif too_bright and not too_dark:
        print("→ Suggestion: CLOSE lens slightly (reduce brightness)")
    elif too_bright and too_dark:
        print("→ Contrast problem: adjust lens AND check projector distance")
    else:
        print("✓ Exposure looks good")

    print("--------------------------------------\n")

try:
    while True:
        analyse_frame()
        time.sleep(1.0)

except KeyboardInterrupt:
    print("Stopping...")
    picam2.stop()
