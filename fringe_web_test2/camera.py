"""
camera.py

Camera setup and auto-exposure routines for structured light scanning.

This version includes:
  - Picamera2 initialization
  - Grayscale/RGB capture helpers
  - Automatic object detection to determine the AE region
  - AE routine that locks exposure based on the detected object only
"""

from __future__ import annotations

import time
import threading
import numpy as np
import cv2
from picamera2 import Picamera2


class CameraController:
    """
    Wrapper for Picamera2 providing:
      - video configuration
      - automatic, object-focused AE using mid-grey projection
      - safe frame capture
    """

    # =====================================================================
    # INITIALIZATION
    # =====================================================================
    def __init__(self,
                 size_main=(1280, 720),
                 size_lores=(640, 480),
                 framerate=30):

        self.picam2 = Picamera2()
        self.camera_lock = threading.Lock()

        cfg = self.picam2.create_video_configuration(
            main={"size": size_main, "format": "RGB888"},
            lores={"size": size_lores, "format": "YUV420"},
            display=None,
            buffer_count=3,
            controls={"FrameRate": framerate},
        )

        self.picam2.configure(cfg)
        self.picam2.start()

        time.sleep(0.5)

        print("[CAMERA] Picamera2 initialized.")

    # =====================================================================
    # FRAME CAPTURE
    # =====================================================================

    def capture_rgb(self) -> np.ndarray:
        """Capture RGB frame from main stream."""
        with self.camera_lock:
            frame = self.picam2.capture_array("main")
        return frame

    def capture_gray(self) -> np.ndarray:
        """Capture grayscale frame."""
        frame = self.capture_rgb()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # =====================================================================
    # OBJECT DETECTION (for AE region)
    # =====================================================================

    def _detect_object_roi(self, gray: np.ndarray):
        """
        Detect the object in the scene using simple thresholding + contours.

        Returns
        -------
        (x0, y0, x1, y1) bounding box of detected object,
        or None if detection fails.
        """

        H, W = gray.shape

        # Normalize slightly to reduce noise
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Otsu threshold (adaptive)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("[CAMERA-AE] No object contours detected.")
            return None

        # Pick largest contour (assumed to be object)
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area < 500:  # too small
            print("[CAMERA-AE] Object contour found but too small.")
            return None

        x, y, w, h = cv2.boundingRect(cnt)

        print(f"[CAMERA-AE] Object detected: x={x}, y={y}, w={w}, h={h}")

        return (x, y, x + w, y + h)

    # =====================================================================
    # AUTO EXPOSURE WITH OBJECT DETECTION
    # =====================================================================

    def auto_expose_with_midgrey(
        self,
        set_midgrey_surface_callback,
        target_mean=120,
        tolerance=5,
        max_iters=6,
        settle_time=0.3,
        fallback_roi_frac=0.35,
    ):
        """
        Auto-exposure using only light reflected from the object.

        Steps:
          1. Project mid-grey
          2. Enable AE/AWB
          3. Detect object bounding box
          4. Use bounding box as AE area
          5. If detection fails → use central ROI
          6. Lock AE/AWB off after convergence
        """

        print("[CAMERA-AE] Starting auto exposure (object-detected region)...")

        # Step 1: project mid-grey frame
        set_midgrey_surface_callback()
        time.sleep(0.4)

        # Step 2: enable AE/AWB
        self.picam2.set_controls({"AeEnable": True, "AwbEnable": True})

        # Capture a frame for object detection
        gray_init = self.capture_gray()
        H, W = gray_init.shape

        # Try to detect object
        roi = self._detect_object_roi(gray_init)

        # If detection fails → fallback to center region
        if roi is None:
            roi_w = int(W * fallback_roi_frac)
            roi_h = int(H * fallback_roi_frac)
            x0 = (W - roi_w) // 2
            y0 = (H - roi_h) // 2
            x1 = x0 + roi_w
            y1 = y0 + roi_h

            print(f"[CAMERA-AE] Using fallback center ROI: {x0}:{x1}, {y0}:{y1}")
        else:
            x0, y0, x1, y1 = roi
            print(f"[CAMERA-AE] Using detected object ROI: {x0}:{x1}, {y0}:{y1}")

        # Step 3: iterative AE adjustment
        for i in range(max_iters):
            time.sleep(settle_time)
            gray = self.capture_gray()
            roi_pixels = gray[y0:y1, x0:x1]
            mean_val = roi_pixels.mean()

            print(f"[CAMERA-AE] Iter {i}: ROI mean={mean_val:.1f}")

            if abs(mean_val - target_mean) < tolerance:
                break

        # Step 4: read AE settings
        md = self.picam2.capture_metadata()
        exposure_us = md.get("ExposureTime")
        analogue_gain = md.get("AnalogueGain")

        print(f"[CAMERA-AE] AE settled → Exposure={exposure_us}us Gain={analogue_gain:.3f}")

        # Step 5: lock exposure and AWB
        self.picam2.set_controls({
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": exposure_us,
            "AnalogueGain": analogue_gain,
        })

        time.sleep(0.1)

        print("[CAMERA-AE] Exposure locked (object-focused).")

        return exposure_us, analogue_gain
# EOF