# Fringe Projection Scanner

This repository contains a structured-light fringe projection scanner prototype using a projector, a camera, and a Flask web UI. It includes two test branches (`fringe_web_test1`, `fringe_web_test2`) and experimental utilities for the HDMI + camera setup.

✅ Key features:
- Projector output via pygame (KMSDRM fullscreen) for HDMI projection
- Picamera2 integration for Raspberry Pi camera capture
- Structured-light patterns and scan pipeline
- Flask web UI to view live camera and trigger scans

---

## Quick start

1. Ensure you are running on a Linux machine with a connected HDMI display and a camera (e.g., Raspberry Pi + Picamera2).
2. Create and activate a Python virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install dependencies (example):
```bash
pip install numpy opencv-python flask pygame picamera2 scipy
```
Note: On Raspberry Pi you may need to use system package managers and the Raspberry Pi OS-specific installation for `picamera2` and its system dependencies (libcamera).

4. Run the scanner example (fringe_web_test2):
```bash
cd /home/pi/fringe/fringe_web_test2
python3 main.py
```

5. Open the web UI at http://<device-ip>:5000 and use the page to view the camera and trigger scans.

---

## Development / Important files
- `fringe_web_test2/` - Newer test code with modular structure and `main.py` entrypoint.
  - `main.py` - Startup script that initialises projector, camera, scan controller and starts the Flask web server
  - `server.py` - Flask web UI and live MJPEG endpoint
  - `camera.py`, `scan.py`, `psp.py`, `patterns.py`, `projector.py`, `unwrap.py`, `masking.py` - core modules
- `fringe_web_test1/` - older, single-file prototype that combines projector, camera, and Flask UI in `fringe_web.py`
- `hdmicameratest/` - helper scripts for display and camera tests

---

## Configuration
- By default `fringe_web_test2/main.py` uses `SCAN_ROOT = "scans"` as the directory where scan outputs are saved. You can change that constant if needed.

---

## Git / Scans
- The `scans/` folder contains large, generated scan data (e.g., `.npy` files). The repository `.gitignore` already ignores `scans/` and any `*/scans/` subfolders. That avoids accidentally committing large scan data.

---

## Troubleshooting
- If you have problems accessing the camera, ensure `picamera2` is configured correctly and that you can run simple `picamera2` capture scripts.
- If full-screen projector mode doesn't appear, check that `SDL_VIDEODRIVER` is set to `kmsdrm` or your system's display driver (the `Projector` class sets `SDL_VIDEODRIVER=kmsdrm` by default).

---

## Projector calibration (dot grid)
1. Calibrate the camera first at `/calib` (checkerboard). This writes `camera_intrinsics.npz`.
2. Open `/proj_calib` and capture multiple poses. Each pose:
   - Projects a full GrayCode set, locks exposure, captures the stack, and takes a bright checkerboard frame.
   - Decodes projector `(u,v)` per camera pixel, samples projector pixels at detected checkerboard corners, and saves to `calib/projector/session_*/pose_*/`.
3. After 6–10 diverse poses, click **Finish Projector Calibration**. This runs projector intrinsics + stereo calibration and writes `projector_intrinsics.npz` and `stereo_params.npz` (also copied to repo root).
4. Run `/scan` and “Reconstruct” — reconstruction will now use calibrated camera↔projector geometry.
