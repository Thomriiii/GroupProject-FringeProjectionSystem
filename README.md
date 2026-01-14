# Fringe Projection Scanner

Structured-light fringe projection scanner built around a projector, a camera, and a Flask UI.
The system generates phase-shift patterns, captures them with a camera, unwraps phase, and
triangulates camera and projector rays into a 3D point cloud.

## Hardware and OS
- Raspberry Pi or Linux host with Picamera2 support.
- HDMI-connected projector or external display.
- Camera calibrated at the same resolution used for scanning.

## Quick start
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install numpy opencv-python flask pygame picamera2 scipy
   ```
   On Raspberry Pi, install Picamera2 and libcamera from the system packages.
3. Run the scanner:
   ```bash
   cd GroupProject-FringeProjectionSystem
   python3 main.py
   ```
4. Open the UI: `http://<device-ip>:5000`

## How it works
- `core/patterns.py` generates sinusoidal PSP patterns for vertical and horizontal fringes.
- `core/camera.py` captures RGB/gray frames and locks exposure using a mid-grey reference.
- `core/psp.py` computes wrapped phase and quality masks per frequency.
- `core/unwrap.py` temporally unwraps phase from low to high frequency.
- `core/geometry.py` converts unwrapped phase into projector pixel coordinates (u, v).
- `core/triangulation.py` intersects camera and projector rays to form a point cloud.
- `core/scan.py` orchestrates capture, decoding, and output persistence.
- `web/server.py` provides the Flask UI, live MJPEG, and calibration routes.

## Calibration workflow
### Camera calibration (`/calib`)
1. Capture multiple checkerboard views (tilt and move across the frame).
2. Click **Finish** to solve intrinsics.
3. Outputs:
   - `camera_intrinsics.npz`
   - `calibration_report.txt`

### Projector calibration (`/proj_calib`)
1. Capture multiple poses. Each pose projects GrayCode frames, captures them, and
   uses a bright checkerboard frame to link projector pixels to board corners.
2. Click **Finish** to solve projector intrinsics and stereo extrinsics.
3. Outputs (session folder and repo root):
   - `projector_intrinsics.npz`
   - `stereo_params.npz`

### Self-check
Use `python -m calibration.selfcheck --session <session_dir>` to validate reprojection
error against captured poses.

## Scanning and reconstruction
- Start scans from the UI or by calling `ScanController.run_scan`.
- Each scan produces a timestamped folder under `scans/` containing:
  - `phase_final.npy`, `mask_final.npy`
  - `proj_u.npy`, `proj_v.npy`
  - debug images and quality maps
- Reconstruction is triggered from the UI or by calling
  `core.triangulation.reconstruct_3d_from_scan`.
  Outputs include `points_filtered.ply` and a simple single-view mesh.

## Diagnostics tools
- `tools/check_resolution_consistency.py` verifies scan and calibration resolutions.
- `tools/diagnose_undistortion.py` overlays grids and Hough lines for distortion checks.
- `tools/plane_fit.py` fits a plane to a point cloud to spot streak/fan artifacts.

## Configuration
- `main.py` controls scan defaults such as `FREQS`, `N_PHASE`, and output roots.
- `config/uv_convention.json` and `PROJ_UV_TRANSFORM` align scan UVs with calibration UVs.
- `config/fake_projector_config.json` supplies fallback projector parameters.
- Environment flags:
  - `ALLOW_INTRINSIC_RESCALE=1` to rescale intrinsics for pure image resize (no crop).
  - `PROJ_UV_TRANSFORM=<name>` to override UV axis convention.
  - `DIAGNOSE_PROJ_UV_TRANSFORM=1` to print candidate UV transforms.
  - `REQUIRE_PROJECTOR_CALIB=1` to disallow fake projector parameters.

## Project layout
- `core/`: capture, PSP, unwrapping, UV mapping, and reconstruction.
- `calibration/`: camera and projector calibration routines.
- `web/`, `templates/`, `static/`: Flask UI and assets.
- `tools/`: diagnostics and geometry checks.
- `scans/`, `calib/`: output folders created at runtime.
