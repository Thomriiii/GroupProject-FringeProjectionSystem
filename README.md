# Fringe App v2

Structured-light 3D scanner running on a Raspberry Pi. Projects sinusoidal fringe patterns via a connected projector, captures with a Pi camera, and reconstructs 3D surface geometry using multi-frequency phase-shifting profilometry.

## Hardware

- Raspberry Pi (tested on Pi 4/5) with camera module
- HDMI projector connected as a second display
- Optional: motorised turntable with ESP8266 Wi-Fi controller

## Algorithm Pipeline

1. **Structured capture** — N-step PSP at 3 frequencies (1, 4, 16 cycles) × 2 orientations (vertical + horizontal)
2. **Phase extraction** — `atan2(-S, C)` wrapped phase per frequency
3. **Temporal unwrapping** — coarse→fine multi-frequency unwrap
4. **ROI detection** — object segmentation on black background
5. **UV mapping** — absolute phase → projector pixel coordinates
6. **Triangulation** — ray-ray intersection using stereo calibration (camera ↔ projector)
7. **Point cloud** — outlier removal, optional multi-scan merge via turntable
8. **Defect detection** — height-deviation segmentation on flattened surface

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```bash
python -m fringe_app_v2
```

Web UI at `http://<pi-ip>:5000/`

| Page | URL |
|------|-----|
| Main pipeline | `/` |
| Camera calibration | `/calibration-camera` |
| Projector calibration | `/calibration-projector` |
| Turntable | `/turntable` |

## CLI Commands

### Web server

```bash
python -m fringe_app_v2 [--config path/to/config.yaml] [--host 0.0.0.0] [--port 5000]
```

### Defect / analysis

```bash
python -m fringe_app_v2 run-defect        --run <run_id>
python -m fringe_app_v2 run-phase-defect  --run <run_id>
python -m fringe_app_v2 run-3d-defect     --run <run_id>
python -m fringe_app_v2 run-height-defect --run <run_id>
python -m fringe_app_v2 run-scratch-defect --run <run_id>
python -m fringe_app_v2 run-flatten       --run <run_id>
python -m fringe_app_v2 run-phase-debug   --run <run_id>
```

### Turntable calibration

```bash
# Print ChArUco board to physical media
python -m fringe_app_v2 turntable-print-board --output data/turntable/board.png

# Manual capture sequence (prompt per angle)
python -m fringe_app_v2 turntable-new-session --step 15
python -m fringe_app_v2 turntable-capture-sequence --session <session_id>

# Fully automated capture (ESP8266 turntable)
python -m fringe_app_v2 turntable-capture-auto [--ip 192.168.x.x] [--step 15]

# Analyse session and fit rotation axis
python -m fringe_app_v2 turntable-analyse --session <session_id>
python -m fringe_app_v2 turntable-report  --session <session_id>
```

## Calibration Workflows

### Camera intrinsics (checkerboard)

1. Open `/calibration-camera` in the web UI
2. Create a session
3. Capture ≥10 checkerboard poses from different angles
4. Click **Calibrate**

Outputs: `data/calibration/camera/intrinsics_latest.json`

### Projector stereo calibration

1. Open `/calibration-projector` in the web UI
2. Create or resume a session
3. Place ChArUco board in view and capture multiple positions
4. Click **Calibrate**

Per-view data saved under `data/calibration/projector/sessions/<session_id>/views/`.
Outputs:
- `data/calibration/projector/stereo_latest.json`
- `data/calibration/projector/stereo_latest.npz`

### Turntable axis calibration

Run `turntable-capture-auto` or `turntable-capture-sequence`, then `turntable-analyse`. The fitted axis vector and translation are written to:
- `data/turntable/sessions/<session_id>/axis_fit.json`
- `data/turntable/sessions/<session_id>/alignment.json`

Copy the axis values into `multi_scan.rotation_axis_vector` / `axis_x_m` / `axis_y_m` in `config/default.yaml`.

## Configuration

Main config: `fringe_app_v2/config/default.yaml`

Override at runtime:
```bash
python -m fringe_app_v2 --config /path/to/my_config.yaml
```

Key sections:

| Section | Controls |
|---------|----------|
| `web` | host, port, debug |
| `camera` | type (`picamera2`/`mock`), exposure, gain |
| `scan` | resolution, n_steps, frequencies, orientations, settle_ms |
| `phase` | saturation thresholds, modulation threshold, mask cleanup |
| `unwrap` | ROI-constrained unwrap, mask post-processing |
| `roi` | background percentile, morphology, post-cleanup |
| `reconstruction` | z range, reprojection error gate, statistical outlier removal |
| `defect` | depth threshold, area gates, edge suppression |
| `turntable` | IP, auto-discover, settle_ms |
| `multi_scan` | angles, rotation axis, ICP refinement |
| `turntable_calibration` | ChArUco board spec, capture settings |
| `calibration` | intrinsics / stereo paths |

## Data Layout

```
data/
  calibration/
    camera/
      intrinsics_latest.json
      sessions/<session_id>/
    projector/
      stereo_latest.json
      stereo_latest.npz
      sessions/<session_id>/
  turntable/
    sessions/<session_id>/

fringe_app_v2/runs/<run_id>/
  structured/
    vertical/freq_<f>/   step_000.png … step_N.png
    horizontal/freq_<f>/
  phase/
    vertical/freq_<f>/   phi_wrapped.npy  A.npy  B.npy  mask*.npy
    horizontal/freq_<f>/
  unwrap/
    vertical/   phi_abs.npy  mask_unwrap.npy  residual.npy
    horizontal/
  phase_quality/
  roi/          roi_mask.npy  roi_meta.json
  reconstruction/
    xyz.npy           (H×W×3, NaN = invalid)
    cloud.ply
    depth.npy
    reproj_err_cam.npy  reproj_err_proj.npy
    reconstruction_meta.json
  defect/
    defect_mask.npy
    defect_overlay.png
    defect_report.json
```

## Notes

- **Mock camera**: set `camera.type: mock` in config and place PNG/JPEG frames in `mock_data/` for development without hardware.
- **Projector display**: the app drives the projector via Pygame fullscreen on the second HDMI output. Set `display.screen_index` if display detection is unreliable.
- **Multi-scan**: requires a working turntable axis calibration. The rotation axis vector in `config/default.yaml` was calibrated from session `20260501_135837`.
- **3D reconstruction** requires both camera intrinsics and projector stereo calibration to be present under `data/calibration/`.
