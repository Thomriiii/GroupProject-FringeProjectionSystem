# Fringe App

Fringe projection system for Raspberry Pi with:
- synchronized projector pattern display + camera capture
- multi-frequency PSP (`[1, 4, 16]` default)
- ROI-aware masking and temporal unwrapping
- UV correspondence map generation
- projector stereo calibration (camera ↔ projector)
- stage-3 triangulation and point cloud export

## Requirements

- Python 3.11+
- Raspberry Pi camera stack (`libcamera`/`Picamera2`) for hardware capture
- OpenCV (`cv2`) for checkerboard detection/calibration/triangulation utilities

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run Server

```bash
python -m fringe_app
```

Pages:
- Pipeline: `http://<pi-ip>:8000/`
- Camera calibration: `http://<pi-ip>:8000/calibration`
- Projector calibration: `http://<pi-ip>:8000/projector-calibration`
- Reconstruction: `http://<pi-ip>:8000/reconstruction`

## Core CLI

Single-orientation scan:
```bash
python -m fringe_app scan --n 8 --frequencies 1 4 16 --orientation vertical --settle-ms 150
```

Phase, unwrap, score:
```bash
python -m fringe_app phase --run <run_id>
python -m fringe_app unwrap --run <run_id> --use-roi auto
python -m fringe_app score --run <run_id>
```

Safe pipeline (single orientation):
```bash
python -m fringe_app pipeline-run-safe --print-hints
```

UV pipeline (vertical + horizontal + projector UV):
```bash
python -m fringe_app pipeline-run-uv --print-hints
```

3D pipeline (UV + triangulation):
```bash
python -m fringe_app pipeline-run-3d --print-hints
```

Reconstruct from an existing UV run:
```bash
python -m fringe_app reconstruct --run <run_id>_uv
```

Quality state:
```bash
python -m fringe_app quality-state --show
python -m fringe_app quality-state --reset
```

## Camera Calibration (Checkerboard)

Workflow:
1. Open `/calibration`
2. Create session
3. Capture checkerboard poses
4. Calibrate

Saved under:
- `data/calibration/camera/sessions/<session_id>/...`
- `data/calibration/camera/intrinsics_latest.json`

Only captures with successful checkerboard detection are used for solving.

## Projector Calibration (Stereo)

Workflow:
1. Open `/projector-calibration`
2. Create or continue session
3. Capture multiple checkerboard views
4. Run projector stereo calibration

Per-view data includes:
- camera frame + checkerboard corners
- UV artifacts (`u.npy`, `v.npy`, `mask_uv.npy`, `uv_meta.json`)
- camera↔projector corner correspondences
- diagnostics (`view_diag.json`, corner validity overlays)

Calibration outputs:
- `data/calibration/projector/sessions/<session_id>/results/stereo.json`
- `data/calibration/projector/sessions/<session_id>/results/stereo.npz`
- `data/calibration/projector/stereo_latest.json`

## Reconstruction Outputs

For UV run `data/runs/<run_id>_uv/`:

`reconstruction/`
- `xyz.npy` (H×W×3, NaN invalid)
- `xyzrgb.npy` (optional exported xyz+rgb points)
- `depth.npy`
- `depth_debug_fixed.png`
- `depth_debug_autoscale.png`
- `cloud.ply`
- `reprojection errors` (`reproj_err_cam.npy`, `reproj_err_proj.npy`)
- `reconstruction_meta.json`
- `masks/mask_uv.npy|png`, `masks/mask_recon.npy|png`

Triangulation is done in the camera coordinate frame with distortion-aware normalized-point triangulation.

## Data Layout

### Runs

Single-orientation run:
- `data/runs/<run_id>/captures/`
- `data/runs/<run_id>/phase/`
- `data/runs/<run_id>/unwrap/`
- `data/runs/<run_id>/roi/`
- `data/runs/<run_id>/normalise/`
- `data/runs/<run_id>/quality_report.json`

Combined UV run:
- `data/runs/<run_id>_uv/vertical/...`
- `data/runs/<run_id>_uv/horizontal/...`
- `data/runs/<run_id>_uv/projector_uv/...`
- `data/runs/<run_id>_uv/reconstruction/...`

### Calibration

- `data/calibration/camera/...`
- `data/calibration/projector/...`

## Configuration

Main config: `config/default.yaml`

Important sections:
- `scan`, `patterns`, `phase`, `unwrap`, `roi`
- `normalise`, `quality_gate`, `quality`, `uv_gate`
- `calibration`, `projector_calibration`
- `reconstruction`

## Notes

- The pipeline is strict about clipping and quality gates unless `--force` is used.
- Projector and camera calibration resolutions must match UV run expectations.
- If 3D reconstruction fails, check:
  - `data/calibration/projector/stereo_latest.json`
  - `data/runs/<run_id>_uv/projector_uv/uv_meta.json`
  - `data/runs/<run_id>_uv/reconstruction/reconstruction_meta.json`
