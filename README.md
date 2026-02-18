# Fringe App

Raspberry Pi fringe projection capture + processing system.

Current pipeline supports:
- Pattern projection and synchronized camera capture
- Multi-frequency PSP phase compute
- ROI detection and mask generation
- Temporal unwrapping
- UV map generation (vertical + horizontal runs)
- Quality gating and safe pipeline execution
- Web UI with live preview, one-click pipeline run, downloads, and checkerboard camera calibration

## Requirements

- Python 3.11+
- Raspberry Pi + libcamera/Picamera2 for real capture
- OpenCV (`cv2`) for checkerboard calibration and corner overlays

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```bash
python -m fringe_app
```

Open:
- Main UI: `http://<pi-ip>:8000/`
- Calibration UI: `http://<pi-ip>:8000/calibration`

## Main UI

Main page is intentionally minimal:
- Start end-to-end pipeline scan button
- Live camera preview
- Run ZIP downloads

## Checkerboard Calibration UI

Workflow:
1. Open `/calibration`
2. Click `New Session`
3. Capture checkerboard images (varied poses/angles/distances)
4. Click `Run Calibrate`

Saved artifacts:
- `data/calibration/sessions/<session_id>/captures/capture_XXX.png`
- `data/calibration/sessions/<session_id>/captures/capture_XXX_overlay.png`
- `data/calibration/sessions/<session_id>/detections/capture_XXX.json`
- `data/calibration/sessions/<session_id>/intrinsics.json`
- `data/calibration/intrinsics_latest.json`

Important rule:
- Calibration uses only captures where checkerboard detection has `found=true`.
- If too few valid detections are available, calibration returns:
  - `Need at least 10 valid checkerboard detections.`

## CLI Commands

Basic capture:

```bash
python -m fringe_app scan --n 8 --frequencies 1 4 16 --orientation vertical --settle-ms 150
```

Phase + unwrap + score for existing run:

```bash
python -m fringe_app phase --run <run_id>
python -m fringe_app unwrap --run <run_id> --use-roi auto
python -m fringe_app score --run <run_id>
```

Safe full pipeline (single orientation):

```bash
python -m fringe_app pipeline-run-safe --print-hints
```

Safe UV pipeline (vertical + horizontal + UV outputs):

```bash
python -m fringe_app pipeline-run-uv --print-hints
```

Quality state:

```bash
python -m fringe_app quality-state --show
python -m fringe_app quality-state --reset
```

## Data Layout

### Regular scan runs

`data/runs/<run_id>/` contains:
- `captures/`
- `phase/`
- `unwrap/`
- `roi/`
- `overlays/`
- `normalise/`
- `quality_report.json`
- `meta.json`

For UV combined runs:
- `vertical/` and `horizontal/` sub-runs
- `projector_uv/` with `u.npy`, `v.npy`, `mask_uv.npy`, debug images, `uv_meta.json`

### Calibration

`data/calibration/` contains:
- `sessions/<session_id>/...`
- `intrinsics_latest.json`

## Key Config

Main configuration file: `config/default.yaml`

Notable sections:
- `scan` (steps, frequencies, settle/flush timing, retries)
- `patterns` (contrast, brightness offset, min intensity)
- `phase` (A/B thresholds, saturation threshold, mask cleanup)
- `unwrap` (unwrap mask post-processing)
- `roi` (ROI detection and post-processing)
- `normalise` (auto exposure/gain normalization policy)
- `quality_gate` / `quality` (phase+unwrap thresholds and staged residual policy)
- `uv_gate` (UV validation thresholds)
- `calibration` (checkerboard config and min valid detections)

## Notes

- For stable results, use manual camera controls via safe pipeline commands.
- If run quality fails, inspect `quality_report.json` and per-stage metadata.
- Calibration is checkerboard-only in this version.

## Out of Scope (Current)

- Projector calibration
- Camera-projector stereo calibration
- Triangulation / point clouds
- Charuco workflow
