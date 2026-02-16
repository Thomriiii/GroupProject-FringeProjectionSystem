# Fringe App

Minimal fringe projection capture system for Raspberry Pi using pygame for projector output and FastAPI for control.

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

Open the UI at `http://<pi-ip>:8000/`.

## Multi-Frequency Unwrap (Temporal)

You can capture multiple frequencies and compute an absolute phase map:

```bash
python -m fringe_app scan --n 4 --frequencies 4 16 --orientation vertical --settle-ms 150
python -m fringe_app phase --run <run_id>
python -m fringe_app unwrap --run <run_id> --use-roi auto
```

## Camera

- **Picamera2 (real camera)**: set `camera.type: picamera2` in `config/default.yaml`.
- **Mock (dev only)**: set `camera.type: mock` and place images in `mock_data/`.

## Projector Display Selection

- Use `display.screen_index` in `config/default.yaml` to select the target display.
- `display.fullscreen` controls fullscreen mode.

## Self Test

1. Start the app: `python -m fringe_app`.
2. Start a scan:

```bash
curl -X POST http://localhost:8000/api/scan/start \
  -H 'Content-Type: application/json' \
  -d '{"n_steps":4,"frequency":8.0,"orientation":"vertical","brightness":1.0,"settle_ms":10,"preview_fps":5}'
```

4. Verify output under `data/runs/<timestamp>/` with `captures/` and `meta.json`.

## ROI + Masking Notes

ROI:
- ROI is detected from the first capture frame (object brighter than black background).
- ROI mask is saved under `data/runs/<run_id>/roi/roi_mask.png`.

Masking:
- The phase valid mask uses: `valid = (B>=B_thresh) & (A>=A_min) & (max_frame<=sat_high)`.
- Low-saturation is ignored (dark troughs don’t invalidate).
- The stored `mask.png` is ROI-gated (valid pixels inside ROI only).
- `valid_in_roi.png` shows valid∩ROI pixels explicitly.

Example metrics:
- Before: `roi_valid_ratio ~ 0.004` (wrong denominator + low-sat rejection).
- After: `roi_valid_ratio ~ 0.55` (valid∩ROI / ROI).

## Why Not Multi-Frequency Yet

Multiple frequencies are for unwrapping/absolute phase, not for improving mask quality. Mask quality should be stable on a single frequency first. Higher frequencies also reduce modulation due to blur/MTF, which shrinks valid masks.
