# Projector Calibration v2

How to calibrate in 5 steps:

1. Open `/calibration/projector-v2`.
2. Capture views until coverage shows **Sufficient coverage**.
3. Inspect the grid; re-capture or delete rejected views.
4. Click **Solve** to run raw + pruned stereo solve.
5. Download the session zip from the page (or use the API endpoint) and use `export/stereo.json`.

Notes:
- Only accepted views are used in solve.
- All artifacts are stored under one session folder:
  `data/calibration/projector_v2/<session_id>/`.
- `export/stereo.json` keeps the reconstruction-compatible stereo schema.
