# Fringe System Analysis

This document records the source-grounded behavior that `fringe_app_v2` must preserve. The new app is an orchestration and packaging layer around the proven calibration, phase, unwrapping, UV, and reconstruction math in `src/fringe_app`.

## Data Flow

### Camera Calibration

```text
checkerboard captures
  -> detect_checkerboard / calibrate_intrinsics
  -> data/calibration/camera/sessions/<session>/intrinsics.json
  -> data/calibration/camera/intrinsics_latest.json
```

The current camera calibration manager is `fringe_app.calibration.manager.CalibrationManager`. It stores calibration sessions below `data/calibration/camera/sessions/<session_id>/` and writes the selected latest intrinsics to `data/calibration/camera/intrinsics_latest.json`.

The loader contract used by reconstruction expects:

```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist_coeffs": [k1, k2, p1, p2, k3],
  "image_size": [width, height]
}
```

The active artifact in this repo is `data/calibration/camera/intrinsics_latest.json`; it contains a 3x3 camera matrix, 5 distortion coefficients, and `image_size` `[1024, 768]`.

### Projector Calibration v2

```text
camera intrinsics
  + accepted ChArUco projector-calibration views
  + per-view projector UV samples
  -> OpenCV projector intrinsics solve
  -> OpenCV stereoCalibrate(camera fixed, projector estimated)
  -> data/calibration/projector_v2/<session>/export/stereo.json
  -> compatibility copy: data/calibration/projector/stereo_latest.json
```

The v2 session store is `fringe_app.calibration_v2.session_store`. It creates sessions below `data/calibration/projector_v2/<session_id>/` with:

```text
session.json
views/<view_id>/charuco.json
views/<view_id>/view_report.json
solve/stereo_raw.json
solve/stereo_pruned.json
solve/stereo_raw.npz
solve/stereo_pruned.npz
export/stereo.json
```

`fringe_app.calibration_v2.stereo_solve.solve_session` gathers accepted views, loads `charuco.json` object/camera points and `view_report.json` projector corner samples, estimates the projector as an inverse camera, and solves camera-projector stereo extrinsics.

The reconstruction-compatible stereo schema is:

```json
{
  "schema_version": 1,
  "camera_matrix": [[...], [...], [...]],
  "camera_dist_coeffs": [...],
  "projector_matrix": [[...], [...], [...]],
  "projector_dist_coeffs": [...],
  "R": [[...], [...], [...]],
  "T": [tx, ty, tz],
  "E": [[...], [...], [...]],
  "F": [[...], [...], [...]],
  "rectification": {"R1": [...], "R2": [...], "P1": [...], "P2": [...], "Q": [...]},
  "projector": {"width": 1920, "height": 1080}
}
```

`R` and `T` transform camera-frame 3D points into projector-frame coordinates:

```text
X_projector = R * X_camera + T
```

The v2 export and `data/calibration/projector/stereo_latest.json` currently share the same core schema, so both must remain loadable.

### Structured Capture and Phase

```text
for each orientation and frequency:
  generate N sinusoidal projector patterns
  project pattern
  wait settle_ms
  flush stale camera frames
  capture step image
  save captures/f_<freq>/step_###.png
```

The canonical pattern generator is `fringe_app.patterns.generator.FringePatternGenerator`. For vertical patterns, phase varies along camera/projector `x`; for horizontal patterns, phase varies along `y`. Frequency semantics default to `cycles_across_dimension`.

The canonical PSP implementation is `fringe_app.phase.psp.PhaseShiftProcessor.compute_phase`. For an N-step stack `I_k(x,y)`:

```text
theta_k = 2*pi*k/N
A = mean_k(I_k)
C = sum_k(cos(theta_k) * I_k)
S = sum_k(sin(theta_k) * I_k)
phi_wrapped = atan2(-S, C)
B = (2/N) * sqrt(C^2 + S^2)
```

The sign convention is explicitly `atan2(-S,C)` and must not change.

Masking starts with:

```text
invalid_high = any step >= sat_high
mask_raw = not invalid_high and B >= B_thresh and A >= A_min
```

Low intensity troughs are intentionally not rejected by saturation logic. Additional mask products are saved for different consumers:

```text
mask_raw.npy
mask_clean.npy
mask_for_unwrap.npy
mask_for_defects.npy
mask_for_display.npy
clipped_any.npy
```

`mask_for_unwrap` is the mask consumed by temporal unwrapping. `mask_for_defects` is more inclusive and is intended for later defect analysis.

### Multi-Frequency Unwrapping

```text
wrapped phases at frequencies [f0, f1, ...]
  -> sort coarse-to-fine by frequency
  -> anchor lowest frequency in [0, 2*pi)
  -> propagate to each higher frequency with rounding
  -> unwrap/phi_abs.npy
  -> unwrap/mask_unwrap.npy
  -> unwrap/residual.npy
  -> unwrap/unwrap_meta.json
```

The canonical implementation is `fringe_app.unwrap.temporal.unwrap_multi_frequency`. For each coarse-to-fine step:

```text
r = f_high / f_low
k = round((r * Phi_low - phi_high) / (2*pi))
Phi_high = phi_high + 2*pi*k
residual = wrap_to_pi(r * Phi_low - Phi_high)
```

The valid mask at each stage is the intersection of the previous valid mask, current frequency mask, finite phase values, and ROI if enabled. Invalid pixels are stored as `NaN` in `phi_abs.npy`.

### Projector UV Mapping

```text
vertical absolute phase + horizontal absolute phase
  -> phase_to_uv
  -> projector_uv/u.npy
  -> projector_uv/v.npy
  -> projector_uv/mask_uv.npy
  -> projector_uv/uv_meta.json
```

The canonical mapper is `fringe_app.uv.phase_to_uv.phase_to_uv`.

For `cycles_across_dimension`:

```text
span_u = 2*pi*freq_u
span_v = 2*pi*freq_v
u = mod(phi_vertical - phase_origin_u, span_u) / span_u * projector_width
v = mod(phi_horizontal - phase_origin_v, span_v) / span_v * projector_height
```

For `pixels_per_period`, cycles are `projector_dimension / period_px`.

The mapper preserves the ROI by forcing pixels outside `mask_uv` to `NaN`.

### 3D Reconstruction

```text
camera intrinsics + projector stereo model + projector_uv
  -> undistort camera pixel coordinates
  -> undistort projector UV coordinates
  -> triangulatePoints in normalized camera/projector coordinates
  -> filter by finite XYZ, z range, reprojection error
  -> reconstruction/depth.npy
  -> reconstruction/xyz.npy
  -> reconstruction/uv_map.npy
  -> reconstruction/masks/*
```

The canonical reconstruction implementation is `fringe_app.recon.triangulate.reconstruct_uv_run`.

Projection matrices in normalized coordinates are:

```text
P_camera = [I | 0]
P_projector = [R | T]
```

Triangulated points are in the camera coordinate frame. `depth.npy` is the `Z_camera` channel. The current output writer preserves high-resolution arrays (`xyz.npy`, `depth.npy`, `uv_map.npy`) and writes a downsampled point cloud only for export.

## Required Compatibility

The new app must load these existing files without conversion:

```text
data/calibration/camera/intrinsics_latest.json
data/calibration/projector/stereo_latest.json
data/calibration/projector_v2/<session>/export/stereo.json
data/runs/<run>/projector_uv/u.npy
data/runs/<run>/projector_uv/v.npy
data/runs/<run>/projector_uv/mask_uv.npy
data/runs/<run>/unwrap/phi_abs.npy
data/runs/<run>/phase/f_<freq>/phi_wrapped.npy
```

Existing shape assumptions:

```text
camera images: H x W, currently 768 x 1024
projector coordinates: u in [0, projector_width), v in [0, projector_height)
calibration units: meters for stereo translation and reconstruction XYZ
phase arrays: float32 with invalid pixels represented by NaN after unwrap/UV stages
masks: boolean NumPy arrays, with PNG aliases for visual inspection
```

## Assumptions for `fringe_app_v2`

1. Calibration math, PSP formulas, temporal unwrap rounding, UV mapping, and triangulation are reused from the existing modules.
2. `fringe_app_v2` may reorganize orchestration and web control, but it must not alter coordinate systems or sign conventions.
3. A complete 3D scan requires both vertical and horizontal structured-light captures so phase can map to projector `u` and `v`.
4. ROI captured before structured light is the authoritative downstream mask. Phase and unwrap masks may further restrict it, but no stage may expand outside it.
5. Height maps and masks are preserved at camera resolution for later defect detection and ML workflows.
