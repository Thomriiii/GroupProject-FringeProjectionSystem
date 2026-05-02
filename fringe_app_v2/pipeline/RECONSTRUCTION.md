# 3D Reconstruction Pipeline - Technical Documentation

## Overview

The reconstruction pipeline implements **true geometric 3D triangulation** using structured light phase encoding. The system converts unwrapped phase maps to 3D point clouds through calibrated ray-ray intersection.

## Architecture

```
Phase Maps (horizontal + vertical)
    ↓
Phase → Projector Coordinates (u, v)
    ↓
Calibrated Ray Construction
    ├─ Camera rays: from camera origin through camera pixels
    └─ Projector rays: from projector origin through projector pixels
    ↓
Ray-Ray Intersection (Triangulation)
    ↓
Reprojection Error Filtering
    ↓
3D Point Cloud + Depth Map
```

## Core Components

### 1. Phase-to-Projector Conversion (`phase_to_projector.py`)

**Purpose:** Convert unwrapped phase to projector pixel coordinates (u, v)

**Why Two Phase Directions?**

Structured light encodes position via phase:
- **Horizontal patterns** (left-right sweeping) → encode projector X coordinate (u)
- **Vertical patterns** (top-bottom sweeping) → encode projector Y coordinate (v)

Without both directions, you have only 1D correspondence (ambiguous projection). With both, you have full 2D projector pixel → 3D world correspondence.

**Mathematical Formula:**

```
u = (phase_horizontal / (2π × cycles_u)) × projector_width
v = (phase_vertical / (2π × cycles_v)) × projector_height
```

**Key Points:**
- Phase must be unwrapped (continuous, not wrapped to [0, 2π))
- Phase origin accounts for systematic offsets
- Multiple cycles = integer multiples of 2π added back by unwrapping
- Frequency semantics: "cycles_across_dimension" vs "pixels_per_period"

### 2. Triangulation (`triangulation.py`)

**Core Algorithm:** Ray-Ray Intersection via Least Squares

For each corresponding pixel pair (u_cam, v_cam) ↔ (u_proj, v_proj):

1. **Undistort** pixel coordinates using camera intrinsics
2. **Build rays**:
   - Camera ray: origin C, direction d_cam
   - Projector ray: origin P, direction d_proj
3. **Solve overdetermined system**:
   ```
   C + λ d_cam = P + μ d_proj
   
   Rearranged: [d_cam | -d_proj] [λ; μ] = P - C
   
   Solved via: SVD / LSTSQ
   ```
4. **Triangulated point**: X = C + λ d_cam

**Why Rays?**

Each pixel in an image corresponds to a ray in 3D space. A camera pixel (x, y) defines all points that project to that pixel. Combining two such rays (camera + projector) typically intersects at a unique point (or comes very close, hence least squares).

**Reprojection Error Filtering:**

After triangulation, points are re-projected back to both images:
- Compute expected pixel for triangulated X in camera image
- Compute expected pixel for triangulated X in projector image
- Reject if error > threshold (typically 2-3 pixels)

This eliminates false correspondences and phase-unwrap errors.

### 3. Pipeline Integration (`reconstruct.py`)

**Workflow:**

1. Load unwrapped phase maps (horizontal, vertical)
2. Convert phase → projector coordinates (u, v)
3. Validate UV map (coverage, range, edge-pinning)
4. Build camera & projector models from calibration
5. Triangulate from UV maps
6. Save outputs (XYZ, depth, masks, reprojection errors)
7. Generate debug images
8. Export point cloud (PLY format)

**Validation (`validation.py`):**

Comprehensive tests verify reconstruction quality:
- **Point density:** Sufficient coverage (>50% typical)
- **Depth range:** Points within physical limits
- **Flatness:** Local Z variance < threshold (for flat objects)
- **No tilt:** Fitted plane angle < threshold
- **Low reprojection errors:** Mean error < 3px
- **No edge artifacts:** Points distributed across image

## Mathematical Details

### Ray Construction

**Camera ray (origin at camera center C = [0,0,0]):**

```
undistorted_pixel = undistortPoints(pixel_cam, K_cam, dist_cam)
normalized_point = [x_n, y_n, 1] (in normalized image plane)
ray_direction = normalize([x_n, y_n, 1])
```

**Projector ray (origin at projector center P in world coords):**

```
undistorted_pixel = undistortPoints(pixel_proj, K_proj, dist_proj)
normalized_point_proj_frame = [x_n, y_n, 1]
ray_direction_world = normalize(R^T @ [x_n, y_n, 1])
ray_origin_world = -R^T @ t
```

Where R, t are the extrinsic transform: **projector in camera frame**.

### Triangulation Solution

Minimize: ||C + λ d_cam - P - μ d_proj||²

Solutions:
- If lines intersect exactly: λ, μ are unique
- If lines are skew (typical): least-squares finds closest approach
- Triangulated point: midpoint or weighted average of closest points

**SVD Solution:**
```
A = [d_cam | -d_proj]       (3×2)
b = P - C                     (3×1)

lstsq(A, b) → [λ; μ]
X = C + λ d_cam
```

## Why Previous Method Failed

**Incorrect approach: Phase → Height scaling**

```
height = phase / (2π) × scale_factor
```

**Problems:**
1. **Uses only ONE phase direction** — underdetermined mapping
2. **No triangulation** — no ray-ray intersection
3. **No geometric constraints** — ignores camera/projector geometry
4. **No error filtering** — includes all phase values uncritically
5. **Results:** Wavy surfaces, slanted planes, artifacts

**Correct approach: Full triangulation**

✓ Uses BOTH phase directions for full 2D correspondence
✓ Constructs calibrated rays from both cameras
✓ Applies SVD triangulation (rigorous geometry)
✓ Filters by reprojection error
✓ Results: Flat planes, consistent depth, proper 3D geometry

## Calibration Integration

The system relies on:
1. **Camera intrinsics** K_cam, dist_cam
2. **Projector intrinsics** K_proj, dist_proj
3. **Stereo extrinsics** R, t (projector relative to camera)

These come from offline calibration and are stored in JSON:
- `data/calibration/camera/intrinsics_latest.json`
- `data/calibration/projector/stereo_latest.json`

## Performance Characteristics

**Typical reconstruction quality:**
- Point density: 70-90% of image pixels
- Depth accuracy: ±5-15 mm (depending on object distance)
- Reprojection error: mean 0.5-2 px, max 2-3 px
- Flatness: local std < 1 mm for flat objects
- Surface curvature: Accurate to ~1-2 mm

**Computational cost:**
- Phase → projector coords: O(1) per pixel
- Ray construction: O(1) per pixel (vectorized)
- Triangulation: O(N) via LSTSQ (N = valid pixels, typically 100k-300k)
- Filtering & output: O(N)
- Total: ~100-500 ms for 1024×768 images

## Debug Outputs

When `debug_reconstruct: true` in config:

```
reconstruct/
├── debug/
│   ├── depth_map.png          # Grayscale depth visualization
│   ├── reproj_err_camera.png  # Reprojection error (camera image)
│   ├── reproj_err_projector.png  # Reprojection error (projector image)
│   ├── projector_u.png        # Projector X coordinates
│   └── projector_v.png        # Projector Y coordinates
```

These help diagnose:
- Calibration issues (high reprojection errors)
- Unwrap errors (coordinate discontinuities)
- Phase-wrapping problems (edges, clipping)
- UV range problems (too small object or wrong scaling)

## Validation Examples

**Good reconstruction (flat object):**
```
✓ Point Cloud Density: 85.2% (required 50%)
✓ Depth Range: [0.485, 0.510]m within [0.05, 5.0]m
✓ Flatness: mean local std = 0.000340m (max 0.000612m, threshold 0.01m)
✓ No Tilt: 0.42° (threshold 2.0°)
✓ Low Reprojection Errors: camera 0.82px, projector 0.91px, threshold 3.0px
✓ No Edge Artifacts: 345 center pixels, 45 edge pixels
```

**Problem reconstruction:**
```
✗ Flatness: mean local std = 0.0156m >> 0.01m threshold
  → Suggests calibration error, unwrap error, or phase ambiguity
✗ No Tilt: 4.2° >> 2.0° threshold
  → Systematic slant; check projector alignment
✗ Reprojection Errors: camera 5.2px >> 3.0px threshold
  → Phase mapping inconsistent; check frequency / phase origin
```

## Configuration

**Key parameters in `config.yaml`:**

```yaml
scan:
  frequencies: [4, 8, 16]              # Multi-frequency phases
  frequency_semantics: "cycles_across_dimension"  # or "pixels_per_period"
  phase_origin_rad: 0.0                # Phase offset (0 to 2π)

reconstruction:
  z_min_m: 0.05                        # Minimum depth filter
  z_max_m: 5.0                         # Maximum depth filter
  max_reproj_err_px: 2.0               # Reprojection error threshold

uv_gate:
  enabled: true                        # Enable UV validation
  min_valid_ratio: 0.03                # Minimum 3% valid pixels
  min_u_range_px: 40.0                 # Min projector X spread
  min_v_range_px: 40.0                 # Min projector Y spread
  max_edge_pct: 0.10                   # Max 10% edge saturation
  max_zero_pct: 0.01                   # Max 1% zero-value pixels

debug_reconstruct: false               # Enable debug outputs
```

## Testing & Validation

Run validation on a reconstruction:

```python
from fringe_app_v2.pipeline.validation import validate_reconstruction

result = validate_reconstruction(Path("runs/20260428_234355/reconstruct"))
# Returns 0 if all tests pass, 1 if any fail
```

Or use CLI:

```bash
cd /home/pi/fringe
python -m fringe_app_v2.pipeline.validation runs/20260428_234355/reconstruct
```

## File Structure

```
fringe_app_v2/pipeline/
├── reconstruct.py           # Main pipeline orchestrator
├── triangulation.py         # Ray triangulation core algorithm
├── phase_to_projector.py    # Phase → UV mapping
└── validation.py            # Quality validation tests
```

## Future Enhancements

1. **Multi-camera stereo:** Support >2 cameras for overdetermined solution
2. **Dynamic filtering:** Adaptive thresholds based on local confidence
3. **Temporal filtering:** Smooth across frames in video mode
4. **Color mapping:** RGB per point from camera texture
5. **Mesh reconstruction:** Delaunay/Poisson from point cloud
6. **GPU acceleration:** CUDA triangulation for real-time
7. **Bundle adjustment:** Joint optimization of calibration + 3D points

## References

- Salvi et al. (2010): "A state of the art in structured light patterns for 3D imaging"
- Hartley & Zisserman (2003): "Multiple View Geometry in Computer Vision"
- OpenCV documentation: calibrateCamera, triangulatePoints, projectPoints
