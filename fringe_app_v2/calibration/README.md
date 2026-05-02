# calibration/ — Camera and Projector Calibration

Calibration determines the intrinsic and extrinsic parameters needed to convert image observations into 3D measurements. The system requires two calibration procedures: camera intrinsics (once, when the camera is set up) and stereo projector calibration (once per camera–projector alignment).

---

## 1. Camera Model

All cameras and projectors in this system are modelled using the pinhole camera with polynomial distortion (the OpenCV / Brown–Conrady model).

### Intrinsic Matrix

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

where:
- $f_x, f_y$ — focal lengths in pixels (different if pixel aspect ratio $\neq 1$)
- $c_x, c_y$ — principal point (optical axis intersection with image plane, ideally near image centre)

### Distortion Model (Brown–Conrady)

The projection of a 3D point $\mathbf{X}$ proceeds in three steps:

1. **Normalised coordinates**: $\mathbf{x}_n = [X/Z, Y/Z]^\top$

2. **Distortion correction**:

$$x' = x_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2p_1 x_n y_n + p_2(r^2 + 2x_n^2)$$
$$y' = y_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1(r^2 + 2y_n^2) + 2p_2 x_n y_n$$

where $r^2 = x_n^2 + y_n^2$

3. **Pixel coordinates**: $\mathbf{u} = K [x', y', 1]^\top$

The distortion coefficients are stored as $D = [k_1, k_2, p_1, p_2, k_3]$.

---

## 2. Camera Intrinsics Calibration — `camera/session.py`, `camera/charuco.py`

### Calibration Target: ChArUco Board

A ChArUco board combines a checkerboard pattern with ArUco fiducial markers. This provides two advantages over a plain checkerboard:
1. Individual squares can be identified by their ArUco marker ID, allowing partial board detection (the board does not need to be fully in view).
2. Corner positions can be refined to sub-pixel accuracy using the surrounding marker geometry.

Board specification: 9 × 7 squares, 10 mm side length, DICT_4X4_50, 300 DPI print.

### Corner Detection Pipeline

1. **ArUco detection**: `cv2.aruco.detectMarkers` — finds markers in the image using adaptive thresholding and contour analysis, and decodes the binary ID pattern.
2. **ChArUco interpolation**: `cv2.aruco.interpolateCornersCharuco` — for each pair of adjacent detected markers, the checkerboard corner position between them is localised to sub-pixel accuracy. The sub-pixel refinement uses the local image gradient (Förstner operator / Harris corner refinement).
3. **3D object points**: The board geometry defines a planar reference frame $(Z = 0)$ with corner positions in metres.

### Zhang's Calibration Method

Given $M$ views of the calibration board, each view $i$ provides:
- 3D points: $\{\mathbf{X}_{ij}\}$ (known from board geometry)
- 2D observations: $\{\mathbf{u}_{ij}\}$ (detected corners)

The calibration solves for $K$, $D$, and per-view extrinsics $\{R_i, \mathbf{t}_i\}$:

$$\min_{K, D, \{R_i, \mathbf{t}_i\}} \sum_{i,j} \left\|\mathbf{u}_{ij} - \pi(K, D, R_i, \mathbf{t}_i, \mathbf{X}_{ij})\right\|^2$$

where $\pi$ is the full projection function including distortion.

The solution is obtained in two stages:
1. **Closed-form initialisation**: homography-based (each view gives $H_i = K [r_1 \; r_2 \; t]$); extract $K$ from the constraints on the homographies.
2. **Non-linear refinement**: Levenberg–Marquardt iterative minimisation of reprojection error.

Implemented by `cv2.calibrateCamera` with `CALIB_RATIONAL_MODEL` flag omitted (standard 5-coefficient model: $k_1, k_2, p_1, p_2, k_3$).

**Quality metric**: per-view RMS reprojection error:

$$\text{RMS}_i = \sqrt{\frac{1}{N_i} \sum_j \left\|\mathbf{u}_{ij} - \hat{\mathbf{u}}_{ij}\right\|^2}$$

Typical acceptable range: $< 1.0$ px. Captures with RMS $>$ rejection threshold are excluded from the solve.

### Literature

- Zhang, Z. (2000). *A flexible new technique for camera calibration.* IEEE TPAMI, 22(11), 1330–1334.
- Garrido-Jurado, S., et al. (2014). *Automatic generation and detection of highly reliable fiducial markers under occlusion.* Pattern Recognition, 47(6), 2280–2292.
- Heikkila, J., & Silven, O. (1997). *A four-step camera calibration procedure with implicit image correction.* CVPR, 1106–1112.

---

## 3. Projector Calibration — `projector/solve.py`

### Principle

The projector cannot observe the scene — it can only project. To calibrate it as a "camera", the structured-light pipeline itself is used: the UV map provides correspondences between camera image coordinates and projector pixel coordinates. The ChArUco board corners give camera image coordinates; the UV map evaluated at those corners gives the corresponding projector coordinates.

### Stereo Calibration

The stereo calibration solves for the rigid transform between camera and projector coordinate frames:

$$\mathbf{P}_\text{proj} = R\,\mathbf{P}_\text{cam} + \mathbf{t}$$

where $R \in SO(3)$ and $\mathbf{t} \in \mathbb{R}^3$.

**Stage 1 — Projector intrinsics:**

`cv2.calibrateCamera(obj_pts, proj_pts, proj_size)` estimates $K_p$ and $D_p$ by treating the projector UV coordinates as image observations and the known board 3D points as object points.

**Stage 2 — Stereo calibration with fixed intrinsics:**

`cv2.stereoCalibrate(obj, cam_pts, proj_pts, K_c, D_c, K_p, D_p, cam_size, flags=CALIB_FIX_INTRINSIC)`

Minimises the total stereo reprojection error:

$$\min_{R, \mathbf{t}, \{R_i, \mathbf{t}_i\}} \sum_{i,j} \left[\left\|\mathbf{u}^c_{ij} - \pi(K_c, D_c, R_i, \mathbf{t}_i, \mathbf{X}_{ij})\right\|^2 + \left\|\mathbf{u}^p_{ij} - \pi(K_p, D_p, R R_i, R\mathbf{t}_i + \mathbf{t}, \mathbf{X}_{ij})\right\|^2\right]$$

**Additional outputs:**
- **Essential matrix**: $E = [\mathbf{t}]_\times R$ — satisfies the epipolar constraint $\mathbf{x}_p^\top E\, \mathbf{x}_c = 0$ for corresponding normalised coordinates
- **Fundamental matrix**: $F = K_p^{-\top} E K_c^{-1}$ — same constraint for pixel coordinates

### View Pruning

To improve robustness, views with high projector reprojection error are iteratively pruned (greedy algorithm):

1. Compute per-view projector RMS from the current solve.
2. Remove the view with the highest RMS, if the overall RMS improves by $\geq \delta$ (default 0.05 px).
3. Repeat up to `max_prune_steps` (default 6) times, subject to maintaining $\geq$ `min_views` (default 8).

### UV Subpixel Refinement — `projector/uv_refine.py`

The UV map has discrete sampling — the phase measurement resolution is approximately 1 projector pixel per $2\pi/N$ radians of phase. At a board corner position $(c_x, c_y)$ in the camera image, the projector coordinate is estimated by:

1. **Median of patch**: extract all valid UV values in a $(2r+1)\times(2r+1)$ neighbourhood, take the median.
2. **Local plane fit** (if $\geq 6$ valid pixels): fit $u = a_u x + b_u y + c_u$ and $v = a_v x + b_v y + c_v$ by least squares across the patch, evaluate at $(c_x, c_y)$.
3. **Consistency check**: if $\|[u_\text{plane} - u_\text{median}, v_\text{plane} - v_\text{median}]\| \leq 3$ px, use the plane estimate; otherwise fall back to median.

The plane fit accounts for the linear variation of phase across the board, giving a more accurate interpolated projector coordinate than the raw median.

---

## 4. View Quality Gating — `projector/view_gating.py`

Each captured calibration view is evaluated before being included in the solve. Acceptance criteria:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| ChArUco marker count | ≥ min_markers | Minimum information content |
| Board coverage ratio | 0.05 – 0.60 | Too small → poor conditioning; too large → edge distortion |
| Border clearance | ≥ 20 px from all edges | Boundary pixels have high distortion and less reliable detection |
| Valid UV corner count | ≥ 16 | Minimum correspondences for a well-conditioned projector solve |

**Board tilt estimation**: `cv2.solvePnP(obj, corners, K, D, flags=SOLVEPNP_ITERATIVE)` → $R_\text{board}$. Tilt angle from $\theta = \arccos(|R_{22}|)$ (angle between board normal and camera optical axis). Very high tilt reduces corner localisation accuracy.

---

## 5. Calibration Coordinate Frames

```
World frame = Camera frame (by convention)

Object/Board frame
    ↓  R_i, t_i  (per-view extrinsic, from PnP)
Camera frame
    ↓  R, t  (stereo extrinsic, from stereoCalibrate)
Projector frame
```

The extrinsic stereo transform $\{R, \mathbf{t}\}$ is the critical output: it converts any 3D point measured in the camera frame to the projector frame, enabling the forward projection (reprojection error calculation) and the ray-ray triangulation.
