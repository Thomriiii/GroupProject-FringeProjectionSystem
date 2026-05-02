# turntable/ — Turntable Axis Calibration

A motorised turntable enables 360° surface reconstruction by capturing multiple scans at different rotation angles and merging them into a single point cloud. For accurate merging, the turntable's rotation axis must be precisely calibrated in the camera coordinate frame.

---

## 1. Motivation

The turntable rotates the object about a fixed physical axis. In the camera coordinate frame, this axis has:
- A **direction** $\hat{\mathbf{a}} \in \mathbb{R}^3$, $\|\hat{\mathbf{a}}\| = 1$
- A **position** — any point $\mathbf{p}$ on the axis (typically parameterised as the axis crossing the XY plane of the camera frame: $[p_x, p_y, 0]$, with the Z component absorbed into the axis direction)

Given a 3D point $\mathbf{P}$ observed at turntable angle $\theta_0$, its position at angle $\theta_i$ is:

$$\mathbf{P}(\theta_i) = R(\hat{\mathbf{a}}, \theta_i - \theta_0)\,(\mathbf{P}(\theta_0) - \mathbf{p}) + \mathbf{p}$$

Axis calibration error directly couples into 3D point errors: a 1° axis direction error at 100 mm radius produces ~1.7 mm registration error.

---

## 2. ChArUco Pose Estimation — `charuco_pose.py`

### Perspective-n-Point (PnP)

At each turntable angle, a photograph of a ChArUco board is taken. The board's 3D pose relative to the camera is computed by solving the Perspective-n-Point problem:

Given $N$ correspondences $\{(\mathbf{X}_i, \mathbf{u}_i)\}$ (3D board points, 2D image points):

$$\min_{R, \mathbf{t}} \sum_i \left\|\mathbf{u}_i - \pi(K, D, R, \mathbf{t}, \mathbf{X}_i)\right\|^2$$

Using `cv2.solvePnP` with `SOLVEPNP_ITERATIVE` flag (Levenberg–Marquardt, seeded by the algebraic Direct Linear Transform solution).

The output is a rotation vector $\mathbf{r}$ (Rodrigues encoding: $\|\mathbf{r}\| = \theta$, $\hat{\mathbf{r}} = \hat{\mathbf{a}}$) and translation $\mathbf{t}$, giving the pose of the board in the camera frame.

**Reprojection error** is the key quality metric per frame:

$$\epsilon = \frac{1}{N}\sum_i \|\mathbf{u}_i - \pi(K, D, R, \mathbf{t}, \mathbf{X}_i)\|_2$$

Frames with $\epsilon > \epsilon_\text{max}$ (default 2 px) are flagged as low quality.

### Rodrigues Rotation Vector

OpenCV uses the Rodrigues parameterisation: $\mathbf{r} = \theta\,\hat{\mathbf{a}}$, where $\theta = \|\mathbf{r}\|$ is the rotation angle and $\hat{\mathbf{a}} = \mathbf{r}/\theta$ is the axis direction. The rotation matrix:

$$R = \cos\theta\,\mathbf{I} + (1-\cos\theta)\,\hat{\mathbf{a}}\hat{\mathbf{a}}^\top + \sin\theta\,[\hat{\mathbf{a}}]_\times$$

This is the Rodrigues formula for rotation. `cv2.Rodrigues` converts between the vector and matrix forms.

---

## 3. Rotation Axis Estimation — `axis_fit.py`

### Step 1 — Relative Rotation Extraction

For consecutive frames at angles $\theta_i$ and $\theta_{i+1}$, the relative rotation of the board in the camera frame is:

$$R_\text{rel} = R_{i+1}\,R_i^\top$$

The axis and angle of $R_\text{rel}$ give the rotation applied between the two captures. If the turntable stepped by $\Delta\theta = 15°$, then $\|cv2.\text{Rodrigues}(R_\text{rel})\|$ should equal $\pi/12$ radians.

**Consistency check**: the standard deviation of measured step angles should be $< 0.5°$; larger deviation indicates mechanical play or pose estimation noise.

### Step 2 — Mean Axis Direction

The instantaneous axis directions from all inter-frame rotations are averaged (after sign-alignment to a common convention):

$$\hat{\mathbf{a}} = \frac{1}{M}\sum_{i=1}^{M} \hat{\mathbf{a}}_i$$

(re-normalised). The axis tilt from the camera Z axis is $\theta_\text{tilt} = \arccos(|\hat{a}_z|)$.

### Step 3 — Circle Fit (Taubin Method)

As the board rotates, the board origin (translation vector $\mathbf{t}_i$) traces a circle in 3D space. Projecting onto the camera XY plane, the $(t_x, t_y)$ values form a circle. The circle centre gives the axis position $(p_x, p_y)$.

The **Taubin algebraic circle fit** fits a circle $x^2 + y^2 + Dx + Ey + F = 0$ by minimising a geometric approximation. After mean-centering $(u_i = t_{x,i} - \bar{t}_x,\; v_i = t_{y,i} - \bar{t}_y)$:

$$S_{uu} = \sum u_i^2, \quad S_{vv} = \sum v_i^2, \quad S_{uv} = \sum u_i v_i$$

$$S_{uuu} = \sum u_i^3, \quad S_{uvv} = \sum u_i v_i^2, \quad S_{vvv} = \sum v_i^3$$

The centre offset $(u_c, v_c)$ from the mean is found from:

$$\begin{bmatrix} S_{uu} & S_{uv} \\ S_{uv} & S_{vv} \end{bmatrix} \begin{bmatrix} u_c \\ v_c \end{bmatrix} = \frac{1}{2}\begin{bmatrix} S_{uuu} + S_{uvv} \\ S_{vvv} + S_{vuu} \end{bmatrix}$$

Circle radius: $r = \sqrt{u_c^2 + v_c^2 + (S_{uu} + S_{vv})/N}$

Circle residual (quality): $\sqrt{\frac{1}{N}\sum_i\left(\sqrt{(u_i - u_c)^2 + (v_i - v_c)^2} - r\right)^2}$

### Literature

- Taubin, G. (1991). *Estimation of planar curves, surfaces, and nonplanar space curves defined by implicit equations with applications to edge and range image segmentation.* IEEE TPAMI, 13(11), 1115–1138.
- Horn, B. K. P. (1987). *Closed-form solution of absolute orientation using unit quaternions.* JOSA A, 4(4), 629–642.

---

## 4. 2D Homography Alignment — `alignment.py`

### Purpose

For each captured frame, a 2D homography is computed relative to a reference frame. This provides a fast 2D registration that can be used to check consistency and detect gross misalignments without full 3D reconstruction.

### Homography

A homography $H$ is a $3 \times 3$ projective transformation:

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

For a planar scene (the ChArUco board is planar), the homography exactly relates two views: $H = K' [R | \mathbf{t}'] K^{-1}$ (for pinhole cameras without distortion).

`cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 3.0)` solves for $H$ using the Direct Linear Transform (DLT) with RANSAC outlier rejection (3.0 px inlier threshold).

**DLT**: Each point correspondence $\mathbf{x}' \sim H\mathbf{x}$ gives two linear equations in the 9 elements of $H$. With $N \geq 4$ correspondences:

$$A\,\text{vec}(H) = 0 \quad \Rightarrow \quad H = V_\text{last column}(A)$$

RANSAC randomly samples minimal sets (4 correspondences), computes $H$, counts inliers (reprojection error $< 3$ px), and keeps the $H$ with the most inliers.

---

## 5. ChArUco Board Specification — `board.py`

### Board Design

| Parameter | Value |
|-----------|-------|
| Squares | 9 × 7 |
| Square side | 10 mm |
| Marker dictionary | DICT_4X4_50 |
| Marker side | 7.5 mm |
| Print DPI | 300 |
| Total board size | 90 mm × 70 mm |

The board is generated programmatically using `cv2.aruco.CharucoBoard.generateImage()` and includes:
- A central crosshair for rough axis alignment
- A 10 mm scale bar for verifying print scale accuracy

### Why ChArUco over Checkerboard?

A plain checkerboard requires all corners to be visible for detection. ChArUco boards allow partial detection (as long as enough ArUco marker IDs are visible to infer which corners are present). This is critical for turntable calibration where the board may be partially occluded at some angles.

---

## 6. Automated Capture — `auto_capture.py`

### Protocol

1. Move turntable to angle $\theta_i$ via HTTP command to ESP8266 controller
2. Wait `settle_ms` (default 300 ms) for mechanical oscillation to damp
3. Capture frame
4. Run ChArUco detection and PnP
5. Record: image path, angle, corner count, pose validity, reprojection error
6. Repeat for all $N = \lfloor 360° / \Delta\theta \rfloor$ angles

### Settle Time

The settle time is empirically determined. For the stepper-motor turntable used:
- Mechanical oscillation ring-down: ~150 ms
- Camera AE/AWB (disabled — manual controls): 0 ms
- Safety margin: 150 ms
- Total: 300 ms

Setting `settle_ms` too short introduces motion blur and phase errors from vibration. Setting it too long reduces throughput.
