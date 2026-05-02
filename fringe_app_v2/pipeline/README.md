# pipeline/ — Structured-Light 3D Reconstruction Pipeline

This module implements the complete measurement chain from raw captured images to a 3D point cloud and defect map. Each stage is a pure function that reads from and writes to a `RunPaths` directory structure, making individual stages re-runnable and inspectable.

---

## Pipeline Overview

```
Structured capture (N × M images)
        ↓
Phase extraction (wrapped phase φ_w per frequency × orientation)
        ↓
ROI detection (object mask)
        ↓
Temporal unwrapping (absolute phase φ_abs)
        ↓
Phase → projector coordinates (UV map)
        ↓
Triangulation (3D point cloud XYZ)
        ↓
Statistical outlier removal (cleaned cloud)
        ↓
Surface flattening (height residual map)
        ↓
Defect segmentation (defect mask + features)
```

For multi-angle scans, the single-scan pipeline runs once per turntable angle, and the resulting clouds are merged using the calibrated rotation axis.

---

## 1. Structured Capture — `structured_capture.py`

### Capture Protocol

For each orientation (`vertical`, `horizontal`) and each frequency ($f_1, f_2, \ldots, f_K$), $N$ patterns are projected and captured:

1. Display pattern $n$ on the projector
2. Wait `settle_ms` milliseconds (first step: `settle_ms_first_step`, frequency switch: `settle_ms_freq_switch`) to allow the projector lamp to stabilise and camera sensor to clear charge from the previous frame
3. Flush `flush_frames_per_step` camera frames (discarded) to clear the hardware buffer
4. Capture the measurement frame

The settle delays are critical: if the pattern has not fully rendered before capture, the measured intensity is a temporal mixture of two patterns, which corrupts the phase measurement.

### Total capture count

$$N_\text{total} = |\text{orientations}| \times |\text{frequencies}| \times N_\text{steps}$$

Typical: $2 \times 3 \times 8 = 48$ frames per scan.

---

## 2. Phase Extraction — `phase.py`

See `core/README.md §1` for the N-step PSP algorithm.

### Adaptive Unwrap Threshold

The unwrap seed mask uses an adaptive modulation threshold:

$$B_\text{unwrap} = \begin{cases} \min(B_\text{thresh},\; 0.05 \times \text{median}(B_\text{ROI})) & \text{if } \text{median}(B_\text{ROI}) > 0 \\ B_\text{thresh} & \text{otherwise} \end{cases}$$

This adapts to the measured contrast of the specific scene being scanned, avoiding the case where a fixed threshold is too aggressive or too permissive.

### Mask Policy Rationale

The unwrap mask uses the conservative path (largest component + erosion) to avoid unwrapping into noisy boundary pixels where the phase can jump discontinuously. The defect mask uses the inclusive path (raw + dilation) so that defects at the object boundary are not clipped.

---

## 3. Temporal Phase Unwrapping — `unwrap.py`, `core/temporal_unwrap.py`

### Problem

PSP yields the phase $\phi_\text{wrapped} \in (-\pi, \pi]$. The absolute phase is:

$$\phi_\text{abs} = \phi_\text{wrapped} + 2\pi k, \quad k \in \mathbb{Z}$$

The integer $k$ (fringe order) must be determined to recover the unique depth value. For a single frequency this is impossible without spatial unwrapping; multi-frequency temporal unwrapping resolves it analytically.

### Multi-Frequency Temporal Unwrapping

Given two frequencies $f_\text{low} < f_\text{high}$ (in cycles across the dimension):

1. At $f_\text{low}$, the pattern has only one period across the field — the wrapped phase IS the absolute phase (shifted to $[0, 2\pi)$).
2. At $f_\text{high}$, the fringe order $k$ satisfies:

$$k = \text{round}\!\left(\frac{(f_\text{high}/f_\text{low})\,\phi_\text{abs,low} - \phi_\text{wrapped,high}}{2\pi}\right)$$

$$\phi_\text{abs,high} = \phi_\text{wrapped,high} + 2\pi k$$

3. With three frequencies (e.g. 1, 4, 16 cycles), the process is applied hierarchically:
   - $f_1 = 1$ → absolute directly
   - $f_2 = 4$ → unwrapped using $f_1$
   - $f_3 = 16$ → unwrapped using $f_2$

The frequency ratio $r = f_\text{high}/f_\text{low}$ must satisfy $r \leq 2$ in general for noise-free unwrapping, but with sufficient modulation the method works for larger ratios (ratios 4 and 4 are used here, which is standard in practice).

### Residual Quality Metric

$$\text{residual} = \phi_\text{abs,high} - (f_\text{high}/f_\text{low}) \cdot \phi_\text{abs,low}$$

Ideally zero. The RMS and 95th percentile of the residual are reported as quality metrics. $|\text{residual}| > 1$ rad indicates a fringe-order error.

### Literature

- Towers, C. E., Towers, D. P., & Jones, J. D. C. (2003). *Optimum frequency selection in multifrequency interferometry.* Optics Letters, 28(11), 887–889.
- Sansoni, G., Carocci, M., & Rodella, R. (1999). *Three-dimensional vision based on a combination of gray-code and phase-shift light projection.* Applied Optics, 38(31), 6565–6573.
- Zuo, C., et al. (2016). *Temporal phase unwrapping algorithms for fringe projection profilometry: A comparative review.* Optics and Lasers in Engineering, 85, 84–103.

---

## 4. Phase-to-Projector Coordinate Mapping — `phase_to_projector.py`

### Principle

The projector acts as an inverse camera: it emits light at known pixel coordinates. The absolute phase encodes which projector column (or row) illuminated each camera pixel.

For vertical patterns ($f_U$ cycles across projector width $W_p$):

$$u(x, y) = \frac{\phi_\text{abs,vertical}(x,y)}{2\pi f_U} \cdot W_p$$

For horizontal patterns ($f_V$ cycles across projector height $H_p$):

$$v(x, y) = \frac{\phi_\text{abs,horizontal}(x,y)}{2\pi f_V} \cdot H_p$$

Together, $(u, v)$ provides a dense correspondence field: for every camera pixel $(x, y)$, the corresponding projector pixel $(u, v)$ is known.

### UV Validity Gate

The UV map is rejected if:
- Valid pixel ratio $< 3\%$ of the full frame — too few correspondences for reliable calibration
- UV range $< 40$ px in either dimension — the object does not subtend enough projector pixels
- Edge pixel fraction $> 10\%$ — too many correspondences near the projector boundary where the calibration is less reliable

---

## 5. Triangulation — `triangulation.py`

### Geometric Setup

The system is a stereo pair: a camera and a projector, with known relative geometry from stereo calibration (rotation $\mathbf{R}$, translation $\mathbf{t}$). A 3D point $\mathbf{P}$ satisfies:

- In the camera frame: $\mathbf{P}$ lies on the ray through pixel $(x_c, y_c)$
- In the projector frame: $\mathbf{P}$ lies on the ray through pixel $(u, v)$

Triangulation finds $\mathbf{P}$ as the closest-approach midpoint of these two rays.

### Step 1 — Normalised Image Coordinates

Distorted pixel coordinates are converted to normalised image coordinates:

$$\tilde{\mathbf{x}} = K^{-1} \mathbf{x}_\text{undistorted}$$

OpenCV's `undistortPoints` handles both radial ($k_1, k_2, k_3$) and tangential ($p_1, p_2$) distortion using Brown–Conrady model:

$$x' = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2p_1 xy + p_2(r^2 + 2x^2)$$

where $r^2 = x^2 + y^2$.

### Step 2 — Ray Definitions

Camera ray (camera at origin):
$$\mathbf{o}_c = \mathbf{0}, \quad \hat{\mathbf{d}}_c = \frac{[\tilde{x}_c, \tilde{y}_c, 1]^\top}{\|[\tilde{x}_c, \tilde{y}_c, 1]\|}$$

Projector centre in camera coordinates:
$$\mathbf{o}_p = -\mathbf{R}^\top \mathbf{t}$$

Projector ray direction in camera frame:
$$\hat{\mathbf{d}}_p = \frac{\mathbf{R}^\top [\tilde{x}_p, \tilde{y}_p, 1]^\top}{\|\mathbf{R}^\top [\tilde{x}_p, \tilde{y}_p, 1]^\top\|}$$

### Step 3 — Least-Squares Ray Intersection

The two rays are generally skew. The system of equations is:

$$\mathbf{o}_c + \lambda\,\hat{\mathbf{d}}_c = \mathbf{o}_p + \mu\,\hat{\mathbf{d}}_p$$

Rearranged as a $3 \times 2$ linear system:

$$\begin{bmatrix} \hat{\mathbf{d}}_c & -\hat{\mathbf{d}}_p \end{bmatrix} \begin{bmatrix} \lambda \\ \mu \end{bmatrix} = \mathbf{o}_p - \mathbf{o}_c$$

Solved via `np.linalg.lstsq` (SVD-based minimum-norm solution):

$$\mathbf{P} = \mathbf{o}_c + \lambda\,\hat{\mathbf{d}}_c$$

### Alternative: OpenCV DLT (`cv2.triangulatePoints`)

The `core/calibration.py` path uses the homogeneous DLT method. Given camera projection matrices $P_c = K_c [\mathbf{I} | \mathbf{0}]$ and $P_p = K_p [\mathbf{R} | \mathbf{t}]$:

$$\mathbf{x}_h = \text{triangulatePoints}(P_c, P_p, \tilde{\mathbf{x}}_c, \tilde{\mathbf{x}}_p)$$

$$\mathbf{P} = \mathbf{x}_h[:3] / \mathbf{x}_h[3]$$

Both methods are algebraically equivalent for exact correspondences and give very similar results in practice.

### Reprojection Error

For quality control, the triangulated point is re-projected into each view:

$$\epsilon_c = \|\pi(K_c, D_c, \mathbf{P}) - \mathbf{x}_c\|_2, \qquad \epsilon_p = \|\pi(K_p, D_p, \mathbf{R}\mathbf{P} + \mathbf{t}) - \mathbf{x}_p\|_2$$

Points with $\max(\epsilon_c, \epsilon_p) > \epsilon_\text{max}$ (default 15 px) are rejected.

### Literature

- Hartley, R. I., & Sturm, P. (1997). *Triangulation.* Computer Vision and Image Understanding, 68(2), 146–157.
- Zhang, S. (2018). *High-speed 3D shape measurement with structured light methods: A review.* Optics and Lasers in Engineering, 106, 119–131.
- Scharstein, D., & Szeliski, R. (2002). *A taxonomy and evaluation of dense two-frame stereo correspondence algorithms.* IJCV, 47(1), 7–42.

---

## 6. 3D Reconstruction Post-Processing — `reconstruct.py`

### Component Filter (Image Space)

Isolated valid pixels in the reconstruction mask are removed using connected component labelling (`scipy.ndimage.label`, 4-connected). Any component smaller than `min_component_px` pixels is rejected. This is applied in image space (2D) because the projective structure of the camera means that spatially isolated valid pixels almost always correspond to phase noise, not real object geometry.

### Statistical Outlier Removal (SOR)

After triangulation, outlier points are removed using the SOR algorithm:

1. Build a k-d tree on the $N_\text{valid}$ 3D points.
2. For each point $\mathbf{P}_i$, compute the mean distance to its $k$ nearest neighbours:

$$\bar{d}_i = \frac{1}{k} \sum_{j \in \mathcal{N}_k(i)} \|\mathbf{P}_i - \mathbf{P}_j\|_2$$

3. Compute the global statistics: $\mu_d = \text{mean}(\bar{d}_i)$, $\sigma_d = \text{std}(\bar{d}_i)$

4. Inlier condition:

$$\bar{d}_i \leq \mu_d + \alpha\,\sigma_d$$

Default: $k = 20$, $\alpha = 3.0$. This removes points that are far from their neighbours (disconnected noise) while preserving genuine geometric features.

This is equivalent to the SOR filter in the PCL (Point Cloud Library) / Open3D toolkits.

### PLY Export

The output point cloud is saved in PLY (Polygon File Format / Stanford Triangle Format), the standard format for point cloud interchange with software such as MeshLab, CloudCompare, and Open3D.

---

## 7. Surface Flattening — `flatten.py`

### Motivation

Defect detection requires measuring local height deviations from the expected object surface. The triangulated point cloud contains a global tilt (from the object placement relative to the camera) and low-frequency surface curvature. These must be removed to isolate small-scale defects.

### Step 1 — Global Plane Removal (Least Squares)

For all valid pixels $(x_i, y_i)$ with height $h_i$ (Z coordinate in camera frame):

$$\min_{a, b, c} \sum_i \left(a x_i + b y_i + c - h_i\right)^2$$

Solved via the normal equations using `np.linalg.lstsq`. The plane is subtracted:

$$h' = h - (ax + by + c)$$

The plane normal $\hat{\mathbf{n}} = [-a, -b, 1]^\top / \|[-a, -b, 1]\|$ is reported; its angle with the Z axis indicates the object tilt.

### Step 2 — Low-Frequency Shape Removal (NaN-Aware Gaussian)

A large-scale shape (curvature) is estimated by Gaussian smoothing of the tilt-removed height map:

$$\hat{h}(x,y) = \frac{\int h'(x', y')\, G_\sigma(x-x', y-y')\, m(x',y')\, dx'\, dy'}{\int G_\sigma(x-x', y-y')\, m(x',y')\, dx'\, dy'}$$

where $m(x,y) \in \{0,1\}$ is the valid pixel mask and $G_\sigma$ is a Gaussian kernel with standard deviation $\sigma$ (default $\sigma = 8$ px).

In practice, this is computed as:

$$\hat{h} = \frac{F * (h' \cdot m)}{F * m}$$

where $F * (\cdot)$ denotes Gaussian convolution (`scipy.ndimage.gaussian_filter`). This is a standard technique for handling missing data in spatial filters, equivalent to locally-normalised weighted averaging.

The flat surface residual is:

$$\Delta h = h' - \hat{h}$$

---

## 8. Multi-Scan Merging — `merge_clouds.py`

### Problem

A single scan captures only the side of the object facing the camera. To reconstruct a complete 360° surface, the object is rotated on a turntable and multiple scans are captured. Each scan must be transformed to a common reference frame.

### Rotation Transform

Each scan is captured at turntable angle $\theta_i$. The rotation is about a calibrated axis: unit direction $\hat{\mathbf{a}}$ and a point $\mathbf{p}$ on the axis. The rotation matrix about an arbitrary axis is (Rodrigues formula):

$$R(\hat{\mathbf{a}}, \theta) = \cos\theta\,\mathbf{I} + (1-\cos\theta)\,\hat{\mathbf{a}}\hat{\mathbf{a}}^\top + \sin\theta\,[\hat{\mathbf{a}}]_\times$$

where $[\hat{\mathbf{a}}]_\times$ is the skew-symmetric matrix of $\hat{\mathbf{a}}$.

To transform a scan at angle $\theta_i$ back to the reference frame ($\theta = 0$):

$$\mathbf{P}_\text{ref} = R(\hat{\mathbf{a}}, -\theta_i)(\mathbf{P}_i - \mathbf{p}) + \mathbf{p}$$

### ICP Refinement

After the rigid rotation alignment, residual misregistration (from axis calibration error or mechanical play) is corrected using Iterative Closest Point (ICP):

**One ICP iteration:**
1. Find correspondences: for each source point $\mathbf{s}_i$, find nearest target point $\mathbf{t}_{\sigma(i)}$ in the k-d tree.
2. Compute the optimal rigid transform using singular value decomposition:

$$H = \sum_i (\mathbf{s}_i - \bar{\mathbf{s}})(\mathbf{t}_{\sigma(i)} - \bar{\mathbf{t}})^\top$$

$$H = U\Sigma V^\top \quad \Rightarrow \quad R_* = V U^\top$$

If $\det(R_*) < 0$ (reflection), the last column of $V$ is negated.

$$\mathbf{t}_* = \bar{\mathbf{t}} - R_*\,\bar{\mathbf{s}}$$

3. Apply: $\mathbf{s} \leftarrow R_*\,\mathbf{s} + \mathbf{t}_*$, accumulate $R_\text{total}$, $\mathbf{t}_\text{total}$.

**Convergence**: stop when median correspondence distance changes by $< 10^{-5}$ m or `max_iterations` reached.

**Guards**: if accumulated rotation $> 2°$ or translation $> 10$ mm, the ICP result is rejected (the initial alignment was too poor for ICP to correct).

### Literature

- Besl, P. J., & McKay, N. D. (1992). *A method for registration of 3-D shapes.* IEEE TPAMI, 14(2), 239–256.
- Rodrigues, O. (1840). *Des lois géométriques qui régissent les déplacements d'un système solide.* Journal de Mathématiques Pures et Appliquées, 5, 380–440.

---

## 9. Defect Orchestration — `defect.py`

See `defect/README.md` for the segmentation and classification algorithms.

---

## 10. Reconstruction Validation — `validation.py`

### Quality Tests

| Test | Metric | Default threshold |
|------|--------|-------------------|
| Point cloud density | valid px / total px | ≥ 50% |
| Depth range | min/max Z values | within [z_min, z_max] |
| Local flatness | mean(std(Z in 64×64 windows)) | ≤ 10 mm |
| Tilt | angle(surface normal, Z-axis) | ≤ 2° |
| Camera reprojection | mean $\epsilon_c$ | ≤ 3 px |
| Projector reprojection | mean $\epsilon_p$ | ≤ 3 px |
| Edge artefacts | fraction valid at boundary | ≤ 30% |

The tilt angle is computed from the fitted plane coefficients $(a, b)$:

$$\theta_\text{tilt} = \arccos\!\left(\frac{1}{\sqrt{a^2 + b^2 + 1}}\right)$$
