# defect/ — Surface Defect Detection and Classification

This module detects and characterises surface defects from the reconstructed height map. The approach is model-free: rather than training on labelled defect examples, it uses signal-processing methods to identify regions where the local surface height deviates significantly from the expected (smooth) surface.

---

## 1. Problem Formulation

After 3D reconstruction and surface flattening (`pipeline/flatten.py`), the height residual map $\Delta h(x,y)$ contains:
- **Measurement noise** — random, zero-mean, characterised by the system's depth resolution
- **Real surface variation** — gradual shape (low spatial frequency)
- **Defects** — localised regions of anomalous height (either depressed: pits, scratches; or raised: burrs, contamination)

Defect detection = separating the third category from the first two.

---

## 2. Defect Segmentation — `segment.py`

### Step 1 — Reference Surface Estimation

A local reference surface $\hat{h}(x,y)$ is estimated by NaN-aware Gaussian smoothing with standard deviation $\sigma_\text{smooth}$ (default 12 px):

$$\hat{h}(x,y) = \frac{\iint \Delta h(x',y')\, G_\sigma(x-x', y-y')\, m(x',y')\, dx'\, dy'}{\iint G_\sigma(x-x', y-y')\, m(x',y')\, dx'\, dy'}$$

where $m$ is the valid-pixel mask. This is computed efficiently as:

$$\hat{h} = \frac{\texttt{gaussian\_filter}(\Delta h \cdot m,\; \sigma)}{\texttt{gaussian\_filter}(m,\; \sigma)}$$

The denominator acts as a locally-normalised weight, correctly handling the absence of measurements at object boundaries and within holes (see `defect/utils.py`).

**Physical interpretation**: $\hat{h}$ captures the global shape of the object — its macroscopic curvature and form. Subtracting it isolates the residual fine structure.

**Parameter choice**: $\sigma$ should be larger than the expected defect size but smaller than the object's overall curvature radius. A typical defect is $<5$ mm; $\sigma = 12$ px at 0.165 mm/px corresponds to ~2 mm, which is appropriate for distinguishing defect-scale features from part-scale curvature.

### Step 2 — Residual Computation

$$r(x,y) = \Delta h(x,y) - \hat{h}(x,y)$$

$r > 0$: surface raised above local mean (protrusion, bump, contamination)
$r < 0$: surface below local mean (pit, scratch, void)

### Step 3 — Thresholding

$$M_\text{raw}(x,y) = |r(x,y)| > \tau$$

where $\tau = $ `depth_threshold_mm` converted to metres (default 1.5 mm → 0.0015 m).

### Step 4 — Morphological Cleanup

The binary defect mask is refined through morphological operations using a disk structuring element $\mathcal{B}_r = \{(dy, dx) : dy^2 + dx^2 \leq r^2\}$:

1. **Opening** (erode then dilate, $r = r_\text{open}$): removes isolated single-pixel noise
2. **Closing** (dilate then erode, $r = r_\text{close}$): fills small holes inside defect regions
3. **Area filter**: remove components with fewer than `min_area_px` pixels (default 30)
4. **Boundary exclusion**: erode the valid object mask by `boundary_exclusion_px` (default 12) — defects touching the object boundary are excluded (phase noise is highest there)

### Step 5 — Edge Suppression

Surface edges (sharp geometric features like corners, ribs) produce large height gradients that can be mistaken for defects. These are suppressed by:

1. Compute the surface gradient magnitude from the reference surface:

$$g(x,y) = \left\|\nabla \hat{h}\right\|_2 = \sqrt{\left(\frac{\partial \hat{h}}{\partial x}\right)^2 + \left(\frac{\partial \hat{h}}{\partial y}\right)^2}$$

computed using `np.gradient` (second-order central differences).

2. Threshold at the `edge_percentile`-th percentile (default 90%) of $g$ values → edge mask $M_\text{edge}$.

3. Dilate $M_\text{edge}$ by `edge_exclusion_radius_px` (default 12 px).

4. Any defect component $C_k$ where:

$$\frac{|C_k \cap M_\text{edge}|}{|C_k|} > \text{edge\_overlap\_reject}$$

(default 0.25) is removed.

### Literature

- Malamas, E. N., et al. (2003). *A survey on industrial vision systems, applications and tools.* Image and Vision Computing, 21(2), 171–188.
- Kumar, A. (2008). *Computer-vision-based fabric defect detection: A survey.* IEEE Transactions on Industrial Electronics, 55(1), 348–363.

---

## 3. Feature Extraction — `features.py`

For each connected defect component, geometric features are computed for classification and reporting.

### Spatial Features

| Feature | Formula |
|---------|---------|
| Area | $A = N_\text{px} \times (\text{mm/px})^2$ |
| Centroid | $(\bar{x}_\text{mm}, \bar{y}_\text{mm}) = (\text{mean}(x_i), \text{mean}(y_i)) \times \text{mm/px}$ |
| Bounding box | $[x_\text{min}, y_\text{min}, W, H]$ in mm |
| Aspect ratio | $AR = \text{major axis} / \text{minor axis}$ |

### Depth Features

| Feature | Formula |
|---------|---------|
| Mean depth | $\bar{r} = \text{mean}(r_i)$ where $r_i$ are residual values in mm |
| Peak depth | $r_\text{peak} = \max_i |r_i|$ |
| Std depth | $\sigma_r = \text{std}(r_i)$ |

### Volumetric Features

| Feature | Formula |
|---------|---------|
| Signed volume | $V_s = (\text{mm/px})^2 \sum_i r_i$ |
| Absolute volume | $V_a = (\text{mm/px})^2 \sum_i |r_i|$ |
| Positive volume | $V_+ = (\text{mm/px})^2 \sum_i \max(r_i, 0)$ |
| Negative volume | $V_- = (\text{mm/px})^2 \sum_i \max(-r_i, 0)$ |
| Volume fill factor | $V_a / (W \times H \times r_\text{peak})$ |

The `mm_per_pixel` scale factor is critical. It should be calibrated from the known object dimensions or computed from the reconstruction geometry (the 3D extent of the point cloud divided by its pixel extent at the mean depth).

---

## 4. Rule-Based Classification — `models.py`

Defects are classified into three categories using geometric rules on the extracted features:

### Classification Logic

```
IF aspect_ratio ≥ linear_aspect_ratio (default 4.0)
    → scratch / linear defect

ELSE IF minor_axis/major_axis ≥ point_aspect_min (default 0.8)
     AND major_axis ≤ point_max_major_mm (default 5.0 mm)
    → pit / point defect

ELSE
    → surface defect (diffuse/large)
```

### Rationale

- **Scratches** are elongated (high aspect ratio). A threshold of 4:1 is conservative; highly directional machining marks may require a lower threshold.
- **Pits** are small and roughly isotropic (aspect ratio close to 1). The maximum size limit (5 mm) prevents large surface regions from being classified as pits.
- **Surface defects** are the catch-all: large, irregularly shaped regions of height deviation that do not fit either of the above categories.

This is a deliberately simple rule-based classifier. More sophisticated approaches (e.g. SVM on the feature vector, or deep learning on the height-residual patch) could be used for higher classification accuracy, at the cost of requiring labelled training data.

---

## 5. Shared Utilities — `utils.py`

### NaN-Aware Gaussian Smoothing

See `pipeline/flatten.py` section for the mathematical formulation. This is the central utility used by both surface flattening and defect segmentation.

### Morphological Operations

Disk structuring element: $\mathcal{B}_r = \{(dy, dx) : dy^2 + dx^2 \leq r^2\}$ (unit disk approximated on a pixel grid).

Operations defined on binary masks $M$:

$$\text{dilate}(M, r)[y,x] = \exists (dy,dx) \in \mathcal{B}_r : M[y+dy, x+dx] = 1$$

$$\text{erode}(M, r)[y,x] = \forall (dy,dx) \in \mathcal{B}_r : M[y+dy, x+dx] = 1$$

$$\text{open}(M, r) = \text{dilate}(\text{erode}(M, r), r) \quad \text{(removes small objects)}$$

$$\text{close}(M, r) = \text{erode}(\text{dilate}(M, r), r) \quad \text{(fills small holes)}$$

Implemented using `scipy.ndimage.binary_dilation/erosion` with the disk kernel, with a pure-Python BFS fallback.

### Connected Components

`scipy.ndimage.label` with an 8-connected structure (3×3 all-ones kernel):

$$M_\text{labelled}[y,x] = k \iff \text{pixel is in component } k$$

Properties per component (area, bounding box, centroid) are extracted using `scipy.ndimage.find_objects` and boolean indexing.
