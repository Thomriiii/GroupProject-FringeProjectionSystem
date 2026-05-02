# phase_quality/ — Phase Quality Assessment

This module provides independent quality metrics for the phase measurement, separate from the PSP computation itself. It acts as a self-diagnostic layer: if the phase quality fails here, the reconstruction and defect detection will be unreliable.

---

## 1. Phase Quality Validation — `validation.py`

### Independent Modulation Calculation

The modulation (fringe contrast) $B$ is recomputed from the captured image stack independently of the main PSP processor:

$$B(x,y) = \frac{2}{N} \sqrt{\left(\sum_{n=0}^{N-1} I_n \cos\frac{2\pi n}{N}\right)^2 + \left(\sum_{n=0}^{N-1} I_n \sin\frac{2\pi n}{N}\right)^2}$$

This is identical to the PSP formula — the purpose is to provide a clean, standalone validation path with configurable thresholds separate from the main pipeline's mask policies.

### Mean Intensity

$$A(x,y) = \frac{1}{N}\sum_{n=0}^{N-1} I_n(x,y)$$

### Pixel-Level Quality Gate

A pixel is declared "good quality" if:

$$B(x,y) \geq B_\text{min} \quad \text{and} \quad A_\text{min} \leq A(x,y) \leq A_\text{max}$$

- $B_\text{min}$ (default 10 DN): minimum modulation — too low means the pattern is barely visible against the surface texture or ambient light
- $A_\text{min}$ (default 5 DN): minimum intensity — avoids dark pixels with poor SNR
- $A_\text{max}$ (default 250 DN): saturation guard

### Frame-Level Statistics

Per capture:
- **Valid fraction**: proportion of pixels passing the quality gate
- **Modulation histogram**: distribution of $B$ values over valid pixels
- **Intensity histogram**: distribution of $A$ values

These statistics are saved as JSON and used to flag problematic captures before reconstruction proceeds.

### Relationship to Phase Uncertainty

The phase uncertainty of the N-step PSP estimator is approximately:

$$\sigma_\phi \approx \frac{\sigma_I \sqrt{2/N}}{B}$$

where $\sigma_I$ is the image noise standard deviation. This shows that high modulation $B$ directly reduces phase noise, motivating the minimum modulation threshold.

### Literature

- Surrel, Y. (1993). *Phase stepping: a new self-calibrating algorithm.* Applied Optics, 32(19), 3598–3600.
- Hibino, K., et al. (1995). *Phase shifting for nonsinusoidal waveforms with phase-shift errors.* JOSA A, 12(4), 761–768.

---

## 2. Phase Diagnostics — `diagnostics.py`

### Wrapped Phase Diagnostics

Saved for each orientation × frequency combination after phase extraction.

**Phase visualisation**: two PNG images are saved:
- **Autoscale**: normalised to the 1st–99th percentile of valid phase values (enhances contrast for scene-specific patterns)
- **Fixed**: normalised to $[-\pi, \pi]$ (consistent across captures, useful for comparing frequency sets)

### Unwrapped Phase Diagnostics

After temporal unwrapping, the absolute phase map is saved in two modes:
- **Fixed**: phase modulo $2\pi f_\text{max}$ (shows the fringe structure)
- **Autoscale**: percentile-normalised (shows the full depth range)

### Phase Gradient Magnitude

$$g_\phi(x,y) = \|\nabla\phi_\text{abs}\|_2 = \sqrt{\left(\frac{\partial\phi}{\partial x}\right)^2 + \left(\frac{\partial\phi}{\partial y}\right)^2}$$

computed using `np.gradient` (central differences):

$$\frac{\partial\phi}{\partial x} \approx \frac{\phi[y, x+1] - \phi[y, x-1]}{2}$$

High gradient magnitude is expected at object edges (genuine surface slope). Anomalously high gradients in the interior indicate phase discontinuities (unwrapping errors).

### Stripe Artefact Score

$$\text{stripe\_score} = \text{std}\!\left(\frac{\partial\phi}{\partial x}\bigg|_\text{valid}\right) + \text{std}\!\left(\frac{\partial\phi}{\partial y}\bigg|_\text{valid}\right)$$

For a well-conditioned phase map, the gradient should vary smoothly. High standard deviation in the gradient (uneven fringe spacing) indicates:
- Gamma nonlinearity in the projector (non-sinusoidal patterns)
- Vibration during capture
- Surface with very strong specular reflections

---

## 3. Gamma Correction — `gamma.py`

### Projector Nonlinearity Problem

Consumer displays follow the sRGB standard: the emitted light intensity is related to the input digital value $v \in [0, 255]$ by:

$$L = \left(\frac{v}{255}\right)^\gamma, \qquad \gamma \approx 2.2$$

If the PSP patterns are generated with linear intensity values (as the math requires) but displayed on a gamma-correcting projector, the physical fringe pattern is distorted. A sinusoidal input $A + B\cos(\phi)$ is rendered as $(A + B\cos(\phi))^{2.2}$, which is a sum of harmonics. This introduces systematic phase errors of order $B^2/A$ in the PSP measurement.

### Correction

Pre-distort the pattern before display:

$$v = \text{clip}\!\left(\left(\frac{I_\text{linear} - I_\text{min}}{1 - I_\text{min}}\right)^{1/\gamma},\; 0, 1\right) \times 255$$

where $I_\text{min}$ prevents division by zero when the pattern minimum is close to zero.

### LUT Implementation

In practice, the correction is applied via a 256-entry look-up table (LUT):

$$\text{LUT}[v] = \text{round}\!\left(v^{1/\gamma} \times 255 / 255^\gamma\right)$$

Pre-computing the LUT allows $O(1)$ gamma correction per pixel (a single array index), which is important for real-time pattern generation.

### Effect on Phase Quality

Without gamma correction, the phase error from harmonic distortion for $N$-step PSP is approximately:

$$\Delta\phi \approx \frac{a_2 \sin(2\phi)}{2}$$

where $a_2$ is the amplitude of the second harmonic term. This produces a periodic phase error at twice the fringe frequency, visible as a systematic error in the reconstruction. Gamma correction eliminates this to first order.

### Literature

- Guo, H., He, H., & Chen, M. (2004). *Gamma correction for digital fringe projection profilometry.* Applied Optics, 43(14), 2906–2914.
- Zhang, S., & Yau, S.-T. (2007). *Generic nonsinusoidal phase error correction for three-dimensional shape measurement using a digital video projector.* Applied Optics, 46(1), 36–43.

---

## 4. Calibration Consistency Check — `calibration_check.py`

### Intrinsic Sanity Checks

After loading camera calibration, the following are verified:

| Check | Condition | Rationale |
|-------|-----------|-----------|
| Focal lengths positive | $f_x > 0,\; f_y > 0$ | Mathematical requirement |
| Focal lengths finite | $f_x < \infty$ | Detects degenerate calibration |
| Principal point inside image | $0 < c_x < W$, $0 < c_y < H$ | Guarantees valid projection |
| Square pixels | $f_x / f_y \approx 1$ | Expected for digital sensors |
| Centred principal point | $|c_x - W/2| / W < 0.2$ | Most lenses have near-centred principal point |

### Reported Quantities for the Dissertation

| Quantity | Interpretation |
|----------|---------------|
| $f_x, f_y$ (px) | Focal length; convert to mm via $f_\text{mm} = f_x \times \text{pixel pitch (mm)}$ |
| $c_x, c_y$ (px) | Principal point; should be near $(W/2, H/2)$ |
| $f_x / f_y$ | Pixel aspect ratio (deviation from 1 indicates non-square pixels) |
| $k_1, k_2$ | Radial distortion; $|k_1| > 0.3$ indicates significant barrel/pincushion distortion |
| Per-view RMS | Calibration quality; typical values 0.2–0.8 px for a good calibration |

The RMS reprojection error is the standard quality metric for OpenCV-style calibrations and should be reported in the dissertation alongside the calibration procedure.
