# core/ — Hardware Interfaces and Signal Processing Primitives

This module contains the lowest-level components: hardware drivers, the fundamental phase-measurement algorithm, and the signal-processing primitives on which every pipeline stage depends.

---

## 1. N-Step Phase-Shifting Profilometry (PSP) — `psp.py`

### Concept

Phase-Shifting Profilometry (PSP) is the primary measurement principle of the system. A sinusoidal intensity pattern is projected onto the surface and a camera records the scene. The surface height modulates the local position of each fringe, encoding depth as a phase offset. By projecting $N$ patterns that are phase-shifted relative to one another by $2\pi/N$, the phase at every camera pixel can be computed analytically.

### Pattern Equation

Each projected pattern has intensity:

$$I_n(x, y) = A(x,y) + B(x,y) \cos\!\left(2\pi f x + \frac{2\pi n}{N}\right), \quad n = 0, 1, \ldots, N-1$$

where:
- $A$ — background (DC) intensity, determined by ambient light and surface reflectance
- $B$ — modulation (fringe contrast), determined by surface reflectance and pattern visibility
- $f$ — spatial frequency of the fringe pattern (cycles across the projector dimension)
- $\phi = 2\pi f x$ — the wrapped phase to be recovered

### Phase Extraction

The phase is recovered from the $N$ captured frames using the Discrete Fourier Transform at the fundamental frequency. The quadrature components are:

$$C = \sum_{n=0}^{N-1} I_n \cos\!\left(\frac{2\pi n}{N}\right), \qquad S = \sum_{n=0}^{N-1} I_n \sin\!\left(\frac{2\pi n}{N}\right)$$

Wrapped phase:

$$\phi_\text{wrapped}(x,y) = \arctan\!\left(\frac{-S}{C}\right) \in (-\pi, \pi]$$

Mean intensity and modulation:

$$A = \frac{1}{N}\sum_{n=0}^{N-1} I_n, \qquad B = \frac{2}{N}\sqrt{C^2 + S^2}$$

The sign convention $\arctan2(-S, C)$ is used so that increasing fringe position maps to increasing phase.

### Validity Masking

A pixel is considered valid (reliable phase measurement) if and only if:

$$B(x,y) \geq B_\text{thresh} \quad \text{and} \quad A(x,y) \geq A_\text{min} \quad \text{and} \quad \max_n I_n < I_\text{sat}$$

The saturation check rejects pixels where the camera response has clipped — clipped pixels introduce systematic phase errors because the sinusoidal assumption is violated.

### Key Parameters (configured in `default.yaml → phase`)

| Parameter | Role |
|-----------|------|
| `n_steps` | Number of phase-shift steps $N$ (typically 8) |
| `B_thresh` | Minimum modulation threshold (rejects low-contrast areas) |
| `A_min` | Minimum mean intensity (rejects very dark areas) |
| `sat_high` | Saturation DN level |

### Literature

- Srinivasan, V., Liu, H. C., & Halioua, M. (1984). *Automated phase-measuring profilometry of 3-D diffuse objects.* Applied Optics, 23(18), 3105–3108.
- Creath, K. (1988). *Phase-measurement interferometry techniques.* Progress in Optics, 26, 349–393.
- Zuo, C., et al. (2018). *Phase shifting algorithms for fringe projection profilometry: A review.* Optics and Lasers in Engineering, 109, 23–59.

---

## 2. Fringe Pattern Generation — `pattern_generator.py`

### Pattern Synthesis

The projector renders grayscale images of the form:

$$I_n(x, y) = \text{clip}\!\left[I_\text{min} + (1 - I_\text{min})\!\left(\frac{1}{2} + \frac{1}{2}\cos\!\left(\frac{2\pi f x}{W} + \frac{2\pi n}{N} + \phi_0\right)\right), \; I_\text{min}, 1\right]$$

where $W$ is the projector width in pixels and $\phi_0$ is the phase origin offset. A contrast scaling parameter allows trading fringe visibility against the risk of saturation.

### Gamma Correction

Consumer projectors apply a nonlinear mapping $\tilde{I} = I^\gamma$ (typically $\gamma \approx 2.2$) between the input digital value and the emitted light intensity. If the patterns are generated assuming a linear projector, the displayed patterns will not be sinusoidal, introducing harmonic distortion into the PSP measurement. This is corrected by pre-distorting the patterns: if the desired linear intensity is $I_\text{linear}$, the digital value sent is:

$$I_\text{digital} = I_\text{linear}^{1/\gamma}$$

See also `phase_quality/gamma.py`.

---

## 3. Object ROI Detection — `object_roi.py`

### Problem

The projected patterns illuminate the entire scene but the object occupies only part of the camera image. Phase measurements outside the object boundary are unreliable (background, shadows, reflections). The ROI mask confines all subsequent processing to the object region.

### Algorithm

1. **Downscale** the reference image to at most 640 px wide (nearest-neighbour) to reduce computation.
2. **Box-blur** with a $k \times k$ kernel (default $k = 7$) to suppress high-frequency texture.
3. **Adaptive threshold**: background level estimated as the $p$-th percentile of pixel intensities (default $p = 82$). Threshold = background + offset. The percentile-based background estimate is robust to partial scene illumination.
4. **Morphological cleanup**: close (dilate then erode, structuring element = 3×3 box), open (erode then dilate) to remove isolated noise and bridge small gaps.
5. **Largest connected component** selection to keep only the primary object.
6. **Post-processing** (optional): fill small holes, dilate by `post_dilate_radius_px` to include the full object silhouette.

The result is upscaled back to the original resolution (nearest-neighbour).

---

## 4. Phase Mask Post-Processing — `masking.py`

### Purpose

The raw validity mask from PSP contains noise — small isolated valid clusters (phase noise that happens to pass the threshold) and small invalid holes inside the valid region. Post-processing enforces physical plausibility.

### Connected Component Analysis

Connected components are identified using a BFS flood-fill (4-connectivity). A component with fewer than `min_component_area` pixels is removed. This is equivalent to a morphological area opening, but applied directly to the binary mask without requiring structuring elements.

### Hole Filling

Small enclosed invalid regions (holes) inside the valid mask are filled. A hole is detected if a connected component of invalid pixels does not touch the image border. Holes smaller than `max_hole_area` pixels are filled to avoid false defect detections in the interior of the object.

### Morphological Operations

Binary dilation and erosion are implemented using the 8-connected neighbourhood (3×3 box structuring element):

$$\text{dilate}(M)[y,x] = \bigcup_{(dy,dx) \in \mathcal{N}} M[y+dy, x+dx]$$

$$\text{erode}(M)[y,x] = \bigcap_{(dy,dx) \in \mathcal{N}} M[y+dy, x+dx]$$

### Four Mask Types

The phase stage produces four masks for different downstream uses:

| Mask | Purpose |
|------|---------|
| `mask_raw` | Unprocessed threshold output |
| `mask_clean` | Component-filtered + hole-filled |
| `mask_for_unwrap` | Largest component + erode (conservative, avoids unwrapping into noisy boundary) |
| `mask_for_defects` | Raw + largest component + dilate (inclusive, preserves defect signatures at borders) |
| `mask_for_display` | Smoothed-B threshold (visually stable for live preview) |

---

## 5. Camera Hardware — `camera_base.py`, `camera_mock.py`, `camera_picamera2.py`

### Camera Model Used (Picamera2)

The physical camera is a Raspberry Pi HQ Camera (Sony IMX477 sensor). It is controlled via the `picamera2` / `libcamera` stack, which provides direct access to sensor controls:
- **ExposureTime** (µs) — integration time
- **AnalogueGain** — sensor amplification (ISO analogue)
- **AwbEnable** / **AwbMode** — white balance

For structured light, automatic exposure and white balance are disabled. Manual control ensures consistent, repeatable frame intensities across all $N$ phase steps — a fundamental requirement for PSP validity.

### Dual-Stream Capture

Picamera2 is configured with two simultaneous streams:
- **Main** — full resolution (e.g. 1024×768), RGB888, used for phase measurement
- **Lores** — quarter-resolution, YUV420, used for live preview

The YUV420 → RGB conversion is:

$$\begin{pmatrix} R \\ G \\ B \end{pmatrix} = \begin{pmatrix} 1 & 0 & 1.402 \\ 1 & -0.344 & -0.714 \\ 1 & 1.772 & 0 \end{pmatrix} \begin{pmatrix} Y \\ U-128 \\ V-128 \end{pmatrix}$$

(ITU-R BT.601 full-range)

---

## 6. Projector Display — `pygame_display.py`

The projector is driven as a second HDMI display. Pygame is used to render fullscreen grayscale images in a dedicated thread. The key constraint is timing: the projector must complete a full frame refresh (at its native refresh rate, typically 60 Hz) before the camera captures, to avoid partial-frame artefacts.

The display thread calls `time.sleep(0.008)` after each flip to allow the compositor and GPU buffer pipeline to settle before the camera shutter opens. This empirical delay is a system-specific tuning parameter.

---

## 7. Data Model — `models.py`

`ScanParams` is the central configuration object passed through the entire pipeline. It carries all parameters needed to generate patterns, drive the camera, and interpret the captured images:

- `resolution` $(W, H)$ — projector and camera image size
- `n_steps` $N$ — number of phase-shift steps
- `frequencies` — list of spatial frequencies $[f_1, f_2, \ldots]$ (e.g. `[1.0, 4.0, 16.0]`)
- `frequency_semantics` — whether `frequencies` are in cycles-across-dimension or pixels-per-period
- `orientation` — `"vertical"` (encodes projector X) or `"horizontal"` (encodes projector Y)
- `brightness_offset`, `contrast`, `min_intensity` — pattern generation controls
- `phase_origin_rad` — phase origin $\phi_0$
