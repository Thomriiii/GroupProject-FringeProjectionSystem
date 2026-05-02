# utils/ — I/O and Mathematical Utilities

Supporting utilities used throughout the pipeline. These modules have no algorithmic content unique to structured light, but they define the data storage layout and several mathematical helpers referenced in the pipeline.

---

## 1. Run Storage Layout — `io.py`

### RunPaths

The `RunPaths` dataclass defines the canonical directory tree for one scan run:

```
<run_root>/<run_id>/
    structured/
        vertical/
            freq_1.0000/   step_000.png ... step_N.png
            freq_4.0000/
            freq_16.0000/
        horizontal/
            ...
    phase/
        vertical/
            freq_1.0000/
                phi_wrapped.npy     (float32, H×W)
                A.npy               (float32, H×W)
                B.npy               (float32, H×W)
                mask.npy            (bool, H×W)
                mask_raw.npy
                mask_clean.npy
                mask_for_unwrap.npy
                mask_for_defects.npy
                mask_for_display.npy
                clipped_any.npy
                phase_meta.json
    unwrap/
        vertical/
            phi_abs.npy         (float32, H×W)
            mask_unwrap.npy     (bool, H×W)
            residual.npy        (float32, H×W)
            unwrap_meta.json
        horizontal/
    roi/
        roi_mask.npy            (bool, H×W)
        roi_meta.json
    phase_quality/
        validation/
        diagnostics/
    reconstruction/
        xyz.npy                 (float32, H×W×3)
        depth.npy               (float32, H×W)
        cloud.ply
        reproj_err_cam.npy
        reproj_err_proj.npy
        reconstruction_meta.json
    flatten/
        height_flat.npy         (float32, H×W)
        height_plane_removed.npy
        flatten_meta.json
    defect/
        defect_mask.npy         (bool, H×W)
        residual.npy
        defect_report.json
        defect_overlay.png
```

All NumPy arrays are saved in `.npy` format (little-endian, raw binary) for fast I/O. Metadata is saved as human-readable JSON.

### Frequency Tag Convention

Frequencies are serialised to string keys as `freq_1.0000`, `freq_4.0000`, `freq_16.0000`. The 4-decimal-place format ensures consistent sorting and avoids floating-point serialisation ambiguity.

---

## 2. Mathematical Utilities — `math_utils.py`

### Grayscale Conversion

$$I_\text{gray} = 0.299\,R + 0.587\,G + 0.114\,B$$

ITU-R BT.601 luminance coefficients. Used for converting camera RGB frames to grayscale for phase computation, ROI detection, and calibration.

### JSON Serialisation of NumPy Types

NumPy scalars and arrays are not directly JSON-serialisable. The `json_safe` utility recursively converts:
- `np.ndarray` → `list` (via `.tolist()`)
- `np.integer` → `int`
- `np.floating` → `float`
- `np.bool_` → `bool`

This is used throughout the pipeline to serialise metadata dictionaries to JSON files.

---

## 3. Key Array Conventions

Throughout the pipeline, the following conventions are used consistently:

| Variable | Shape | Dtype | Units |
|----------|-------|-------|-------|
| `phi_wrapped` | H×W | float32 | radians ∈ (-π, π] |
| `phi_abs` | H×W | float32 | radians ∈ [0, 2πf_max] |
| `A` | H×W | float32 | DN (8-bit digital numbers) |
| `B` | H×W | float32 | DN |
| `mask_*` | H×W | bool | — |
| `u`, `v` | H×W | float32 | projector pixels |
| `xyz` | H×W×3 | float32 | metres |
| `depth` | H×W | float32 | metres |
| `reproj_err_*` | H×W | float32 | pixels |
| `height_flat` | H×W | float32 | metres |
| `residual` | H×W | float32 | metres |

NaN is used throughout to indicate invalid/missing values, allowing NumPy's `np.nan*` functions (nanmean, nanmedian, etc.) to operate on masked arrays without requiring explicit mask arguments.
