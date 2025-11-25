"""
scan.py

Central scanning pipeline for structured-light fringe projection.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import cv2
from scipy.ndimage import median_filter   # <-- NEW

from camera import CameraController
from psp import run_psp_per_frequency
from masking import merge_frequency_masks
from unwrap import temporal_unwrap


class ScanController:
    # ... __init__ unchanged ...

    def __init__(
        self,
        camera: CameraController,
        patterns: Dict[int, List[object]],
        midgrey_surface: object,
        set_surface_callback,
        freqs: List[int],
        n_phase: int,
        scan_root: str = "scans",
        pattern_settle_time: float = 0.15,
    ):
        self.camera = camera
        self.patterns = patterns
        self.midgrey_surface = midgrey_surface
        self.set_surface_callback = set_surface_callback
        self.freqs = freqs
        self.n_phase = n_phase
        self.scan_root = scan_root
        self.pattern_settle_time = pattern_settle_time

        os.makedirs(self.scan_root, exist_ok=True)

    # ... _make_scan_dir, _run_auto_exposure, _capture_psp_sequence unchanged ...

    def _make_scan_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scan_dir = os.path.join(self.scan_root, timestamp)
        os.makedirs(scan_dir, exist_ok=True)
        return scan_dir

    def _run_auto_exposure(self):
        def set_midgrey():
            self.set_surface_callback(self.midgrey_surface)

        exp, gain = self.camera.auto_expose_with_midgrey(
            set_midgrey_surface_callback=set_midgrey,
            target_mean=90,     # slightly darker target now
            tolerance=5,
            max_iters=6,
        )
        print(f"[SCAN] AE locked: Exposure={exp}us Gain={gain}")

    def _capture_psp_sequence(self, scan_dir: str) -> Dict[Tuple[int, int], np.ndarray]:
        I_dict = {}
        for f in self.freqs:
            for n in range(self.n_phase):
                self.set_surface_callback(self.patterns[f][n])
                time.sleep(self.pattern_settle_time)
                gray = self.camera.capture_gray().astype(np.float32)
                I_dict[(f, n)] = gray
                fname = f"f{f:03d}_n{n:02d}.png"
                cv2.imwrite(os.path.join(scan_dir, fname), gray)
                print(f"[SCAN] Captured {fname} (mean={gray.mean():.2f})")
        return I_dict

    def _save_phase_debug(self, Phi_final, mask_final, out_dir):
        phase_display = Phi_final.copy()
        phase_display[~mask_final] = np.nan
        finite_vals = phase_display[np.isfinite(phase_display)]
        if finite_vals.size > 0:
            p_min, p_max = np.percentile(finite_vals, [1, 99])
            phase_display = np.clip(phase_display, p_min, p_max)
        else:
            p_min, p_max = -np.pi, np.pi

        norm = (phase_display - p_min) / (p_max - p_min + 1e-9)
        norm[~np.isfinite(norm)] = 0.0

        img_u8 = (norm * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_TURBO)
        cv2.imwrite(os.path.join(out_dir, "phase_debug.png"), img_color)
        print("[SCAN] Saved phase_debug.png")

    # ------------------ MAIN PIPELINE ------------------ #
    def run_scan(self) -> str:
        print("[SCAN] Starting scan...")
        scan_dir = self._make_scan_dir()

        # Auto exposure
        self._run_auto_exposure()

        # Capture
        I_dict = self._capture_psp_sequence(scan_dir)

        print("[SCAN] Running PSP analysis...")
        psp_res = run_psp_per_frequency(
            I_dict,
            freqs=self.freqs,
            n_phase=self.n_phase,
            apply_input_blur=False,
            median_phase_filter=True,
        )

        print("[SCAN] Cleaning masks...")
        masks_final = merge_frequency_masks(psp_res.mask, self.freqs)

        print("[SCAN] Unwrapping temporally...")
        unwrap_res = temporal_unwrap(
            phi_wrapped=psp_res.phi_wrapped,
            mask_merged=masks_final,
            freqs=self.freqs,
        )

        Phi_final = unwrap_res.Phi_final
        mask_final = unwrap_res.mask_final

        # ----- NEW: apply median filter to final phase (inside mask) ----- #
        print("[SCAN] Applying median filter to final phase...")
        Phi_med = median_filter(Phi_final, size=3)
        Phi_final_smoothed = Phi_final.copy()
        Phi_final_smoothed[mask_final] = Phi_med[mask_final]
        Phi_final = Phi_final_smoothed
        # ---------------------------------------------------------------- #

        np.save(os.path.join(scan_dir, "phase_final.npy"), Phi_final)
        np.save(os.path.join(scan_dir, "mask_final.npy"), mask_final)
        print("[SCAN] Saved phase_final.npy and mask_final.npy")

        self._save_phase_debug(Phi_final, mask_final, scan_dir)

        print(f"[SCAN] COMPLETE: results saved in {scan_dir}")
        return scan_dir
