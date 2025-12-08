"""
scan.py

Central scanning pipeline for structured-light fringe projection (PSP).

Features:
  - Normal scan mode:
      * Uses full frequency set (e.g. [4, 8, 16, 32])
      * Uses psp.run_psp_per_frequency
      * Uses masking.merge_frequency_masks
      * Temporal unwrapping across all freqs
      * Outputs phase_final.npy, mask_final.npy, phase_debug.png
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import cv2
from scipy.ndimage import median_filter
import pygame

from core.camera import CameraController
from core.psp import run_psp_per_frequency
from core.masking import merge_frequency_masks
from core.unwrap import temporal_unwrap

class ScanController:
    def __init__(
        self,
        camera: CameraController,
        patterns: Dict[int, List[object]],
        midgrey_surface: object,
        set_surface_callback,
        freqs: List[int],
        n_phase: int,
        patterns_horiz: Dict[int, List[object]] | None = None,
        scan_root: str = "scans",
        calib_root: str = "calib",
        pattern_settle_time: float = 0.15,
    ):
        self.camera = camera
        self.patterns = patterns              # dict[freq] -> list[pygame.Surface]
        self.midgrey_surface = midgrey_surface
        self.set_surface_callback = set_surface_callback
        self.freqs = freqs
        self.n_phase = n_phase
        self.patterns_horiz = patterns_horiz  # optional dict for horizontal fringes
        self.scan_root = scan_root
        self.calib_root = calib_root
        self.pattern_settle_time = pattern_settle_time
        self.proj_w = patterns[freqs[0]][0].get_width()
        self.proj_h = patterns[freqs[0]][0].get_height()


        os.makedirs(self.scan_root, exist_ok=True)
        os.makedirs(self.calib_root, exist_ok=True)

        # Calibration session state
        self.calib_session_root: str | None = None
        self.calib_pose_idx: int = 0

    # =====================================================================
    # INTERNAL HELPERS
    # =====================================================================

    def _make_scan_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scan_dir = os.path.join(self.scan_root, timestamp)
        os.makedirs(scan_dir, exist_ok=True)
        return scan_dir

    def _ensure_calib_session(self) -> str:
        """
        Ensure there is an active calibration session directory.
        Returns the session root path.

        A session has the form:
            calib/session_YYYYMMDD_HHMMSS
        and contains pose_XXX subfolders.
        """
        if self.calib_session_root is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.calib_session_root = os.path.join(
                self.calib_root, f"session_{timestamp}"
            )
            os.makedirs(self.calib_session_root, exist_ok=True)
            self.calib_pose_idx = 0
            print(f"[CALIB] New calibration session at {self.calib_session_root}")
        return self.calib_session_root

    def _make_calib_pose_dir(self) -> str:
        session_root = self._ensure_calib_session()
        pose_name = f"pose_{self.calib_pose_idx:03d}"
        self.calib_pose_idx += 1
        pose_dir = os.path.join(session_root, pose_name)
        os.makedirs(pose_dir, exist_ok=True)
        print(f"[CALIB] Creating calibration pose directory: {pose_dir}")
        return pose_dir

    def _run_auto_exposure(self):
        def set_midgrey():
            self.set_surface_callback(self.midgrey_surface)

        exp, gain = self.camera.auto_expose_with_midgrey(
            set_midgrey_surface_callback=set_midgrey,
            target_mean=90,     # slightly darker target
            tolerance=5,
            max_iters=6,
        )
        print(f"[SCAN] AE locked: Exposure={exp}us Gain={gain}")

    def _capture_psp_sequence(
        self,
        out_dir: str,
        patterns: Dict[int, List[object]],
        prefix: str = "",
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Capture a full PSP sequence for the given patterns dict.

        patterns[f][n] is the pygame.Surface to show for frequency f
        and phase index n.
        """
        I_dict: Dict[Tuple[int, int], np.ndarray] = {}

        for f in self.freqs:
            for n in range(self.n_phase):
                self.set_surface_callback(patterns[f][n])
                time.sleep(self.pattern_settle_time)
                gray = self.camera.capture_gray().astype(np.float32)
                I_dict[(f, n)] = gray

                if prefix:
                    fname = f"{prefix}_f{f:03d}_n{n:02d}.png"
                else:
                    fname = f"f{f:03d}_n{n:02d}.png"

                cv2.imwrite(os.path.join(out_dir, fname), gray)
                print(f"[SCAN] Captured {fname} (mean={gray.mean():.2f})")

        return I_dict

    def _save_phase_debug(self, Phi_final, mask_final, out_dir, filename: str = "phase_debug.png"):
        """
        Save a false-colour debug image of the final unwrapped phase.
        """
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
        out_path = os.path.join(out_dir, filename)
        cv2.imwrite(out_path, img_color)
        print(f"[SCAN] Saved {filename}")


    # =====================================================================
    # NORMAL SCAN PIPELINE
    # =====================================================================

    def run_scan(self) -> str:
        """
        Full normal scan:
          - AE on midgrey
          - capture vertical PSP sequence (self.freqs)
          - PSP, mask merging, temporal unwrapping
          - median filter final phase
          - save phase_final.npy, mask_final.npy, phase_debug.png

        Returns
        -------
        scan_dir : str
            Directory where scan results were saved.
        """
        print("[SCAN] Starting scan...")
        scan_dir = self._make_scan_dir()

        # Auto exposure
        self._run_auto_exposure()

        # Capture vertical PSP sequence (for object scanning)
        I_dict = self._capture_psp_sequence(scan_dir, self.patterns, prefix="scan")

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

        # Median filter final phase (inside mask)
        print("[SCAN] Applying median filter to final phase...")
        Phi_med = median_filter(Phi_final, size=3)
        Phi_final_smoothed = Phi_final.copy()
        Phi_final_smoothed[mask_final] = Phi_med[mask_final]
        Phi_final = Phi_final_smoothed

        np.save(os.path.join(scan_dir, "phase_final.npy"), Phi_final)
        np.save(os.path.join(scan_dir, "mask_final.npy"), mask_final)
        print("[SCAN] Saved phase_final.npy and mask_final.npy")

        self._save_phase_debug(Phi_final, mask_final, scan_dir, filename="phase_debug.png")

        print(f"[SCAN] COMPLETE: results saved in {scan_dir}")
        return scan_dir
