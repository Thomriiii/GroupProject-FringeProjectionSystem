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
from scipy.ndimage import median_filter

from camera import CameraController
from psp import run_psp_per_frequency
from masking import merge_frequency_masks
from unwrap import temporal_unwrap


class ScanController:
    def __init__(
        self,
        camera: CameraController,
        patterns_vert: Dict[int, List[object]],
        midgrey_surface: object,
        set_surface_callback,
        freqs: List[int],
        n_phase: int,
        scan_root: str = "scans",
        calib_root: str = "calib",
        pattern_settle_time: float = 0.15,
        calib_patterns_horiz: Dict[int, List[object]] | None = None,
    ):
        self.camera = camera
        self.patterns_vert = patterns_vert
        self.patterns_horiz = calib_patterns_horiz if calib_patterns_horiz is not None else patterns_vert
        self.midgrey_surface = midgrey_surface
        self.set_surface_callback = set_surface_callback
        self.freqs = freqs
        self.n_phase = n_phase
        self.scan_root = scan_root
        self.calib_root = calib_root
        self.pattern_settle_time = pattern_settle_time

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
        """
        if self.calib_session_root is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.calib_session_root = os.path.join(self.calib_root, f"session_{timestamp}")
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
            target_mean=90,     # slightly darker target now
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
    # MAIN SCAN PIPELINE
    # =====================================================================

    def run_scan(self) -> str:
        print("[SCAN] Starting scan...")
        scan_dir = self._make_scan_dir()

        # Auto exposure
        self._run_auto_exposure()

        # Capture (vertical fringes for normal scanning)
        I_dict = self._capture_psp_sequence(scan_dir, self.patterns_vert, prefix="scan")

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

    # =====================================================================
    # CALIBRATION POSE CAPTURE
    # =====================================================================

    def run_calib_pose(self) -> str:
        """
        Capture a single calibration pose:
          - Auto-expose on mid-grey
          - Capture vertical PSP sequence (for projector X)
          - Capture horizontal PSP sequence (for projector Y)
          - Run PSP + unwrap for each
          - Save final phase + mask arrays and debug images

        Returns
        -------
        pose_dir : str
            Directory where this pose's data was saved.
        """
        print("[CALIB] Starting calibration pose capture...")
        pose_dir = self._make_calib_pose_dir()

        # Auto exposure
        self._run_auto_exposure()

        # -------- Vertical fringes (projector X) --------
        print("[CALIB] Capturing VERTICAL PSP sequence...")
        I_vert = self._capture_psp_sequence(pose_dir, self.patterns_vert, prefix="vert")

        print("[CALIB] Running PSP for vertical fringes...")
        psp_vert = run_psp_per_frequency(
            I_vert,
            freqs=self.freqs,
            n_phase=self.n_phase,
            apply_input_blur=False,
            median_phase_filter=True,
        )

        print("[CALIB] Cleaning masks for vertical fringes...")
        masks_vert = merge_frequency_masks(psp_vert.mask, self.freqs)

        print("[CALIB] Temporal unwrapping for vertical fringes...")
        unwrap_vert = temporal_unwrap(
            phi_wrapped=psp_vert.phi_wrapped,
            mask_merged=masks_vert,
            freqs=self.freqs,
        )

        Phi_vert = unwrap_vert.Phi_final
        mask_vert = unwrap_vert.mask_final

        print("[CALIB] Applying median filter to vertical phase...")
        Phi_vert_med = median_filter(Phi_vert, size=3)
        Phi_vert_smoothed = Phi_vert.copy()
        Phi_vert_smoothed[mask_vert] = Phi_vert_med[mask_vert]
        Phi_vert = Phi_vert_smoothed

        np.save(os.path.join(pose_dir, "phase_vert_final.npy"), Phi_vert)
        np.save(os.path.join(pose_dir, "mask_vert_final.npy"), mask_vert)
        self._save_phase_debug(Phi_vert, mask_vert, pose_dir, filename="phase_vert_debug.png")

        # -------- Horizontal fringes (projector Y) --------
        print("[CALIB] Capturing HORIZONTAL PSP sequence...")
        I_horiz = self._capture_psp_sequence(pose_dir, self.patterns_horiz, prefix="horiz")

        print("[CALIB] Running PSP for horizontal fringes...")
        psp_horiz = run_psp_per_frequency(
            I_horiz,
            freqs=self.freqs,
            n_phase=self.n_phase,
            apply_input_blur=False,
            median_phase_filter=True,
        )

        print("[CALIB] Cleaning masks for horizontal fringes...")
        masks_horiz = merge_frequency_masks(psp_horiz.mask, self.freqs)

        print("[CALIB] Temporal unwrapping for horizontal fringes...")
        unwrap_horiz = temporal_unwrap(
            phi_wrapped=psp_horiz.phi_wrapped,
            mask_merged=masks_horiz,
            freqs=self.freqs,
        )

        Phi_horiz = unwrap_horiz.Phi_final
        mask_horiz = unwrap_horiz.mask_final

        print("[CALIB] Applying median filter to horizontal phase...")
        Phi_horiz_med = median_filter(Phi_horiz, size=3)
        Phi_horiz_smoothed = Phi_horiz.copy()
        Phi_horiz_smoothed[mask_horiz] = Phi_horiz_med[mask_horiz]
        Phi_horiz = Phi_horiz_smoothed

        np.save(os.path.join(pose_dir, "phase_horiz_final.npy"), Phi_horiz)
        np.save(os.path.join(pose_dir, "mask_horiz_final.npy"), mask_horiz)
        self._save_phase_debug(Phi_horiz, mask_horiz, pose_dir, filename="phase_horiz_debug.png")

        print(f"[CALIB] Calibration pose COMPLETE: data saved in {pose_dir}")
        return pose_dir
