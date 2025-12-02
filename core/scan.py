"""
scan.py

Central scanning pipeline for structured-light fringe projection.

Features:
  - Normal scan mode:
      * Uses full frequency set (e.g. [4, 8, 16, 32])
      * Uses psp.run_psp_per_frequency
      * Uses masking.merge_frequency_masks
      * Temporal unwrapping across all freqs
      * Outputs phase_final.npy, mask_final.npy, phase_debug.png

  - Calibration mode (run_calib_pose):
      * Uses only low frequencies [4, 8] for calibration
      * Uses psp_calib.run_psp_calibration (relaxed thresholds)
      * Uses masking_calib.merge_masks_calibration (OR-merging)
      * Temporal unwrapping across [4, 8]
      * Captures both vertical and horizontal fringes
      * Saves:
          phase_vert_final.npy,  mask_vert_final.npy
          phase_horiz_final.npy, mask_horiz_final.npy
      * Raw fringe images saved as:
          vert_fXXX_nYY.png, horiz_fXXX_nYY.png
        under calib/session_YYYYMMDD_HHMMSS/pose_XXX/
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

from calibration.psp_calib import run_psp_calibration
from calibration.masking_calib import merge_masks_calibration
from calibration.build_projector_dataset_psp import build_projector_dataset_psp
from calibration import projector_intrinsics



class ScanController:
    def __init__(
        self,
        camera: CameraController,
        patterns: Dict[int, List[object]],
        midgrey_surface: object,
        set_surface_callback,
        freqs: List[int],
        n_phase: int,
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
        # Gray-code calibration session state
        self.graycode_root = "calib_graycode"
        os.makedirs(self.graycode_root, exist_ok=True)

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

    # =====================================================================
    # GRAY CODE CAPTURE (for projector calibration)
    # =====================================================================
    def run_graycode_pose(self, patterns_gray: list[tuple[str, object]], session_root: str) -> str:
        """
        Capture a single Gray code pose.

        Parameters
        ----------
        patterns_gray : list[(label, pygame.Surface)]
            Gray code patterns to project.
        session_root : str
            Root session directory under calib_graycode.

        Returns
        -------
        pose_dir : str
        """
        pose_idx = 0
        while True:
            candidate = os.path.join(session_root, f"pose_{pose_idx:03d}")
            if not os.path.exists(candidate):
                pose_dir = candidate
                break
            pose_idx += 1
        os.makedirs(pose_dir, exist_ok=True)
        print(f"[GRAY] Capturing Gray code pose at {pose_dir}")

        images = {}
        for i, (label, surf) in enumerate(patterns_gray):
            self.set_surface_callback(surf)
            time.sleep(self.pattern_settle_time)
            gray = self.camera.capture_gray().astype(np.float32)
            images[label] = gray
            fname = f"{label}.png"
            cv2.imwrite(os.path.join(pose_dir, fname), gray)
            print(f"[GRAY] Captured {fname} (mean={gray.mean():.2f})")

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
    # psp SCAN PIPELINE
    # =====================================================================
    def start_psp_session(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_psp_session = os.path.join(self.calib_root, f"psp_session_{timestamp}")
        os.makedirs(self.current_psp_session, exist_ok=True)
        self.psp_pose_index = 0
        return self.current_psp_session


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

    # =====================================================================
    # CALIBRATION POSE CAPTURE
    # =====================================================================

    def run_calib_pose(self) -> str:
        """
        Capture a single calibration pose.

        Steps:
          - Ensure / create a calibration session under self.calib_root.
          - Auto exposure on midgrey.
          - Capture VERTICAL fringes for all frequencies in self.freqs, saving:
                vert_f{f:03d}_n{n:02d}.png
          - Capture HORIZONTAL fringes by rotating patterns 90 deg, saving:
                horiz_f{f:03d}_n{n:02d}.png
          - Run calibration PSP on low freqs [4, 8]:
                psp_calib.run_psp_calibration + masking_calib.merge_masks_calibration
                + unwrap.temporal_unwrap
          - Save:
                phase_vert_final.npy,  mask_vert_final.npy
                phase_horiz_final.npy, mask_horiz_final.npy

        Returns
        -------
        pose_dir : str
            Path to this pose's directory.
        """
        print("[CALIB] Starting calibration pose capture...")

        pose_dir = self._make_calib_pose_dir()

        # Auto exposure
        self._run_auto_exposure()

        # -----------------------------
        # Capture VERTICAL images
        # -----------------------------
        print("[CALIB] Capturing VERTICAL PSP sequence...")
        I_vert: Dict[Tuple[int, int], np.ndarray] = {}

        for f in self.freqs:
            for k in range(self.n_phase):
                surface = self.patterns[f][k]
                self.set_surface_callback(surface)
                time.sleep(self.pattern_settle_time)

                gray = self.camera.capture_gray().astype(np.float32)
                I_vert[(f, k)] = gray
                fname = f"vert_f{f:03d}_n{k:02d}.png"
                cv2.imwrite(os.path.join(pose_dir, fname), gray)
                print(f"[CALIB] Captured {fname} (mean={gray.mean():.2f})")

        # -----------------------------
        # Capture HORIZONTAL images
        # -----------------------------
        print("[CALIB] Capturing HORIZONTAL PSP sequence...")
        I_horiz: Dict[Tuple[int, int], np.ndarray] = {}

        for f in self.freqs:
            for k in range(self.n_phase):
                pattern = self.patterns[f][k]
                pattern_h = pygame.transform.rotate(pattern, 90)
                pattern_h = pygame.transform.smoothscale(
                    pattern_h, (self.proj_w, self.proj_h)
                )

                self.set_surface_callback(pattern_h)
                time.sleep(self.pattern_settle_time)

                gray = self.camera.capture_gray().astype(np.float32)
                I_horiz[(f, k)] = gray
                fname = f"horiz_f{f:03d}_n{k:02d}.png"
                cv2.imwrite(os.path.join(pose_dir, fname), gray)
                print(f"[CALIB] Captured {fname} (mean={gray.mean():.2f})")

        # -----------------------------
        # PSP for calibration (low freqs only)
        # -----------------------------
        calib_freqs = [4, 8]  # low frequencies for calibration

        # Restrict intensity dictionaries to calibration freqs
        I_vert_calib = {
            (f, k): img
            for (f, k), img in I_vert.items()
            if f in calib_freqs
        }
        I_horiz_calib = {
            (f, k): img
            for (f, k), img in I_horiz.items()
            if f in calib_freqs
        }

        # VERTICAL calibration PSP
        print("[CALIB] Running PSP for vertical fringes (calibration)...")
        phi_v_wrapped, masks_v_raw = run_psp_calibration(
            I_vert_calib,
            freqs=calib_freqs,
            n_phase=self.n_phase,
        )
        masks_v_merged = merge_masks_calibration(masks_v_raw, freqs=calib_freqs)
        unwrap_v = temporal_unwrap(
            phi_wrapped=phi_v_wrapped,
            mask_merged=masks_v_merged,
            freqs=calib_freqs,
        )
        Phi_vert = unwrap_v.Phi_final
        mask_vert = unwrap_v.mask_final

        np.save(os.path.join(pose_dir, "phase_vert_final.npy"), Phi_vert)
        np.save(os.path.join(pose_dir, "mask_vert_final.npy"), mask_vert)
        self._save_phase_debug(Phi_vert, mask_vert, pose_dir, filename="phase_vert_debug.png")

        # HORIZONTAL calibration PSP
        print("[CALIB] Running PSP for horizontal fringes (calibration)...")
        phi_h_wrapped, masks_h_raw = run_psp_calibration(
            I_horiz_calib,
            freqs=calib_freqs,
            n_phase=self.n_phase,
        )
        masks_h_merged = merge_masks_calibration(masks_h_raw, freqs=calib_freqs)
        unwrap_h = temporal_unwrap(
            phi_wrapped=phi_h_wrapped,
            mask_merged=masks_h_merged,
            freqs=calib_freqs,
        )
        Phi_horiz = unwrap_h.Phi_final
        mask_horiz = unwrap_h.mask_final

        np.save(os.path.join(pose_dir, "phase_horiz_final.npy"), Phi_horiz)
        np.save(os.path.join(pose_dir, "mask_horiz_final.npy"), mask_horiz)
        self._save_phase_debug(Phi_horiz, mask_horiz, pose_dir, filename="phase_horiz_debug.png")

        print(f"[CALIB] Calibration pose COMPLETE: data saved in {pose_dir}")
        return pose_dir
    
    def run_calib_pose_psp(self):
        session = self.current_psp_session
        pose_dir = os.path.join(session, f"pose_{self.psp_pose_index:03d}")
        os.makedirs(pose_dir, exist_ok=True)
        self.psp_pose_index += 1

        # AE
        def set_midgrey():
            self.set_surface_callback(self.midgrey_surface)
        self.camera.auto_expose_with_midgrey(set_midgrey_surface_callback=set_midgrey)

        # VERTICAL capture
        I_vert = {}
        for f in self.freqs:
            for k in range(self.n_phase):
                self.set_surface_callback(self.patterns[f][k])
                time.sleep(self.pattern_settle_time)
                gray = self.camera.capture_gray().astype(np.float32)
                I_vert[(f, k)] = gray
                cv2.imwrite(os.path.join(pose_dir, f"vert_f{f:03d}_n{k:02d}.png"), gray)

        # HORIZONTAL capture
        I_horiz = {}
        for f in self.freqs:
            for k in range(self.n_phase):
                surf = pygame.transform.rotate(self.patterns[f][k], 90)
                self.set_surface_callback(surf)
                time.sleep(self.pattern_settle_time)
                gray = self.camera.capture_gray().astype(np.float32)
                I_horiz[(f, k)] = gray
                cv2.imwrite(os.path.join(pose_dir, f"horiz_f{f:03d}_n{k:02d}.png"), gray)

        # PSP analysis (low freqs only)
        f_low = [4, 8]
        psp_v = psp_calib.run_psp_calibration(I_vert, f_low, self.n_phase)
        psp_h = psp_calib.run_psp_calibration(I_horiz, f_low, self.n_phase)

        mask_v = masking_calib.merge_masks_calibration(psp_v.mask, f_low)
        mask_h = masking_calib.merge_masks_calibration(psp_h.mask, f_low)

        unwrap_v = temporal_unwrap(psp_v.phi_wrapped, mask_v, f_low)
        unwrap_h = temporal_unwrap(psp_h.phi_wrapped, mask_h, f_low)

        np.save(os.path.join(pose_dir, "phi_vert.npy"), unwrap_v.Phi_final)
        np.save(os.path.join(pose_dir, "mask_vert.npy"), unwrap_v.mask_final)
        np.save(os.path.join(pose_dir, "phi_horiz.npy"), unwrap_h.Phi_final)
        np.save(os.path.join(pose_dir, "mask_horiz.npy"), unwrap_h.mask_final)

        return pose_dir

    def solve_psp_calibration(self, session):
        dataset = build_projector_dataset_psp(
            session_dir=session,
            proj_width=self.proj_w,
            proj_height=self.proj_h,
            freqs=self.freqs,          # ← important
        )
        out = projector_intrinsics.solve_from_dataset(dataset)
        return out

