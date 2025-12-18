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
from core.geometry import compute_projector_uv_from_phase
from core.psp import run_psp_per_frequency
from core.masking import merge_frequency_masks, clean_mask
from core.unwrap import temporal_unwrap

THRESH_GAMMA_PHASE1 = 0.25
THRESH_B_PHASE1 = 20.0
U_JUMP_MAX = 8.0
V_JUMP_MAX = 8.0

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
        graycode: object | None = None,
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
        self.graycode = graycode


        os.makedirs(self.scan_root, exist_ok=True)
        os.makedirs(self.calib_root, exist_ok=True)
        self._logged_capture_shape = False

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
                if not self._logged_capture_shape:
                    h, w = gray.shape[:2]
                    print(f"[SCAN] Camera capture_gray() shape: {w}x{h}")
                    self._logged_capture_shape = True
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

    def _save_psp_quality_maps(self, scan_dir: str, orientation: str, psp_result):
        """
        Persist per-frequency PSP quality metrics for debugging/diagnostics.
        """
        f_hi = max(self.freqs)
        prefix = "vert" if orientation == "vert" else "horiz"
        for f in self.freqs:
            np.save(os.path.join(scan_dir, f"A_{prefix}_f{f}.npy"), psp_result.A[f])
            np.save(os.path.join(scan_dir, f"B_{prefix}_f{f}.npy"), psp_result.B[f])
            np.save(os.path.join(scan_dir, f"gamma_{prefix}_f{f}.npy"), psp_result.gamma[f])
            if hasattr(psp_result, "saturated"):
                np.save(os.path.join(scan_dir, f"sat_{prefix}_f{f}.npy"), psp_result.saturated[f])
        np.save(os.path.join(scan_dir, f"A_{prefix}.npy"), psp_result.A[f_hi])
        np.save(os.path.join(scan_dir, f"B_{prefix}.npy"), psp_result.B[f_hi])
        np.save(os.path.join(scan_dir, f"gamma_{prefix}.npy"), psp_result.gamma[f_hi])
        if hasattr(psp_result, "saturated"):
            np.save(os.path.join(scan_dir, f"sat_{prefix}.npy"), psp_result.saturated[f_hi])


    # =====================================================================
    # NORMAL SCAN PIPELINE
    # =====================================================================

    def run_scan(self, polished: bool = False) -> str:
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
        print(f"[SCAN] Starting scan (polished={polished})...")
        scan_dir = self._make_scan_dir()

        # Auto exposure
        self._run_auto_exposure()

        # -----------------------------------------------------------------
        # Capture and decode VERTICAL patterns (u direction)
        # -----------------------------------------------------------------
        print("[SCAN] Capturing vertical PSP sequence...")
        I_vert = self._capture_psp_sequence(scan_dir, self.patterns, prefix="scan")

        print("[SCAN] Running PSP analysis (vertical)...")
        psp_vert = run_psp_per_frequency(
            I_vert,
            freqs=self.freqs,
            n_phase=self.n_phase,
            apply_input_blur=False,
            median_phase_filter=True,
        )
        self._save_psp_quality_maps(scan_dir, "vert", psp_vert)

        print("[SCAN] Cleaning masks (vertical)...")
        masks_vert = merge_frequency_masks(psp_vert.mask, self.freqs)

        print("[SCAN] Unwrapping temporally (vertical)...")
        unwrap_vert = temporal_unwrap(
            phi_wrapped=psp_vert.phi_wrapped,
            mask_merged=masks_vert,
            freqs=self.freqs,
            spatial_axis=1,   # vertical fringes vary along x → unwrap rows
        )

        Phi_vert = unwrap_vert.Phi_final
        mask_vert = unwrap_vert.mask_final

        # Median filter vertical phase (inside mask)
        print("[SCAN] Applying median filter to vertical phase...")
        Phi_vert_med = median_filter(Phi_vert, size=3)
        Phi_vert_smoothed = Phi_vert.copy()
        Phi_vert_smoothed[mask_vert] = Phi_vert_med[mask_vert]
        Phi_vert = Phi_vert_smoothed

        # -----------------------------------------------------------------
        # Capture and decode HORIZONTAL patterns (v direction)
        # -----------------------------------------------------------------
        if self.patterns_horiz is None:
            raise RuntimeError("Horizontal pattern set is not available but horizontal PSP is required.")

        print("[SCAN] Capturing horizontal PSP sequence...")
        I_horiz = self._capture_psp_sequence(scan_dir, self.patterns_horiz, prefix="scanh")

        print("[SCAN] Running PSP analysis (horizontal)...")
        psp_horiz = run_psp_per_frequency(
            I_horiz,
            freqs=self.freqs,
            n_phase=self.n_phase,
            apply_input_blur=False,
            median_phase_filter=True,
        )
        self._save_psp_quality_maps(scan_dir, "horiz", psp_horiz)

        print("[SCAN] Cleaning masks (horizontal)...")
        masks_horiz = merge_frequency_masks(psp_horiz.mask, self.freqs)

        print("[SCAN] Unwrapping temporally (horizontal)...")
        unwrap_horiz = temporal_unwrap(
            phi_wrapped=psp_horiz.phi_wrapped,
            mask_merged=masks_horiz,
            freqs=self.freqs,
            spatial_axis=0,   # horizontal fringes vary along y → unwrap columns
        )

        Phi_horiz = unwrap_horiz.Phi_final
        mask_horiz = unwrap_horiz.mask_final

        # Median filter horizontal phase (inside mask)
        print("[SCAN] Applying median filter to horizontal phase...")
        Phi_h_med = median_filter(Phi_horiz, size=3)
        Phi_horiz_smoothed = Phi_horiz.copy()
        Phi_horiz_smoothed[mask_horiz] = Phi_h_med[mask_horiz]
        Phi_horiz = Phi_horiz_smoothed

        # in scan.py after Phi_horiz smoothing:
        self._save_phase_debug(Phi_horiz, mask_horiz, scan_dir, filename="phase_horiz_debug.png")

        # -----------------------------------------------------------------
        # Phase-1 quality gating and smoothing
        # -----------------------------------------------------------------
        f_hi = max(self.freqs)
        qual_vert = (
            (psp_vert.gamma[f_hi] > THRESH_GAMMA_PHASE1) &
            (psp_vert.B[f_hi] > THRESH_B_PHASE1) &
            (~psp_vert.saturated[f_hi])
        )
        qual_horiz = (
            (psp_horiz.gamma[f_hi] > THRESH_GAMMA_PHASE1) &
            (psp_horiz.B[f_hi] > THRESH_B_PHASE1) &
            (~psp_horiz.saturated[f_hi])
        )

        mask_quality = (
            (mask_vert & qual_vert) |
            (mask_horiz & qual_horiz)
        )
        mask_quality = clean_mask(mask_quality.astype(np.uint8) > 0, kernel_size=3)
        mask_quality_pre_uv = mask_quality.copy()
        np.save(os.path.join(scan_dir, "mask_quality.npy"), mask_quality)

        print("[SCAN] Applying quality-gated median filter to phases...")
        Phi_vert_filtered = Phi_vert.copy()
        Phi_horiz_filtered = Phi_horiz.copy()
        Phi_vert_med_q = median_filter(Phi_vert, size=3)
        Phi_horiz_med_q = median_filter(Phi_horiz, size=3)
        Phi_vert_filtered[mask_quality] = Phi_vert_med_q[mask_quality]
        Phi_horiz_filtered[mask_quality] = Phi_horiz_med_q[mask_quality]
        Phi_vert_filtered[~mask_quality] = np.nan
        Phi_horiz_filtered[~mask_quality] = np.nan

        np.save(os.path.join(scan_dir, "phase_vert_filtered.npy"), Phi_vert_filtered)
        np.save(os.path.join(scan_dir, "phase_horiz_filtered.npy"), Phi_horiz_filtered)
        Phi_vert = Phi_vert_filtered
        Phi_horiz = Phi_horiz_filtered


        # -----------------------------------------------------------------
        # Combine masks and compute projector UV mapping
        # -----------------------------------------------------------------
        # Use strict intersection for combined validity prior to UV mapping
        mask_both = mask_vert & mask_horiz

        print("[SCAN] Computing projector UV maps from phase...")
        u_map, v_map, mask_final = compute_projector_uv_from_phase(
            phase_vert=Phi_vert,
            phase_horiz=Phi_horiz,
            freqs=self.freqs,
            proj_size=(self.proj_w, self.proj_h),
            mask_vert=mask_vert,
            mask_horiz=mask_horiz,
            apply_affine_normalisation=False,
        )
        mask_final = mask_final & mask_both
        if np.isfinite(u_map).any() and np.isfinite(v_map).any():
            print(f"[SCAN][UV] u range: {np.nanmin(u_map):.2f} .. {np.nanmax(u_map):.2f} (proj_w={self.proj_w})")
            print(f"[SCAN][UV] v range: {np.nanmin(v_map):.2f} .. {np.nanmax(v_map):.2f} (proj_h={self.proj_h})")

        # UV-gradient rejection inside quality mask
        if mask_quality.any():
            mask_uv_smooth = np.ones_like(mask_quality, dtype=bool)
            mq = mask_quality
            # right neighbor
            du_r = np.abs(u_map[:, :-1] - u_map[:, 1:])
            dv_r = np.abs(v_map[:, :-1] - v_map[:, 1:])
            bad_u_r = (du_r > U_JUMP_MAX) & mq[:, :-1] & mq[:, 1:]
            bad_v_r = (dv_r > V_JUMP_MAX) & mq[:, :-1] & mq[:, 1:]
            # down neighbor
            du_d = np.abs(u_map[:-1, :] - u_map[1:, :])
            dv_d = np.abs(v_map[:-1, :] - v_map[1:, :])
            bad_u_d = (du_d > U_JUMP_MAX) & mq[:-1, :] & mq[1:, :]
            bad_v_d = (dv_d > V_JUMP_MAX) & mq[:-1, :] & mq[1:, :]

            mask_uv_smooth[:, :-1] &= ~bad_u_r
            mask_uv_smooth[:, 1:] &= ~bad_u_r
            mask_uv_smooth[:, :-1] &= ~bad_v_r
            mask_uv_smooth[:, 1:] &= ~bad_v_r

            mask_uv_smooth[:-1, :] &= ~bad_u_d
            mask_uv_smooth[1:, :] &= ~bad_u_d
            mask_uv_smooth[:-1, :] &= ~bad_v_d
            mask_uv_smooth[1:, :] &= ~bad_v_d

            mask_quality = mask_quality & mask_uv_smooth
            np.save(os.path.join(scan_dir, "mask_quality.npy"), mask_quality)
            total_pix = mask_uv_smooth.size
            pre_cnt = np.count_nonzero(mask_quality_pre_uv)
            post_cnt = np.count_nonzero(mask_quality)
            rej_uv_pct = 100.0 * max(pre_cnt - post_cnt, 0) / float(pre_cnt if pre_cnt > 0 else 1)
            remaining_pct = 100.0 * post_cnt / float(total_pix)
            print(f"[SCAN][UV] Rejected by UV-gradient: {rej_uv_pct:.2f}% | Remaining mask_quality: {remaining_pct:.2f}%")
        else:
            rej_uv_pct = 0.0
            remaining_pct = 0.0

        # Diagnostics for summary
        diag = {
            "uv_reject_pct": rej_uv_pct,
            "mask_quality_pct": remaining_pct,
        }
        if mask_quality.any():
            b_vals = []
            g_vals = []
            if psp_vert.B:
                b_vals.append(psp_vert.B[f_hi][mask_quality])
                g_vals.append(psp_vert.gamma[f_hi][mask_quality])
            if psp_horiz.B:
                b_vals.append(psp_horiz.B[f_hi][mask_quality])
                g_vals.append(psp_horiz.gamma[f_hi][mask_quality])
            if b_vals:
                med_b = float(np.nanmedian(np.concatenate([bv.ravel() for bv in b_vals])))
                diag["median_B_mask_quality"] = med_b
                print(f"[SCAN][DIAG] Median B inside mask_quality: {med_b:.2f}")
            if g_vals:
                med_g = float(np.nanmedian(np.concatenate([gv.ravel() for gv in g_vals])))
                diag["median_gamma_mask_quality"] = med_g
                print(f"[SCAN][DIAG] Median gamma inside mask_quality: {med_g:.4f}")
        diag_path = os.path.join(scan_dir, "scan_diag_scan.json")
        with open(diag_path, "w", encoding="utf-8") as f:
            import json as _json
            _json.dump(diag, f, indent=2)

        # -----------------------------------------------------------------
        # Save outputs
        # -----------------------------------------------------------------
        np.save(os.path.join(scan_dir, "phase_final.npy"), Phi_vert)  # keep legacy vertical phase
        np.save(os.path.join(scan_dir, "mask_final.npy"), mask_final)
        np.save(os.path.join(scan_dir, "proj_u.npy"), u_map.astype(np.float32))
        np.save(os.path.join(scan_dir, "proj_v.npy"), v_map.astype(np.float32))
        print("[SCAN] Saved phase_final.npy, mask_final.npy, proj_u.npy, proj_v.npy")

        self._save_phase_debug(Phi_vert, mask_final, scan_dir, filename="phase_debug.png")

        print(f"[SCAN] COMPLETE: results saved in {scan_dir}")
        return scan_dir
