"""
Validation tests for 3D reconstruction quality.

Tests verify that the reconstruction pipeline produces:
  - Flat planes with minimal Z variance
  - Consistent depth across valid regions
  - Low reprojection errors
  - Proper outlier filtering
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np


class ReconstructionValidator:
    """Validates reconstruction output quality."""

    def __init__(self, reconstruct_dir: Path):
        """
        Load reconstruction from directory.

        Args:
            reconstruct_dir: Path to reconstruct/ output directory
        """
        self.dir = Path(reconstruct_dir)
        self._load_data()

    def _load_data(self) -> None:
        """Load all reconstruction arrays."""
        self.xyz = np.load(self.dir / "xyz.npy")
        self.depth = np.load(self.dir / "depth.npy")
        self.mask = np.load(self.dir / "masks" / "mask_reconstruct.npy")
        self.reproj_cam = np.load(self.dir / "reproj_err_cam.npy")
        self.reproj_proj = np.load(self.dir / "reproj_err_proj.npy")

        # Load metadata
        meta_path = self.dir / "reconstruction_meta.json"
        self.meta = json.loads(meta_path.read_text())

    def test_point_cloud_density(self, min_coverage: float = 0.5) -> tuple[bool, str]:
        """
        Check that we have sufficient 3D points.

        Args:
            min_coverage: Minimum fraction of valid pixels (default 50%)

        Returns:
            Tuple of (pass, message)
        """
        total_pixels = self.mask.size
        valid_pixels = np.count_nonzero(self.mask)
        coverage = valid_pixels / total_pixels

        if coverage >= min_coverage:
            return True, f"✓ Point cloud coverage: {coverage:.1%} (required {min_coverage:.1%})"
        else:
            return False, f"✗ Point cloud coverage: {coverage:.1%} (required {min_coverage:.1%})"

    def test_depth_range(self) -> tuple[bool, str]:
        """
        Check that depth values are within expected range.

        Returns:
            Tuple of (pass, message)
        """
        z_min = float(self.meta.get("z_min_m", 0.05))
        z_max = float(self.meta.get("z_max_m", 5.0))
        z_values = self.depth[self.mask]

        if z_values.size == 0:
            return False, "✗ No valid depth values"

        actual_min = float(np.min(z_values))
        actual_max = float(np.max(z_values))

        if actual_min >= z_min and actual_max <= z_max:
            return True, (
                f"✓ Depth range [{actual_min:.3f}, {actual_max:.3f}]m "
                f"within limits [{z_min}, {z_max}]m"
            )
        else:
            return False, (
                f"✗ Depth range [{actual_min:.3f}, {actual_max:.3f}]m "
                f"exceeds limits [{z_min}, {z_max}]m"
            )

    def test_flatness(self, window_size: int = 64, max_std: float = 0.01) -> tuple[bool, str]:
        """
        Test flatness by checking Z variance in local windows.

        For flat object: local standard deviation should be very small.

        Args:
            window_size: Size of sliding window (pixels)
            max_std: Maximum acceptable std deviation (meters)

        Returns:
            Tuple of (pass, message)
        """
        h, w = self.mask.shape
        local_stds = []

        # Slide window across image
        for y in range(0, h - window_size, window_size):
            for x in range(0, w - window_size, window_size):
                window = self.depth[y : y + window_size, x : x + window_size]
                window_mask = self.mask[y : y + window_size, x : x + window_size]

                if np.count_nonzero(window_mask) > 4:
                    valid_z = window[window_mask]
                    std = np.std(valid_z)
                    local_stds.append(std)

        if not local_stds:
            return False, "✗ Too few valid regions for flatness test"

        mean_std = np.mean(local_stds)
        max_observed_std = np.max(local_stds)

        if mean_std <= max_std:
            return True, (
                f"✓ Flatness test passed: mean local std = {mean_std:.6f}m "
                f"(max {max_observed_std:.6f}m, threshold {max_std}m)"
            )
        else:
            return False, (
                f"✗ Flatness test failed: mean local std = {mean_std:.6f}m "
                f"(exceeds threshold {max_std}m). Surface may be warped/slanted."
            )

    def test_no_tilt(self, max_tilt_deg: float = 2.0) -> tuple[bool, str]:
        """
        Test for systematic tilt by fitting a plane to the data.

        Args:
            max_tilt_deg: Maximum acceptable tilt angle (degrees)

        Returns:
            Tuple of (pass, message)
        """
        if np.count_nonzero(self.mask) < 100:
            return False, "✗ Too few points for tilt estimation"

        # Extract valid points
        ys, xs = np.where(self.mask)
        zs = self.depth[ys, xs]

        # Fit plane: Z = ax*X + ay*Y + c
        A = np.column_stack([xs, ys, np.ones(len(xs))])
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(A, zs, rcond=None)
        except np.linalg.LinAlgError:
            return False, "✗ Failed to estimate plane"

        ax, ay, c = coeffs

        # Compute tilt angle from normal vector (ax, ay, -1) → angle to Z
        normal = np.array([ax, ay, -1.0])
        normal = normal / np.linalg.norm(normal)
        z_axis = np.array([0, 0, 1.0])

        # Angle between normal and Z axis
        cos_angle = abs(np.dot(normal, z_axis))
        tilt_rad = np.arccos(np.clip(cos_angle, 0, 1))
        tilt_deg = np.degrees(tilt_rad)

        if tilt_deg <= max_tilt_deg:
            return True, (
                f"✓ No systematic tilt: {tilt_deg:.2f}° "
                f"(threshold {max_tilt_deg:.2f}°)"
            )
        else:
            return False, (
                f"✗ Systematic tilt detected: {tilt_deg:.2f}° "
                f"(exceeds threshold {max_tilt_deg:.2f}°)"
            )

    def test_reprojection_errors(self, max_err_px: float = 3.0) -> tuple[bool, str]:
        """
        Check that reprojection errors are within acceptable range.

        Args:
            max_err_px: Maximum acceptable error (pixels)

        Returns:
            Tuple of (pass, message)
        """
        valid_err_cam = self.reproj_cam[self.mask]
        valid_err_proj = self.reproj_proj[self.mask]

        if valid_err_cam.size == 0:
            return False, "✗ No valid reprojection errors"

        mean_cam = np.mean(valid_err_cam)
        mean_proj = np.mean(valid_err_proj)
        max_cam = np.max(valid_err_cam)
        max_proj = np.max(valid_err_proj)

        cam_ok = mean_cam <= max_err_px
        proj_ok = mean_proj <= max_err_px

        if cam_ok and proj_ok:
            return True, (
                f"✓ Low reprojection errors: "
                f"camera {mean_cam:.2f}px (max {max_cam:.2f}px), "
                f"projector {mean_proj:.2f}px (max {max_proj:.2f}px), "
                f"threshold {max_err_px:.2f}px"
            )
        else:
            reason = []
            if not cam_ok:
                reason.append(f"camera {mean_cam:.2f}px > {max_err_px:.2f}px")
            if not proj_ok:
                reason.append(f"projector {mean_proj:.2f}px > {max_err_px:.2f}px")
            return False, f"✗ High reprojection errors: {', '.join(reason)}"

    def test_no_edge_artifacts(self, edge_fraction: float = 0.1) -> tuple[bool, str]:
        """
        Check that valid pixels don't cluster at image boundaries.

        Args:
            edge_fraction: Fraction of image to exclude as edge (default 10%)

        Returns:
            Tuple of (pass, message)
        """
        h, w = self.mask.shape
        margin = int(max(h, w) * edge_fraction / 2)

        center_region = np.zeros_like(self.mask)
        if margin < h and margin < w:
            center_region[margin : h - margin, margin : w - margin] = True
        else:
            center_region[:, :] = True

        edge_pixels = np.count_nonzero(self.mask & ~center_region)
        center_pixels = np.count_nonzero(self.mask & center_region)
        total_pixels = np.count_nonzero(self.mask)

        if total_pixels == 0:
            return False, "✗ No valid pixels"

        edge_fraction_actual = edge_pixels / total_pixels

        if edge_fraction_actual <= 0.3:  # Allow up to 30% edge pixels
            return True, (
                f"✓ No excessive edge artifacts: "
                f"{center_pixels} center pixels, {edge_pixels} edge pixels"
            )
        else:
            return False, (
                f"✗ Excessive edge clustering: "
                f"{edge_fraction_actual:.1%} of points are at edges"
            )

    def run_all_tests(self) -> dict[str, Any]:
        """
        Run all validation tests.

        Returns:
            Dict with test results and summary
        """
        tests = [
            ("Point Cloud Density", self.test_point_cloud_density()),
            ("Depth Range", self.test_depth_range()),
            ("Flatness", self.test_flatness()),
            ("No Tilt", self.test_no_tilt()),
            ("Reprojection Errors", self.test_reprojection_errors()),
            ("No Edge Artifacts", self.test_no_edge_artifacts()),
        ]

        results = {}
        passed = 0
        failed = 0

        print("\n" + "=" * 70)
        print("RECONSTRUCTION VALIDATION REPORT")
        print("=" * 70)

        for name, (passed_test, message) in tests:
            results[name] = {"passed": passed_test, "message": message}
            print(f"\n{name}:")
            print(f"  {message}")
            if passed_test:
                passed += 1
            else:
                failed += 1

        print("\n" + "=" * 70)
        print(f"SUMMARY: {passed}/{len(tests)} tests passed")
        if failed == 0:
            print("✓ All validation checks passed!")
        else:
            print(f"✗ {failed} test(s) failed - check warnings above")
        print("=" * 70 + "\n")

        return {
            "passed": passed,
            "failed": failed,
            "total": len(tests),
            "tests": results,
        }


def validate_reconstruction(reconstruct_dir: Path) -> int:
    """
    Standalone function to validate a reconstruction directory.

    Args:
        reconstruct_dir: Path to reconstruct/ output

    Returns:
        0 if all tests pass, 1 if any test fails
    """
    validator = ReconstructionValidator(reconstruct_dir)
    results = validator.run_all_tests()
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python validation.py <reconstruct_dir>")
        sys.exit(1)

    reconstruct_dir = Path(sys.argv[1])
    if not reconstruct_dir.exists():
        print(f"Error: {reconstruct_dir} does not exist")
        sys.exit(1)

    exit_code = validate_reconstruction(reconstruct_dir)
    sys.exit(exit_code)
