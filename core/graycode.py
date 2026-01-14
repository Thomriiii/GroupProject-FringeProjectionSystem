"""
graycode.py

Binary GrayCode pattern generation and decoding for projector calibration.

This module generates bitwise GrayCode patterns (with inverted pairs) for both
projector axes and decodes a captured sequence into projector pixel coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pygame


@dataclass
class GrayPattern:
    """Single GrayCode pattern frame and its metadata."""
    axis: str               # "u" or "v".
    bit: int                # Bit index (0 = MSB).
    inverted: bool
    surface: pygame.Surface
    raw: np.ndarray         # float32 in [0..1].
    name: str


@dataclass
class GrayCodeSet:
    """Collection of GrayCode patterns plus projector metadata."""
    width: int
    height: int
    bits_u: int
    bits_v: int
    patterns: List[GrayPattern]


def _gray_encode(n: np.ndarray) -> np.ndarray:
    """Convert integer values to Gray code."""
    return n ^ (n >> 1)


def _make_bit_planes(length: int, bits: int) -> np.ndarray:
    """
    Returns array of shape (bits, length) with MSB-first GrayCode bits.
    """
    coords = np.arange(length, dtype=np.uint32)
    gray = _gray_encode(coords)
    planes = np.zeros((bits, length), dtype=np.uint8)
    for b in range(bits):
        shift = bits - 1 - b
        planes[b] = ((gray >> shift) & 1).astype(np.uint8)
    return planes


def _apply_gamma(img: np.ndarray, gamma: float | None) -> np.ndarray:
    """Apply optional gamma compensation to a linear [0..1] image."""
    if gamma is None:
        return img
    eps = 1e-6
    img = np.clip(img, eps, 1.0 - eps)
    return img ** (1.0 / gamma)


def _make_surface_from_mask(mask: np.ndarray, gamma_proj: float | None) -> Tuple[pygame.Surface, np.ndarray]:
    """
    mask: float array in [0,1]
    Returns pygame surface and linear float image.
    """
    img = mask.astype(np.float32)
    img = _apply_gamma(img, gamma_proj)
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    rgb = np.dstack([img_u8] * 3)
    surf = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
    return surf, img.copy()


def generate_graycode_patterns(
    width: int,
    height: int,
    gamma_proj: float | None = None,
    brightness_scale: float = 1.0,
) -> GrayCodeSet:
    """
    Generate GrayCode patterns (with inverted pairs) covering width/height.

    Returns
    -------
    GrayCodeSet with ordered patterns list. The capture/decoding order is the
    same as the patterns list (u bits first, then v bits), each as a normal
    frame followed by its inverted frame.
    """
    bits_u = math.ceil(math.log2(width))
    bits_v = math.ceil(math.log2(height))

    planes_u = _make_bit_planes(width, bits_u)
    planes_v = _make_bit_planes(height, bits_v)

    patterns_list: List[GrayPattern] = []

    # U direction (horizontal codes varying along x).
    for b in range(bits_u):
        base = np.tile(planes_u[b][None, :], (height, 1)).astype(np.float32)
        mask = base * brightness_scale
        surf, raw = _make_surface_from_mask(mask, gamma_proj)
        patterns_list.append(GrayPattern(axis="u", bit=b, inverted=False, surface=surf, raw=raw, name=f"u_b{b:02d}_pos"))

        mask_inv = (1.0 - base) * brightness_scale
        surf_inv, raw_inv = _make_surface_from_mask(mask_inv, gamma_proj)
        patterns_list.append(GrayPattern(axis="u", bit=b, inverted=True, surface=surf_inv, raw=raw_inv, name=f"u_b{b:02d}_neg"))

    # V direction (vertical codes varying along y).
    for b in range(bits_v):
        base = np.tile(planes_v[b][:, None], (1, width)).astype(np.float32)
        mask = base * brightness_scale
        surf, raw = _make_surface_from_mask(mask, gamma_proj)
        patterns_list.append(GrayPattern(axis="v", bit=b, inverted=False, surface=surf, raw=raw, name=f"v_b{b:02d}_pos"))

        mask_inv = (1.0 - base) * brightness_scale
        surf_inv, raw_inv = _make_surface_from_mask(mask_inv, gamma_proj)
        patterns_list.append(GrayPattern(axis="v", bit=b, inverted=True, surface=surf_inv, raw=raw_inv, name=f"v_b{b:02d}_neg"))

    return GrayCodeSet(
        width=width,
        height=height,
        bits_u=bits_u,
        bits_v=bits_v,
        patterns=patterns_list,
    )


def _gray_to_binary(gray_bits: np.ndarray) -> np.ndarray:
    """
    gray_bits: (..., bits) bool
    Returns binary bits (..., bits) bool.
    """
    binary = np.zeros_like(gray_bits, dtype=np.uint8)
    if gray_bits.shape[-1] == 0:
        return binary

    binary[..., 0] = gray_bits[..., 0]
    for i in range(1, gray_bits.shape[-1]):
        binary[..., i] = np.bitwise_xor(binary[..., i - 1], gray_bits[..., i])
    return binary


def _bits_to_values(binary_bits: np.ndarray) -> np.ndarray:
    """
    binary_bits: (..., bits) uint8 where bits[0] is MSB.
    Returns integer values.
    """
    vals = np.zeros(binary_bits.shape[:-1], dtype=np.int32)
    bits = binary_bits.shape[-1]
    for i in range(bits):
        shift = bits - 1 - i
        vals |= (binary_bits[..., i].astype(np.int32) << shift)
    return vals


def decode_graycode(
    frames: List[np.ndarray],
    pattern_set: GrayCodeSet,
    min_contrast: float = 5.0,
    norm_thresh: float = 0.04,
    min_fraction_bits: float = 0.6,
    debug_dir: Optional[str | Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Decode a captured GrayCode sequence.

    Parameters
    ----------
    frames : list[np.ndarray]
        Grayscale images captured in the same order as pattern_set.patterns.
    pattern_set : GrayCodeSet
    min_contrast : float
        Minimum absolute difference between a normal/inverted pair to accept
        the bit for a pixel.

    Returns
    -------
    proj_u, proj_v, valid_mask, debug_dict
    """
    if len(frames) != len(pattern_set.patterns):
        raise ValueError(f"Expected {len(pattern_set.patterns)} frames, got {len(frames)}")

    H, W = frames[0].shape[:2]
    if H == 0 or W == 0:
        raise ValueError("Empty frames supplied to decode_graycode.")

    frames_f = [f.astype(np.float32) for f in frames]

    # Group by axis/bit.
    normals_u = [None] * pattern_set.bits_u
    inverses_u = [None] * pattern_set.bits_u
    normals_v = [None] * pattern_set.bits_v
    inverses_v = [None] * pattern_set.bits_v

    for frame, pat in zip(frames_f, pattern_set.patterns):
        if pat.axis == "u":
            if pat.inverted:
                inverses_u[pat.bit] = frame
            else:
                normals_u[pat.bit] = frame
        else:
            if pat.inverted:
                inverses_v[pat.bit] = frame
            else:
                normals_v[pat.bit] = frame

    def _decode_axis(normals, inverses, bits, axis_name):
        """Decode GrayCode for one axis (u or v)."""
        if bits == 0:
            return (
                np.zeros((H, W), dtype=np.float32),
                np.zeros((H, W), dtype=bool),
                np.zeros((H, W), dtype=np.float32),
                np.zeros((H, W, 0), dtype=np.float32),
            )

        diff_stack = []
        valid_stack = []
        for i in range(bits):
            a = normals[i]
            b = inverses[i]
            if a is None or b is None:
                raise ValueError(f"Missing frames for {axis_name} bit {i}")

            diff = a - b
            norm = np.abs(a) + np.abs(b) + 1e-6
            valid = (np.abs(diff) > min_contrast) | (np.abs(diff / norm) > norm_thresh)
            diff_stack.append(diff)
            valid_stack.append(valid)

        diff_stack = np.stack(diff_stack, axis=-1)
        valid_stack = np.stack(valid_stack, axis=-1)
        bits_gray = (diff_stack > 0).astype(np.uint8)
        bits_gray[~valid_stack] = 0

        bits_bin = _gray_to_binary(bits_gray)
        vals = _bits_to_values(bits_bin)
        required = max(1, int(math.ceil(bits * min_fraction_bits)))
        valid_axis = (valid_stack.sum(axis=-1) >= required)
        return vals.astype(np.float32), valid_axis, diff_stack.mean(axis=-1), diff_stack

    u_vals, valid_u, diff_u, diff_stack_u = _decode_axis(normals_u, inverses_u, pattern_set.bits_u, "u")
    v_vals, valid_v, diff_v, diff_stack_v = _decode_axis(normals_v, inverses_v, pattern_set.bits_v, "v")

    u_vals = np.clip(u_vals, 0, pattern_set.width - 1).astype(np.float32)
    v_vals = np.clip(v_vals, 0, pattern_set.height - 1).astype(np.float32)

    valid = valid_u & valid_v

    u_vals[~valid] = np.nan
    v_vals[~valid] = np.nan

    debug = {
        "valid": valid,
        "valid_u": valid_u,
        "valid_v": valid_v,
        "diff_u": diff_u,
        "diff_v": diff_v,
        "diff_stack_u": diff_stack_u,
        "diff_stack_v": diff_stack_v,
    }

    if debug_dir is not None:
        try:
            import cv2
        except ImportError:
            pass
        else:
            dpath = Path(debug_dir)
            dpath.mkdir(parents=True, exist_ok=True)

            def _save_color(arr: np.ndarray, max_val: float, name: str):
                arr_disp = arr.copy()
                arr_disp[~np.isfinite(arr_disp)] = 0
                arr_disp = np.clip(arr_disp, 0, max_val)
                norm = (arr_disp / (max_val + 1e-6) * 255.0).astype(np.uint8)
                img = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
                cv2.imwrite(str(dpath / name), img)

            _save_color(u_vals, pattern_set.width - 1, "decoded_u.png")
            _save_color(v_vals, pattern_set.height - 1, "decoded_v.png")
            cv2.imwrite(str(dpath / "valid_mask.png"), (valid.astype(np.uint8) * 255))

            for i in range(diff_stack_u.shape[-1]):
                diff = diff_stack_u[..., i]
                norm = np.abs(diff)
                norm = np.clip(norm / (norm.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
                cv2.imwrite(str(dpath / f"u_bit{i:02d}_diff.png"), norm)
            for i in range(diff_stack_v.shape[-1]):
                diff = diff_stack_v[..., i]
                norm = np.abs(diff)
                norm = np.clip(norm / (norm.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
                cv2.imwrite(str(dpath / f"v_bit{i:02d}_diff.png"), norm)

    return u_vals, v_vals, valid, debug


def decode_with_cleaning(
    frames: List[np.ndarray],
    pattern_set: GrayCodeSet,
    min_contrast: float = 5.0,
    norm_thresh: float = 0.04,
    min_fraction_bits: float = 0.6,
    morph_kernel: int = 3,
    min_component: int = 500,
    debug_dir: Optional[str | Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Decode and optionally clean the validity mask morphologically.
    """
    u_vals, v_vals, valid, debug = decode_graycode(
        frames,
        pattern_set,
        min_contrast=min_contrast,
        norm_thresh=norm_thresh,
        min_fraction_bits=min_fraction_bits,
        debug_dir=debug_dir,
    )

    try:
        import cv2
    except ImportError:
        return u_vals, v_vals, valid, debug

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    valid_clean = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    valid_clean = cv2.morphologyEx(valid_clean, cv2.MORPH_CLOSE, kernel)
    valid_clean = cv2.medianBlur(valid_clean, 3)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(valid_clean, connectivity=8)
    keep = np.zeros_like(valid_clean, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_component:
            keep[labels == i] = 1
    valid_clean = keep.astype(bool)

    u_vals[~valid_clean] = np.nan
    v_vals[~valid_clean] = np.nan
    debug["valid_clean"] = valid_clean
    return u_vals, v_vals, valid_clean, debug


def generate_midgray_surface(width: int, height: int, level: float = 0.5, gamma_proj: float | None = None) -> pygame.Surface:
    """
    Generate a pure mid-gray surface for GrayCode auto-exposure (no brightness scaling).
    """
    img = np.full((height, width), float(np.clip(level, 0.0, 1.0)), dtype=np.float32)
    img = _apply_gamma(img, gamma_proj)
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    rgb = np.dstack([img_u8] * 3)
    return pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
