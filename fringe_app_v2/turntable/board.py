"""
ChArUco calibration board generator for turntable calibration.

Physical design:
  - 9×7 squares, 10 mm each → 90 mm × 70 mm
  - DICT_4X4_50 markers (globally unique IDs; partial views are valid)
  - 300 DPI print resolution
  - Center crosshair: align with turntable rotation axis
  - Scale bar for print verification
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


_DEFAULTS = dict(
    squares_x=9,
    squares_y=7,
    square_length_m=0.010,
    marker_length_m=0.0075,
    dict_name="DICT_4X4_50",
    dpi=300,
    add_center_mark=True,
    add_scale_bar=True,
)


def _mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def _build_board(cfg: dict):
    dict_name = cfg.get("dict_name", _DEFAULTS["dict_name"])
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    board = cv2.aruco.CharucoBoard(
        (cfg.get("squares_x", _DEFAULTS["squares_x"]),
         cfg.get("squares_y", _DEFAULTS["squares_y"])),
        cfg.get("square_length_m", _DEFAULTS["square_length_m"]),
        cfg.get("marker_length_m", _DEFAULTS["marker_length_m"]),
        aruco_dict,
    )
    return aruco_dict, board


def _draw_crosshair(img: np.ndarray, cx: int, cy: int, arm: int, thickness: int = 2) -> None:
    color = (0, 0, 0)
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), color, thickness)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), color, thickness)
    cv2.circle(img, (cx, cy), max(2, arm // 6), color, thickness)


def _draw_scale_bar(img: np.ndarray, dpi: int, margin_px: int) -> None:
    bar_px = _mm_to_px(10, dpi)
    h = img.shape[0]
    x0 = margin_px
    x1 = x0 + bar_px
    y = h - margin_px // 2
    thick = max(1, _mm_to_px(0.3, dpi))
    color = (0, 0, 0)
    cv2.line(img, (x0, y), (x1, y), color, thick)
    cv2.line(img, (x0, y - thick * 3), (x0, y + thick * 3), color, thick)
    cv2.line(img, (x1, y - thick * 3), (x1, y + thick * 3), color, thick)
    fs = max(0.3, _mm_to_px(2.5, dpi) / 40)
    cv2.putText(img, "10 mm", (x0, y - thick * 5), cv2.FONT_HERSHEY_SIMPLEX,
                fs, color, max(1, thick))


def generate_board_image(cfg: dict | None = None) -> np.ndarray:
    """Generate printable ChArUco board image (BGR uint8)."""
    if cfg is None:
        cfg = {}

    squares_x = cfg.get("squares_x", _DEFAULTS["squares_x"])
    squares_y = cfg.get("squares_y", _DEFAULTS["squares_y"])
    sq_m = cfg.get("square_length_m", _DEFAULTS["square_length_m"])
    dpi = cfg.get("dpi", _DEFAULTS["dpi"])
    add_center = cfg.get("add_center_mark", _DEFAULTS["add_center_mark"])
    add_scale = cfg.get("add_scale_bar", _DEFAULTS["add_scale_bar"])

    sq_mm = sq_m * 1000.0
    sq_px = _mm_to_px(sq_mm, dpi)
    board_w_px = squares_x * sq_px
    board_h_px = squares_y * sq_px

    margin_px = _mm_to_px(8, dpi)
    img_w = board_w_px + 2 * margin_px
    img_h = board_h_px + 2 * margin_px

    _, board = _build_board(cfg)
    board_img = board.generateImage((board_w_px, board_h_px), marginSize=0, borderBits=1)
    if board_img.ndim == 2:
        board_img = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)

    canvas = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
    canvas[margin_px:margin_px + board_h_px, margin_px:margin_px + board_w_px] = board_img

    if add_center:
        cx = margin_px + board_w_px // 2
        cy = margin_px + board_h_px // 2
        arm = max(sq_px // 2, _mm_to_px(5, dpi))
        _draw_crosshair(canvas, cx, cy, arm, thickness=max(1, _mm_to_px(0.4, dpi)))

    if add_scale:
        _draw_scale_bar(canvas, dpi, margin_px)

    phys_w_mm = squares_x * sq_mm
    phys_h_mm = squares_y * sq_mm
    info = (f"ChArUco {squares_x}×{squares_y}  sq={sq_mm:.0f}mm  "
            f"board={phys_w_mm:.0f}×{phys_h_mm:.0f}mm  {dpi}dpi  "
            f"align crosshair with turntable axis")
    fs = max(0.3, _mm_to_px(2.5, dpi) / 40)
    cv2.putText(canvas, info, (margin_px, margin_px - _mm_to_px(2, dpi)),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (80, 80, 80), 1, cv2.LINE_AA)

    return canvas


def save_board(output_path: str | Path, cfg: dict | None = None) -> Path:
    """
    Generate and save the calibration board PNG with embedded DPI metadata.

    The DPI is embedded so that any standard print dialog will default to the
    correct physical size without needing to manually set scale. If Pillow is
    not installed the file is saved without DPI metadata and a warning is shown.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg is None:
        cfg = {}
    img = generate_board_image(cfg)
    dpi = int(cfg.get("dpi", _DEFAULTS["dpi"]))
    sq_m = cfg.get("square_length_m", _DEFAULTS["square_length_m"])
    squares_x = cfg.get("squares_x", _DEFAULTS["squares_x"])
    squares_y = cfg.get("squares_y", _DEFAULTS["squares_y"])

    # Embed DPI metadata so the image prints at the correct physical size.
    # cv2.imwrite writes no DPI tag, causing printers to scale the image.
    saved_with_dpi = False
    try:
        from PIL import Image
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.save(str(output_path), dpi=(dpi, dpi))
        saved_with_dpi = True
    except ImportError:
        cv2.imwrite(str(output_path), img)

    print(f"Board saved: {output_path}")
    print(f"  Physical size: {squares_x * sq_m * 1000:.0f} mm × {squares_y * sq_m * 1000:.0f} mm")
    if saved_with_dpi:
        print(f"  DPI metadata embedded ({dpi} dpi) — print at 'Actual Size' / 100% scale")
    else:
        print(f"  WARNING: Pillow not installed, DPI metadata not embedded.")
        print(f"  In your print dialog, manually set scale so 10 mm scale bar = 10 mm on paper.")
    print(f"  Align the crosshair with the turntable rotation axis")
    return output_path
