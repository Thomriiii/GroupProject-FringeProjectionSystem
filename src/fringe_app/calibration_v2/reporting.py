"""Report and plot writers for projector calibration v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _plot_canvas(width: int = 960, height: int = 540, bg: tuple[int, int, int] = (245, 246, 248)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (int(width), int(height)), bg)
    draw = ImageDraw.Draw(img)
    return img, draw


def _draw_axes(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int) -> None:
    draw.line((x0, y0, x0, y1), fill=(70, 70, 70), width=2)
    draw.line((x0, y1, x1, y1), fill=(70, 70, 70), width=2)


def save_coverage_plot(coverage: dict[str, Any], out_path: Path) -> None:
    grid_x = max(1, int(coverage.get("grid_x", 3)))
    grid_y = max(1, int(coverage.get("grid_y", 3)))
    visited = coverage.get("visited_bins", []) or []
    if len(visited) != grid_y or any(len(row) != grid_x for row in visited):
        visited = [[0 for _ in range(grid_x)] for _ in range(grid_y)]

    cell = 60
    pad = 40
    img = Image.new("RGB", (grid_x * cell + pad * 2, grid_y * cell + pad * 2 + 60), (24, 26, 30))
    draw = ImageDraw.Draw(img)

    for y in range(grid_y):
        for x in range(grid_x):
            x0 = pad + x * cell
            y0 = pad + y * cell
            x1 = x0 + cell - 4
            y1 = y0 + cell - 4
            hit = int(visited[y][x]) > 0
            fill = (82, 172, 96) if hit else (70, 72, 78)
            edge = (35, 120, 58) if hit else (110, 110, 110)
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=edge, width=2)

    bins_filled = int(coverage.get("bins_filled", 0))
    min_bins = int(coverage.get("min_bins_filled", 0))
    bucket_counts = [int(v) for v in (coverage.get("bucket_counts", []) or [])]
    min_bucket_counts = [int(v) for v in (coverage.get("min_bucket_counts", []) or [])]
    indicator = str(coverage.get("indicator", "Need more variety"))
    draw.text((pad, grid_y * cell + pad + 10), f"bins={bins_filled} / min={min_bins}", fill=(230, 230, 230))
    draw.text((pad, grid_y * cell + pad + 28), f"tilt_counts={bucket_counts} / min={min_bucket_counts}", fill=(230, 230, 230))
    draw.text((pad, grid_y * cell + pad + 46), f"{indicator}", fill=(255, 220, 120))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_reprojection_plot(
    per_view_errors: list[dict[str, Any]],
    *,
    key: str,
    title: str,
    out_path: Path,
) -> None:
    img, draw = _plot_canvas()
    x0, y0, x1, y1 = 70, 40, 920, 470
    _draw_axes(draw, x0, y0, x1, y1)

    vals = np.asarray([float(v.get(key, np.nan)) for v in per_view_errors], dtype=np.float64)
    vals = np.where(np.isfinite(vals), vals, np.nan)
    finite = vals[np.isfinite(vals)]
    vmax = float(np.max(finite)) if finite.size > 0 else 1.0
    vmax = max(vmax, 1e-6)

    n = len(per_view_errors)
    bar_w = max(4, int((x1 - x0 - 12) / max(1, n)))
    for i, item in enumerate(per_view_errors):
        val = float(item.get(key, np.nan))
        if not np.isfinite(val):
            continue
        px0 = x0 + 6 + i * bar_w
        px1 = min(x1 - 1, px0 + bar_w - 2)
        h = int((val / vmax) * (y1 - y0 - 8))
        py0 = y1 - h
        draw.rectangle((px0, py0, px1, y1 - 1), fill=(70, 130, 220), outline=(35, 75, 160))

    draw.text((x0, 12), title, fill=(30, 30, 30))
    if finite.size > 0:
        draw.text((x0, 490), f"min={float(np.min(finite)):.4f}px  median={float(np.median(finite)):.4f}px  max={float(np.max(finite)):.4f}px", fill=(30, 30, 30))
    else:
        draw.text((x0, 490), "No finite errors", fill=(30, 30, 30))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_residual_histogram(
    camera_residuals_px: np.ndarray,
    projector_residuals_px: np.ndarray,
    out_path: Path,
) -> None:
    cam = np.asarray(camera_residuals_px, dtype=np.float64).reshape(-1)
    proj = np.asarray(projector_residuals_px, dtype=np.float64).reshape(-1)
    cam = cam[np.isfinite(cam)]
    proj = proj[np.isfinite(proj)]

    img, draw = _plot_canvas()
    x0, y0, x1, y1 = 70, 40, 920, 470
    _draw_axes(draw, x0, y0, x1, y1)

    all_vals = np.concatenate([cam, proj], axis=0) if cam.size > 0 or proj.size > 0 else np.asarray([1.0], dtype=np.float64)
    vmax = float(np.max(all_vals)) if all_vals.size > 0 else 1.0
    vmax = max(vmax, 1e-3)
    bins = np.linspace(0.0, vmax, 26)

    cam_hist, _ = np.histogram(cam, bins=bins)
    proj_hist, _ = np.histogram(proj, bins=bins)
    hmax = max(int(np.max(cam_hist)) if cam_hist.size > 0 else 0, int(np.max(proj_hist)) if proj_hist.size > 0 else 0, 1)

    n = len(bins) - 1
    bw = max(3, int((x1 - x0 - 20) / max(1, n)))
    for i in range(n):
        base_x = x0 + 10 + i * bw
        ch = int((int(cam_hist[i]) / hmax) * (y1 - y0 - 10))
        ph = int((int(proj_hist[i]) / hmax) * (y1 - y0 - 10))
        if ch > 0:
            draw.rectangle((base_x, y1 - ch, base_x + max(1, bw // 2) - 1, y1 - 1), fill=(80, 150, 230))
        if ph > 0:
            draw.rectangle((base_x + max(1, bw // 2), y1 - ph, base_x + bw - 1, y1 - 1), fill=(240, 130, 80))

    draw.text((x0, 12), "Residual Histogram (camera blue / projector orange)", fill=(30, 30, 30))
    draw.text((x0, 490), f"camera_n={cam.size} projector_n={proj.size} max_bin={hmax}", fill=(30, 30, 30))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def build_solve_report(
    *,
    raw_result: dict[str, Any],
    pruned_result: dict[str, Any],
    selected_model: str,
    prune_steps: list[dict[str, Any]],
    suggestions: list[str],
) -> dict[str, Any]:
    return {
        "raw_rms_stereo": float(raw_result.get("rms_stereo", float("nan"))),
        "pruned_rms_stereo": float(pruned_result.get("rms_stereo", float("nan"))),
        "raw_views_used": int(raw_result.get("views_used", 0)),
        "pruned_views_used": int(pruned_result.get("views_used", 0)),
        "selected_model": str(selected_model),
        "prune_steps": prune_steps,
        "suggestions": [str(s) for s in suggestions],
        "raw_per_view": raw_result.get("per_view_errors", []),
        "pruned_per_view": pruned_result.get("per_view_errors", []),
    }
