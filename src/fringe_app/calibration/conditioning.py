"""Projector-space conditioning accumulator for calibration capture guidance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np


@dataclass(slots=True)
class ConditioningConfig:
    grid_w: int = 64
    grid_h: int = 36
    min_projector_coverage_ratio: float = 0.6
    min_uniformity_metric: float = 0.40
    min_edge_coverage_ratio: float = 0.25
    stop_when_sufficient: bool = True

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "ConditioningConfig":
        c = dict(cfg or {})
        return cls(
            grid_w=max(4, int(c.get("grid_w", c.get("grid_size_x", 64)))),
            grid_h=max(4, int(c.get("grid_h", c.get("grid_size_y", 36)))),
            min_projector_coverage_ratio=float(c.get("min_projector_coverage_ratio", 0.6)),
            min_uniformity_metric=float(c.get("min_uniformity_metric", 0.40)),
            min_edge_coverage_ratio=float(c.get("min_edge_coverage_ratio", 0.25)),
            stop_when_sufficient=bool(c.get("stop_when_sufficient", True)),
        )


class ConditioningAccumulator:
    def __init__(self, projector_size: tuple[int, int], cfg: ConditioningConfig, grid: np.ndarray | None = None) -> None:
        self.projector_size = (max(1, int(projector_size[0])), max(1, int(projector_size[1])))
        self.cfg = cfg
        if grid is None:
            self.grid = np.zeros((cfg.grid_h, cfg.grid_w), dtype=np.int32)
        else:
            arr = np.asarray(grid, dtype=np.int32)
            if arr.shape != (cfg.grid_h, cfg.grid_w):
                self.grid = np.zeros((cfg.grid_h, cfg.grid_w), dtype=np.int32)
            else:
                self.grid = arr.copy()

    def update(self, projector_points: np.ndarray) -> None:
        pts = np.asarray(projector_points, dtype=np.float64).reshape(-1, 2)
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if pts.size == 0:
            return
        pw, ph = self.projector_size
        u = np.clip(pts[:, 0], 0.0, float(pw - 1))
        v = np.clip(pts[:, 1], 0.0, float(ph - 1))
        bx = np.floor((u / float(pw)) * self.cfg.grid_w).astype(np.int32)
        by = np.floor((v / float(ph)) * self.cfg.grid_h).astype(np.int32)
        bx = np.clip(bx, 0, self.cfg.grid_w - 1)
        by = np.clip(by, 0, self.cfg.grid_h - 1)
        for x, y in zip(bx, by):
            self.grid[y, x] += 1

    def compute_conditioning_score(self) -> dict[str, Any]:
        occupied = self.grid > 0
        coverage_ratio = float(np.mean(occupied))

        edge_mask = np.zeros_like(occupied, dtype=bool)
        edge_mask[0, :] = True
        edge_mask[-1, :] = True
        edge_mask[:, 0] = True
        edge_mask[:, -1] = True
        edge_total = int(np.count_nonzero(edge_mask))
        edge_coverage_ratio = (
            float(np.count_nonzero(occupied & edge_mask) / max(edge_total, 1))
            if edge_total > 0
            else 0.0
        )

        occ_counts = self.grid[occupied].astype(np.float64)
        if occ_counts.size == 0:
            uniformity_metric = 0.0
        else:
            mean = float(np.mean(occ_counts))
            std = float(np.std(occ_counts))
            cv = std / max(mean, 1e-9)
            uniformity_metric = float(1.0 / (1.0 + cv))

        guidance: list[str] = []
        row_cov = occupied.mean(axis=1)
        col_cov = occupied.mean(axis=0)
        rmin = int(np.argmin(row_cov))
        cmin = int(np.argmin(col_cov))
        if float(np.min(row_cov)) < 0.25:
            if rmin < (self.cfg.grid_h // 3):
                guidance.append("Need more top coverage")
            elif rmin > (2 * self.cfg.grid_h // 3):
                guidance.append("Need more bottom coverage")
            else:
                guidance.append("Need more vertical span")
        if float(np.min(col_cov)) < 0.25:
            if cmin < (self.cfg.grid_w // 3):
                guidance.append("Need more left coverage")
            elif cmin > (2 * self.cfg.grid_w // 3):
                guidance.append("Need more right coverage")
            else:
                guidance.append("Need more horizontal span")

        sufficient = bool(
            coverage_ratio >= self.cfg.min_projector_coverage_ratio
            and uniformity_metric >= self.cfg.min_uniformity_metric
        )
        return {
            "grid_size": [int(self.cfg.grid_w), int(self.cfg.grid_h)],
            "bins_x": int(self.cfg.grid_w),
            "bins_y": int(self.cfg.grid_h),
            "projector_size": [int(self.projector_size[0]), int(self.projector_size[1])],
            "coverage_ratio": coverage_ratio,
            "edge_coverage_ratio": edge_coverage_ratio,
            "uniformity_metric": uniformity_metric,
            "guidance": sorted(set(guidance)),
            "sufficient": sufficient,
            "occupied_bins": int(np.count_nonzero(occupied)),
            "covered_bins": int(np.count_nonzero(occupied)),
            "total_bins": int(occupied.size),
            "bins_total": int(occupied.size),
            "row_coverage": row_cov.astype(float).tolist(),
            "col_coverage": col_cov.astype(float).tolist(),
            "grid": self.grid.astype(int).tolist(),
        }

    def to_dict(self) -> dict[str, Any]:
        d = self.compute_conditioning_score()
        d["config"] = {
            "grid_w": int(self.cfg.grid_w),
            "grid_h": int(self.cfg.grid_h),
            "min_projector_coverage_ratio": float(self.cfg.min_projector_coverage_ratio),
            "min_uniformity_metric": float(self.cfg.min_uniformity_metric),
            "min_edge_coverage_ratio": float(self.cfg.min_edge_coverage_ratio),
            "stop_when_sufficient": bool(self.cfg.stop_when_sufficient),
        }
        return d

    @classmethod
    def from_file(
        cls,
        path: Path,
        projector_size: tuple[int, int],
        cfg: ConditioningConfig,
    ) -> "ConditioningAccumulator":
        if not path.exists():
            return cls(projector_size, cfg)
        try:
            data = json.loads(path.read_text())
            grid = np.asarray(data.get("grid", []), dtype=np.int32)
            proj = data.get("projector_size", projector_size)
            psize = (int(proj[0]), int(proj[1])) if isinstance(proj, (list, tuple)) and len(proj) == 2 else projector_size
            return cls(psize, cfg, grid=grid)
        except Exception:
            return cls(projector_size, cfg)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
