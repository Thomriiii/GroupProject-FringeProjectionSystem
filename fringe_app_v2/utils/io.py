"""Filesystem and serialization helpers for the v2 run layout."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from .math_utils import json_safe


RUN_SUBDIRS = ("raw", "roi", "structured", "phase", "unwrap", "reconstruct")


@dataclass(frozen=True, slots=True)
class RunPaths:
    run_id: str
    root: Path
    raw: Path
    roi: Path
    structured: Path
    phase: Path
    unwrap: Path
    reconstruct: Path


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2))


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    save_image(path, np.where(mask.astype(bool), 255, 0).astype(np.uint8))


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def load_image_stack(paths: list[Path]) -> list[np.ndarray]:
    return [load_image(path) for path in paths]


def timestamp_id(suffix: str | None = None) -> str:
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{suffix}" if suffix else base


def create_run(root: Path) -> RunPaths:
    root.mkdir(parents=True, exist_ok=True)
    run_id = timestamp_id()
    run_root = root / run_id
    index = 0
    while run_root.exists():
        index += 1
        run_id = f"{timestamp_id()}_{index:02d}"
        run_root = root / run_id
    paths = RunPaths(
        run_id=run_id,
        root=run_root,
        raw=run_root / "raw",
        roi=run_root / "roi",
        structured=run_root / "structured",
        phase=run_root / "phase",
        unwrap=run_root / "unwrap",
        reconstruct=run_root / "reconstruct",
    )
    for name in RUN_SUBDIRS:
        (run_root / name).mkdir(parents=True, exist_ok=True)
    write_json(
        run_root / "run.json",
        {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "layout": {name: name for name in RUN_SUBDIRS},
            "status": "created",
        },
    )
    return paths


def freq_tag(freq: float) -> str:
    f = float(freq)
    if f.is_integer():
        return f"f_{int(f):03d}"
    return "f_" + str(f).replace(".", "p")


def sorted_step_images(directory: Path) -> list[Path]:
    return sorted(directory.glob("step_*.png"))
