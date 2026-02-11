"""Run storage utilities."""

from __future__ import annotations

import json
import zipfile
from dataclasses import asdict
import re
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image

from fringe_app.core.models import ScanParams, RunMeta
from fringe_app.phase.psp import PhaseResult
from fringe_app.phase import visualize
from fringe_app.vision.object_roi import ObjectRoiResult


class RunStore:
    """Filesystem-backed storage for scan runs."""

    def __init__(self, root: str = "data/runs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run(self, params: ScanParams, device_info: dict, preview_enabled: bool) -> tuple[str, Path, RunMeta]:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "captures").mkdir(exist_ok=True)
        (run_dir / "patterns").mkdir(exist_ok=True)

        meta = RunMeta(
            run_id=run_id,
            params=params.to_dict(),
            started_at=datetime.now().isoformat(),
            finished_at=None,
            status="running",
            error=None,
            device_info=device_info,
            total_frames=params.N,
            saved_frames=0,
            preview_enabled=preview_enabled,
        )
        self.save_meta(run_dir, meta)
        return run_id, run_dir, meta

    def save_capture(self, run_dir: Path, index: int, image: np.ndarray) -> Path:
        out_path = run_dir / "captures" / f"frame_{index:03d}.png"
        Image.fromarray(image).save(out_path)
        return out_path

    def save_pattern(self, run_dir: Path, index: int, image: np.ndarray) -> Path:
        out_path = run_dir / "patterns" / f"pattern_{index:03d}.png"
        Image.fromarray(image).save(out_path)
        return out_path

    def save_meta(self, run_dir: Path, meta: RunMeta) -> None:
        out_path = run_dir / "meta.json"
        out_path.write_text(json.dumps(meta.to_dict(), indent=2))

    def update_meta(self, run_dir: Path, **updates) -> RunMeta:
        meta = self.load_meta(run_dir)
        for k, v in updates.items():
            setattr(meta, k, v)
        self.save_meta(run_dir, meta)
        return meta

    def load_meta(self, run_dir: Path) -> RunMeta:
        data = json.loads((run_dir / "meta.json").read_text())
        return RunMeta(**data)

    def list_runs(self) -> List[RunMeta]:
        if not self.root.exists():
            return []
        metas: List[RunMeta] = []
        for run_dir in sorted(self.root.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            meta_path = run_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                metas.append(RunMeta(**json.loads(meta_path.read_text())))
            except Exception:
                continue
        return metas

    def zip_run(self, run_id: str) -> Path:
        run_dir = self.root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(run_id)
        zip_path = self.root / f"{run_id}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in run_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=path.relative_to(run_dir))
        return zip_path

    def load_captures(self, run_id: str) -> List[np.ndarray]:
        run_dir = self.root / run_id
        cap_dir = run_dir / "captures"
        if not cap_dir.exists():
            raise FileNotFoundError(f"Captures not found for run {run_id}")
        pattern = re.compile(r"frame_(\d+)\.png$")
        indexed = []
        for p in cap_dir.glob("frame_*.png"):
            m = pattern.match(p.name)
            if m:
                indexed.append((int(m.group(1)), p))
        if not indexed:
            raise FileNotFoundError("No capture frames found")
        indexed.sort(key=lambda x: x[0])
        images = []
        for _, path in indexed:
            images.append(np.array(Image.open(path)))
        return images

    def save_phase_outputs(self, run_id: str, result: PhaseResult) -> None:
        run_dir = self.root / run_id
        phase_dir = run_dir / "phase"
        phase_dir.mkdir(exist_ok=True)

        np.save(phase_dir / "phi_wrapped.npy", result.phi_wrapped)
        np.save(phase_dir / "A.npy", result.A)
        np.save(phase_dir / "B.npy", result.B)

        try:
            visualize.save_mask_png(result.mask, str(phase_dir / "mask.png"))
            np.save(phase_dir / "mask.npy", result.mask)
            perc = result.debug.get("debug_percentiles", [1.0, 99.0])
            visualize.save_phase_png_autoscale(
                result.phi_wrapped,
                result.mask,
                (float(perc[0]), float(perc[1])),
                str(phase_dir / "phi_debug_autoscale.png"),
            )
            visualize.save_phase_png_fixed(
                result.phi_wrapped,
                result.mask,
                str(phase_dir / "phi_debug_fixed.png"),
            )
            visualize.save_modulation_png(
                result.B,
                result.mask,
                str(phase_dir / "B_debug.png"),
            )
        except Exception as exc:
            result.debug["visualize_error"] = str(exc)

        meta_path = phase_dir / "phase_meta.json"
        meta_path.write_text(json.dumps(result.debug, indent=2))

    def save_roi(self, run_id: str, roi_mask: np.ndarray, bbox, roi_meta: dict) -> None:
        run_dir = self.root / run_id
        roi_dir = run_dir / "roi"
        roi_dir.mkdir(exist_ok=True)
        try:
            visualize.save_mask_png(roi_mask, str(roi_dir / "roi_mask.png"))
        except Exception:
            pass
        meta = {"bbox": bbox, **roi_meta}
        (roi_dir / "roi_meta.json").write_text(json.dumps(meta, indent=2))

    def load_reference_image(self, run_id: str) -> np.ndarray:
        images = self.load_captures(run_id)
        return images[0]
