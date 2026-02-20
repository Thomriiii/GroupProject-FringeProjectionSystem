"""Run storage utilities."""

from __future__ import annotations

import json
import zipfile
from dataclasses import asdict
import re
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Iterable

import numpy as np
from PIL import Image

from fringe_app.core.models import ScanParams, RunMeta
from fringe_app.phase.psp import PhaseResult
from fringe_app.phase import visualize
from fringe_app.vision.object_roi import ObjectRoiResult, build_reference_from_stack


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

    def _freq_tag(self, freq: float) -> str:
        if float(freq).is_integer():
            return f"f_{int(freq):03d}"
        safe = str(freq).replace(".", "p")
        return f"f_{safe}"

    def save_capture(self, run_dir: Path, index: int, image: np.ndarray, freq: float | None = None) -> Path:
        cap_dir = run_dir / "captures"
        if freq is not None:
            cap_dir = cap_dir / self._freq_tag(freq)
            cap_dir.mkdir(exist_ok=True, parents=True)
        out_path = cap_dir / f"step_{index:03d}.png" if freq is not None else cap_dir / f"frame_{index:03d}.png"
        Image.fromarray(image).save(out_path)
        return out_path

    def save_pattern(self, run_dir: Path, index: int, image: np.ndarray, freq: float | None = None) -> Path:
        pat_dir = run_dir / "patterns"
        if freq is not None:
            pat_dir = pat_dir / self._freq_tag(freq)
            pat_dir.mkdir(exist_ok=True, parents=True)
        out_path = pat_dir / f"step_{index:03d}.png" if freq is not None else pat_dir / f"pattern_{index:03d}.png"
        Image.fromarray(image).save(out_path)
        return out_path

    def save_meta(self, run_dir: Path, meta: RunMeta) -> None:
        out_path = run_dir / "meta.json"
        out_path.write_text(json.dumps(meta.to_dict(), indent=2))

    def save_normalise(self, run_dir: Path, data: dict) -> Path:
        norm_dir = run_dir / "normalise"
        norm_dir.mkdir(exist_ok=True, parents=True)
        out = norm_dir / "normalise.json"
        out.write_text(json.dumps(data, indent=2))
        # Backward compatibility with existing tools expecting run-level normalise.json.
        (run_dir / "normalise.json").write_text(json.dumps(data, indent=2))
        return out

    def save_step_sanity(self, run_dir: Path, freq: float, report: dict) -> Path:
        cap_dir = run_dir / "captures" / self._freq_tag(freq)
        cap_dir.mkdir(exist_ok=True, parents=True)
        out = cap_dir / "step_sanity.json"
        out.write_text(json.dumps(report, indent=2))
        return out

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

    def load_captures(self, run_id: str, freq: float | None = None) -> List[np.ndarray]:
        images: List[np.ndarray] = []
        for path in self.capture_paths(run_id, freq=freq):
            images.append(np.array(Image.open(path)))
        return images

    def capture_paths(self, run_id: str, freq: float | None = None) -> List[Path]:
        run_dir = self.root / run_id
        cap_dir = run_dir / "captures"
        if not cap_dir.exists():
            raise FileNotFoundError(f"Captures not found for run {run_id}")
        if freq is not None:
            cap_dir = cap_dir / self._freq_tag(freq)
            pattern = re.compile(r"step_(\d+)\.png$")
        else:
            # Backward compat: if subdirs exist and no freq provided, use top-level frame_*.png.
            pattern = re.compile(r"frame_(\d+)\.png$")
        indexed = []
        for p in cap_dir.glob("*.png"):
            m = pattern.match(p.name)
            if m:
                indexed.append((int(m.group(1)), p))
        if not indexed:
            raise FileNotFoundError("No capture frames found")
        indexed.sort(key=lambda x: x[0])
        return [p for _, p in indexed]

    def iter_captures(self, run_id: str, freq: float | None = None) -> Iterable[np.ndarray]:
        for path in self.capture_paths(run_id, freq=freq):
            yield np.array(Image.open(path))

    def save_phase_outputs(self, run_id: str, result: PhaseResult, freq: float | None = None) -> None:
        run_dir = self.root / run_id
        phase_dir = run_dir / "phase"
        if freq is not None:
            phase_dir = phase_dir / self._freq_tag(freq)
        phase_dir.mkdir(exist_ok=True, parents=True)

        np.save(phase_dir / "phi_wrapped.npy", result.phi_wrapped)
        np.save(phase_dir / "A.npy", result.A)
        np.save(phase_dir / "B.npy", result.B)

        try:
            mask_raw = result.mask_raw.astype(bool) if hasattr(result, "mask_raw") else result.mask.astype(bool)
            mask_clean = result.mask_clean.astype(bool) if hasattr(result, "mask_clean") else result.mask.astype(bool)
            mask_for_unwrap = (
                result.mask_for_unwrap.astype(bool)
                if hasattr(result, "mask_for_unwrap")
                else mask_clean
            )
            mask_for_defects = (
                result.mask_for_defects.astype(bool)
                if hasattr(result, "mask_for_defects")
                else mask_clean
            )
            mask_for_display = (
                result.mask_for_display.astype(bool)
                if hasattr(result, "mask_for_display")
                else mask_for_defects
            )

            visualize.save_mask_png(mask_raw, str(phase_dir / "mask_raw.png"))
            np.save(phase_dir / "mask_raw.npy", mask_raw)
            visualize.save_mask_png(mask_clean, str(phase_dir / "mask_clean.png"))
            np.save(phase_dir / "mask_clean.npy", mask_clean)
            visualize.save_mask_png(mask_for_unwrap, str(phase_dir / "mask_for_unwrap.png"))
            np.save(phase_dir / "mask_for_unwrap.npy", mask_for_unwrap)
            visualize.save_mask_png(mask_for_defects, str(phase_dir / "mask_for_defects.png"))
            np.save(phase_dir / "mask_for_defects.npy", mask_for_defects)
            visualize.save_mask_png(mask_for_display, str(phase_dir / "mask_for_display.png"))
            np.save(phase_dir / "mask_for_display.npy", mask_for_display)

            # Default mask path is explicitly RAW from now on.
            visualize.save_mask_png(mask_raw, str(phase_dir / "mask.png"))
            np.save(phase_dir / "mask.npy", mask_raw)
            visualize.save_mask_png(result.clipped_any_map, str(phase_dir / "clipped_any.png"))
            np.save(phase_dir / "clipped_any.npy", result.clipped_any_map)
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

        result.debug["mask_files"] = {
            "raw": "mask_raw.npy",
            "clean": "mask_clean.npy",
            "for_unwrap": "mask_for_unwrap.npy",
            "for_defects": "mask_for_defects.npy",
            "for_display": "mask_for_display.npy",
            "default_mask_npy_is": "raw",
        }
        meta_path = phase_dir / "phase_meta.json"
        meta_path.write_text(json.dumps(result.debug, indent=2))

    def save_roi(
        self,
        run_id: str,
        roi_mask: np.ndarray,
        bbox,
        roi_meta: dict,
        roi_raw: np.ndarray | None = None,
        roi_post: np.ndarray | None = None,
    ) -> None:
        run_dir = self.root / run_id
        roi_dir = run_dir / "roi"
        roi_dir.mkdir(exist_ok=True)
        try:
            visualize.save_mask_png(roi_mask, str(roi_dir / "roi_mask.png"))
            if roi_raw is not None:
                visualize.save_mask_png(roi_raw, str(roi_dir / "roi_raw.png"))
            if roi_post is not None:
                visualize.save_mask_png(roi_post, str(roi_dir / "roi_post.png"))
        except Exception:
            pass
        meta = {"bbox": bbox, **roi_meta}
        (roi_dir / "roi_meta.json").write_text(json.dumps(meta, indent=2))

    def load_reference_image(self, run_id: str, ref_method: str = "median_over_frames") -> np.ndarray:
        run_dir = self.root / run_id
        cap_dir = run_dir / "captures"
        if not cap_dir.exists():
            raise FileNotFoundError("Captures not found")
        paths: List[Path] = []
        subdirs = [p for p in cap_dir.iterdir() if p.is_dir() and p.name.startswith("f_")]
        if subdirs:
            for sd in subdirs:
                paths.extend(sorted(sd.glob("step_*.png")))
        else:
            paths = sorted(cap_dir.glob("frame_*.png"))
        if not paths:
            raise FileNotFoundError("No capture frames found")
        gray_frames: list[np.ndarray] = []
        for p in paths:
            arr = np.array(Image.open(p))
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr_f = arr.astype(np.float32)
                gray = 0.299 * arr_f[:, :, 0] + 0.587 * arr_f[:, :, 1] + 0.114 * arr_f[:, :, 2]
                gray_frames.append(np.clip(np.rint(gray), 0, 255).astype(np.uint8))
            elif arr.ndim == 2:
                gray_frames.append(arr.astype(np.uint8))
            else:
                raise ValueError(f"Unsupported capture format in {p}")
        stack_u8 = np.stack(gray_frames, axis=0)
        return build_reference_from_stack(stack_u8, ref_method=ref_method)  # type: ignore[arg-type]
