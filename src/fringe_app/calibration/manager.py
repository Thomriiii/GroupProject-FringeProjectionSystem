"""Calibration session persistence and orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .checkerboard import (
    calibrate_intrinsics,
    detect_checkerboard,
    save_detection_json,
    save_image,
)


@dataclass(slots=True)
class CalibrationConfig:
    root: str = "data/calibration"
    checkerboard_cols: int = 9
    checkerboard_rows: int = 6
    square_size_mm: float = 25.0
    min_valid_detections: int = 10


class CalibrationManager:
    """Filesystem-backed checkerboard calibration manager."""

    def __init__(self, cfg: CalibrationConfig) -> None:
        self.cfg = cfg
        self.root = Path(cfg.root)
        self.sessions_root = self.root / "sessions"
        self.sessions_root.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> dict[str, Any]:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = stamp
        i = 0
        while (self.sessions_root / session_id).exists():
            i += 1
            session_id = f"{stamp}_{i:02d}"
        session_dir = self.sessions_root / session_id
        (session_dir / "captures").mkdir(parents=True, exist_ok=True)
        (session_dir / "detections").mkdir(parents=True, exist_ok=True)
        session = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "checkerboard": {
                "cols": int(self.cfg.checkerboard_cols),
                "rows": int(self.cfg.checkerboard_rows),
                "square_size_mm": float(self.cfg.square_size_mm),
            },
            "min_valid_detections": int(self.cfg.min_valid_detections),
            "captures": [],
            "calibration": None,
        }
        self._save_session(session_id, session)
        return session

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for p in sorted(self.sessions_root.glob("*"), reverse=True):
            if not p.is_dir():
                continue
            meta_path = p / "session.json"
            if not meta_path.exists():
                continue
            try:
                sessions.append(json.loads(meta_path.read_text()))
            except Exception:
                continue
        return sessions

    def load_session(self, session_id: str) -> dict[str, Any]:
        p = self._session_dir(session_id) / "session.json"
        if not p.exists():
            raise FileNotFoundError(f"Calibration session not found: {session_id}")
        return json.loads(p.read_text())

    def capture(self, session_id: str, image: np.ndarray) -> dict[str, Any]:
        session = self.load_session(session_id)
        idx = len(session.get("captures", []))
        capture_id = f"capture_{idx:03d}"
        sdir = self._session_dir(session_id)
        image_rel = Path("captures") / f"{capture_id}.png"
        overlay_rel = Path("captures") / f"{capture_id}_overlay.png"
        detection_rel = Path("detections") / f"{capture_id}.json"

        detection, overlay = detect_checkerboard(
            image=image,
            cols=int(self.cfg.checkerboard_cols),
            rows=int(self.cfg.checkerboard_rows),
            refine_subpix=True,
        )

        save_image(sdir / image_rel, image)
        save_image(sdir / overlay_rel, overlay)
        save_detection_json(
            sdir / detection_rel,
            detection,
            extra={
                "capture_id": capture_id,
                "session_id": session_id,
                "checkerboard": {
                    "cols": int(self.cfg.checkerboard_cols),
                    "rows": int(self.cfg.checkerboard_rows),
                    "square_size_mm": float(self.cfg.square_size_mm),
                },
            },
        )

        record = {
            "capture_id": capture_id,
            "index": idx,
            "timestamp": datetime.now().isoformat(),
            "image_path": str(image_rel),
            "overlay_path": str(overlay_rel),
            "detection_path": str(detection_rel),
            "found": bool(detection.found),
            "corner_count": int(detection.corner_count),
            "image_size": list(detection.image_size),
        }
        session.setdefault("captures", []).append(record)
        self._save_session(session_id, session)
        return {"record": record, "detection": detection.to_dict()}

    def calibrate(self, session_id: str) -> dict[str, Any]:
        session = self.load_session(session_id)
        sdir = self._session_dir(session_id)
        captures = session.get("captures", [])

        found_dets: list[dict[str, Any]] = []
        for rec in captures:
            if not rec.get("found", False):
                continue
            dpath = sdir / rec["detection_path"]
            if not dpath.exists():
                continue
            det = json.loads(dpath.read_text())
            if det.get("found"):
                found_dets.append(det)

        min_valid = int(self.cfg.min_valid_detections)
        if len(found_dets) < min_valid:
            raise ValueError(f"Need at least {min_valid} valid checkerboard detections.")

        intrinsics = calibrate_intrinsics(
            detections=found_dets,
            cols=int(self.cfg.checkerboard_cols),
            rows=int(self.cfg.checkerboard_rows),
            square_size_mm=float(self.cfg.square_size_mm),
        )
        intrinsics["session_id"] = session_id
        intrinsics["created_at"] = datetime.now().isoformat()
        intrinsics["captures_total"] = int(len(captures))
        intrinsics["captures_found"] = int(len(found_dets))

        intrinsics_path = sdir / "intrinsics.json"
        intrinsics_path.write_text(json.dumps(intrinsics, indent=2))
        latest_path = self.root / "intrinsics_latest.json"
        latest_path.write_text(json.dumps(intrinsics, indent=2))

        session["calibration"] = {
            "rms": float(intrinsics["rms"]),
            "captures_found": int(len(found_dets)),
            "updated_at": datetime.now().isoformat(),
            "intrinsics_path": "intrinsics.json",
        }
        self._save_session(session_id, session)
        return intrinsics

    def capture_image_path(self, session_id: str, capture_id: str) -> Path:
        return self._session_dir(session_id) / "captures" / f"{capture_id}.png"

    def overlay_image_path(self, session_id: str, capture_id: str) -> Path:
        return self._session_dir(session_id) / "captures" / f"{capture_id}_overlay.png"

    def detection_path(self, session_id: str, capture_id: str) -> Path:
        return self._session_dir(session_id) / "detections" / f"{capture_id}.json"

    def intrinsics_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "intrinsics.json"

    def _session_dir(self, session_id: str) -> Path:
        return self.sessions_root / session_id

    def _save_session(self, session_id: str, payload: dict[str, Any]) -> None:
        p = self._session_dir(session_id) / "session.json"
        p.write_text(json.dumps(payload, indent=2))
