"""Session create / load / save for turntable calibration."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _angle_label(angle_deg: float) -> str:
    return f"angle_{int(round(angle_deg)):03d}"


@dataclass
class FrameRecord:
    angle_deg: float
    angle_label: str
    image_path: str
    charuco_ok: bool = False
    n_corners: int = 0
    pose_ok: bool = False
    reprojection_error_px: float = 0.0
    note: str = ""

    @property
    def status(self) -> str:
        if not self.charuco_ok or self.n_corners < 4:
            return "bad"
        if not self.pose_ok or self.reprojection_error_px > 2.0:
            return "warning"
        return "good"


@dataclass
class TurntableSession:
    session_id: str
    root: Path
    nominal_step_deg: float = 15.0
    frames: list[FrameRecord] = field(default_factory=list)
    analysed: bool = False
    notes: str = ""

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def calibration_dir(self) -> Path:
        return self.root / "calibration"

    @property
    def session_json(self) -> Path:
        return self.root / "session.json"

    def frame_dir(self, angle_label: str) -> Path:
        return self.frames_dir / angle_label

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)
        data = {
            "session_id": self.session_id,
            "nominal_step_deg": self.nominal_step_deg,
            "analysed": self.analysed,
            "notes": self.notes,
            "frames": [asdict(f) for f in self.frames],
        }
        self.session_json.write_text(json.dumps(data, indent=2))

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "root": str(self.root),
            "nominal_step_deg": self.nominal_step_deg,
            "analysed": self.analysed,
            "notes": self.notes,
            "n_frames": len(self.frames),
            "n_good": sum(1 for f in self.frames if f.status == "good"),
            "frames": [
                {
                    "angle_deg": f.angle_deg,
                    "angle_label": f.angle_label,
                    "status": f.status,
                    "n_corners": f.n_corners,
                    "reprojection_error_px": f.reprojection_error_px,
                    "note": f.note,
                }
                for f in self.frames
            ],
        }


def new_session(storage_root: str | Path, nominal_step_deg: float = 15.0) -> TurntableSession:
    storage_root = Path(storage_root)
    session_id = _timestamp_id()
    root = storage_root / session_id
    session = TurntableSession(
        session_id=session_id,
        root=root,
        nominal_step_deg=nominal_step_deg,
    )
    session.save()
    return session


def load_session(storage_root: str | Path, session_id: str) -> TurntableSession:
    path = Path(storage_root) / session_id / "session.json"
    if not path.exists():
        raise FileNotFoundError(f"Session not found: {path}")
    data = json.loads(path.read_text())
    frames = [FrameRecord(**f) for f in data.get("frames", [])]
    return TurntableSession(
        session_id=data["session_id"],
        root=Path(storage_root) / session_id,
        nominal_step_deg=data.get("nominal_step_deg", 15.0),
        frames=frames,
        analysed=data.get("analysed", False),
        notes=data.get("notes", ""),
    )


def list_sessions(storage_root: str | Path) -> list[dict[str, Any]]:
    storage_root = Path(storage_root)
    sessions = []
    if not storage_root.exists():
        return sessions
    for d in sorted(storage_root.iterdir(), reverse=True):
        jf = d / "session.json"
        if jf.exists():
            try:
                data = json.loads(jf.read_text())
                sessions.append({
                    "session_id": data.get("session_id", d.name),
                    "nominal_step_deg": data.get("nominal_step_deg", 15.0),
                    "n_frames": len(data.get("frames", [])),
                    "analysed": data.get("analysed", False),
                })
            except Exception:
                pass
    return sessions


def add_frame(
    session: TurntableSession,
    angle_deg: float,
    image_path: str,
    note: str = "",
) -> FrameRecord:
    label = _angle_label(angle_deg)
    rec = FrameRecord(
        angle_deg=angle_deg,
        angle_label=label,
        image_path=image_path,
        note=note,
    )
    session.frames.append(rec)
    return rec
