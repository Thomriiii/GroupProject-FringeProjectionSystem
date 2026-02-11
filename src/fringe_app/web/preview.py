"""Preview broadcaster for WebSocket streaming."""

from __future__ import annotations

import threading
from io import BytesIO
from typing import Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image


class PreviewBroadcaster:
    """Thread-safe storage of latest preview JPEG bytes + metadata."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_bytes: Optional[bytes] = None
        self._latest_meta: Optional[Dict[str, Any]] = None

    def update(self, frame: np.ndarray, run_id: str, step: int, total: int) -> None:
        img = Image.fromarray(frame)
        if img.width > 640:
            scale = 640 / float(img.width)
            new_size = (640, int(img.height * scale))
            img = img.resize(new_size)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=75)
        data = buffer.getvalue()
        meta = {"run_id": run_id, "step": step, "total": total}
        with self._lock:
            self._latest_bytes = data
            self._latest_meta = meta

    def get_latest(self) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        with self._lock:
            if self._latest_bytes is None or self._latest_meta is None:
                return None
            return self._latest_bytes, dict(self._latest_meta)
