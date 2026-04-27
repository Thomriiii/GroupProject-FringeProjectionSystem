"""Capture-stage wrapper for the stable scan pipeline."""

from __future__ import annotations

from typing import Any

from fringe_app.cli import cmd_scan


def run_scan_command(args: Any) -> int:
    """Run the existing scan capture command path unchanged."""
    return int(cmd_scan(args))
