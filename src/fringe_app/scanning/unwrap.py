"""Unwrap wrapper for the stable scan pipeline."""

from __future__ import annotations

from typing import Any

from fringe_app.cli import cmd_unwrap


def run_unwrap_command(args: Any) -> int:
    """Run temporal unwrap via existing command implementation unchanged."""
    return int(cmd_unwrap(args))
