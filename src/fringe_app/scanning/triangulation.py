"""Triangulation/reconstruction wrapper for the stable scan pipeline."""

from __future__ import annotations

from typing import Any

from fringe_app.cli.commands import cmd_reconstruct


def run_reconstruction_command(args: Any) -> int:
    """Run UV triangulation/reconstruction stage unchanged."""
    return int(cmd_reconstruct(args))
