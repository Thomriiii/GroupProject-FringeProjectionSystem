"""Phase decode wrapper for the stable scan pipeline."""

from __future__ import annotations

from typing import Any

from fringe_app.cli import cmd_phase


def run_phase_decode_command(args: Any) -> int:
    """Run phase decoding via existing command implementation unchanged."""
    return int(cmd_phase(args))
