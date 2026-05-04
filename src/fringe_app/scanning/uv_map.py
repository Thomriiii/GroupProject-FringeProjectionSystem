"""UV mapping wrapper for the stable scan pipeline."""

from __future__ import annotations

from typing import Any

from fringe_app.cli.commands import cmd_pipeline_run_uv


def run_uv_map_command(args: Any) -> int:
    """Run capture->phase->unwrap->UV pipeline stage unchanged."""
    return int(cmd_pipeline_run_uv(args))
