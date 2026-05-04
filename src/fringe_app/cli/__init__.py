"""CLI command package.

Stable entrypoints live in :mod:`fringe_app.cli.commands`.
"""

from .commands import (  # noqa: F401
    build_parser,
    main,
    cmd_calibrate_camera,
    cmd_calibrate_projector,
    cmd_run_2d_defect,
    cmd_scan,
    cmd_phase,
    cmd_unwrap,
    cmd_score,
    cmd_pipeline_run_3d,
    cmd_projector_calibrate_v2,
)

__all__ = [
    "build_parser",
    "main",
    "cmd_calibrate_camera",
    "cmd_calibrate_projector",
    "cmd_run_2d_defect",
    "cmd_scan",
    "cmd_phase",
    "cmd_unwrap",
    "cmd_score",
    "cmd_pipeline_run_3d",
    "cmd_projector_calibrate_v2",
]
