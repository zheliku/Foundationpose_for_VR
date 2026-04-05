"""Compatibility wrapper for the new structured RealSense pipeline API."""

from __future__ import annotations

from pipeline.realsense_pipeline import (  # noqa: F401
    RealSenseStereoPosePipeline,
    build_arg_parser,
    build_realsense_pipeline,
    main,
    parse_args,
    run_realsense_pipeline,
    validate_paths,
)

__all__ = [
    "RealSenseStereoPosePipeline",
    "build_arg_parser",
    "build_realsense_pipeline",
    "parse_args",
    "run_realsense_pipeline",
    "validate_paths",
    "main",
]


if __name__ == "__main__":
    main()
