"""Compatibility wrapper for the new structured Quest pipeline API."""

from __future__ import annotations

from pipeline.quest_pipeline import (  # noqa: F401
    QuestStereoPosePipeline,
    StereoCalibration,
    build_arg_parser,
    build_quest_pipeline,
    main,
    parse_args,
    run_quest_pipeline,
    validate_paths,
)

__all__ = [
    "StereoCalibration",
    "QuestStereoPosePipeline",
    "build_arg_parser",
    "build_quest_pipeline",
    "parse_args",
    "run_quest_pipeline",
    "validate_paths",
    "main",
]


if __name__ == "__main__":
    main()
