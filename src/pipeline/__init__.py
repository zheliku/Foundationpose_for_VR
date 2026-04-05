"""Pipeline 包入口：导出构建函数，供外部按需创建 Quest/RealSense Pipeline。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .quest_pipeline import PosePipelineOutput as QuestPosePipelineOutput
    from .quest_pipeline import QuestStereoPosePipeline
    from .realsense_pipeline import PosePipelineOutput as RealSensePosePipelineOutput
    from .realsense_pipeline import RealSenseStereoPosePipeline


def build_realsense_pipeline(args):
    """Lazily import and build RealSense pipeline to avoid optional dependency coupling."""
    from .realsense_pipeline import build_realsense_pipeline as _build

    return _build(args)


def build_quest_pipeline(args):
    """Lazily import and build Quest pipeline to avoid optional dependency coupling."""
    from .quest_pipeline import build_quest_pipeline as _build

    return _build(args)


__all__ = [
    "build_realsense_pipeline",
    "build_quest_pipeline",
]
