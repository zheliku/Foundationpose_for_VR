"""Pipeline 包入口：导出构建函数，供外部按需创建 Quest/RealSense Pipeline。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .quest_pipeline import PosePipelineOutput as QuestPosePipelineOutput
    from .quest_pipeline import QuestStereoPosePipeline
    from .realsense_pipeline import PosePipelineOutput as RealSensePosePipelineOutput
    from .realsense_pipeline import RealSenseStereoPosePipeline


def build_realsense_pipeline(args):
    """懒加载并构建 RealSense Pipeline，避免导入包时强制加载 RealSense/深度学习依赖。

    设计原因：
    - RealSense、YOLO、FFS、FoundationPose 都属于重依赖；
    - 外部只想查看包导出或构建 Quest Pipeline 时，不应因 RealSense SDK 缺失而失败。
    """
    from .realsense_pipeline import build_realsense_pipeline as _build

    return _build(args)


def build_quest_pipeline(args):
    """懒加载并构建 Quest Pipeline，避免导入包时提前初始化模型与网络依赖。"""
    from .quest_pipeline import build_quest_pipeline as _build

    return _build(args)


__all__ = [
    "build_realsense_pipeline",
    "build_quest_pipeline",
]
