from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class PipelineStatus(Enum):
    DETECTING = "detecting"
    TRACKING = "tracking"
    LOST = "lost"


@dataclass(slots=True)
class RGBDFrame:
    color_bgr: NDArray[np.uint8]
    depth_m: NDArray[np.float64]
    timestamp_s: float


@dataclass(slots=True)
class MaskResult:
    mask_u8: NDArray[np.uint8] | None
    score: float | None
    label: str | None


@dataclass(slots=True)
class DepthResult:
    depth_m: NDArray[np.float64]
    valid_ratio: float
    meta: dict[str, float]


@dataclass(slots=True)
class PoseResult:
    pose_4x4: NDArray[np.float64] | None
    vis_bgr: NDArray[np.uint8]


@dataclass(slots=True)
class PipelineResult:
    status: PipelineStatus
    pose: NDArray[np.float64] | None
    vis_bgr: NDArray[np.uint8]
    mask_u8: NDArray[np.uint8] | None
    debug: dict[str, float | str]
