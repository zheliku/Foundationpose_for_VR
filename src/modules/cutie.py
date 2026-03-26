from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from VOT import Cutie


@dataclass(slots=True)
class CutieConfig:
    cutie_seg_threshold: float = 0.1
    erosion_size: int = 5


class CutieTracker2D:
    def __init__(self, config: CutieConfig) -> None:
        self.config = config
        self._tracker = Cutie(
            cutie_seg_threshold=config.cutie_seg_threshold,
            erosion_size=config.erosion_size,
        )
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self, color_bgr: np.ndarray, mask_u8: np.ndarray) -> list[int]:
        bbox = self._tracker.initialize(color_bgr, init_info={"mask": mask_u8})
        self._initialized = True
        return [int(v) for v in bbox]

    def track(self, color_bgr: np.ndarray) -> list[int]:
        if not self._initialized:
            return [-1, -1, 0, 0]
        bbox = self._tracker.track(color_bgr)
        return [int(v) for v in bbox]

    @staticmethod
    def bbox_center_xy(bbox_xywh: list[int]) -> tuple[float, float] | None:
        x, y, w, h = bbox_xywh
        if w <= 0 or h <= 0:
            return None
        return (float(x) + float(w) * 0.5, float(y) + float(h) * 0.5)
