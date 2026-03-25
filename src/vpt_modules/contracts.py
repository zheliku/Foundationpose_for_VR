from __future__ import annotations

from typing import Protocol

from .types import DepthResult, MaskResult, PoseResult, RGBDFrame


class RGBDSource(Protocol):
    def start(self) -> None: ...

    def read(self) -> RGBDFrame | None: ...

    def stop(self) -> None: ...


class Segmenter(Protocol):
    def segment(self, frame: RGBDFrame) -> MaskResult: ...


class StereoDepthEstimator(Protocol):
    def estimate(
        self, left_bgr, right_bgr, fx: float, baseline_m: float
    ) -> DepthResult: ...


class PoseEstimator(Protocol):
    def initialize(self, frame: RGBDFrame, mask_u8) -> PoseResult: ...

    def track(self, frame: RGBDFrame) -> PoseResult: ...
