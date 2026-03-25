from __future__ import annotations

import time

from vpt_modules.contracts import PoseEstimator, Segmenter
from vpt_modules.types import PipelineResult, PipelineStatus, RGBDFrame


class FirstFrameMaskPipeline:
    def __init__(self, segmenter: Segmenter, pose_estimator: PoseEstimator) -> None:
        self.segmenter = segmenter
        self.pose_estimator = pose_estimator
        self._initialized = False

    def process(self, frame: RGBDFrame) -> PipelineResult:
        t0 = time.perf_counter()

        if not self._initialized:
            mask_result = self.segmenter.segment(frame)
            if mask_result.mask_u8 is None:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return PipelineResult(
                    status=PipelineStatus.DETECTING,
                    pose=None,
                    vis_bgr=frame.color_bgr,
                    mask_u8=None,
                    debug={"elapsed_ms": elapsed_ms, "stage": "segment"},
                )

            pose_result = self.pose_estimator.initialize(frame, mask_result.mask_u8)
            self._initialized = pose_result.pose_4x4 is not None
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            h, w = frame.color_bgr.shape[:2]
            mask_area_ratio = float((mask_result.mask_u8 > 0).sum()) / float(h * w)
            return PipelineResult(
                status=(
                    PipelineStatus.TRACKING
                    if self._initialized
                    else PipelineStatus.LOST
                ),
                pose=pose_result.pose_4x4,
                vis_bgr=pose_result.vis_bgr,
                mask_u8=mask_result.mask_u8,
                debug={
                    "elapsed_ms": elapsed_ms,
                    "stage": "init",
                    "mask_score": (
                        mask_result.score if mask_result.score is not None else -1.0
                    ),
                    "mask_area_ratio": mask_area_ratio,
                },
            )

        pose_result = self.pose_estimator.track(frame)
        if pose_result.pose_4x4 is None:
            self._initialized = False
            status = PipelineStatus.LOST
        else:
            status = PipelineStatus.TRACKING

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PipelineResult(
            status=status,
            pose=pose_result.pose_4x4,
            vis_bgr=pose_result.vis_bgr,
            mask_u8=None,
            debug={"elapsed_ms": elapsed_ms, "stage": "track"},
        )
