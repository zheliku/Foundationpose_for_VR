from __future__ import annotations

import numpy as np

from .base_encoder import BaseEncoder
from ..message.pose_msg import PoseMsg


class PoseEncoder(BaseEncoder):
    """Encode pose_server output to single payload bytes."""

    def encode(
        self,
        *,
        timestamp_ms: float,
        frame_id: int,
        stage: int,
        phase: str,
        det_count: int,
        depth_valid_ratio: float,
        fps: float,
        timing_ms: dict[str, float],
        pose_4x4: np.ndarray | None,
    ) -> bytes | None:
        timing = timing_ms or {}
        pose_matrix_flat: list[float] | None = None
        if pose_4x4 is not None:
            pose_matrix_flat = [
                float(item)
                for item in np.asarray(pose_4x4, dtype=np.float64).reshape(16).tolist()
            ]

        message = PoseMsg(
            timestamp_ms=float(timestamp_ms),
            frame_id=int(frame_id),
            stage=int(stage),
            phase=str(phase),
            det_count=int(det_count),
            depth_valid_ratio=float(depth_valid_ratio),
            fps=float(fps),
            has_pose=pose_matrix_flat is not None,
            pose_matrix_flat=pose_matrix_flat,
            yolo_ms=float(timing.get("yolo", 0.0)),
            depth_ms=float(timing.get("depth", 0.0)),
            cutie_ms=float(timing.get("cutie", 0.0)),
            pose_ms=float(timing.get("pose", 0.0)),
        )
        return message.serialize()
