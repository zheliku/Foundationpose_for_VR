from __future__ import annotations

import numpy as np

from .base_encoder import BaseEncoder
from ..message.pose import PoseMsg


class PoseEncoder(BaseEncoder):
    """Encode pose_server output to one-part JSON payload."""

    def encode(
        self,
        *,
        timestamp_ms: float,
        stage: int,
        phase: str,
        det_count: int,
        depth_valid_ratio: float,
        fps: float,
        timing_ms: dict[str, float],
        pose_4x4: np.ndarray | None,
    ) -> list[bytes]:
        message = PoseMsg.from_runtime(
            timestamp_ms=timestamp_ms,
            stage=stage,
            phase=phase,
            det_count=det_count,
            depth_valid_ratio=depth_valid_ratio,
            fps=fps,
            timing_ms=timing_ms,
            pose_4x4=np.asarray(pose_4x4) if pose_4x4 is not None else None,
        )
        return [message.to_json_bytes()]
