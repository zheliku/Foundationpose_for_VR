from __future__ import annotations

import json

import numpy as np

from .base_encoder import BaseEncoder


class PoseServerEncoder(BaseEncoder):
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
        pose_matrix = None
        if pose_4x4 is not None:
            pose_matrix = np.asarray(pose_4x4, dtype=np.float64).reshape(4, 4).tolist()

        payload = {
            "timestamp_ms": float(timestamp_ms),
            "stage": int(stage),
            "phase": str(phase),
            "det_count": int(det_count),
            "depth_valid_ratio": float(depth_valid_ratio),
            "fps": float(fps),
            "has_pose": pose_matrix is not None,
            "pose_matrix": pose_matrix,
            "timing_ms": {
                "yolo": float(timing_ms.get("yolo", 0.0)),
                "depth": float(timing_ms.get("depth", 0.0)),
                "cutie": float(timing_ms.get("cutie", 0.0)),
                "pose": float(timing_ms.get("pose", 0.0)),
            },
        }

        return [json.dumps(payload, ensure_ascii=False).encode("utf-8")]
