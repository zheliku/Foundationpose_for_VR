from __future__ import annotations

from .base_decoder import PayloadDecoder
from ..message.pose_msg import PoseMsg


class PoseDecoder(PayloadDecoder):
    """Decode single pose payload bytes."""

    def decode(self, payload: bytes) -> dict[str, object] | None:
        message = PoseMsg.deserialize(payload)
        if message is None:
            return None

        return {
            "timestamp_ms": float(message.timestamp_ms),
            "stage": int(message.stage),
            "phase": str(message.phase),
            "det_count": int(message.det_count),
            "depth_valid_ratio": float(message.depth_valid_ratio),
            "fps": float(message.fps),
            "has_pose": bool(message.has_pose),
            "pose_matrix_flat": message.pose_matrix_flat,
            "yolo_ms": float(message.yolo_ms),
            "depth_ms": float(message.depth_ms),
            "cutie_ms": float(message.cutie_ms),
            "pose_ms": float(message.pose_ms),
        }
