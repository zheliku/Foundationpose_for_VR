from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import msgpack


@dataclass(frozen=True)
class PoseMsg:
    """Pose 传输消息：用于 pose 编码器与解码器之间统一交换。"""

    timestamp_ms: float  # 该条消息时间戳（毫秒）。
    frame_id: int  # 对应输入双目帧号（用于 Unity 本地匹配发送时参考位姿）。
    stage: int  # Pipeline 阶段编号。
    phase: str  # 阶段名称（例如 REGISTER / TRACK）。
    det_count: int  # 检测框数量。
    depth_valid_ratio: float  # 深度有效像素占比（0~1）。
    fps: float  # 实时帧率估计值。
    has_pose: bool  # 是否存在有效位姿。
    pose_matrix_flat: list[float] | None  # 4x4 位姿矩阵展平后的 16 个元素（行优先）。
    yolo_ms: float  # YOLO 阶段耗时（毫秒）。
    depth_ms: float  # 深度估计阶段耗时（毫秒）。
    cutie_ms: float  # Cutie 跟踪阶段耗时（毫秒）。
    pose_ms: float  # Pose 求解阶段耗时（毫秒）。

    @classmethod
    def deserialize(cls, payload: bytes) -> PoseMsg | None:
        """从 MessagePack 字节反序列化为 Pose 消息对象。"""
        try:
            data = msgpack.unpackb(payload, raw=False, strict_map_key=False)
            if not isinstance(data, dict):
                return None

            pose_matrix_flat_raw = data.get("pose_matrix_flat")
            pose_matrix_flat: list[float] | None = None
            if pose_matrix_flat_raw is not None:
                pose_matrix_flat = [float(item) for item in pose_matrix_flat_raw]
                if len(pose_matrix_flat) != 16:
                    return None

            return cls(
                timestamp_ms=float(data.get("timestamp_ms", 0.0)),
                frame_id=int(data.get("frame_id", 0)),
                stage=int(data.get("stage", 0)),
                phase=str(data.get("phase", "")),
                det_count=int(data.get("det_count", 0)),
                depth_valid_ratio=float(data.get("depth_valid_ratio", 0.0)),
                fps=float(data.get("fps", 0.0)),
                has_pose=bool(data.get("has_pose", pose_matrix_flat is not None)),
                pose_matrix_flat=pose_matrix_flat,
                yolo_ms=float(data.get("yolo_ms", 0.0)),
                depth_ms=float(data.get("depth_ms", 0.0)),
                cutie_ms=float(data.get("cutie_ms", 0.0)),
                pose_ms=float(data.get("pose_ms", 0.0)),
            )
        except (TypeError, ValueError):
            return None

    def serialize(self) -> bytes:
        """编码为 MessagePack 字节。"""
        payload = {
            "timestamp_ms": float(self.timestamp_ms),
            "frame_id": int(self.frame_id),
            "stage": int(self.stage),
            "phase": str(self.phase),
            "det_count": int(self.det_count),
            "depth_valid_ratio": float(self.depth_valid_ratio),
            "fps": float(self.fps),
            "has_pose": bool(self.has_pose),
            "pose_matrix_flat": self.pose_matrix_flat,
            "yolo_ms": float(self.yolo_ms),
            "depth_ms": float(self.depth_ms),
            "cutie_ms": float(self.cutie_ms),
            "pose_ms": float(self.pose_ms),
        }
        return cast(bytes, msgpack.packb(payload, use_bin_type=True))
