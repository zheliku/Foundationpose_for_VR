from __future__ import annotations

from .base_decoder import PayloadDecoder
from ..message.pose_msg import PoseMsg


class PoseDecoder(PayloadDecoder):
    """将单帧 Pose payload 解码为便于 Python 调试使用的字典。

    说明：
    - 主链路中 Pose 通常由 Python 发送、Unity 接收；
    - 该解码器主要用于 Python 侧测试/回环/诊断；
    - 字段名保持与 PoseMsg 一致，避免协议转换时引入歧义。
    """

    def decode(self, payload: bytes) -> dict[str, object] | None:
        # 先复用 PoseMsg 的统一反序列化与基础校验逻辑。
        message = PoseMsg.deserialize(payload)
        if message is None:
            return None

        # 输出普通 dict，便于日志、测试断言和临时脚本直接使用。
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
