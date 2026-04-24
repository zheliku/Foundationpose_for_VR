"""Quest 相机静态信息解码器。"""

from __future__ import annotations

from .base_decoder import PayloadDecoder
from ..message.quest_camera_info_msg import QuestCameraInfoMsg


class CameraInfoDecoder(PayloadDecoder):
    """Quest 相机信息 payload 解码器。"""

    def decode(self, payload: bytes) -> QuestCameraInfoMsg | None:
        """将单帧 payload 字节解码为 QuestCameraInfoMsg。"""
        return QuestCameraInfoMsg.deserialize(payload)
