"""Quest 相机静态信息编码器。"""

from __future__ import annotations

from .base_encoder import BaseEncoder
from ..message.quest_camera_info_msg import QuestCameraInfoMsg


class CameraInfoEncoder(BaseEncoder):
    """Quest 相机信息 payload 编码器。"""

    def encode(self, msg: QuestCameraInfoMsg, **kwargs: object) -> bytes | None:
        """将 QuestCameraInfoMsg 编码为 MessagePack 字节。"""
        return msg.serialize()
