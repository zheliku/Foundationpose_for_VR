"""Quest 相机静态信息解码器。"""

from __future__ import annotations

from .base_decoder import PayloadDecoder
from ..message.quest_camera_info_msg import QuestCameraInfoMsg


class CameraInfoDecoder(PayloadDecoder):
    """Quest 相机信息 payload 解码器。

    设计约定：
    - 输入 payload 必须是 QuestCameraInfoMsg.serialize() 产生的 MessagePack 字节；
    - 解码成功后返回不可变 QuestCameraInfoMsg，便于上层安全缓存；
    - 解码失败返回 None，由接收器统计 decode_failed 并丢弃该帧。
    """

    def decode(self, payload: bytes) -> QuestCameraInfoMsg | None:
        """将单帧 payload 字节解码为 QuestCameraInfoMsg。"""
        return QuestCameraInfoMsg.deserialize(payload)
