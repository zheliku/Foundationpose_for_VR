"""Quest 相机静态信息编码器。"""

from __future__ import annotations

from .base_encoder import BaseEncoder
from ..message.quest_camera_info_msg import QuestCameraInfoMsg


class CameraInfoEncoder(BaseEncoder):
    """Quest 相机信息 payload 编码器。

    该编码器主要用于 Python 侧测试或回环：
    - 正常 Quest 主链路中，相机信息由 Unity 侧 QuestCameraInfoEncoder 低频发送；
    - 若需要 Python 主动发布/转发 camera_info，可复用本类保持同一 MessagePack 协议。
    """

    def encode(self, msg: QuestCameraInfoMsg, **kwargs: object) -> bytes | None:
        """将 QuestCameraInfoMsg 编码为 MessagePack 字节。"""
        # QuestCameraInfoMsg 是 frozen dataclass，serialize 内部只做字段到 dict 的稳定映射。
        # kwargs 预留给未来扩展（例如压缩/版本号），当前不参与编码。
        return msg.serialize()
