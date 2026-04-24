from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .base_decoder import PayloadDecoder
from ..message.quest_stereo_msg import QuestStereoMsg


class StereoDecoder(PayloadDecoder):
    """Quest 双目 payload 解码器（单帧 MessagePack 协议）。"""

    def decode(self, payload: bytes) -> QuestStereoMsg | None:
        """将单帧 payload 字节解码为 QuestStereoMsg。"""
        message = QuestStereoMsg.deserialize(payload)
        if message is None:
            return None

        if message.left_image_jpeg is None or message.right_image_jpeg is None:
            return None

        left = cv2.imdecode(
            np.frombuffer(message.left_image_jpeg, np.uint8), cv2.IMREAD_COLOR
        )
        right = cv2.imdecode(
            np.frombuffer(message.right_image_jpeg, np.uint8), cv2.IMREAD_COLOR
        )
        if left is None or right is None:
            return None

        message.left = cast(NDArray[np.uint8], left)
        message.right = cast(NDArray[np.uint8], right)
        return message
