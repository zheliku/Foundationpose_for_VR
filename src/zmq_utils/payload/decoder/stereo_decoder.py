from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .base_decoder import PayloadDecoder
from ..message.stereo import QuestStereoMsg


class StereoDecoder(PayloadDecoder):
    """Quest 双目 payload 解码器（单帧 MessagePack 协议）。"""

    def decode(self, payload: bytes) -> QuestStereoMsg | None:
        """将单帧 payload 字节解码为 QuestStereoMsg。"""
        message = QuestStereoMsg.deserialize(payload)
        if message is None:
            return None

        if message.packed_image is not None:
            # 当前协议为左右拼接 JPEG：先整体解码，再按左右半幅切分。
            packed = cv2.imdecode(
                np.frombuffer(message.packed_image, np.uint8), cv2.IMREAD_COLOR
            )
            if packed is None:
                return None

            _, width = packed.shape[:2]
            if width < 2:
                return None

            mid = width // 2
            left = packed[:, :mid]
            right = packed[:, mid:]

            message.left = cast(NDArray[np.uint8], left)
            message.right = cast(NDArray[np.uint8], right)
            return message

        return None
