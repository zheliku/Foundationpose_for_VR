from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .base_decoder import PayloadDecoder
from ..message.stereo import QuestStereoMsg


class StereoDecoder(PayloadDecoder):
    """Quest 双目 payload 解码器（支持可选元数据尾帧）。

    支持格式：
    - [packed_stereo_jpg]
    - [left_jpg, right_jpg]
    - [packed_stereo_jpg, metadata_json]
    - [left_jpg, right_jpg, metadata_json]
    """

    def decode(self, parts: list[bytes]) -> QuestStereoMsg | None:
        """将 multipart 字节解码为 QuestStereoMsg。"""
        message = QuestStereoMsg.from_parts(parts)
        if message is None:
            return None

        if message.packed_image is not None:
            # Packed 模式：先解码整张图，再按左右半幅切分。
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

        if message.left_image is None or message.right_image is None:
            return None

        # Dual 模式：分别解码左右两张图像。
        left = cv2.imdecode(
            np.frombuffer(message.left_image, np.uint8), cv2.IMREAD_COLOR
        )
        right = cv2.imdecode(
            np.frombuffer(message.right_image, np.uint8), cv2.IMREAD_COLOR
        )

        if left is None or right is None:
            return None

        message.left = cast(NDArray[np.uint8], left)
        message.right = cast(NDArray[np.uint8], right)
        return message
