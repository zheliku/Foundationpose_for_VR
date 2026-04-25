from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .base_decoder import PayloadDecoder
from ..message.rgbd_msg import RGBDMsg


class RGBDDecoder(PayloadDecoder):
    """将单帧 RGBD payload 解码为 OpenCV 可用的彩色图与深度图。

    协议约定：
    - 彩色图通常使用 JPEG 编码，解码后为 BGR uint8；
    - 深度图通常使用 PNG 无损编码，解码后统一转为 uint16；
    - 返回 None 表示消息结构非法或图像解码失败。
    """

    def decode(
        self, payload: bytes
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint16]] | None:
        # 第一步：先反序列化 MessagePack，取出图像压缩字节。
        message = RGBDMsg.deserialize(payload)
        if message is None:
            return None

        # 第二步：分别用 OpenCV 解码彩色图和深度图。
        color = cv2.imdecode(
            np.frombuffer(message.color_image, np.uint8), cv2.IMREAD_COLOR
        )
        depth = cv2.imdecode(
            np.frombuffer(message.depth_image, np.uint8), cv2.IMREAD_UNCHANGED
        )

        # 任意一路解码失败都视为整帧无效，避免上层处理半帧数据。
        if color is None or depth is None:
            return None

        return cast(NDArray[np.uint8], color), depth.astype(np.uint16)
