from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .base_encoder import BaseEncoder
from ..message.rgbd_msg import RGBDMsg


class RGBDEncoder(BaseEncoder):
    """将彩色图和深度图编码为单帧 RGBD payload。

    编码策略：
    - color 使用 JPEG，减小网络带宽；
    - depth 使用 PNG，保持 uint16 深度值不被有损压缩破坏；
    - 最终再封装为 RGBDMsg 的 MessagePack 字节。
    """

    def encode(
        self,
        color: NDArray[np.uint8],
        depth: NDArray[np.uint16],
        quality: int = 80,
        *args: object,
        **kwargs: object,
    ) -> bytes | None:
        # 彩色图走 JPEG，可通过 quality 权衡画质与带宽。
        success, color_buf = cv2.imencode(
            ".jpg", color, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not success:
            return None

        # 深度图必须走无损格式，避免毫米级/像素级深度值被 JPEG 破坏。
        success, depth_buf = cv2.imencode(".png", depth)
        if not success:
            return None

        # MessagePack 只承载压缩后的 bytes，不直接承载大数组。
        payload = RGBDMsg(
            color_image=color_buf.tobytes(),
            depth_image=depth_buf.tobytes(),
        )
        return payload.serialize()
